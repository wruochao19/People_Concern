# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:59:17 2019

@author: Wruoc
"""
def getStopWords():
    import re
    wordsStr = open('stopWords.txt').read()
    listOfTokens = re.split('\s',wordsStr)
    listOfWords = set(listOfTokens)
    return listOfWords
#*************************************************************************************************8
def getTopWords(ny,sf):
    #import operator
    #vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];topSF=[]
    N = []; S = []
    for i in range(len(p0V)):
        if p0V[i]>-6.0:topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0:topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print ("LA**LA**LA**LA**LA**LA**LA**LA**LA**LA**LA**LA**LA**LA**")
    for item in sortedSF:
        S.append(item[0])
        #print (item[0])
    print (S)
    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        N.append(item[0])
        #print (item[0])
    print (N)
#*********************************************************************************************************************
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
#****************************************************************************************************8
def trainNBO(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)   #计算p(w0|1)p(w1|1),避免其中一个概率值为0，最后的乘积为0
    p0Demo=2.0;p1Demo=2.0  #初始化概率
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
               p1Num+=trainMatrix[i]
               p1Demo+=sum(trainMatrix[i])
        else:
               p0Num+=trainMatrix[i]
               p0Demo+=sum(trainMatrix[i])
    #p1Vect=p1Num/p1Demo
    #p0Vect=p0Num/p0Demo
    p1Vect=np.log(p1Num/p1Demo) #计算p(w0|1)p(w1|1)时，大部分因子都非常小，程序会下溢出或得不到正确答案（相乘许多很小数，最后四舍五入会得到0）
    p0Vect=np.log(p0Num/p0Demo)
    return p0Vect,p1Vect,pAbusive
#****************************************************************************************************
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)  #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
                returnVec[vocabList.index(word)]+=1
    return returnVec
#************************************************************************************************
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}
    for token in vocabList:  #遍历词汇表中的每个词
        freqDict[token]=fullText.count(token)  #统计每个词在文本中出现的次数
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)  #根据每个词出现的次数从高到底对字典进行排序
    return sortedFreq[:30]
#************************************************************************************************

def createVocabList(dataSet):
    vocabSet=set([])    #创建一个空集
    for document in dataSet:
        vocabSet=vocabSet|set(document)   #创建两个集合的并集
    return list(vocabSet)
#************************************************************************************************
def textParse(bigString):
    import re
    clean_string = re.compile(r'[^a-zA-z]|\d')
    list_string = clean_string.split(bigString)
    list_string_lower = [string.lower() for string in list_string if len(list_string) > 0]
    return list_string_lower
#*******************************************************************************************************
import feedparser
import random
import numpy as np
ny = feedparser.parse('https://newyork.craigslist.org/search/biz?format=rss')
sf = feedparser.parse('https://sfbay.craigslist.org/search/biz?format=rss')
feed1 = ny
feed0 = sf
docList=[];classList=[];fullText=[]
minLen=min(len(feed1['entries']),len(feed0['entries']))
#print('num of information from RSS feed is',minLen)
#print(feed2['entries'][24]['title'])
#print(feed2_1['entries'][0]['title'])

#print(len(feed2_1['entries']))
#print(len(feed2['feed']['title']))
#print(feed2['feed']['title'])
#for post in feed2.feed:
    #print(post.title)
for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['title'])#每次访问一条RSS
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0['entries'][i]['title'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
vocabList=createVocabList(docList)
#top30Words=calcMostFreq(vocabList,fullText)
#print (top30Words)
#for pairW in top30Words:
    #if pairW[0] in vocabList:
        #vocabList.remove(pairW[0])
stopWords = getStopWords()
for words in stopWords:
    if words in vocabList:
        vocabList.remove(words)
top30Words=calcMostFreq(vocabList,fullText)
print(top30Words)
trainingSet = list(range(2*minLen))
testSet = []
for i in range(5):
        randIndex=int(random.uniform(0,len(trainingSet)))
        #print('sss',randIndex)
        testSet.append(trainingSet[randIndex])
        #print(testSet)
        #del(trainingSet[randIndex])
        #print(trainingSet)
trainMat = []
trainClasses = []
for docIndex in trainingSet:
    #print('docIndex',docIndex)
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
p0V,p1V,pSpam=trainNBO(np.array(trainMat),np.array(trainClasses))
errorCount=0
for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
print('the error rate is:',float(errorCount)/len(testSet))

def craigslist(*keywords):
    vocSet = set(keywords)
    wordVector = bagOfWords2VecMN(vocabList, vocSet)
    region  = classifyNB(np.array(wordVector),p0V,p1V,pSpam)
    if region == 1:
        print('NY')
    else:
        print('SF')
    