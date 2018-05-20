from gensim.models.fasttext import FastText
from functools import reduce
from collections import Counter
import nltk
import csv
import pickle
from math import log10

def gatherTitlesLyrics(a,b):
  art,title,url,lyr = b
  a['Titles'].append(title)
  a['Lyrics'].append(lyr)
  return a

def loadTitlesLyrics():
  with open("./data/songdata.csv") as lyrics:
    return reduce(gatherTitlesLyrics,csv.reader(lyrics),{"Titles":[],"Lyrics":[]})

def updateCounts(a,b):
  for u in set(b):
    a[u] = a.get(u,0) + 1
  return a

def IdfTokenized(documents):
  doc_freqs = reduce(updateCounts,documents,{})
  idfs = {k:log10(1 + len(documents)/v) for k,v in doc_freqs.items()}
  return idfs
    
if __name__ == "__main__":
  d = loadTitlesLyrics()
  d["Lyrics"] = [nltk.tokenize.word_tokenize(l) for l in d["Lyrics"]]
  lyricVectors = FastText(d['Lyrics'],min_count=2,workers=2,size=100)
  lyricVectors.save("LyricVectors.pkl")
  idfs = IdfTokenized(d["Lyrics"])
  pickle.dump(idfs,open("LyricTokenIDFs.pkl","wb"))
