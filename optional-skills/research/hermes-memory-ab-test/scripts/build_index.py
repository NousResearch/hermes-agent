#!/usr/bin/env python3
"""
Entity index builder. Run once after storing memory to update the sidecar index.
"""
import argparse, json, re, sqlite3
from pathlib import Path
import jieba.posseg as pseg

STOPWORDS = set("的了在是我有和就不人都一个上也很到说要去你会着没看好自己这".strip())
EN_SW = {"the","a","an","is","are","was","were","do","does","did","has","have",
         "had","will","would","could","should","may","might","can","what","why",
         "how","when","where","who","which"}

def extract(text):
    seen = set()
    results = []
    for word, flag in pseg.cut(text):
        w = word.strip()
        if len(w) < 2 or w.lower() in EN_SW or w in STOPWORDS:
            continue
        key = w.lower()
        if key in seen:
            continue
        if flag == 'eng' and w[0].isupper():
            seen.add(key); results.append((w, 'proper_noun'))
        elif flag.startswith(('nr','ns','nt','nz')):
            seen.add(key); results.append((w, {'nr':'person','ns':'place','nt':'org','nz':'proper_noun'}.get(flag[:2],'entity')))
        elif flag.startswith(('n','v','a','l')):
            seen.add(key); results.append((w, 'concept'))
    return results

def build_index(conversation, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS entities (entity TEXT, turn_idx INT, content TEXT, tag TEXT)")
    conn.execute("DELETE FROM entities")
    for idx, turn in enumerate(conversation):
        for entity, tag in extract(turn):
            conn.execute("INSERT INTO entities VALUES (?, ?, ?, ?)",
                         (entity.lower(), idx, turn[:200], tag))
    conn.commit(); conn.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(Path.home()/".hermes/entity_index/index.db"))
    p.add_argument("--add", help="conversation turn text to index")
    args = p.parse_args()
    if args.add:
        build_index([args.add], args.db)
