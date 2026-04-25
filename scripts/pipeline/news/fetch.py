#!/usr/bin/env python3
# ============================================================================
# Layer 1: 新聞收集
# 每小時執行
# 職責：抓取 RSS，寫入 cache/news/raw-*.json
# 特性：純 Python，無 AI，快速執行
# ============================================================================

import subprocess
import re
import json
from datetime import datetime
import sys
import os

CACHE_DIR = "/home/ubuntu/.openclaw/cache/news"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")
LOG_FILE = "/tmp/pipeline-news-fetch.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{TIMESTAMP}] {msg}\n")

log("=== 開始收集新聞 ===")

# RSS 來源
RSS_SOURCES = [
    ("Yahoo財經", "https://tw.news.yahoo.com/rss"),
    ("東森", "https://news.ebc.net.tw/rss"),
    ("中央社", "https://www.cna.com.tw/rss/all.aspx"),
]

all_news = []

for name, url in RSS_SOURCES:
    log(f"抓取: {name}")
    
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '15', '-A', 'Mozilla/5.0', url],
            capture_output=True, text=True, timeout=20
        )
        
        xml = result.stdout
        
        # 解析標題
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', xml)
        # 解析連結
        links = re.findall(r'<link>(https?://[^<]+)</link>', xml)
        
        log(f"  找到 {len(titles)} 則")
        
        for i, title in enumerate(titles[1:], 0):  # 跳過第一個 (頻道標題)
            if i < len(links):
                link = links[i]
            else:
                link = ""
            
            all_news.append({
                "source": name,
                "title": title.strip(),
                "link": link,
                "time": datetime.now().isoformat()
            })
            
    except Exception as e:
        log(f"  錯誤: {e}")

log(f"總共收集 {len(all_news)} 則")

# 寫入快取
os.makedirs(CACHE_DIR, exist_ok=True)
cache_file = f"{CACHE_DIR}/raw-{TIMESTAMP}.json"

with open(cache_file, "w", encoding="utf-8") as f:
    json.dump(all_news, f, ensure_ascii=False, indent=2)

log(f"寫入快取: {cache_file}")

# 清理舊的快取 (保留最近 24 小時)
try:
    import time
    now = time.time()
    for f in os.listdir(CACHE_DIR):
        if f.startswith("raw-") and f.endswith(".json"):
            filepath = os.path.join(CACHE_DIR, f)
            if now - os.path.getmtime(filepath) > 86400:  # 24 小時
                os.remove(filepath)
                log(f"刪除舊快取: {f}")
except Exception as e:
    log(f"清理錯誤: {e}")

log("=== 收集完成 ===")
print(f"收集了 {len(all_news)} 則新聞")