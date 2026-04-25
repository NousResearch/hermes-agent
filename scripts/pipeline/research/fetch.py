#!/usr/bin/env python3
# ============================================================================
# Layer 1: 研究資料收集
# 每週一 10:00 執行
# 職責：從多個來源收集市場研究資料
# ============================================================================

import subprocess
import re
import json
from datetime import datetime
import os

CACHE_DIR = "/home/ubuntu/.openclaw/cache/research"
LOG_FILE = "/tmp/pipeline-research-fetch.log"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")

os.makedirs(CACHE_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{TIMESTAMP}] {msg}\n")

log("=== 研究資料收集 ===")

# 研究來源
SOURCES = [
    ("HackerNews", "https://hnrss.org/frontpage"),
    ("Reddit-tech", "https://www.reddit.com/r/technology/.rss"),
    ("Reddit-investing", "https://www.reddit.com/r/investing/.rss"),
]

all_data = []

for name, url in SOURCES:
    log(f"抓取: {name}")
    
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '15', '-A', 'Mozilla/5.0', url],
            capture_output=True, text=True, timeout=20
        )
        
        xml = result.stdout
        
        # 解析標題和連結
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', xml)
        links = re.findall(r'<link>(https?://[^<]+)</link>', xml)
        
        items = []
        for i, title in enumerate(titles[1:6]):  # 每源取5則
            link = links[i] if i < len(links) else ""
            items.append({
                "title": title.strip(),
                "link": link,
                "source": name
            })
        
        all_data.append({
            "source": name,
            "items": items,
            "fetched_at": datetime.now().isoformat()
        })
        
        log(f"  抓到 {len(items)} 則")
        
    except Exception as e:
        log(f"  錯誤: {e}")

# 保存
output_file = f"{CACHE_DIR}/raw-{TIMESTAMP}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

log(f"=== 收集完成: {len(all_data)} 個來源 ===")
print(f"收集了 {len(all_data)} 個來源的資料")