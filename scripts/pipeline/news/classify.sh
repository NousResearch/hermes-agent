#!/bin/bash
# ============================================================================
# Layer 2: 新聞分類
# 每小時執行 (收集後立即)
# 職責：讀取 raw-*.json，關鍵字分類，寫入 classified.json
# 特性：純 Shell + Python，無 AI，快速執行
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/news"
LOG_FILE="/tmp/pipeline-news-classify.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 開始分類 ==="

# 找最新的 raw 快取
RAW_FILE=$(ls -t "$CACHE_DIR"/raw-*.json 2>/dev/null | head -1)

if [ -z "$RAW_FILE" ]; then
    log "無原始新聞資料"
    exit 0
fi

log "使用: $RAW_FILE"

# 分類
python3 << PYEOF
import json
from datetime import datetime

with open("$RAW_FILE") as f:
    news = json.load(f)

# 關鍵字分類
CATEGORIES = {
    "🔥 重要": ["台積電", "華為", "川普", "戰爭", "金融風暴", "病毒", "肺炎", "核", "導彈"],
    "💰 金融": ["利率", "匯率", "台幣", "央行", " Fed", "聯準會", "升息", "降息", "QE"],
    "📈 股市": ["股市", "ETF", "漲跌", "IPO", "上市", "下市", "跌深", "反彈", "多头", "空头"],
    "🏭 總經": ["GDP", "CPI", "通膨", "景氣", "出口", "進口", "貿易戰", "關稅", "制裁"],
    "🌍 國際": ["美中", "中美", "伊朗", "北韓", "俄羅斯", "烏克蘭", "以色列", "沙烏地"],
    "🏛️ 政治": ["總統", "選舉", "立法院", "罷免", "公投", "內閣", "政黨"],
    "🏠 房產": ["房價", "房市", "打房", "實價登錄", "都更", "建商"],
    "💱 匯市": ["美元", "台幣", "日圓", "歐元", "人民幣", "韓元", "貶值", "升值"],
}

# 分類
for item in news:
    title = item.get("title", "")
    item["category"] = "📋 其他"
    
    for cat, keywords in CATEGORIES.items():
        if any(k in title for k in keywords):
            item["category"] = cat
            break

# 統計
stats = {}
for item in news:
    cat = item.get("category", "📋 其他")
    stats[cat] = stats.get(cat, 0) + 1

# 輸出分類結果
output = {
    "news": news,
    "stats": stats,
    "classified_at": datetime.now().isoformat(),
    "source_file": "$RAW_FILE"
}

with open("$CACHE_DIR/classified.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("分類完成:")
for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

PYEOF

log "=== 分類完成 ==="