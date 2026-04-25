#!/bin/bash
# ============================================================================
# Layer 3: 研究報告格式化
# 每週一 11:00 執行
# 職責：讀取研究資料，格式化，發送到 Discord
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/research"
LOG_FILE="/tmp/pipeline-research-format.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

# Webhook
WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
print(config['webhooks']['stock_monitor']['url'])
" 2>/dev/null)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 研究報告格式化 ==="

# 找最新的研究資料
RAW_FILE=$(ls -t "$CACHE_DIR"/raw-*.json 2>/dev/null | head -1)

if [ -z "$RAW_FILE" ]; then
    log "無研究資料"
    exit 0
fi

format_output() {
    python3 << PYEOF
import json
from datetime import datetime

with open("$RAW_FILE") as f:
    data = json.load(f)

now = datetime.now()
weekday = now.strftime("%A")

msg = f"🔬 **市場研究 {now.strftime('%Y-%m-%d')}**\n"
msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"

# 關鍵字分類
keywords = {
    "AI/科技": ["AI", "artificial intelligence", "machine learning", "OpenAI", "Google", "Meta", "Microsoft", "晶片", "半導體"],
    "投資": ["stock", "market", "ETF", "invest", "dividend", "revenue", "earnings", "財報"],
    "創業": ["startup", "funding", "Series", "launch", "product", "release", "創業"],
    "趨勢": ["trend", "future", "prediction", "growth", "趨勢", "預測"]
}

for source_data in data:
    source = source_data.get("source", "")
    items = source_data.get("items", [])
    
    msg += f"📰 **{source}**\n"
    
    for item in items[:3]:
        title = item.get("title", "")[:70]
        link = item.get("link", "")
        
        # 簡單關鍵字標記
        tags = []
        for cat, kws in keywords.items():
            if any(k.lower() in title.lower() for k in kws):
                tags.append(cat.split("/")[0])
        
        tag_str = " ".join([f"`{t}`" for t in tags[:2]]) if tags else ""
        
        msg += f"• {title}...\n"
        if tag_str:
            msg += f"  {tag_str}\n"
    
    msg += "\n"

msg += "---\n"
msg += f"_🕐 更新於 {now.strftime('%H:%M')}_"

print(msg)
PYEOF
}

MSG=$(format_output)

# 發送
response=$(echo "$MSG" | jq -Rs '{content: .}' | curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @- \
    "$WEBHOOK_URL")

if [ $? -eq 0 ]; then
    log "✅ 發送成功"
    echo "✅ Research report sent"
else
    log "❌ 發送失敗"
    echo "❌ Failed"
fi

log "=== 完成 ==="