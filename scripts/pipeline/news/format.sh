#!/bin/bash
# ============================================================================
# Layer 3: 新聞格式化輸出
# 07:00, 13:00, 20:00 執行
# 職責：讀取 classified.json，格式化，發送到 Discord
# 特性：純 Shell，無 AI，快速執行
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/news"
LOG_FILE="/tmp/pipeline-news-format.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

# 讀取 webhook URL
WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
print(config['webhooks'].get('news_brief', {}).get('url', ''))
" 2>/dev/null)

# 如果 news_brief 沒有設定，使用 stock_monitor
if [ -z "$WEBHOOK_URL" ]; then
    WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
print(config['webhooks']['stock_monitor']['url'])
")
fi

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 開始格式化輸出 ==="

# 檢查 webhook
if [ -z "$WEBHOOK_URL" ]; then
    log "錯誤: 無法取得 Webhook URL"
    exit 1
fi

# 格式化輸出
format_output() {
    python3 << PYEOF
import json
from datetime import datetime

cache_dir = "$CACHE_DIR"

try:
    with open(f"{cache_dir}/classified.json") as f:
        data = json.load(f)
except:
    print("無法讀取分類資料")
    exit(1)

news = data.get("news", [])
stats = data.get("stats", {})
classified_at = data.get("classified_at", "")

now = datetime.now().strftime("%Y-%m-%d %H:%M")
msg = f"📰 **新聞簡報 {now}**\n"
msg += f"📊 {len(news)} 則 | 來源: Yahoo News\n\n"

# 按分類顯示
category_order = [
    "🔥 重要",
    "💰 金融",
    "📈 股市",
    "🏭 總經",
    "🌍 國際",
    "🏛️ 政治",
    "🏠 房產",
    "💱 匯市",
    "📋 其他",
]

# 分組新聞
grouped = {}
for item in news:
    cat = item.get("category", "📋 其他")
    if cat not in grouped:
        grouped[cat] = []
    grouped[cat].append(item)

# 格式化
for cat in category_order:
    items = grouped.get(cat, [])
    if not items:
        continue
    
    count = len(items)
    emoji = cat.split()[0]
    name = cat.split()[1] if len(cat.split()) > 1 else cat
    
    msg += f"{emoji} **{name}** ({count} 則)\n"
    
    for item in items[:3]:  # 每類最多3則
        title = item.get("title", "")[:40]
        source = item.get("source", "")[:10]
        msg += f"  • {title}...\n"
    
    if len(items) > 3:
        msg += f"  ... 還有 {len(items) - 3} 則\n"
    
    msg += "\n"

msg += "---\n"
msg += f"_更新: {datetime.now().strftime('%H:%M')}_"

print(msg)
PYEOF
}

MSG=$(format_output)

# 發送到 Discord
log "發送訊息到 Discord..."

response=$(echo "$MSG" | jq -Rs '{content: .}' | curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @- \
    "$WEBHOOK_URL")

# 檢查結果
if [ $? -eq 0 ]; then
    log "✅ 發送成功"
    echo "✅ News brief sent"
else
    log "❌ 發送失敗: $response"
    echo "❌ Failed to send: $response"
fi

log "=== 格式化輸出完成 ==="