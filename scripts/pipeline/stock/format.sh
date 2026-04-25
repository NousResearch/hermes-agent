#!/bin/bash
# ============================================================================
# Layer 3: 股票格式化輸出
# 每 15 分鐘執行 (交易時段 9:00-14:00)
# 職責：讀取 cache，格式化，發送到 Discord
# 特性：純 Shell，無 AI，快速執行
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/stock"
LOG_FILE="/tmp/pipeline-stock-format.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

# 讀取 webhook URL
WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
print(config['webhooks']['stock_monitor']['url'])
" 2>/dev/null)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 開始格式化輸出 ==="

# 檢查 webhook
if [ -z "$WEBHOOK_URL" ]; then
    log "錯誤: 無法取得 Webhook URL"
    exit 1
fi

# 讀取 portfolio
load_portfolio() {
    if [ -f "/home/ubuntu/.openclaw/memory/portfolio.json" ]; then
        PORTFOLIO=$(cat /home/ubuntu/.openclaw/memory/portfolio.json)
    else
        PORTFOLIO="{}"
    fi
}

# 格式化輸出
format_output() {
    python3 << PYEOF
import json
from datetime import datetime
import sys

cache_dir = "$CACHE_DIR"
portfolio = json.loads('$PORTFOLIO')

# 讀取所有快取
stocks = ["04006C", "2303", "3008"]
data = {}
for code in stocks:
    try:
        with open(f"{cache_dir}/{code}.json") as f:
            data[code] = json.load(f)
    except:
        data[code] = None

# 格式化訊息
now = datetime.now().strftime("%H:%M")
msg = f"📊 **股票監控 {now}**\n\n"

# 持有
holdings = portfolio.get("holdings", [])
if holdings:
    msg += "📈 **【持有】**\n"
    msg += "```\n"
    for h in holdings:
        code = h["code"]
        if data.get(code):
            d = data[code]
            price = d.get("price", 0)
            change_pct = d.get("change_pct", 0)
            cost = h.get("avgCost", 0)
            profit = (price - cost) * h.get("quantity", 0)
            profit_pct = ((price - cost) / cost * 100) if cost > 0 else 0
            
            emoji = "📈" if profit >= 0 else "📉"
            sign = "+" if profit >= 0 else ""
            
            msg += f"{code} {h.get('name', code)} | 買 {price} | {emoji} {sign}{profit_pct:.2f}% | 成本 ${cost} | {sign}${profit:,.0f}\n"
    msg += "```\n\n"

# 觀察
msg += "👁️ **【觀察】**\n"
msg += "```\n"
watchlist = ["2303", "3008"]
for code in watchlist:
    if data.get(code):
        d = data[code]
        name = d.get("name", code)
        price = d.get("price", 0)
        change_pct = d.get("change_pct", 0)
        sign = "+" if change_pct >= 0 else ""
        msg += f"{code} {name[:6]:<6} | 買 {price:<8} | {sign}{change_pct}%\n"
msg += "```\n"

# 顯示快取狀態
cached_count = sum(1 for d in data.values() if d is not None)
msg += f"\n_快取: {cached_count}/{len(stocks)} 更新_"

print(msg)
PYEOF
}

# 主要邏輯
load_portfolio

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
    echo "✅ Stock update sent"
else
    log "❌ 發送失敗: $response"
    echo "❌ Failed to send: $response"
fi

log "=== 格式化輸出完成 ==="