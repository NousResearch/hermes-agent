#!/bin/bash
# ============================================================================
# Layer 3: Morning Brief 格式化輸出
# 07:00 執行
# 職責：讀取 cache/，格式化，發送到 Discord
# 特性：純 Shell，無 AI，快速執行
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache"
LOG_FILE="/tmp/pipeline-morning-format.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

# 讀取 webhook URL
WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
# 發送到 #一般 頻道
print(config['webhooks']['stock_monitor']['url'])
" 2>/dev/null)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 開始 Morning Brief 格式化 ==="

format_output() {
    python3 << 'PYEOF'
import json
from datetime import datetime, date

cache_dir = "/home/ubuntu/.openclaw/cache"

now = datetime.now()
today = now.strftime("%Y-%m-%d")
weekday = now.strftime("%A")

# 星期幾中文
weekday_cn = {
    "Monday": "星期一",
    "Tuesday": "星期二",
    "Wednesday": "星期三",
    "Thursday": "星期四",
    "Friday": "星期五",
    "Saturday": "星期六",
    "Sunday": "星期日",
}

msg = f"🌅 **晨報 {today} {weekday_cn.get(weekday, weekday)} {now.strftime('%H:%M')}**\n"
msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

# 天氣
try:
    with open(f"{cache_dir}/weather.json") as f:
        weather = json.load(f)
    
    current = weather.get("current_condition", [{}])[0]
    temp = current.get("temp_C", "?")
    desc = current.get("weatherDesc", [{}])[0].get("value", "?")
    humidity = current.get("humidity", "?")
    
    msg += f"🌤️ **台北** {temp}°C, {desc}\n"
    msg += f"   濕度: {humidity}%\n\n"
except Exception as e:
    msg += f"🌤️ **台北** 天氣載入失敗\n\n"

# 行事曆
try:
    with open(f"{cache_dir}/calendar.json") as f:
        calendar = json.load(f)
    
    if calendar:
        msg += "📅 **今日行事曆**\n"
        if isinstance(calendar, list):
            for event in calendar[:3]:
                if isinstance(event, dict):
                    time = event.get("start", "").get("dateTime", "")[11:16] if "start" in event else ""
                    title = event.get("summary", "無標題")
                    msg += f"  • {time} {title}\n"
                else:
                    msg += f"  • {str(event)[:50]}\n"
        else:
            msg += f"  • {str(calendar)[:50]}\n"
        msg += "\n"
    else:
        msg += "📅 **今日行事曆** 無活動\n\n"
except Exception as e:
    msg += "📅 **今日行事曆** 載入失敗\n\n"

# 任務
try:
    with open(f"{cache_dir}/tasks.json") as f:
        tasks_data = json.load(f)
    
    pending = tasks_data.get("pending", 0)
    total = tasks_data.get("count", 0)
    
    if total > 0:
        msg += f"📋 **待辦任務** {pending}/{total} 項待完成\n"
        tasks = tasks_data.get("tasks", [])
        pending_tasks = [t for t in tasks if not t.get("done", False)][:3]
        for t in pending_tasks:
            text = t.get("text", "")[:40]
            msg += f"  • ⏳ {text}...\n"
        if pending > 3:
            msg += f"  ... 還有 {pending - 3} 項\n"
        msg += "\n"
    else:
        msg += "📋 **待辦任務** 無待辦\n\n"
except Exception as e:
    msg += "📋 **待辦任務** 載入失敗\n\n"

# 持股
try:
    with open(f"{cache_dir}/holdings.json") as f:
        holdings = json.load(f)
    
    holding_list = holdings.get("holdings", [])
    cash = holdings.get("cash", 0)
    
    if holding_list:
        msg += "📈 **持有**\n"
        for h in holding_list:
            code = h.get("code", "?")
            name = h.get("name", "")[:10]
            qty = h.get("quantity", 0)
            cost = h.get("avgCost", 0)
            
            # 讀取最新報價
            try:
                with open(f"/home/ubuntu/.openclaw/cache/stock/{code}.json") as sf:
                    stock = json.load(sf)
                price = stock.get("price", 0)
                change_pct = stock.get("change_pct", 0)
                profit = (price - cost) * qty if price > 0 else 0
                profit_pct = ((price - cost) / cost * 100) if cost > 0 else 0
                sign = "+" if profit >= 0 else ""
                
                msg += f"  • {code} {qty} 股 @ ${cost}\n"
                msg += f"    現價 ${price} | {sign}{profit_pct:.1f}% | {sign}${profit:,.0f}\n"
            except:
                msg += f"  • {code} {qty} 股 @ ${cost} (無報價)\n"
        
        msg += f"  💰 現金 ${cash:,.0f}\n\n"
    else:
        msg += "📈 **持有** 無持股\n"
        msg += f"💰 現金 ${cash:,.0f}\n\n"
except Exception as e:
    msg += f"📈 **持有** 載入失敗\n\n"

msg += "---\n"
msg += f"_🕐 更新於 {now.strftime('%H:%M')}_"

print(msg)
PYEOF
}

MSG=$(format_output)

# 發送到 Discord
log "發送 Morning Brief..."

response=$(echo "$MSG" | jq -Rs '{content: .}' | curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @- \
    "$WEBHOOK_URL")

if [ $? -eq 0 ]; then
    log "✅ 發送成功"
    echo "✅ Morning brief sent"
else
    log "❌ 發送失敗"
    echo "❌ Failed"
fi

log "=== 完成 ==="