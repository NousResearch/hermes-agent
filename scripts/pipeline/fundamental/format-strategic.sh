#!/bin/bash
# ============================================================================
# Strategic Screening Report - 策略選股格式化輸出
# ============================================================================

CACHE_DIR="/home/ubuntu/.openclaw/cache/fundamental"
LOG_FILE="/tmp/pipeline-strategic-format.log"
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

log "=== 開始格式化策略選股報告 ==="

# 格式化輸出
format_output() {
    python3 << 'PYEOF'
import json
from datetime import datetime

cache_dir = "/home/ubuntu/.openclaw/cache/fundamental"
report_file = f"{cache_dir}/strategic_report.json"

with open(report_file) as f:
    report = json.load(f)

now = datetime.now().strftime("%Y-%m-%d %H:%M")

# 選擇重點策略顯示
key_strategies = ["價值投資", "高股息", "低估藍籌"]

msg = f"📊 **策略導向選股報告**\n"
msg += f"更新: {now}\n"
msg += f"分析總檔數: {report['total_stocks_analyzed']}\n"
msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

# 每個策略顯示前 5 名
for strategy_key in key_strategies:
    if strategy_key not in report["strategies"]:
        continue
    
    data = report["strategies"][strategy_key]
    stocks = data["stocks"][:5]  # 只取前 5 名
    
    msg += f"**◆ {strategy_key}**\n"
    msg += f"_{data['description']}_\n"
    msg += "```\n"
    msg += f"{'排名':<3} {'代碼':<6} {'名稱':<8} {'P/E':>6} {'P/B':>6} {'殖利率':>7}\n"
    msg += f"{'-'*3} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*7}\n"
    
    for i, stock in enumerate(stocks, 1):
        msg += f"{i:<3} {stock['code']:<6} {stock['name']:<8} {str(stock['PE'])[:6]:>6} {str(stock['PB'])[:6]:>6} {str(stock['yield'])[:7]:>7}\n"
    
    msg += "```\n"

# 精選標的
msg += "**📌 精選標的**\n"
msg += "━━━━━━━━━━━━━━━━━━━━\n"

# 價值投資首選
value_pick = report["strategies"].get("價值投資", {}).get("stocks", [{}])[0]
if value_pick:
    msg += f"• **{value_pick['code']} {value_pick['name']}**: P/E {value_pick['PE']}, P/B {value_pick['PB']}, 殖利率 {value_pick['yield']}%\n"

# 高股息首選
dividend_pick = report["strategies"].get("高股息", {}).get("stocks", [{}])[0]
if dividend_pick:
    msg += f"• **{dividend_pick['code']} {dividend_pick['name']}**: P/E {dividend_pick['PE']}, P/B {dividend_pick['PB']}, 殖利率 {dividend_pick['yield']}%\n"

# 低估藍籌首選
bluechip_pick = report["strategies"].get("低估藍籌", {}).get("stocks", [{}])[0]
if bluechip_pick:
    msg += f"• **{bluechip_pick['code']} {bluechip_pick['name']}**: P/E {bluechip_pick['PE']}, P/B {bluechip_pick['PB']}, 殖利率 {bluechip_pick['yield']}%\n"

msg += f"\n_資料日期: {report['trade_date']} · 每策略僅顯示前 5 名_"

print(msg)
PYEOF
}

# 主要邏輯
if [ ! -f "$CACHE_DIR/strategic_report.json" ]; then
    echo "❌ 報告不存在，先執行 strategic_screening.py"
    exit 1
fi

MSG=$(format_output)

# 發送到 Discord
log "發送到 Discord..."

response=$(echo "$MSG" | jq -Rs '{content: .}' | curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @- \
    "$WEBHOOK_URL")

if [ $? -eq 0 ]; then
    log "✅ 發送成功"
    echo "✅ Strategic screening sent"
else
    log "❌ 發送失敗: $response"
    echo "❌ Failed: $response"
fi