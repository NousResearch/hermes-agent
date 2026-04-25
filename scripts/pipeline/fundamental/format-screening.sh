#!/bin/bash
# ============================================================================
# Combined Screening Report - 綜合選股報告格式化輸出
# ============================================================================

CACHE_DIR="/home/ubuntu/.openclaw/cache/fundamental"
LOG_FILE="/tmp/pipeline-screening-format.log"
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

log "=== 開始格式化 ==="

# 格式化輸出
format_output() {
    python3 << 'PYEOF'
import json
from datetime import datetime

cache_dir = "/home/ubuntu/.openclaw/cache/fundamental"
report_file = f"{cache_dir}/combined_report.json"

# 讀取報告
with open(report_file) as f:
    report = json.load(f)

now = datetime.now().strftime("%Y-%m-%d %H:%M")

# 格式化 message
msg = f"📊 **綜合選股報告**\n"
msg += f"更新: {now}\n"
msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

msg += "```\n"
msg += f"{'代碼':<6} {'名稱':<8} {'P/E':>8} {'P/B':>8} {'殖利率':>8} {'評分':>6} {'等級':<6}\n"
msg += f"{'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}\n"

for stock in report.get("stocks", []):
    if "error" in stock:
        msg += f"{stock['code']:<6} {stock['name']:<8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>6} {'⚠️':<6}\n"
        continue
    
    pe = str(stock.get('PE', 'N/A'))[:8]
    pb = str(stock.get('PB', 'N/A'))[:8]
    y = str(stock.get('yield', 'N/A'))[:8]
    score = stock.get('fund_score', 0)
    grade = stock.get('grade', '⚠️')
    
    msg += f"{stock['code']:<6} {stock['name']:<8} {pe:>8} {pb:>8} {y:>8} {score:>6} {grade:<6}\n"

msg += "```\n\n"

# 評價摘要
msg += "📈 **評價摘要**\n"

for stock in report.get("stocks", []):
    if "error" in stock:
        continue
    
    name = stock.get('name', stock['code'])
    grade = stock.get('grade', '⚠️')
    rec = stock.get('recommendation', '')
    
    if stock.get('fund_percentage', 0) >= 70:
        status = "✅"
    elif stock.get('fund_percentage', 0) >= 40:
        status = "⚠️"
    else:
        status = "❌"
    
    msg += f"{status} **{name}**: {grade} {rec}\n"

msg += f"\n_基本面具備分 (PE/PB/殖利率) · 資料日期: {report.get('trade_date', 'N/A')}_"

print(msg)
PYEOF
}

# 主要邏輯
if [ ! -f "$CACHE_DIR/combined_report.json" ]; then
    echo "❌ 報告不存在，先執行 combined_screening.py"
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
    echo "✅ Screening report sent"
else
    log "❌ 發送失敗: $response"
    echo "❌ Failed: $response"
fi