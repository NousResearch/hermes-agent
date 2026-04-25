#!/bin/bash
# ============================================================================
# Unified Morning Brief - 發送到 Discord
# ============================================================================

CACHE_DIR="/home/ubuntu/.openclaw/cache/fundamental"
WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
print(config['webhooks']['stock_monitor']['url'])
" 2>/dev/null)

# 格式化訊息
format_msg() {
    python3 << 'PYEOF'
import json
from datetime import datetime

cache_dir = "/home/ubuntu/.openclaw/cache/fundamental"
report_file = f"{cache_dir}/morning_report.json"

with open(report_file) as f:
    report = json.load(f)

now = datetime.now().strftime("%Y-%m-%d %H:%M")
stocks = report.get("stocks", [])

msg = f"📊 **投資晨報** {now}\n"
msg += f"資料日期: {report['trade_date']}\n"
msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

# 基本面 + 技術面
msg += "**【基本面 + 技術面】**\n"
msg += "```\n"
msg += f"{'代碼':<6} {'名稱':<8} {'價格':>10} {'P/E':>6} {'殖利率':>7} {'評分':>5} {'技術':<6}\n"
msg += f"{'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*7} {'-'*5} {'-'*6}\n"

for r in stocks:
    pe = str(r['PE'])[:6] if r['PE'] else 'N/A'
    y = str(r['yield'])[:7] if r['yield'] else 'N/A'
    trend_icon = "📈" if r['trend'] == "多頭" else "📉" if r['trend'] == "空頭" else "➡️"
    msg += f"{r['code']:<6} {r['name']:<8} {str(r['price']):>10} {pe:>6} {y:>7} {r['grade']:<5} {trend_icon} {r['trend']:<5}\n"

msg += "```\n"

# 技術面燈號
msg += "**【技術面燈號】**\n"
for r in stocks:
    if not r.get('rsi'):
        continue
    
    signals = []
    if r['rsi'] > 70:
        signals.append("RSI超買")
    elif r['rsi'] < 30:
        signals.append("RSI超賣")
    
    if r['macd']:
        signals.append("MACD多" if r['macd'] > 0 else "MACD空")
    
    signal_str = " / ".join(signals) if signals else "中性"
    msg += f"• {r['code']} {r['name']}: {signal_str}\n"

# 策略建議
msg += "\n**【策略建議】**\n"

value_picks = [r for r in stocks if r['fund_pct'] >= 60 and r['trend'] == '多頭']
if value_picks:
    picks_str = ', '.join([f"{r['code']}{r['name']}" for r in value_picks[:3]])
    msg += f"價值 + 多頭: {picks_str}\n"

bluechip = [r for r in stocks if r['fund_pct'] >= 50 and r['code'] in ['2330', '2317', '2454']]
if bluechip:
    blue_str = ', '.join([f"{r['code']}{r['name']}" for r in bluechip[:3]])
    msg += f"藍籌關注: {blue_str}\n"

msg += f"\n_每日 08:00 自動更新_"

print(msg)
PYEOF
}

# 執行
echo "產生晨報..."
python3 /home/ubuntu/.openclaw/scripts/pipeline/morning/unified_brief.py 2>/dev/null

echo "發送到 Discord..."
MSG=$(format_msg)

response=$(echo "$MSG" | jq -Rs '{content: .}' | curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @- \
    "$WEBHOOK_URL")

if [ $? -eq 0 ]; then
    echo "✅ 晨報已發送"
else
    echo "❌ 發送失敗: $response"
fi