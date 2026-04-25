#!/bin/bash
# ============================================================================
# Fundamental Pipeline - Layer 2: 格式化輸出
# 每天 07:00 執行 (晨報時)
# ============================================================================

CACHE_DIR="/home/ubuntu/.openclaw/cache/fundamental"
LOG_FILE="/tmp/pipeline-fundamental-format.log"
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

log "=== 開始基本面格式化 ==="

# 檢查 webhook
if [ -z "$WEBHOOK_URL" ]; then
    log "錯誤: 無法取得 Webhook URL"
    exit 1
fi

# 格式化輸出
format_output() {
    python3 << 'PYEOF'
import json
from datetime import datetime

cache_dir = "/home/ubuntu/.openclaw/cache/fundamental"
stocks = ["2330", "2317", "2454", "2303", "3008"]

# 讀取快取
data = {}
for code in stocks:
    try:
        with open(f"{cache_dir}/{code}.json") as f:
            data[code] = json.load(f)
    except:
        data[code] = None

now = datetime.now().strftime("%Y-%m-%d %H:%M")

msg = f"📊 **基本面觀察 {now}**\n"
msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

msg += "```\n"
msg += f"{'代碼':<6} {'名稱':<8} {'價格':>10} {'本益比':>8} {'股價淨值比':>10} {'殖利率':>8}\n"
msg += f"{'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}\n"

for code in stocks:
    if data.get(code):
        d = data[code]
        name = d.get('name', code)[:6]
        price = d.get('price', 'N/A')
        pe = d.get('PE', 'N/A')
        pb = d.get('PB', 'N/A')
        yield_val = d.get('yield', 'N/A')
        
        msg += f"{code:<6} {name:<8} {price:>10} {pe:>8} {pb:>10} {yield_val:>8}\n"

msg += "```\n\n"

# 簡單評估
msg += "📈 **觀察摘要**\n"

for code in stocks:
    if data.get(code):
        d = data[code]
        name = d.get('name', code)
        pe = d.get('PE', '0')
        yield_val = d.get('yield', '0')
        
        try:
            pe_val = float(pe) if pe != 'N/A' else 0
            yield_val_f = float(yield_val) if yield_val != 'N/A' else 0
            
            # 簡單評估
            if pe_val > 0:
                if pe_val < 15:
                    pe_status = "🟢低估"
                elif pe_val < 25:
                    pe_status = "🟡適中"
                else:
                    pe_status = "🔴偏高"
            else:
                pe_status = "⚪無資料"
            
            if yield_val_f > 4:
                yield_status = "高殖利率"
            elif yield_val_f > 2:
                yield_status = "中殖利率"
            else:
                yield_status = "低殖利率"
            
            msg += f"• {code} {name}: PE {pe} ({pe_status}), 殖利率 {yield_val}% ({yield_status})\n"
        except:
            msg += f"• {code} {name}: PE {pe}, 殖利率 {yield_val}%\n"

msg += f"\n_資料更新: {data.get('2330', {}).get('timestamp', 'N/A')[:16]} UTC_"

print(msg)
PYEOF
}

# 主要邏輯
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
    echo "✅ Fundamental update sent"
else
    log "❌ 發送失敗: $response"
    echo "❌ Failed to send: $response"
fi

log "=== 基本面格式化完成 ==="