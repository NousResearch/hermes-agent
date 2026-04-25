#!/bin/bash
# ============================================================================
# Analysis Pipeline - 投資分析
# Layer 1: 收集資料 (價格、新聞、持股)
# Layer 2: 分析信號 (可選 AI)
# Layer 3: 格式化報告
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/analysis"
ALERTS_DIR="/home/ubuntu/.openclaw/alerts/analysis"
LOG_FILE="/tmp/pipeline-analysis.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$CACHE_DIR" "$ALERTS_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

# ============================================================================
# Layer 1: 收集
# ============================================================================
layer1_fetch() {
    log "=== Layer 1: 收集資料 ==="
    
    # 1. 收集股票快取
    if [ -f "/home/ubuntu/.openclaw/cache/stock/04006C.json" ]; then
        cp /home/ubuntu/.openclaw/cache/stock/04006C.json "$CACHE_DIR/holdings.json"
        log "股票資料: ✅"
    else
        echo '{}' > "$CACHE_DIR/holdings.json"
        log "股票資料: ⚠️ 無"
    fi
    
    # 2. 收集持股成本
    if [ -f "/home/ubuntu/.openclaw/memory/portfolio.json" ]; then
        python3 << 'PYEOF'
import json

with open("/home/ubuntu/.openclaw/memory/portfolio.json") as f:
    portfolio = json.load(f)

# 添加計算欄位
for h in portfolio.get("holdings", []):
    code = h.get("code", "")
    
    # 嘗試讀取最新報價
    try:
        with open(f"/home/ubuntu/.openclaw/cache/stock/{code}.json") as f:
            stock = json.load(f)
        h["current_price"] = stock.get("price", 0)
        h["change_pct"] = stock.get("change_pct", 0)
    except:
        h["current_price"] = 0
        h["change_pct"] = 0

with open("/home/ubuntu/.openclaw/cache/analysis/holdings.json", "w") as f:
    json.dump(portfolio, f, ensure_ascii=False, indent=2)
PYEOF
        log "持股分析: ✅"
    fi
    
    # 3. 收集今日頭條
    if [ -f "/home/ubuntu/.openclaw/cache/news/classified.json" ]; then
        head=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/cache/news/classified.json') as f:
    data = json.load(f)
news = data.get('news', [])
headlines = [n.get('title', '')[:50] for n in news[:5]]
print('|'.join(headlines))
" 2>/dev/null || echo "")
        echo "$head" > "$CACHE_DIR/headlines.txt"
        log "頭條收集: ✅"
    fi
    
    log "=== Layer 1 完成 ==="
}

# ============================================================================
# Layer 2: 分析
# ============================================================================
layer2_analyze() {
    log "=== Layer 2: 分析 ==="
    
    python3 << 'PYEOF'
import json
from datetime import datetime

# 讀取持股
try:
    with open("/home/ubuntu/.openclaw/cache/analysis/holdings.json") as f:
        portfolio = json.load(f)
except:
    print("無法讀取持股資料")
    exit(0)

holdings = portfolio.get("holdings", [])
cash = portfolio.get("cash", 0)

analysis = {
    "timestamp": datetime.now().isoformat(),
    "holdings_analysis": [],
    "signals": [],
    "summary": {}
}

for h in holdings:
    code = h.get("code", "")
    name = h.get("name", code)
    qty = h.get("quantity", 0)
    cost = h.get("avgCost", 0)
    price = h.get("current_price", 0)
    change_pct = h.get("change_pct", 0)
    
    # 計算損益
    if price > 0 and cost > 0:
        profit = (price - cost) * qty
        profit_pct = (price - cost) / cost * 100
        
        # 風險評估
        risk_level = "normal"
        if profit_pct < -5:
            risk_level = "warning"
        if profit_pct < -10:
            risk_level = "critical"
        if profit_pct > 15:
            risk_level = "take_profit"
        
        analysis["holdings_analysis"].append({
            "code": code,
            "name": name,
            "qty": qty,
            "cost": cost,
            "price": price,
            "profit": profit,
            "profit_pct": profit_pct,
            "risk_level": risk_level
        })
        
        # 信號
        if risk_level in ["critical", "take_profit"]:
            analysis["signals"].append({
                "type": risk_level,
                "code": code,
                "message": f"{code} {'觸及停損' if risk_level == 'critical' else '達標'} {profit_pct:.1f}%"
            })

# 總結
total_profit = sum(h.get("profit", 0) for h in analysis["holdings_analysis"])
total_value = cash + sum(h.get("price", 0) * h.get("qty", 0) for h in holdings)

analysis["summary"] = {
    "cash": cash,
    "holdings_value": total_value - cash,
    "total_value": total_value,
    "total_profit": total_profit
}

# 保存
with open("/home/ubuntu/.openclaw/cache/analysis/result.json", "w") as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

print(f"分析完成:")
print(f"  持有: {len(holdings)} 檔")
print(f"  信號: {len(analysis['signals'])} 個")
for sig in analysis["signals"]:
    print(f"    - [{sig['type']}] {sig['message']}")

PYEOF
    
    log "=== Layer 2 完成 ==="
}

# ============================================================================
# Layer 3: 格式化
# ============================================================================
layer3_format() {
    log "=== Layer 3: 格式化輸出 ==="
    
    # Webhook
    WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
print(config['webhooks']['stock_monitor']['url'])
" 2>/dev/null)
    
    python3 << PYEOF
import json
from datetime import datetime

try:
    with open("/home/ubuntu/.openclaw/cache/analysis/result.json") as f:
        analysis = json.load(f)
except:
    print("無分析結果")
    exit(0)

summary = analysis.get("summary", {})
holdings = analysis.get("holdings_analysis", [])
signals = analysis.get("signals", [])

now = datetime.now()

msg = f"📊 **投資分析 {now.strftime('%Y-%m-%d %H:%M')}**\n"
msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"

# 總結
msg += f"💰 **帳戶總覽**\n"
msg += f"  現金: ${summary.get('cash', 0):,.0f}\n"
msg += f"  持股: ${summary.get('holdings_value', 0):,.0f}\n"
msg += f"  總值: ${summary.get('total_value', 0):,.0f}\n"

total_profit = summary.get('total_profit', 0)
sign = "+" if total_profit >= 0 else ""
msg += f"  損益: {sign}${total_profit:,.0f}\n\n"

# 警示信號
if signals:
    msg += "🚨 **警示信號**\n"
    for sig in signals:
        emoji = "🔴" if sig["type"] == "critical" else "🎯"
        msg += f"  {emoji} {sig['message']}\n"
    msg += "\n"

# 持有明細
if holdings:
    msg += "📈 **持有明細**\n"
    msg += "```\n"
    for h in holdings:
        code = h.get("code", "?")
        name = h.get("name", "")[:8]
        qty = h.get("qty", 0)
        cost = h.get("cost", 0)
        price = h.get("price", 0)
        profit = h.get("profit", 0)
        profit_pct = h.get("profit_pct", 0)
        
        sign = "+" if profit >= 0 else ""
        emoji = "📈" if profit >= 0 else "📉"
        
        msg += f"{emoji} {code} {qty}股 @${cost}\n"
        msg += f"   現價${price} | {sign}{profit_pct:.1f}% | {sign}${profit:,.0f}\n"
    msg += "```\n"

msg += f"\n_🕐 {now.strftime('%H:%M')}_"

print(msg)

# 發送到 Discord
import subprocess
WEBHOOK_URL = "$(python3 -c "import json; print(json.load(open('/home/ubuntu/.openclaw/config/system-config.json'))['webhooks']['stock_monitor']['url'])")"
result = subprocess.run(
    f"echo '{msg}' | jq -Rs '{{content: .}}' | curl -s -X POST -H 'Content-Type: application/json' -d @- '{WEBHOOK_URL}'",
    shell=True, capture_output=True
)
print("\n✅ 分析報告已發送" if result.returncode == 0 else "\n❌ 發送失敗")

PYEOF
    
    log "=== Layer 3 完成 ==="
}

# ============================================================================
# 主流程
# ============================================================================
main() {
    layer1_fetch
    layer2_analyze
    layer3_format
}

main "$@"