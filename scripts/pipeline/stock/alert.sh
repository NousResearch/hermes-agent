#!/bin/bash
# ============================================================================
# Layer 2: 股票警示判斷
# 每 5 分鐘執行 (交易時段 9:00-14:00)
# 職責：檢查是否觸及停損/目標價，寫入 alerts/
# 特性：純 Shell，無 AI，快速執行
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/stock"
ALERTS_DIR="/home/ubuntu/.openclaw/alerts/stock"
LOG_FILE="/tmp/pipeline-stock-alert.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$ALERTS_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 開始警示檢查 ==="

# 讀取 portfolio 取得風險控制設定
load_portfolio() {
    if [ -f "/home/ubuntu/.openclaw/memory/portfolio.json" ]; then
        PORTFOLIO=$(cat /home/ubuntu/.openclaw/memory/portfolio.json)
    else
        PORTFOLIO="{}"
    fi
}

# 檢查警示
check_alert() {
    local code=$1
    
    # 檢查是否有快取
    if [ ! -f "$CACHE_DIR/$code.json" ]; then
        log "無快取資料: $code"
        return
    fi
    
    # 讀取快取
    PRICE=$(python3 -c "import json; print(json.load(open('$CACHE_DIR/$code.json')).get('price', 0))")
    
    # 取得持有成本和停損設定
    COST=$(echo "$PORTFOLIO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(next((h['avgCost'] for h in d.get('holdings',[]) if h['code']=='$code'), 0))" 2>/dev/null || echo "0")
    STOP_LOSS_PCT=$(echo "$PORTFOLIO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(next((r['stop_loss_pct'] for r in d.get('risk_control',[]) if r.get('code')=='$code'), -10))" 2>/dev/null || echo "-10")
    TARGET_PCT=$(echo "$PORTFOLIO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(next((r['target_profit_pct'] for r in d.get('risk_control',[]) if r.get('code')=='$code'), 20))" 2>/dev/null || echo "20")
    
    if [ "$COST" = "0" ] || [ "$COST" = "" ]; then
        log "$code: 無持有成本，跳過"
        return
    fi
    
    # 計算停損價和目標價
    STOP_LOSS=$(echo "scale=3; $COST * (1 + $STOP_LOSS_PCT / 100)" | bc)
    TARGET=$(echo "scale=3; $COST * (1 + $TARGET_PCT / 100)" | bc)
    
    # 計算漲跌百分比
    CHANGE_PCT=$(echo "scale=2; (($PRICE - $COST) / $COST) * 100" | bc)
    
    log "$code: 現價=$PRICE, 成本=$COST, 漲跌=$CHANGE_PCT%, 停損<$STOP_LOSS, 目標>$TARGET"
    
    # 檢查停損
    if (( $(echo "$PRICE < $STOP_LOSS" | bc -l) )); then
        ALERT_FILE="$ALERTS_DIR/${code}-STOPLOSS-${TIMESTAMP}.json"
        python3 << PYEOF
import json
with open("$ALERT_FILE", "w") as f:
    json.dump({
        "code": "$code",
        "type": "STOP_LOSS",
        "price": $PRICE,
        "cost": $COST,
        "stop_loss": $STOP_LOSS,
        "change_pct": $CHANGE_PCT,
        "timestamp": "$(date -Iseconds)"
    }, f, ensure_ascii=False)
PYEOF
        log "🚨 停損觸發: $code @ $PRICE < $STOP_LOSS"
    fi
    
    # 檢查目標價
    if (( $(echo "$PRICE > $TARGET" | bc -l) )); then
        ALERT_FILE="$ALERTS_DIR/${code}-TARGET-${TIMESTAMP}.json"
        python3 << PYEOF
import json
with open("$ALERT_FILE", "w") as f:
    json.dump({
        "code": "$code",
        "type": "TARGET",
        "price": $PRICE,
        "cost": $COST,
        "target": $TARGET,
        "change_pct": $CHANGE_PCT,
        "timestamp": "$(date -Iseconds)"
    }, f, ensure_ascii=False)
PYEOF
        log "🎯 目標觸發: $code @ $PRICE > $TARGET"
    fi
}

# 主要邏輯
load_portfolio

# 檢查持有的股票
if [ -f "$CACHE_DIR/04006C.json" ]; then
    check_alert "04006C"
fi

log "=== 警示檢查完成 ==="

# 顯示警示狀態
echo "警示狀態:"
ls -la "$ALERTS_DIR/" | tail -5