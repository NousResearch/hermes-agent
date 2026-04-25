#!/bin/bash
# ============================================================================
# Layer 1: 股票資料收集
# 每 5 分鐘執行 (交易時段 9:00-14:00)
# 職責：抓取報價，寫入 cache/stock/*.json
# 特性：純 Shell，無 AI，快速執行
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/stock"
LOG_FILE="/tmp/pipeline-stock-fetch.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$CACHE_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

# 股票清單
STOCKS=(
    "04006C:04006C"  # 持有
    "2303:2303"      # 觀察
    "3008:3008"      # 觀察
)

# 讀取持股成本
load_portfolio() {
    if [ -f "/home/ubuntu/.openclaw/memory/portfolio.json" ]; then
        HOLDINGS=$(cat /home/ubuntu/.openclaw/memory/portfolio.json)
    else
        HOLDINGS="{}"
    fi
}

log "=== 開始收集股票資料 ==="

# 使用 Python + shioaji 抓取報價
fetch_shioaji() {
    local code=$1
    local name=$2
    
    python3 << PYEOF
import shioaji as sj
import json
from datetime import datetime

try:
    api = sj.Shioaji()
    
    # 登入 (使用環境變數)
    api.login(
        person_id="YOUR_ID",
        passwd="YOUR_PASS"
    )
    
    # 取得報價
    contracts = api.Contracts.Stocks[code]
    snapshot = api.snapshots([contracts])
    
    if snapshot:
        data = {
            "code": "$code",
            "name": "$name",
            "price": snapshot[0]["close"],
            "change": snapshot[0]["change"],
            "change_pct": snapshot[0]["change_ratio"] * 100,
            "quantity": snapshot[0]["total"]["quantity"],
            "timestamp": datetime.now().isoformat()
        }
        
        # 寫入快取
        with open("$CACHE_DIR/$code.json", "w") as f:
            json.dump(data, f, ensure_ascii=False)
        
        print(f"OK: $code @ {data['price']}")
    else:
        print(f"FAIL: $code - No data")
        
except Exception as e:
    print(f"ERROR: $code - {e}")
PYEOF
}

# 主要邏輯
load_portfolio

for entry in "${STOCKS[@]}"; do
    code="${entry%%:*}"
    name="${entry##*:}"
    
    log "抓取 $code ($name)..."
    
    result=$(fetch_shioaji "$code" "$name" 2>&1)
    log "$result"
    
    # 如果失敗，嘗試使用快取
    if echo "$result" | grep -q "ERROR"; then
        if [ -f "$CACHE_DIR/$code.json" ]; then
            log "使用快取: $code"
            # 更新 timestamp
            python3 -c "
import json
with open('$CACHE_DIR/$code.json') as f:
    d = json.load(f)
d['timestamp'] = '$(date -Iseconds)'
d['cached'] = True
with open('$CACHE_DIR/$code.json', 'w') as f:
    json.dump(d, f)
"
        fi
    fi
done

log "=== 收集完成 ==="

# 顯示快取狀態
echo "快取狀態:"
ls -la "$CACHE_DIR/"