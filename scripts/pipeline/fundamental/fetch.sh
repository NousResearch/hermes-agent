#!/bin/bash
# ============================================================================
# Fundamental Pipeline - Layer 1: 抓取基本面資料
# 每天 06:30 執行 (晨報前)
# ============================================================================

CACHE_DIR="/home/ubuntu/.openclaw/cache/fundamental"
LOG_FILE="/tmp/pipeline-fundamental-fetch.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$CACHE_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

# 股票清單
STOCKS="04006C 2303 3008 2330 2317 2454 0050"

log "=== 開始基本面收集 ==="

# 取得最近交易日
get_trading_date() {
    python3 -c "
import urllib.request
import json

for days_ago in range(5):
    from datetime import datetime, timedelta
    d = (datetime.now() - timedelta(days=days_ago))
    day_of_week = d.weekday()
    
    if day_of_week >= 5:  # 週末
        continue
        
    date_str = d.strftime('%Y%m%d')
    url = f'https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={date_str}&stockNo=0050&response=json'
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        if '很抱歉' not in data.get('stat', ''):
            print(date_str)
            return
    except:
        pass
print('20260417')  # fallback
"
}

TRADE_DATE=$(get_trading_date)
log "最近交易日: $TRADE_DATE"

# 抓取個股基本面
fetch_stock() {
    local code=$1
    
    python3 -c "
import urllib.request
import json
from datetime import datetime

code = '$code'
trade_date = '$TRADE_DATE'
cache_dir = '$CACHE_DIR'

url = f'https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={trade_date}&stockNo={code}&response=json'

try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    
    if data.get('data'):
        for row in data['data']:
            if len(row) >= 7:
                result = {
                    'code': code,
                    'date': row[0],
                    'name': row[1],
                    'price': row[2],
                    'PE': row[4] if row[4] != '-' else None,
                    'PB': row[5] if row[5] != '-' else None,
                    'yield': row[6] if row[6] != '-' else None,
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(f'{cache_dir}/{code}.json', 'w') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    
                print(f'OK: {code} PE={result[\"PE\"]} PB={result[\"PB\"]} Yield={result[\"yield\"]}')
                return
    print(f'FAIL: {code}')
except Exception as e:
    print(f'ERROR: {code}: {e}')
"
}

# 主要邏輯
for code in $STOCKS; do
    log "抓取 $code..."
    result=$(fetch_stock "$code" 2>&1)
    log "$result"
done

log "=== 基本面收集完成 ==="

echo "快取:"
ls "$CACHE_DIR/" 2>/dev/null || echo "空"
