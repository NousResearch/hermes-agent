#!/bin/bash
# ============================================================================
# Auto-Heal - 簡化版自動修復
# ============================================================================

CACHE_DIR="/home/ubuntu/.openclaw/cache"
LOG_FILE="/tmp/pipeline-auto-heal.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

# 檢查並修復
check_and_fix() {
    local name="$1"
    local cache="$2"
    local fetch="$3"
    local max_age="${4:-7200}"
    
    if [ ! -f "$cache" ]; then
        log "⚠️ $name cache 不存在，嘗試修復..."
        [ -f "$fetch" ] && bash "$fetch" >> "$LOG_FILE" 2>&1
        return
    fi
    
    file_time=$(stat -c %Y "$cache" 2>/dev/null || echo "0")
    current_time=$(date +%s)
    age=$((current_time - file_time))
    
    if [ $age -gt $max_age ]; then
        log "⚠️ $name cache 過期 (${age}s)，嘗試修復..."
        [ -f "$fetch" ] && bash "$fetch" >> "$LOG_FILE" 2>&1
    fi
}

# 主要邏輯
log "=== Auto-Heal ==="

check_and_fix "stock" "$CACHE_DIR/stock/04006C.json" "/home/ubuntu/.openclaw/scripts/pipeline/stock/fetch.sh" 3600
check_and_fix "news" "$CACHE_DIR/news/classified.json" "/home/ubuntu/.openclaw/scripts/pipeline/news/fetch.py" 7200
check_and_fix "weather" "$CACHE_DIR/weather.json" "/home/ubuntu/.openclaw/scripts/pipeline/morning/fetch-all.sh" 14400
check_and_fix "tasks" "$CACHE_DIR/tasks.json" "/home/ubuntu/.openclaw/scripts/pipeline/task/fetch.sh" 14400

log "=== Auto-Heal 完成 ==="