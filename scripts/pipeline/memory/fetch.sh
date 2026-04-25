#!/bin/bash
# ============================================================================
# Layer 1: 記憶收集
# 每日 22:00 執行
# 職責：收集今日日誌，寫入 memory/YYYY-MM-DD.md
# ============================================================================

set -e

MEMORY_DIR="/home/ubuntu/.openclaw/memory"
CACHE_DIR="/home/ubuntu/.openclaw/cache"
LOG_FILE="/tmp/pipeline-memory-fetch.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)
TODAY=$(date +%Y-%m-%d)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 記憶收集 ==="

# 建立今日日誌檔案
DAILY_LOG="$MEMORY_DIR/$TODAY.md"

if [ -f "$DAILY_LOG" ]; then
    log "今日日誌已存在: $DAILY_LOG"
else
    # 建立新日誌
    cat > "$DAILY_LOG" << EOF
# 每日日誌 $TODAY

## 日期時間
- 日期: $TODAY
- 時間: $(date +%H:%M:%S)
- 星期: $(date +%A)

## 系統狀態
### Cron Jobs
| Job | 狀態 |
|-----|------|
EOF

    # 檢查各 Pipeline 狀態
    for log_file in /tmp/pipeline-*.log; do
        if [ -f "$log_file" ]; then
            name=$(basename "$log_file" .log)
            last_line=$(tail -1 "$log_file" 2>/dev/null || echo "無記錄")
            echo "| $name | $last_line |" >> "$DAILY_LOG"
        fi
    done

    echo "" >> "$DAILY_LOG"
    echo "### 發現的問題" >> "$DAILY_LOG"
    echo "- (待填寫)" >> "$DAILY_LOG"
    echo "" >> "$DAILY_LOG"
    echo "## 待辦事項" >> "$DAILY_LOG"
    echo "- (待填寫)" >> "$DAILY_LOG"

    log "✅ 建立今日日誌: $DAILY_LOG"
fi

# 收集 cache 狀態到 memory
cat >> "$DAILY_LOG" << EOF

## Cache 狀態 ($(date +%H:%M:%S))
EOF

for cache_file in "$CACHE_DIR"/*.json; do
    if [ -f "$cache_file" ]; then
        name=$(basename "$cache_file" .json)
        size=$(stat -c%s "$cache_file" 2>/dev/null || echo "0")
        mtime=$(stat -c %y "$cache_file" 2>/dev/null | cut -d' ' -f1 || echo "未知")
        echo "- $name: ${size} bytes (更新: $mtime)" >> "$DAILY_LOG"
    fi
done

log "=== 記憶收集完成 ==="

echo "✅ Daily log: $DAILY_LOG"