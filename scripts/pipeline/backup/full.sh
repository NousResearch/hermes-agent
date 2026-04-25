#!/bin/bash
# ============================================================================
# Backup Pipeline: 完整備份
# 每日 04:00 執行
# 職責：備份重要資料到備份目錄
# 特性：純 Shell，無 AI，快速執行
# ============================================================================

set -e

BACKUP_DIR="/home/ubuntu/.openclaw/backups"
LOG_FILE="/tmp/pipeline-backup.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)
DATE=$(date +%Y%m%d)

mkdir -p "$BACKUP_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 開始備份 ==="

# 要備份的目錄/檔案
BACKUP_ITEMS=(
    "/home/ubuntu/.openclaw/memory:/memory"
    "/home/ubuntu/.openclaw/workspace-frontdesk:/workspace-frontdesk"
    "/home/ubuntu/.openclaw/config:/config"
)

# 快取目錄不備份（太大）
# /home/ubuntu/.openclaw/cache  # 不備份

BACKUP_FILE="$BACKUP_DIR/backup-$DATE.tar.gz"

# 創建備份
log "創建備份: $BACKUP_FILE"

# 使用 tar 備份
TAR_CMD="tar -czf $BACKUP_FILE"
for item in "${BACKUP_ITEMS[@]}"; do
    src="${item%%:*}"
    if [ -e "$src" ]; then
        TAR_CMD="$TAR_CMD -C $(dirname $src) $(basename $src)"
        log "加入: $src"
    else
        log "略過 (不存在): $src"
    fi
done

# 執行 tar
$TAR_CMD 2>&1 | while read line; do log "tar: $line"; done

# 檢查結果
if [ -f "$BACKUP_FILE" ]; then
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    log "✅ 備份成功: $SIZE"
    echo "✅ Backup created: $BACKUP_FILE ($SIZE)"
else
    log "❌ 備份失敗"
    echo "❌ Backup failed"
fi

# 清理舊備份 (保留 7 天)
log "清理舊備份 (保留 7 天)..."
find "$BACKUP_DIR" -name "backup-*.tar.gz" -mtime +7 -delete 2>/dev/null
find "$BACKUP_DIR" -name "backup-*.tar.gz" -type f | while read f; do
    log "保留: $(basename $f)"
done

log "=== 備份完成 ==="