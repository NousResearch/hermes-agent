#!/bin/bash
# ============================================================================
# GitHub Backup Pipeline - 整合 Vault 的備份系統
# 使用 GitHub 作為異地備份
# ============================================================================

set -e

BACKUP_DIR="/home/ubuntu/.openclaw/backups"
GIT_DIR="/tmp/yao-github-backup"
SOURCE_DIR="/home/ubuntu/.openclaw"
GIT_REPO="https://github.com/puppy0808-ops/yao-vault.git"
LOG_FILE="/tmp/pipeline-github-backup.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$BACKUP_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

# ============================================================================
# 備份策略
# ============================================================================

# 完整備份 (每週日)
backup_full() {
    log "=== 完整備份 ==="
    
    DATE=$(date +%Y%m%d)
    BACKUP_FILE="$BACKUP_DIR/backup-full-$DATE.tar.gz"
    
    # 備份的目錄
    ITEMS=(
        "memory"
        "workspace-frontdesk"
        "config"
        "agents"
    )
    
    # 建立 tarball
    TAR_OPTS="-czf $BACKUP_FILE"
    for item in "${ITEMS[@]}"; do
        if [ -e "$SOURCE_DIR/$item" ]; then
            TAR_OPTS="$TAR_OPTS -C $SOURCE_DIR $item"
        fi
    done
    
    eval "tar $TAR_OPTS" 2>/dev/null
    
    if [ -f "$BACKUP_FILE" ]; then
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log "✅ 完整備份完成: $BACKUP_FILE ($SIZE)"
        echo "✅ Full backup: $BACKUP_FILE ($SIZE)"
        
        # 同步到 GitHub
        sync_to_github "$BACKUP_FILE" "full-backup-$DATE"
    else
        log "❌ 完整備份失敗"
        echo "❌ Full backup failed"
    fi
}

# 增量備份 (每天)
backup_incremental() {
    log "=== 增量備份 ==="
    
    DATE=$(date +%Y%m%d)
    BACKUP_FILE="$BACKUP_DIR/backup-incr-$DATE.tar.gz"
    
    # 只備份當天變更的
    ITEMS=(
        "memory"        # 每日更新
        "cache/pipeline/status.json"  # Pipeline 狀態
        "cache/pipeline/daily-report.json"  # 每日報告
    )
    
    # 建立增量 tarball
    tar -czf "$BACKUP_FILE" \
        -C "$SOURCE_DIR" \
        memory \
        cache/pipeline/status.json \
        cache/pipeline/daily-report.json \
        2>/dev/null
    
    if [ -f "$BACKUP_FILE" ]; then
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log "✅ 增量備份完成: $BACKUP_FILE ($SIZE)"
        echo "✅ Incremental backup: $BACKUP_FILE ($SIZE)"
        
        # 同步到 GitHub
        sync_to_github "$BACKUP_FILE" "incremental-$DATE"
    else
        log "❌ 增量備份失敗"
        echo "❌ Incremental backup failed"
    fi
}

# ============================================================================
# 同步到 GitHub
# ============================================================================

sync_to_github() {
    local backup_file="$1"
    local label="$2"
    
    log "同步到 GitHub: $label"
    
    # Clone 或 pull
    if [ -d "$GIT_DIR/.git" ]; then
        cd "$GIT_DIR"
        git pull origin main 2>/dev/null
        log "Pull 最新"
    else
        rm -rf "$GIT_DIR"
        git clone "$GIT_REPO" "$GIT_DIR" 2>/dev/null
        cd "$GIT_DIR"
        log "Clone 完成"
    fi
    
    # 複製備份檔案
    mkdir -p "$GIT_DIR/backups"
    cp "$backup_file" "$GIT_DIR/backups/"
    
    # 建立版本記錄
    cat >> "$GIT_DIR/backups/manifest.txt" << EOF
$(date -Iseconds) | $label | $(basename "$backup_file")
EOF
    
    # Commit 並 push
    git config user.email "cloud@yao.ai" 2>/dev/null
    git config user.name "Cloud Bot" 2>/dev/null
    
    git add -A
    git commit -m "Backup: $label - $(date '+%Y-%m-%d %H:%M')" 2>/dev/null
    
    if git push origin main 2>&1 | grep -q "Everything up-to-date"; then
        log "✅ GitHub 已最新"
    elif [ $? -eq 0 ]; then
        log "✅ GitHub 同步成功"
        echo "✅ GitHub sync: OK"
    else
        log "⚠️ GitHub 同步可能失敗"
        echo "⚠️ GitHub sync: 可能失敗"
    fi
}

# ============================================================================
# 從 GitHub 恢復
# ============================================================================

restore_from_github() {
    local date="${1:-}"
    
    log "從 GitHub 恢復: $date"
    
    # Clone
    if [ ! -d "$GIT_DIR/.git" ]; then
        git clone "$GIT_REPO" "$GIT_DIR" 2>/dev/null
    fi
    
    cd "$GIT_DIR"
    
    if [ -n "$date" ]; then
        # 恢復特定日期
        BACKUP_FILE=$(ls backups/backup-*$date*.tar.gz 2>/dev/null | head -1)
        if [ -n "$BACKUP_FILE" ]; then
            log "恢復: $BACKUP_FILE"
            tar -xzf "$BACKUP_FILE" -C "$SOURCE_DIR/"
            log "✅ 恢復完成"
            echo "✅ Restored: $BACKUP_FILE"
        else
            log "❌ 找不到備份: $date"
            echo "❌ Backup not found: $date"
        fi
    else
        # 恢復最新
        LATEST=$(ls -t backups/backup-*.tar.gz 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            log "恢復最新: $LATEST"
            tar -xzf "$LATEST" -C "$SOURCE_DIR/"
            log "✅ 恢復完成"
            echo "✅ Restored latest: $LATEST"
        else
            log "❌ 無可恢復的備份"
            echo "❌ No backup available"
        fi
    fi
}

# ============================================================================
# 清理舊備份
# ============================================================================

cleanup() {
    log "清理舊備份..."
    
    # 本地保留 30 天
    find "$BACKUP_DIR" -name "backup-*.tar.gz" -mtime +30 -delete 2>/dev/null
    
    # GitHub 上保留 90 天
    if [ -d "$GIT_DIR/.git" ]; then
        cd "$GIT_DIR"
        # 這個操作比較危險，通常不建議自動刪除 GitHub 上的歷史
        # 所以只清理本地
        log "本地備份已清理 (保留 30 天)"
    fi
    
    echo "✅ Cleanup: 本地保留 30 天"
}

# ============================================================================
# 備份狀態
# ============================================================================

status() {
    echo ""
    echo "📦 GitHub Backup Status"
    echo "=================================================="
    echo ""
    
    echo "本地備份:"
    ls -lh "$BACKUP_DIR"/backup-*.tar.gz 2>/dev/null | tail -5 || echo "  無"
    
    echo ""
    echo "GitHub 狀態:"
    if [ -d "$GIT_DIR/.git" ]; then
        cd "$GIT_DIR"
        COMMITS=$(git log --oneline -5 2>/dev/null | head -5 || echo "無法讀取")
        echo "$COMMITS"
        
        echo ""
        echo "GitHub 備份列表:"
        cat "$GIT_DIR/backups/manifest.txt" 2>/dev/null | tail -5 || echo "  無"
    else
        echo "  未初始化"
    fi
    
    echo ""
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    local action="${1:-status}"
    
    log "=== GitHub Backup: $action ==="
    
    case "$action" in
        "full")
            backup_full
            ;;
        "incremental")
            backup_incremental
            ;;
        "sync")
            sync_to_github "$2" "$3"
            ;;
        "restore")
            restore_from_github "$2"
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            status
            ;;
        "help"|*)
            echo "GitHub Backup"
            echo "============"
            echo ""
            echo "用法:"
            echo "  github-backup.sh full          - 完整備份"
            echo "  github-backup.sh incremental  - 增量備份"
            echo "  github-backup.sh sync         - 同步到 GitHub"
            echo "  github-backup.sh restore [date]- 從 GitHub 恢復"
            echo "  github-backup.sh cleanup      - 清理舊備份"
            echo "  github-backup.sh status       - 顯示狀態"
            echo ""
            echo "排程建議:"
            echo "  週日 04:00: full backup"
            echo "  每日 04:30: incremental backup"
            ;;
    esac
}

main "$@"
