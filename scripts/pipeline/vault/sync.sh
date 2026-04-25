#!/bin/bash
# ============================================================================
# Vault Sync Pipeline - GitHub 同步
# Layer 1: 收集變更
# Layer 2: 比較差異
# Layer 3: 提交同步
# ============================================================================

set -e

VAULT_DIR="/tmp/yao-vault-sync"
SOURCE_DIR="/home/ubuntu/.openclaw"
GIT_REPO="https://github.com/puppy0808-ops/yao-vault.git"
LOG_FILE="/tmp/pipeline-vault.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

# ============================================================================
# Layer 1: 收集變更
# ============================================================================
layer1_fetch() {
    log "=== Layer 1: 收集變更 ==="
    
    mkdir -p "$VAULT_DIR"
    
    # Clone 或 pull
    if [ -d "$VAULT_DIR/.git" ]; then
        cd "$VAULT_DIR"
        git pull origin main 2>/dev/null
        log "Pull 最新"
    else
        rm -rf "$VAULT_DIR"
        git clone "$GIT_REPO" "$VAULT_DIR" 2>/dev/null
        log "Clone 新鮮副本"
    fi
    
    # 需要同步的項目
    SYNC_ITEMS=(
        "workspace-frontdesk"
        "memory"
        "scripts/pipeline"
        "config"
    )
    
    for item in "${SYNC_ITEMS[@]}"; do
        src="$SOURCE_DIR/$item"
        dst="$VAULT_DIR/$item"
        
        if [ -e "$src" ]; then
            mkdir -p "$(dirname "$dst")"
            cp -r "$src" "$dst"
            log "同步: $item"
        fi
    done
    
    log "=== Layer 1 完成 ==="
}

# ============================================================================
# Layer 2: 比較差異
# ============================================================================
layer2_compare() {
    log "=== Layer 2: 比較差異 ==="
    
    cd "$VAULT_DIR"
    
    # 檢查變更
    CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
    
    if [ "$CHANGES" -eq 0 ]; then
        log "無變更"
        echo "無需同步"
        return 1
    fi
    
    log "發現 $CHANGES 個變更"
    
    # 顯示變更
    git status --short | head -20
    
    echo "$CHANGES 個變更待同步"
    
    return 0
}

# ============================================================================
# Layer 3: 提交同步
# ============================================================================
layer3_sync() {
    log "=== Layer 3: 提交同步 ==="
    
    cd "$VAULT_DIR"
    
    # 配置 git
    git config user.email "cloud@yao.ai" 2>/dev/null
    git config user.name "Cloud Bot" 2>/dev/null
    
    # 添加所有變更
    git add -A
    
    # 提交
    COMMIT_MSG="Vault sync - $(date '+%Y-%m-%d %H:%M')"
    git commit -m "$COMMIT_MSG" 2>/dev/null
    
    # Push
    if git push origin main 2>&1 | grep -q "Everything up-to-date"; then
        log "✅ 無需推送 (已最新)"
        echo "✅ Vault 已同步 (無新變更)"
    elif [ $? -eq 0 ]; then
        log "✅ 推送成功"
        echo "✅ Vault 已同步到 GitHub"
    else
        log "❌ 推送失敗"
        echo "❌ Vault 同步失敗"
        return 1
    fi
    
    log "=== Layer 3 完成 ==="
}

# ============================================================================
# 主流程
# ============================================================================
main() {
    log "=== Vault Sync Pipeline 開始 ==="
    
    # Layer 1: 收集
    layer1_fetch
    
    # Layer 2: 比較
    if layer2_compare; then
        # Layer 3: 同步
        layer3_sync
    fi
    
    log "=== Vault Sync Pipeline 完成 ==="
}

main "$@"