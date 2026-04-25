#!/bin/bash
# ============================================================================
# Stock Pipeline - 合併 L1+L2+L3
# ============================================================================

LOG_FILE="/tmp/pipeline-stock.log"
log() {
    echo "[$(date '+%H:%M:%S')] $1" >> "$LOG_FILE"
}

log "=== Stock Pipeline 開始 ==="

# Layer 1: Fetch
log "L1: 收集資料..."
bash /home/ubuntu/.openclaw/scripts/pipeline/stock/fetch.sh >> "$LOG_FILE" 2>&1

# Layer 2: Alert
log "L2: 分析..."
bash /home/ubuntu/.openclaw/scripts/pipeline/stock/alert.sh >> "$LOG_FILE" 2>&1

# Layer 3: Format & Send
log "L3: 格式化發送..."
bash /home/ubuntu/.openclaw/scripts/pipeline/stock/format.sh >> "$LOG_FILE" 2>&1

log "=== Stock Pipeline 完成 ==="
