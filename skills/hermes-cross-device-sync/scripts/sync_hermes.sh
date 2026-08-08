#!/usr/bin/env bash
# Hermes bidirectional sync script
# Termux .hermes/ ←→ /storage/emulated/0/HermesSync/
# Requires: termux-api + Termux:API APK, rsync
# WARNING: Do NOT use --delete with bidirectional rsync — causes data loss.

LOG="$HOME/.hermes/logs/sync.log"
mkdir -p "$(dirname "$LOG")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

sync_out() {
    if [ ! -d "$1" ]; then return; fi
    local count_before=$(find "$2" -type f 2>/dev/null | wc -l)
    rsync -rtu "$1/" "$2/" 2>>"$LOG"
    local count_after=$(find "$2" -type f 2>/dev/null | wc -l)
    log "[OUT] $3: $count_after files in shared storage (was $count_before)"
}

sync_in() {
    if [ ! -d "$1" ]; then return; fi
    local count_before=$(find "$2" -type f 2>/dev/null | wc -l)
    rsync -rtu "$1/" "$2/" 2>>"$LOG"
    local count_after=$(find "$2" -type f 2>/dev/null | wc -l)
    log "[IN] $3: $count_after files in Termux (was $count_before)"
}

log "========== START SYNC =========="

sync_out ~/.hermes/memories/ /storage/emulated/0/HermesSync/memories/ "memories"
sync_in  /storage/emulated/0/HermesSync/memories/ ~/.hermes/memories/ "memories"

sync_out ~/.hermes/skills/ /storage/emulated/0/HermesSync/skills/ "skills"
sync_in  /storage/emulated/0/HermesSync/skills/ ~/.hermes/skills/ "skills"

sync_out ~/.hermes/sessions/ /storage/emulated/0/HermesSync/sessions/ "sessions"
sync_in  /storage/emulated/0/HermesSync/sessions/ ~/.hermes/sessions/ "sessions"

log "========== SYNC DONE =========="
