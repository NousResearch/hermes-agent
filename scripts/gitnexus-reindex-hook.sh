#!/usr/bin/env bash
# GitNexus event-driven re-index hook (post-receive).
#
# Triggered by git push via a post-receive git hook. Re-indexes the
# KenseiAgent code knowledge graph only when HEAD actually moved, with
# full-SHA verification against the GitNexus registry. Detaches to a
# background worker so the push never blocks on indexing.
#
# Wiring (documented, not registered here):
#   1. cp scripts/gitnexus-reindex-hook.sh .git/hooks/post-receive
#   2. chmod +x .git/hooks/post-receive
# The hook receives ref updates on stdin: <old-sha> <new-sha> <ref-name>
#
# Exit codes: 0 = queued/no-op, non-zero = hook error (git ignores
# post-receive exit codes, but we log explicit failures).
set -euo pipefail

REPO="/home/kensei/repos/KenseiAgent"
GITNEXUS="/home/kensei/.hermes/node/bin/gitnexus"
RUNNER="/home/kensei/repos/KenseiAgent/scripts/gitnexus-reindex-runner.sh"
LOG_DIR="/home/kensei/.hermes/logs/gitnexus"
REGISTRY="/home/kensei/.gitnexus/registry.json"
REPO_NAME="KenseiAgent"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/reindex-hook-$(date +%Y%m%d-%H%M%S).log"
HISTORY="$LOG_DIR/reindex-history.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE" >&2; }

REINDEX_NEEDED=0
PUSHED_SHA=""
while read -r old_sha new_sha ref_name; do
    case "$ref_name" in
        refs/heads/*) ;;
        *) continue ;;
    esac
    if [ "$new_sha" = "0000000000000000000000000000000000000000" ]; then
        continue
    fi
    HEAD_BRANCH=$(cd "$REPO" && git symbolic-ref --short HEAD 2>/dev/null || echo "")
    PUSHED_BRANCH="${ref_name#refs/heads/}"
    if [ "$PUSHED_BRANCH" != "$HEAD_BRANCH" ]; then
        continue
    fi
    REINDEX_NEEDED=1
    PUSHED_SHA="$new_sha"
done

if [ "$REINDEX_NEEDED" -ne 1 ]; then
    exit 0
fi

INDEXED_COMMIT=$(python3 -c "
import json
try:
    with open('$REGISTRY') as f:
        d = json.load(f)
    for r in d:
        if r.get('name') == '$REPO_NAME':
            print(r.get('lastCommit', ''))
            break
except Exception:
    pass
" 2>/dev/null || echo "")

if [ -z "$INDEXED_COMMIT" ]; then
    log "WARN: could not read indexed commit from registry; queuing re-index anyway"
    INDEXED_COMMIT="unknown"
fi

if ! echo "$PUSHED_SHA" | grep -qE '^[0-9a-f]{40}$'; then
    log "ERROR: pushed SHA '$PUSHED_SHA' is not a valid 40-char hex SHA - aborting re-index"
    exit 1
fi

if [ "$PUSHED_SHA" = "$INDEXED_COMMIT" ]; then
    log "Index already current at $PUSHED_SHA - no re-index needed"
    exit 0
fi

NEEDS_FULL=false
if "$GITNEXUS" status 2>/dev/null | grep -q "incrementalInProgress"; then
    NEEDS_FULL=true
fi

if [ ! -x "$RUNNER" ]; then
    log "ERROR: runner not found or not executable at $RUNNER - cannot re-index"
    exit 1
fi
if [ ! -x "$GITNEXUS" ]; then
    log "ERROR: gitnexus binary not found or not executable at $GITNEXUS - cannot re-index"
    exit 1
fi

MODE="incremental"
if $NEEDS_FULL; then
    MODE="full"
fi

log "Queuing $MODE re-index: indexed=$INDEXED_COMMIT pushed=$PUSHED_SHA"
setsid "$RUNNER" "$LOG_FILE" "$GITNEXUS" "$REPO" "$MODE" > /dev/null 2>&1 < /dev/null &
disown
log "$MODE re-index started (pid $!) - log: $LOG_FILE"
echo "$(date -Iseconds) hook $MODE reindex queued (pushed=$PUSHED_SHA)" >> "$HISTORY"
exit 0
