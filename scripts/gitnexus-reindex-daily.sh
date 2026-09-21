#!/usr/bin/env bash
# GitNexus daily re-index of KenseiAgent codebase.
# Keeps the code knowledge graph current with HEAD.
# Detaches to background via setsid+disown to avoid cron scheduler timeout.
#
# Repaired 2026-07-15 (C019):
#   - RUNNER path resolved relative to this script (was hardcoded to a
#     .hermes/scripts path that may not exist on fresh installs).
#   - Explicit failure reporting: logs when the runner or gitnexus binary
#     is missing instead of failing silently under set -e.
# P13 isolation (2026-07-28):
#   - GITNEXUS_DRY_RUN=1 short-circuits before starting the indexer so a
#     disposable/local run never spawns a detached background rebuild or
#     touches the gitnexus registry.
#   - GITNEXUS_DIR env parameterises the gitnexus state dir (registry.json)
#     so a dry-run can read a throwaway registry instead of the live one.
set -euo pipefail

REPO="${KENSEI_REPO:-/home/kensei/repos/KenseiAgent}"
GITNEXUS_DIR="${GITNEXUS_DIR:-/home/kensei/.gitnexus}"
GITNEXUS="${GITNEXUS_BIN:-/home/kensei/.hermes/node/bin/gitnexus}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/gitnexus-reindex-runner.sh"
LOG_DIR="${GITNEXUS_LOG_DIR:-/home/kensei/.hermes/logs/gitnexus}"

# P13 dry-run: short-circuit before any indexer spawn or registry mutation.
if [ "${GITNEXUS_DRY_RUN:-0}" = "1" ]; then
    echo "[gitnexus-reindex-daily] DRY-RUN: would re-index $REPO (indexer not started, registry untouched)"
    exit 0
fi

mkdir -p "$LOG_DIR"

# Explicit pre-flight checks (fail with a message, not a silent set -e abort).
if [ ! -x "$RUNNER" ]; then
    echo "[$(date -Iseconds)] ERROR: GitNexus re-index runner not found or not executable: $RUNNER" >&2
    exit 1
fi
if [ ! -x "$GITNEXUS" ]; then
    echo "[$(date -Iseconds)] ERROR: gitnexus binary not found or not executable: $GITNEXUS" >&2
    exit 1
fi

# Get current HEAD (full SHA)
HEAD_COMMIT=$(cd "$REPO" && git rev-parse HEAD)

# Check if index is stale — read from registry.json (has full SHA)
INDEXED_COMMIT=$(python3 -c "
import json,os
d = json.load(open(os.environ.get('GITNEXUS_DIR','/home/kensei/.gitnexus')+'/registry.json'))
for r in d:
    if r['name'] == 'KenseiAgent':
        print(r.get('lastCommit', ''))
" 2>/dev/null || echo "unknown")

if [ "$HEAD_COMMIT" == "$INDEXED_COMMIT" ]; then
    # Index is current — silent
    exit 0
fi

# Check if full rebuild needed (previous run crashed)
NEEDS_FULL=false
if "$GITNEXUS" status 2>/dev/null | grep -q "incrementalInProgress"; then
    NEEDS_FULL=true
fi

LOG_FILE="$LOG_DIR/reindex-$(date +%Y%m%d-%H%M%S).log"

if $NEEDS_FULL; then
    echo "Full rebuild required (previous run incomplete) — starting background re-index"
    # setsid = new session fully detached from cron's process group
    # disown = remove from parent shell's job table
    setsid "$RUNNER" "$LOG_FILE" "$GITNEXUS" "$REPO" full > /dev/null 2>&1 < /dev/null &
    disown
    echo "Full rebuild started (pid $!) — log: $LOG_FILE" >> "$LOG_DIR/reindex-history.log"
    exit 0
else
    # Incremental update — fully detached
    setsid "$RUNNER" "$LOG_FILE" "$GITNEXUS" "$REPO" incremental > /dev/null 2>&1 < /dev/null &
    disown
    echo "Incremental re-index started (pid $!) — log: $LOG_FILE"
    echo "GitNexus re-index started at $(date '+%d/%m/%y %H:%M:%S')"
fi
