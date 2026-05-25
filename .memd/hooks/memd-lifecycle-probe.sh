#!/usr/bin/env bash
# A3-D3 working-memory lifecycle self-test probe.
# Runs store → recall → expire → verify-expired against the live memd server.
# Exits 0 on green, 1 on red. Safe to wire from cron, wake, or checkpoint.
# Appends one-line NDJSON verdict to .memd/state/lifecycle-probe.log so the
# 7-day rolling health artifact lives inside the bundle.
set -euo pipefail

BUNDLE_ROOT="${MEMD_BUNDLE_ROOT:-.memd}"
BASE_URL="${MEMD_BASE_URL:-http://127.0.0.1:8787}"
LOG="${BUNDLE_ROOT}/state/lifecycle-probe.log"
mkdir -p "$(dirname "$LOG")"

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if out=$(memd --base-url "${BASE_URL}" diagnostics lifecycle-probe \
    --output "${BUNDLE_ROOT}" --summary 2>&1); then
    printf '{"ts":"%s","verdict":"green","probe":%s}\n' \
        "$ts" "$(printf %s "$out" | head -1 | sed 's/"/\\"/g' | awk '{printf "\"%s\"", $0}')" \
        >> "$LOG"
    printf '%s\n' "$out"
    exit 0
else
    rc=$?
    printf '{"ts":"%s","verdict":"red","exit":%d,"probe":%s}\n' \
        "$ts" "$rc" "$(printf %s "$out" | head -1 | sed 's/"/\\"/g' | awk '{printf "\"%s\"", $0}')" \
        >> "$LOG"
    printf '%s\n' "$out" >&2
    exit "$rc"
fi
