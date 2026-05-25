#!/usr/bin/env bash
# PostCompact memd restore — NON-BLOCKING.
#
# The post-compaction turn would otherwise see an empty file-interaction
# ledger because compaction wipes session state. This hook copies the
# newest sealed ledger (written by PreCompact seal-ledger) back into the
# active ledger path BEFORE any PreToolUse hook fires — so Read/Edit/Write
# tools inherit the prior session's prime-reads context.
#
# Flag: MEMD_A4_LEDGER_SURVIVAL (default 0 during dogfood).
#   0  → hook exits 0 immediately; ledger survival disabled.
#   1  → hook runs `memd hook restore` and logs the outcome.
#
# Non-blocking: any non-zero exit from `memd hook restore` is logged but
# never propagated. Exit code 2 (no-sealed-ledger) is an expected
# signal that `memd` self-records in the continuity-breach log.
set -euo pipefail

STATE_DIR="${MEMD_HOOK_STATE_DIR:-$HOME/.memd/hook_state}"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/hook.log"

INPUT="$(cat)"
SESSION_ID="$(
  printf '%s' "$INPUT" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("session_id", "unknown"))' 2>/dev/null || printf 'unknown'
)"

echo "[$(date '+%H:%M:%S')] POST-COMPACT session=$SESSION_ID" >> "$LOG"

if [ "${MEMD_A4_LEDGER_SURVIVAL:-0}" != "1" ]; then
  echo "[$(date '+%H:%M:%S')] POST-COMPACT skipped (MEMD_A4_LEDGER_SURVIVAL=0)" >> "$LOG"
  exit 0
fi

BUNDLE_ROOT="${MEMD_BUNDLE_ROOT:-.memd}"
if [ ! -d "$BUNDLE_ROOT" ] && [ -d "$HOME/Documents/projects/memd/.memd" ]; then
  BUNDLE_ROOT="$HOME/Documents/projects/memd/.memd"
fi

if ! command -v memd >/dev/null 2>&1; then
  echo "[$(date '+%H:%M:%S')] POST-COMPACT memd CLI missing; skip" >> "$LOG"
  exit 0
fi

# Restore runs in ≤15s (hook timeout). On exit 2 (no sealed ledger) the CLI
# has already written a breach line; we don't re-log it here.
# B4.9: route through `memd hooks enforce` when MEMD_HOOK_ENFORCE=1 so
# PostCompact gets a contract-gated trace line + budget timer + session lock.
RC=0
if [ "${MEMD_HOOK_ENFORCE:-0}" = "1" ]; then
  memd hooks enforce --event PostCompact --harness claude-code \
    --session-id "$SESSION_ID" --output "$BUNDLE_ROOT" \
    -- memd hook restore --session-id "$SESSION_ID" --output "$BUNDLE_ROOT" \
    >> "$LOG" 2>&1 || RC=$?
else
  memd hook restore --session-id "$SESSION_ID" --output "$BUNDLE_ROOT" \
    >> "$LOG" 2>&1 || RC=$?
fi
case $RC in
  0) echo "[$(date '+%H:%M:%S')] POST-COMPACT restore ok" >> "$LOG" ;;
  2) echo "[$(date '+%H:%M:%S')] POST-COMPACT no-sealed-ledger (breach logged by CLI)" >> "$LOG" ;;
  *) echo "[$(date '+%H:%M:%S')] POST-COMPACT restore rc=$RC (non-fatal)" >> "$LOG" ;;
esac

exit 0
