#!/usr/bin/env bash
# PreCompact memd save — NON-BLOCKING.
#
# Runs in ≤15s (hook timeout). Compaction is lossy, so we snapshot here:
#   1) seal the file-interaction ledger so the continuation session sees it
#   2) write an auto-derived checkpoint summarizing uncommitted work + session
#      so the post-compact turn has a discoverable continuity row
#   3) always allow the compaction to proceed (empty stdout)
#
# Design note: we intentionally do NOT block. The original implementation
# emitted `{"decision":"block"}` unconditionally, which deadlocked /compact.
# Continuous capture happens via UserPromptSubmit + PostToolUse hooks; this
# hook is a seal-and-proceed, not a gate. If you need a richer snapshot,
# call `memd checkpoint` or `memd hook spill --stdin --apply` before /compact
# — the PreCompact hook will NOT force you to.
set -euo pipefail

STATE_DIR="${MEMD_HOOK_STATE_DIR:-$HOME/.memd/hook_state}"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/hook.log"

INPUT="$(cat)"
SESSION_ID="$(
  printf '%s' "$INPUT" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("session_id", "unknown"))' 2>/dev/null || printf 'unknown'
)"
TRIGGER="$(
  printf '%s' "$INPUT" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("trigger", "manual"))' 2>/dev/null || printf 'manual'
)"

echo "[$(date '+%H:%M:%S')] PRE-COMPACT session=$SESSION_ID trigger=$TRIGGER" >> "$LOG"

BUNDLE_ROOT="${MEMD_BUNDLE_ROOT:-.memd}"
if [ ! -d "$BUNDLE_ROOT" ] && [ -d "$HOME/Documents/projects/memd/.memd" ]; then
  BUNDLE_ROOT="$HOME/Documents/projects/memd/.memd"
fi

# (1) Seal the file-interaction ledger. Failures are non-fatal.
# B4.9: route through `memd hooks enforce` when MEMD_HOOK_ENFORCE=1
# so PreCompact gets a contract-gated trace line + budget timer.
if command -v memd >/dev/null 2>&1; then
  if [ "${MEMD_HOOK_ENFORCE:-0}" = "1" ]; then
    memd hooks enforce --event PreCompact --harness claude-code \
      --session-id "$SESSION_ID" --output "$BUNDLE_ROOT" \
      -- memd hook seal-ledger --session-id "$SESSION_ID" --output "$BUNDLE_ROOT" \
      >> "$LOG" 2>&1 || echo "[$(date '+%H:%M:%S')] seal-ledger failed (non-fatal)" >> "$LOG"
  else
    memd hook seal-ledger --session-id "$SESSION_ID" --output "$BUNDLE_ROOT" \
      >> "$LOG" 2>&1 || echo "[$(date '+%H:%M:%S')] seal-ledger failed (non-fatal)" >> "$LOG"
  fi
fi

# (2) Auto-checkpoint: summarize uncommitted work + branch. Cheap, derived.
#     Skipped if memd CLI is missing or the project isn't a git repo.
if command -v memd >/dev/null 2>&1; then
  REPO_ROOT=""
  if command -v git >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$(dirname "$BUNDLE_ROOT")" rev-parse --show-toplevel 2>/dev/null || true)"
  fi

  BRANCH="detached"
  DIRTY_LIST=""
  RECENT_COMMIT=""
  if [ -n "$REPO_ROOT" ]; then
    BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)"
    DIRTY_LIST="$(git -C "$REPO_ROOT" status --short 2>/dev/null | head -20 | tr '\n' '|' || true)"
    RECENT_COMMIT="$(git -C "$REPO_ROOT" log -1 --oneline 2>/dev/null || true)"
  fi

  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CONTENT="auto(pre-compact): session=${SESSION_ID} trigger=${TRIGGER} at=${TS} branch=${BRANCH} last_commit=\"${RECENT_COMMIT}\" dirty=[${DIRTY_LIST}]"

  memd checkpoint \
    --output "$BUNDLE_ROOT" \
    --tag pre-compact --tag auto \
    --ttl-seconds 86400 \
    --content "$CONTENT" \
    >> "$LOG" 2>&1 || echo "[$(date '+%H:%M:%S')] auto-checkpoint failed (non-fatal)" >> "$LOG"
fi

# (3) Allow the compaction. Empty stdout = allow in Claude Code hook spec.
echo "[$(date '+%H:%M:%S')] PRE-COMPACT allow (seal+checkpoint complete)" >> "$LOG"
exit 0
