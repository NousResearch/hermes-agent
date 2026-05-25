#!/usr/bin/env bash
# memd bootstrap hook — runs on UserPromptSubmit in Claude Code.
# Hard enforcement: injects memd wake output before the model reasons.
# Skips if last wake was within MEMD_WAKE_TTL seconds (default 120).
set -euo pipefail

MEMD_WAKE_TTL="${MEMD_WAKE_TTL:-120}"

# Read hook input from stdin (Claude Code sends JSON with session_id, cwd, etc.)
INPUT="$(cat)"
CWD="$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cwd',''))" 2>/dev/null || echo "")"
SESSION_ID="$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || echo "")"
if [ -z "$CWD" ]; then
  CWD="$(pwd)"
fi

# Walk up to find .memd bundle
find_bundle() {
  local dir="$1"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/.memd/config.json" ]; then
      echo "$dir/.memd"
      return
    fi
    dir="$(dirname "$dir")"
  done
  echo ""
}

BUNDLE_ROOT="$(find_bundle "$CWD")"
if [ -z "$BUNDLE_ROOT" ]; then
  # No bundle — nothing to bootstrap
  exit 0
fi

# Helper: emit properly structured hookSpecificOutput JSON
emit_context() {
  local ctx="$1"
  local escaped
  escaped="$(printf '%s' "$ctx" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read())[1:-1])")"
  printf '%s' "{\"hookSpecificOutput\":{\"hookEventName\":\"UserPromptSubmit\",\"additionalContext\":\"${escaped}\"}}"
}

# Staleness check: skip if wake ran recently
MARKER_FILE="$BUNDLE_ROOT/.last-wake"
SESSION_MARKER_DIR="$BUNDLE_ROOT/state/bootstrap-sessions"
SESSION_MARKER_FILE=""
if [ -n "$SESSION_ID" ]; then
  SAFE_SESSION_ID="$(printf '%s' "$SESSION_ID" | tr -c 'A-Za-z0-9._-' '_')"
  SESSION_MARKER_FILE="$SESSION_MARKER_DIR/$SAFE_SESSION_ID"
fi

session_has_live_wake_receipt() {
  [ -n "$SESSION_MARKER_FILE" ] && [ -f "$SESSION_MARKER_FILE" ]
}

stamp_live_wake_receipt() {
  if [ -n "$SESSION_MARKER_FILE" ]; then
    mkdir -p "$SESSION_MARKER_DIR"
    date +%s > "$SESSION_MARKER_FILE"
  fi
}

if [ -f "$MARKER_FILE" ]; then
  LAST_WAKE="$(cat "$MARKER_FILE" 2>/dev/null || echo "0")"
  NOW="$(date +%s)"
  AGE=$(( NOW - LAST_WAKE ))
  if [ "$AGE" -lt "$MEMD_WAKE_TTL" ] && session_has_live_wake_receipt; then
    # Still fresh for this session — inject cached wake instead of re-running
    if [ -f "$BUNDLE_ROOT/wake.md" ]; then
      CACHED="$(cat "$BUNDLE_ROOT/wake.md")"
      emit_context "memd bootstrap (cached ${AGE}s ago):\n${CACHED}"
    fi
    exit 0
  fi
fi

# Run memd wake — the real enforcement
WAKE_OUTPUT="$(memd wake --output "$BUNDLE_ROOT" --write 2>&1 || true)"

if [ -n "$WAKE_OUTPUT" ]; then
  # Stamp the marker
  date +%s > "$MARKER_FILE"
  stamp_live_wake_receipt
  emit_context "memd bootstrap (live):\n${WAKE_OUTPUT}"
elif [ -f "$BUNDLE_ROOT/wake.md" ] && session_has_live_wake_receipt; then
  # Backend down — serve stale cache with warning
  CACHED="$(cat "$BUNDLE_ROOT/wake.md")"
  emit_context "memd bootstrap (stale fallback — backend unreachable):\n${CACHED}"
else
  emit_context "memd bootstrap failure: live wake required for this session before cached wake can be trusted."
fi
