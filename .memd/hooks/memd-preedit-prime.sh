#!/usr/bin/env bash
# PreToolUse guard for Edit/Write/NotebookEdit.
#
# A3 Part 1 stretch: eliminate the "File has not been read yet" round-trip
# after compaction. When Claude tries to Edit a file that:
#   - is NOT in the current session's live file-interaction ledger, AND
#   - IS in a prior session's sealed ledger
# we inject `additionalContext` telling Claude to Read the file first (via
# `memd prime-reads` or direct Read). This surfaces the prime-reads list at
# the exact moment it's needed, instead of waiting for the Read-before-Edit
# error.
#
# Non-blocking by default: emits additionalContext, NOT decision=block.
# Set MEMD_PREEDIT_MODE=block to upgrade to hard blocking. Silent when the
# file is already known to the current session, or was never touched.
set -euo pipefail

MODE="${MEMD_PREEDIT_MODE:-context}"
STATE_DIR="${MEMD_HOOK_STATE_DIR:-$HOME/.memd/hook_state}"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/hook.log"

INPUT="$(cat)"

# Resolve bundle root by walking up from cwd, like memd-bootstrap.sh does.
CWD="$(MEMD_INPUT="$INPUT" python3 -c 'import json,os;print((json.loads(os.environ.get("MEMD_INPUT","")) or {}).get("cwd",""))' 2>/dev/null || echo "")"
[ -z "$CWD" ] && CWD="$(pwd)"
BUNDLE_ROOT=""
dir="$CWD"
while [ "$dir" != "/" ]; do
  if [ -f "$dir/.memd/config.json" ]; then
    BUNDLE_ROOT="$dir/.memd"
    break
  fi
  dir="$(dirname "$dir")"
done
if [ -z "$BUNDLE_ROOT" ]; then
  # No memd bundle in scope — nothing to do.
  exit 0
fi

# Extract session_id, tool_name, file_path from payload. Pass via env var
# so the heredoc remains the Python script body (python3 - reads script from
# stdin, so stdin is not available for the payload).
eval "$(MEMD_INPUT="$INPUT" python3 - <<'PYEOF' 2>/dev/null || true
import json, os, re, sys
try:
    d = json.loads(os.environ.get("MEMD_INPUT", ""))
except Exception:
    sys.exit(0)
sid = d.get("session_id", "")
tn = d.get("tool_name", "")
fp = ""
ti = d.get("tool_input") or {}
if isinstance(ti, dict):
    fp = ti.get("file_path") or ti.get("notebook_path") or ""
safe = lambda s: re.sub(r'[^A-Za-z0-9_./\-~@:+= ]', '', str(s))
print(f'SESSION_ID="{safe(sid)}"')
print(f'TOOL_NAME="{safe(tn)}"')
print(f'FILE_PATH="{safe(fp)}"')
PYEOF
)"

SESSION_ID="${SESSION_ID:-}"
TOOL_NAME="${TOOL_NAME:-}"
FILE_PATH="${FILE_PATH:-}"

# Only act on file-write tools with a path.
case "$TOOL_NAME" in
  Edit|Write|NotebookEdit) ;;
  *) exit 0 ;;
esac
[ -z "$FILE_PATH" ] && exit 0

STATE="$BUNDLE_ROOT/state"
LIVE="$STATE/session-$SESSION_ID/file_interactions.json"

# 1) If this session already Read the file, nothing to do.
if [ -f "$LIVE" ]; then
  HAS_READ="$(python3 - "$LIVE" "$FILE_PATH" <<'PYEOF' 2>/dev/null || echo no
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("no"); sys.exit(0)
target = sys.argv[2]
for e in d.get("entries", []):
    if e.get("path") == target and e.get("op") == "read":
        print("yes"); sys.exit(0)
print("no")
PYEOF
)"
  [ "$HAS_READ" = "yes" ] && exit 0
fi

# 2) Check prior sealed ledgers for this path.
PRIOR_HIT="$(python3 - "$STATE" "$FILE_PATH" "$SESSION_ID" <<'PYEOF' 2>/dev/null || echo ""
import os, json, sys, glob
state, target, self_sid = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.isdir(state):
    sys.exit(0)
found = []
for sess_dir in sorted(glob.glob(os.path.join(state, "session-*"))):
    sid = os.path.basename(sess_dir).removeprefix("session-")
    if sid == self_sid:
        continue
    sealed_dir = os.path.join(sess_dir, "sealed")
    if not os.path.isdir(sealed_dir):
        continue
    sealed_files = sorted(glob.glob(os.path.join(sealed_dir, "*.json")))
    if not sealed_files:
        continue
    try:
        d = json.load(open(sealed_files[-1], encoding="utf-8"))
    except Exception:
        continue
    for e in d.get("entries", []):
        if e.get("path") == target:
            found.append((sid, e.get("op"), e.get("last_ts_ms", 0)))
            break
if found:
    found.sort(key=lambda x: x[2], reverse=True)
    sid, op, ts = found[0]
    print(f"{sid}\t{op}\t{ts}")
PYEOF
)"

if [ -z "$PRIOR_HIT" ]; then
  exit 0
fi

PRIOR_SID="$(printf '%s' "$PRIOR_HIT" | cut -f1)"
PRIOR_OP="$(printf '%s' "$PRIOR_HIT" | cut -f2)"

echo "[$(date '+%H:%M:%S')] PRE-EDIT prime: session=$SESSION_ID tool=$TOOL_NAME path=$FILE_PATH prior_session=$PRIOR_SID prior_op=$PRIOR_OP mode=$MODE" >> "$LOG"

REASON="memd prime-reads: prior session $PRIOR_SID touched '$FILE_PATH' (op=$PRIOR_OP). This session has not Read it yet; the $TOOL_NAME call will fail the Read-before-Edit check. Run: Read('$FILE_PATH') before retrying, or 'memd prime-reads --since-session $PRIOR_SID' for the full list."

if [ "$MODE" = "block" ]; then
  python3 -c "import json; print(json.dumps({'decision':'block','reason':__import__('sys').argv[1]}))" "$REASON"
  exit 0
fi

# Default: emit additionalContext so Claude sees the prime instruction inline.
python3 -c "
import json, sys
print(json.dumps({
  'hookSpecificOutput': {
    'hookEventName': 'PreToolUse',
    'additionalContext': sys.argv[1]
  }
}))
" "$REASON"
exit 0
