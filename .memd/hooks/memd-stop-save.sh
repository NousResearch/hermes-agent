#!/usr/bin/env bash
set -euo pipefail

SAVE_INTERVAL="${MEMD_SAVE_INTERVAL:-15}"
STATE_DIR="${MEMD_HOOK_STATE_DIR:-$HOME/.memd/hook_state}"
mkdir -p "$STATE_DIR"

INPUT="$(cat)"

eval "$(
  printf '%s' "$INPUT" | python3 -c '
import json, re, sys
data = json.load(sys.stdin)
sid = data.get("session_id", "unknown")
sha = data.get("stop_hook_active", False)
tp = data.get("transcript_path", "")
safe = lambda s: re.sub(r"[^a-zA-Z0-9_/.\-~]", "", str(s))
print(f"SESSION_ID=\"{safe(sid)}\"")
print(f"STOP_HOOK_ACTIVE=\"{sha}\"")
print(f"TRANSCRIPT_PATH=\"{safe(tp)}\"")
' 2>/dev/null
)"

TRANSCRIPT_PATH="${TRANSCRIPT_PATH/#\~/$HOME}"

if [[ "$STOP_HOOK_ACTIVE" == "True" || "$STOP_HOOK_ACTIVE" == "true" ]]; then
  echo "{}"
  exit 0
fi

EXCHANGE_COUNT=0
if [[ -f "$TRANSCRIPT_PATH" ]]; then
  EXCHANGE_COUNT="$(
    python3 - "$TRANSCRIPT_PATH" <<'PYEOF'
import json, sys
count = 0
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
            msg = entry.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and "<command-message>" in content:
                    continue
                count += 1
        except Exception:
            pass
print(count)
PYEOF
  )"
fi

LAST_SAVE_FILE="$STATE_DIR/${SESSION_ID}_last_save"
LAST_SAVE=0
if [[ -f "$LAST_SAVE_FILE" ]]; then
  LAST_SAVE="$(cat "$LAST_SAVE_FILE")"
fi

SINCE_LAST=$((EXCHANGE_COUNT - LAST_SAVE))
echo "[$(date '+%H:%M:%S')] Session $SESSION_ID: $EXCHANGE_COUNT exchanges, $SINCE_LAST since last memd save" >> "$STATE_DIR/hook.log"

if [[ "$SINCE_LAST" -ge "$SAVE_INTERVAL" && "$EXCHANGE_COUNT" -gt 0 ]]; then
  echo "$EXCHANGE_COUNT" > "$LAST_SAVE_FILE"
  echo "[$(date '+%H:%M:%S')] TRIGGERING memd stop save at exchange $EXCHANGE_COUNT" >> "$STATE_DIR/hook.log"
  cat <<'HOOKJSON'
{
  "decision": "block",
  "reason": "AUTO-SAVE checkpoint. Before stopping, persist the important state from this session into memd. Prefer compact truth over summary sludge: 1. run memd checkpoint for the current task state, 2. write any durable decisions/corrections/preferences, 3. if you have a compaction packet or turn-state delta, run memd hook spill --output .memd --stdin --apply, 4. then continue."
}
HOOKJSON
else
  echo "{}"
fi
