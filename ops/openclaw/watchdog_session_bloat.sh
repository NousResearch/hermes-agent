#!/usr/bin/env bash
set -euo pipefail

# Guardrail: if the primary Telegram DM session transcript grows too large,
# archive it, write a handover brief, and drop the session pointer so it recreates cleanly.
#
# Exit codes:
# 0 = no action
# 3 = reset performed (archive + handover written)

THRESHOLD_MB=${THRESHOLD_MB:-5}
PREOVERFLOW_RATIO="${OPENCLAW_SESSION_BLOAT_PREOVERFLOW_RATIO:-0.80}"
PREOVERFLOW_THRESHOLD_MB="${OPENCLAW_SESSION_BLOAT_PREOVERFLOW_THRESHOLD_MB:-}"
PREOVERFLOW_MIN_INTERVAL_SEC="${OPENCLAW_SESSION_BLOAT_PREOVERFLOW_MIN_INTERVAL_SEC:-3600}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
KEY="${OPENCLAW_TARGET_SESSION_KEY:-agent:codex-orchestrator-pro:telegram:direct:5936691533}"
TARGET_AGENT_ID="${OPENCLAW_TARGET_AGENT_ID:-$(printf '%s' "$KEY" | cut -d: -f2)}"
if [[ -z "$TARGET_AGENT_ID" || "$TARGET_AGENT_ID" == "$KEY" ]]; then
  TARGET_AGENT_ID="codex-orchestrator-pro"
fi
STORE="${OPENCLAW_SESSION_STORE_PATH:-/home/yeqiuqiu/.openclaw/agents/${TARGET_AGENT_ID}/sessions/sessions.json}"
ARCHDIR=/home/yeqiuqiu/.openclaw/_archives/sessions
HANDOVER_DIR=/home/yeqiuqiu/.openclaw/_handover
SESSION_ARCHIVE_STEM="${OPENCLAW_SESSION_BLOAT_ARCHIVE_STEM:-$(printf '%s' "$KEY" | sed -e 's/[^A-Za-z0-9._-]/_/g')}"
if [[ -z "$SESSION_ARCHIVE_STEM" ]]; then
  SESSION_ARCHIVE_STEM="session_bloat"
fi
VERIFY_SCRIPT="$ROOT/ops/openclaw/continuity/verify_then_resume.sh"
RECONCILE_SCRIPT="$ROOT/ops/openclaw/continuity/reconcile.sh"
CHECKPOINT_SCRIPT="$ROOT/ops/openclaw/continuity/write_checkpoint.sh"
HANDOVER_COMPAT_SCRIPT="$ROOT/ops/openclaw/continuity/render_context_handover_compat.sh"
EVENT_ROUTER="${OPENCLAW_EVENT_ROUTER_SCRIPT:-$ROOT/ops/openclaw/event_router.sh}"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
VERIFY_GATE_LIB="$ROOT/ops/openclaw/lib/verify_gate.sh"
EVENT_COOLDOWN_SEC="${OPENCLAW_SESSION_BLOAT_EVENT_COOLDOWN_SEC:-1800}"
ENFORCE_VERIFY_THEN_RESUME="${OPENCLAW_ENFORCE_VERIFY_THEN_RESUME:-1}"
AUTO_RECONCILE_DRIFT="${OPENCLAW_AUTO_RECONCILE_DRIFT:-1}"

# Run-throttle and archive hygiene (to keep timer spam + archive growth bounded)
STATE_FILE="${OPENCLAW_SESSION_BLOAT_STATE_FILE:-$ROOT/state/cron_watchdog/session_bloat_watchdog_state.json}"
LOCK_FILE="${OPENCLAW_SESSION_BLOAT_LOCK_FILE:-/tmp/openclaw_watchdog_session_bloat.lock}"
MIN_RUN_INTERVAL_SEC="${OPENCLAW_SESSION_BLOAT_MIN_RUN_INTERVAL_SEC:-900}"
RESET_MIN_INTERVAL_SEC="${OPENCLAW_SESSION_BLOAT_RESET_MIN_INTERVAL_SEC:-10800}"
FORCE_RUN="${OPENCLAW_SESSION_BLOAT_FORCE_RUN:-0}"

ARCHIVE_PRUNE_INTERVAL_SEC="${OPENCLAW_SESSION_BLOAT_ARCHIVE_PRUNE_INTERVAL_SEC:-21600}"
ARCHIVE_RETENTION_DAYS="${OPENCLAW_SESSION_BLOAT_ARCHIVE_RETENTION_DAYS:-21}"
ARCHIVE_MAX_MB="${OPENCLAW_SESSION_BLOAT_ARCHIVE_MAX_MB:-80}"
ARCHIVE_HIGH_WATER_MB="${OPENCLAW_SESSION_BLOAT_ARCHIVE_HIGH_WATER_MB:-60}"
FORCE_ARCHIVE_PRUNE="${OPENCLAW_SESSION_BLOAT_FORCE_ARCHIVE_PRUNE:-0}"

mkdir -p "$ARCHDIR" "$HANDOVER_DIR" "$(dirname "$STATE_FILE")"

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"
# shellcheck source=ops/openclaw/lib/verify_gate.sh
source "$VERIFY_GATE_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.session_bloat"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$STORE"

state_update() {
  python3 - "$STATE_FILE" "$@" <<'PY'
import json, os, sys
path = sys.argv[1]
updates = sys.argv[2:]
state = {}
if os.path.exists(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f) or {}
    except Exception:
        state = {}
for item in updates:
    if '=' not in item:
        continue
    k, v = item.split('=', 1)
    vv = v.strip()
    if vv.lower() in ('true', 'false'):
        state[k] = (vv.lower() == 'true')
    else:
        try:
            state[k] = int(vv)
        except Exception:
            state[k] = vv
with open(path, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2, sort_keys=True)
    f.write('\n')
PY
}

read_state_field() {
  local key="$1"
  python3 - "$STATE_FILE" "$key" <<'PY'
import json, os, sys
path, key = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print(0)
    raise SystemExit
try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f) or {}
except Exception:
    data = {}
val = data.get(key, 0)
if isinstance(val, bool):
    print(1 if val else 0)
elif isinstance(val, (int, float)):
    print(int(val))
else:
    try:
        print(int(str(val).strip()))
    except Exception:
        print(0)
PY
}

compute_preoverflow_threshold_mb() {
  python3 - "$THRESHOLD_MB" "$PREOVERFLOW_RATIO" "$PREOVERFLOW_THRESHOLD_MB" <<'PY'
import math
import sys

hard = max(1, int(float(sys.argv[1])))
ratio_raw = (sys.argv[2] or '').strip()
explicit_raw = (sys.argv[3] or '').strip()

if explicit_raw:
    try:
        val = int(float(explicit_raw))
    except Exception:
        val = 0
else:
    try:
        ratio = float(ratio_raw)
    except Exception:
        ratio = 0.80
    if ratio <= 0:
        ratio = 0.80
    val = int(math.floor(hard * ratio))

if val < 1:
    val = 1
if hard > 1 and val >= hard:
    val = hard - 1
elif hard <= 1:
    val = hard

print(val)
PY
}

write_guard_checkpoint_and_handover() {
  local callsite="$1"
  local trigger="$2"
  local status="$3"
  local objective="$4"
  local size_mb="$5"
  local threshold_mb="$6"
  local checkpoint_out checkpoint_rc checkpoint_id handover_out handover_rc handover_rel err

  if [[ ! -x "$CHECKPOINT_SCRIPT" ]]; then
    openclaw_watchdog_route_blocker \
      "session_guard_checkpoint_missing" \
      "session_guard_checkpoint_missing; reason=missing_checkpoint_writer; script=${CHECKPOINT_SCRIPT}" \
      "$CHECKPOINT_SCRIPT" \
      "warn" >/dev/null
    return 1
  fi
  if [[ ! -x "$HANDOVER_COMPAT_SCRIPT" ]]; then
    openclaw_watchdog_route_blocker \
      "session_guard_handover_missing" \
      "session_guard_handover_missing; reason=missing_handover_renderer; script=${HANDOVER_COMPAT_SCRIPT}" \
      "$HANDOVER_COMPAT_SCRIPT" \
      "warn" >/dev/null
    return 1
  fi

  set +e
  checkpoint_out="$(OPENCLAW_INTERNAL_MUTATION=1 OPENCLAW_INTERNAL_MUTATION_CALLSITE="$callsite" "$CHECKPOINT_SCRIPT" \
    --trigger "$trigger" \
    --status "$status" \
    --objective "$objective" \
    --next-action "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/verify_then_resume.sh" \
    --next-action "openclaw sessions --agent ${TARGET_AGENT_ID} --active 1440 --json" \
    --next-action "echo 'Telegram DM thin lane: cockpit only; move heavy orchestration to worker sessions.'" \
    --verify-cmd "openclaw sessions --agent ${TARGET_AGENT_ID} --active 1440 --json >/dev/null" \
    --verify-cmd "openclaw gateway status --json >/dev/null" \
    2>/tmp/watchdog_session_bloat_checkpoint.err)"
  checkpoint_rc=$?
  set -e
  if [[ "$checkpoint_rc" -ne 0 ]]; then
    err="$(cat /tmp/watchdog_session_bloat_checkpoint.err 2>/dev/null || true)"
    openclaw_watchdog_route_blocker \
      "session_guard_checkpoint_failed" \
      "session_guard_checkpoint_failed; trigger=${trigger}; size_mb=${size_mb}; threshold_mb=${threshold_mb}; rc=${checkpoint_rc}; err=${err:0:180}" \
      "$CHECKPOINT_SCRIPT" \
      "warn" >/dev/null
    return 1
  fi

  checkpoint_id="$(python3 -c 'import json,sys; print((json.loads(sys.argv[1]).get("checkpoint_id") or ""))' "$checkpoint_out")"
  if [[ -z "$checkpoint_id" ]]; then
    openclaw_watchdog_route_blocker \
      "session_guard_checkpoint_invalid" \
      "session_guard_checkpoint_invalid; trigger=${trigger}; size_mb=${size_mb}; threshold_mb=${threshold_mb}; reason=missing_checkpoint_id" \
      "$CHECKPOINT_SCRIPT" \
      "warn" >/dev/null
    return 1
  fi

  set +e
  handover_out="$($HANDOVER_COMPAT_SCRIPT --checkpoint "$checkpoint_id" 2>/tmp/watchdog_session_bloat_handover.err)"
  handover_rc=$?
  set -e
  if [[ "$handover_rc" -ne 0 ]]; then
    err="$(cat /tmp/watchdog_session_bloat_handover.err 2>/dev/null || true)"
    openclaw_watchdog_route_blocker \
      "session_guard_handover_failed" \
      "session_guard_handover_failed; trigger=${trigger}; checkpoint=${checkpoint_id}; rc=${handover_rc}; err=${err:0:180}" \
      "$HANDOVER_COMPAT_SCRIPT" \
      "warn" >/dev/null
    return 1
  fi

  handover_rel="$(python3 -c 'import json,sys; print((json.loads(sys.argv[1]).get("handover_path") or "reports/handover_context_latest.md"))' "$handover_out")"
  printf '%s\t%s\n' "$checkpoint_id" "$handover_rel"
}

prune_archive_junk() {
  python3 - "$ARCHDIR" "$HANDOVER_DIR" "$ARCHIVE_RETENTION_DAYS" "$ARCHIVE_MAX_MB" "$ARCHIVE_HIGH_WATER_MB" <<'PY'
import json
import os
import sys
import time

archdir = sys.argv[1]
handover_dir = sys.argv[2]
retention_days = max(0, int(sys.argv[3]))
max_mb = max(1, int(sys.argv[4]))
high_mb = max(1, int(sys.argv[5]))
max_bytes = max_mb * 1024 * 1024
high_bytes = min(max_bytes, high_mb * 1024 * 1024)

report = {
    "ageDeletedCount": 0,
    "ageDeletedBytes": 0,
    "budgetDeletedCount": 0,
    "budgetDeletedBytes": 0,
    "remainingArchiveBytes": 0,
}

cutoff = time.time() - (retention_days * 86400)

for d in (archdir, handover_dir):
    if not os.path.isdir(d):
        continue
    for name in os.listdir(d):
        path = os.path.join(d, name)
        if not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        if retention_days > 0 and st.st_mtime < cutoff:
            report["ageDeletedCount"] += 1
            report["ageDeletedBytes"] += st.st_size
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

files = []
total = 0
if os.path.isdir(archdir):
    for name in os.listdir(archdir):
        path = os.path.join(archdir, name)
        if not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        total += st.st_size
        files.append((st.st_mtime, st.st_size, path))

if total > max_bytes:
    files.sort(key=lambda x: x[0])
    for _, size, path in files:
        if total <= high_bytes:
            break
        try:
            os.remove(path)
            total -= size
            report["budgetDeletedCount"] += 1
            report["budgetDeletedBytes"] += size
        except FileNotFoundError:
            pass

report["remainingArchiveBytes"] = max(0, int(total))
print(json.dumps(report, ensure_ascii=False))
PY
}

NOW_EPOCH="$(date +%s)"

# Prevent overlap if timer fires while prior run is still active.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

LAST_RUN_EPOCH="$(read_state_field lastRunEpoch)"
LAST_ARCHIVE_PRUNE_EPOCH="$(read_state_field lastArchivePruneEpoch)"
LAST_RESET_EPOCH="$(read_state_field lastResetEpoch)"
LAST_PREOVERFLOW_EPOCH="$(read_state_field lastPreoverflowEpoch)"

if [[ "$FORCE_RUN" != "1" ]] && (( MIN_RUN_INTERVAL_SEC > 0 )) && (( LAST_RUN_EPOCH > 0 )); then
  if (( NOW_EPOCH - LAST_RUN_EPOCH < MIN_RUN_INTERVAL_SEC )); then
    exit 0
  fi
fi

state_update "lastRunEpoch=$NOW_EPOCH"

if [[ "$FORCE_ARCHIVE_PRUNE" == "1" ]] || (( ARCHIVE_PRUNE_INTERVAL_SEC <= 0 )) || (( NOW_EPOCH - LAST_ARCHIVE_PRUNE_EPOCH >= ARCHIVE_PRUNE_INTERVAL_SEC )); then
  prune_report="$(prune_archive_junk)"
  state_update "lastArchivePruneEpoch=$NOW_EPOCH"
  pruned_count="$(python3 -c 'import json,sys; print((json.loads(sys.argv[1]).get("ageDeletedCount",0)+json.loads(sys.argv[1]).get("budgetDeletedCount",0)))' "$prune_report")"
  if [[ "$pruned_count" != "0" ]]; then
    echo "archive_pruned: $prune_report"
  fi
fi

if [ ! -f "$STORE" ]; then
  echo "missing sessions store: $STORE" >&2
  exit 0
fi

SESSION_META=$(python3 - "$STORE" "$KEY" <<'PY'
import json
import sys

store = sys.argv[1]
key = sys.argv[2]
with open(store, 'r', encoding='utf-8') as f:
    data = json.load(f)
obj = data.get(key) or {}
print(json.dumps({
  'sessionId': obj.get('sessionId',''),
  'sessionFile': obj.get('sessionFile','')
}))
PY
)

SESSION_ID=$(python3 - <<PY
import json
print(json.loads('''$SESSION_META''').get('sessionId',''))
PY
)
SESSION_FILE=$(python3 - <<PY
import json
print(json.loads('''$SESSION_META''').get('sessionFile',''))
PY
)

if [ -z "$SESSION_FILE" ] || [ ! -f "$SESSION_FILE" ]; then
  exit 0
fi

size_bytes=$(stat -c %s "$SESSION_FILE" 2>/dev/null || echo 0)
size_mb=$((size_bytes/1024/1024))

PREOVERFLOW_THRESHOLD_EFFECTIVE="$(compute_preoverflow_threshold_mb)"

if (( PREOVERFLOW_THRESHOLD_EFFECTIVE > 0 )) && [ "$size_mb" -ge "$PREOVERFLOW_THRESHOLD_EFFECTIVE" ] && [ "$size_mb" -lt "$THRESHOLD_MB" ]; then
  if [[ "$FORCE_RUN" != "1" ]] && (( PREOVERFLOW_MIN_INTERVAL_SEC > 0 )) && (( LAST_PREOVERFLOW_EPOCH > 0 )); then
    if (( NOW_EPOCH - LAST_PREOVERFLOW_EPOCH < PREOVERFLOW_MIN_INTERVAL_SEC )); then
      openclaw_watchdog_route_blocker \
        "session_preoverflow_cooldown" \
        "session_preoverflow_cooldown; size_mb=${size_mb}; preoverflow_mb=${PREOVERFLOW_THRESHOLD_EFFECTIVE}; threshold_mb=${THRESHOLD_MB}; cooldown_sec=${PREOVERFLOW_MIN_INTERVAL_SEC}" \
        "$SESSION_FILE" \
        "warn" >/dev/null
      exit 0
    fi
  fi

  guard_refs="$(write_guard_checkpoint_and_handover \
    "watchdog_session_bloat.sh:preoverflow_checkpoint" \
    "telegram_preoverflow_guard" \
    "PROGRESS" \
    "Telegram direct lane nearing bloat threshold; refresh successor continuity and keep lane thin before overflow." \
    "$size_mb" \
    "$THRESHOLD_MB")" || exit 0
  checkpoint_id="${guard_refs%%$'\t'*}"
  handover_rel="${guard_refs#*$'\t'}"

  openclaw_watchdog_route_blocker \
    "session_preoverflow_guard" \
    "session_preoverflow_guard; size_mb=${size_mb}; preoverflow_mb=${PREOVERFLOW_THRESHOLD_EFFECTIVE}; threshold_mb=${THRESHOLD_MB}; checkpoint=${checkpoint_id}; handover=${handover_rel}; lane=telegram_thin" \
    "$ROOT/$handover_rel" \
    "warn" >/dev/null

  state_update "lastPreoverflowEpoch=$NOW_EPOCH"
  echo "preoverflow telegram session guard: ${size_mb}MB >= ${PREOVERFLOW_THRESHOLD_EFFECTIVE}MB (hard reset at ${THRESHOLD_MB}MB)"
  echo "checkpoint: $checkpoint_id"
  echo "handover: $handover_rel"
  exit 0
fi

if [ "$size_mb" -lt "$THRESHOLD_MB" ]; then
  exit 0
fi

if [[ "$FORCE_RUN" != "1" ]] && (( RESET_MIN_INTERVAL_SEC > 0 )) && (( LAST_RESET_EPOCH > 0 )); then
  if (( NOW_EPOCH - LAST_RESET_EPOCH < RESET_MIN_INTERVAL_SEC )); then
    openclaw_watchdog_route_blocker \
      "session_reset_cooldown" \
      "session_reset_cooldown; size_mb=${size_mb}; threshold_mb=${THRESHOLD_MB}; cooldown_sec=${RESET_MIN_INTERVAL_SEC}" \
      "$SESSION_FILE" \
      "warn" >/dev/null
    exit 0
  fi
fi

openclaw_run_drift_reconcile_best_effort \
  --reconcile-script "$RECONCILE_SCRIPT" \
  --enabled "$AUTO_RECONCILE_DRIFT" \
  --stdout-file "/tmp/session_bloat_reconcile.out" \
  --stderr-file "/tmp/session_bloat_reconcile.err"

if [[ "$ENFORCE_VERIFY_THEN_RESUME" == "1" ]]; then
  if ! openclaw_verify_then_resume_gate \
    --task "session_bloat" \
    --verify-script "$VERIFY_SCRIPT" \
    --verify-report "$ROOT/state/continuity/latest/verify_last.json" \
    --strict-autonomy-regressions \
    --evidence-ref "$STORE" \
    --summary-extra "size_mb=${size_mb}; threshold_mb=${THRESHOLD_MB}" \
    --stdout-file "/tmp/session_bloat_verify.out" \
    --stderr-file "/tmp/session_bloat_verify.err"; then
    exit 0
  fi
fi

guard_refs="$(write_guard_checkpoint_and_handover \
  "watchdog_session_bloat.sh:hard_reset_checkpoint" \
  "telegram_session_bloat_reset" \
  "BLOCKER" \
  "Telegram direct lane exceeded session bloat threshold; prepare successor handover before archive+pointer reset." \
  "$size_mb" \
  "$THRESHOLD_MB")" || exit 0
reset_checkpoint_id="${guard_refs%%$'\t'*}"
reset_handover_rel="${guard_refs#*$'\t'}"

TS=$(date +%s)
ARCHIVE="$ARCHDIR/${SESSION_ARCHIVE_STEM}.${TS}.jsonl"
HANDOVER="$HANDOVER_DIR/${SESSION_ARCHIVE_STEM}.${TS}.md"

# 1) archive transcript
mv "$SESSION_FILE" "$ARCHIVE"

# 2) write handover brief (deterministic; no LLM)
python3 - "$ARCHIVE" "$HANDOVER" "$KEY" "$SESSION_ID" <<'PY'
import json, os, sys
archive = sys.argv[1]
handover = sys.argv[2]
key = sys.argv[3]
session_id = sys.argv[4]

def extract_text(msg):
    # OpenClaw session lines contain message.content as list of {type,text}
    parts=[]
    for c in msg.get('content',[]) or []:
        if isinstance(c, dict) and c.get('type')=='text' and c.get('text'):
            parts.append(c['text'])
    return "\n".join(parts).strip()

rows=[]
with open(archive,'r',errors='ignore') as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        msg=obj.get('message')
        if not isinstance(msg, dict):
            continue
        role=msg.get('role')
        if role not in ('user','assistant'):
            continue
        text=extract_text(msg)
        if not text:
            continue
        rows.append((obj.get('timestamp',''), role, text))

# take the last N messages as context
N=24
rows_tail=rows[-N:]

# very cheap "open tasks" heuristic from the tail
needles=(
    'next step', 'todo', 'to do', 'fix', 'still', 'need to', 'please', 'can you',
    'watchdog', 'handover', 'reset', 'not responding'
)
open_items=[]
for _,role,text in rows_tail:
    t=text.lower()
    if any(n in t for n in needles):
        open_items.append((role,text))
open_items=open_items[-8:]

with open(handover,'w') as out:
    out.write(f"# OpenClaw Telegram DM Handover (auto)\n\n")
    out.write(f"- key: `{key}`\n")
    if session_id:
        out.write(f"- sessionId: `{session_id}`\n")
    out.write(f"- archived transcript: `{archive}`\n")
    out.write(f"- created_at: {os.popen('date -Iseconds').read().strip()}\n\n")

    out.write("## What happened\n")
    out.write("This DM session was automatically reset because the transcript grew too large (risk of context-limit stalls).\n\n")

    if open_items:
        out.write("## Likely open threads (heuristic)\n")
        for role,text in open_items:
            text_one=text.replace('\n',' ').strip()
            if len(text_one)>220:
                text_one=text_one[:217]+'...'
            out.write(f"- **{role}**: {text_one}\n")
        out.write("\n")

    out.write(f"## Last {len(rows_tail)} messages (most recent)\n")
    for ts,role,text in rows_tail:
        text=text.strip()
        if len(text)>800:
            text=text[:800]+'...'
        out.write(f"### {role}\n\n")
        out.write(text)
        out.write("\n\n")
PY

# 3) drop session pointer
python3 - "$STORE" "$KEY" <<'PY'
import json
import sys

store_path = sys.argv[1]
key = sys.argv[2]
with open(store_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
removed = data.pop(key, None)
with open(store_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, sort_keys=True)
print('removed' if removed else 'not_found')
PY

state_update "lastResetEpoch=$NOW_EPOCH"

echo "reset telegram session due to bloat: ${size_mb}MB >= ${THRESHOLD_MB}MB"
echo "reset_checkpoint: $reset_checkpoint_id"
echo "reset_handover: $reset_handover_rel"
echo "archived: $ARCHIVE"
echo "handover: $HANDOVER"

openclaw_watchdog_route_blocker \
  "session_reset_performed" \
  "session_reset_performed; size_mb=${size_mb}; threshold_mb=${THRESHOLD_MB}; checkpoint=${reset_checkpoint_id}; handover=${reset_handover_rel}; lane=telegram_thin" \
  "$HANDOVER" \
  "warn" >/dev/null

exit 3
