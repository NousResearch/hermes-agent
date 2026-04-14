#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
STATE_FILE="${OPENCLAW_MEMORY_CONSOLIDATION_WATCHDOG_STATE_FILE:-$ROOT/state/cron_watchdog/memory_consolidation_watchdog.json}"
CONSOLIDATION_SCRIPT="${OPENCLAW_MEMORY_CONSOLIDATION_SCRIPT:-$ROOT/scripts/memory_consolidation.py}"
DRY_RUN_BLOCKED_THRESHOLD_HOURS="${OPENCLAW_MEMORY_CONSOLIDATION_BLOCKED_THRESHOLD_HOURS:-24}"
NOTIFY_REMINDER_SEC="${OPENCLAW_MEMORY_CONSOLIDATION_NOTIFY_REMINDER_SEC:-7200}"
EVENT_ROUTER="${OPENCLAW_EVENT_ROUTER_SCRIPT:-$ROOT/ops/openclaw/event_router.sh}"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
EVENT_COOLDOWN_SEC="${OPENCLAW_MEMORY_CONSOLIDATION_EVENT_COOLDOWN_SEC:-3600}"

mkdir -p "$(dirname "$STATE_FILE")"

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.memory_consolidation"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$STATE_FILE"

set +e
py_output="$(python3 - "$STATE_FILE" "$CONSOLIDATION_SCRIPT" "$ROOT" "$DRY_RUN_BLOCKED_THRESHOLD_HOURS" "$NOTIFY_REMINDER_SEC" <<'PY'
import json
import os
import pathlib
import subprocess
import sys
import time

state_file = pathlib.Path(sys.argv[1])
consolidation_script = pathlib.Path(sys.argv[2])
root = pathlib.Path(sys.argv[3])
blocked_threshold_hours = max(1, int(sys.argv[4]))
notify_reminder_sec = max(60, int(sys.argv[5]))

def load_state() -> dict:
    if not state_file.exists():
        return {
            "consecutive_failures": 0,
            "first_failure_epoch": 0,
            "last_failure_epoch": 0,
            "last_failure_reason": "",
            "last_notified_epoch": 0,
            "last_notified_signature": "",
            "updated_epoch": 0,
        }
    try:
        obj = json.loads(state_file.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {
        "consecutive_failures": 0,
        "first_failure_epoch": 0,
        "last_failure_epoch": 0,
        "last_failure_reason": "",
        "last_notified_epoch": 0,
        "last_notified_signature": "",
        "updated_epoch": 0,
    }

def save_state(state: dict) -> None:
    state["updated_epoch"] = int(time.time())
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, state_file)

state = load_state()
now = int(time.time())

# Run memory consolidation dry-run
cmd = [
    sys.executable,
    str(consolidation_script),
    "--repo-root",
    str(root),
    "--json",
    "run",
    "--dry-run",
]
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
except subprocess.TimeoutExpired:
    state["consecutive_failures"] += 1
    state["last_failure_reason"] = "timeout"
    if state["first_failure_epoch"] == 0:
        state["first_failure_epoch"] = now
    state["last_failure_epoch"] = now
    save_state(state)
    print("BLOCKER_JSON:" + json.dumps({
        "step": "memory_consolidation_dry_run",
        "reason": "timeout",
        "summary": "memory consolidation dry-run timed out",
    }))
    raise SystemExit(0)

if result.returncode != 0:
    if result.returncode == 2:
        # Treat as blocked governance gate
        state["consecutive_failures"] += 1
        state["last_failure_reason"] = "blocked_governance_gate"
        if state["first_failure_epoch"] == 0:
            state["first_failure_epoch"] = now
        state["last_failure_epoch"] = now
        save_state(state)
        # Check if stale
        blocked_duration = now - state["first_failure_epoch"]
        if blocked_duration >= blocked_threshold_hours * 3600:
            last_notified = state.get("last_notified_epoch", 0)
            if now - last_notified >= notify_reminder_sec:
                state["last_notified_epoch"] = now
                state["last_notified_signature"] = f"blocked_governance_gate_{now}"
                save_state(state)
                print("BLOCKER_JSON:" + json.dumps({
                    "step": "memory_consolidation_governance_gate_blocked",
                    "reason": "stale_blocked",
                    "summary": f"memory consolidation governance gate blocked for {blocked_duration // 3600}h (threshold {blocked_threshold_hours}h)",
                }))
        raise SystemExit(0)
    else:
        state["consecutive_failures"] += 1
        state["last_failure_reason"] = f"exit_code_{result.returncode}"
        if state["first_failure_epoch"] == 0:
            state["first_failure_epoch"] = now
        state["last_failure_epoch"] = now
        save_state(state)
        print("BLOCKER_JSON:" + json.dumps({
            "step": "memory_consolidation_dry_run",
            "reason": "non_zero_exit",
            "summary": f"memory consolidation dry-run failed with exit code {result.returncode}",
        }))
        raise SystemExit(0)

# Parse JSON output
try:
    payload = json.loads(result.stdout.strip())
except json.JSONDecodeError:
    state["consecutive_failures"] += 1
    state["last_failure_reason"] = "invalid_json_output"
    if state["first_failure_epoch"] == 0:
        state["first_failure_epoch"] = now
    state["last_failure_epoch"] = now
    save_state(state)
    print("BLOCKER_JSON:" + json.dumps({
        "step": "memory_consolidation_dry_run",
        "reason": "invalid_json_output",
        "summary": "memory consolidation dry-run produced invalid JSON",
    }))
    raise SystemExit(0)

status = payload.get("status")
if status == "blocked_governance_gate":
    state["consecutive_failures"] += 1
    state["last_failure_reason"] = "blocked_governance_gate"
    if state["first_failure_epoch"] == 0:
        state["first_failure_epoch"] = now
    state["last_failure_epoch"] = now
    save_state(state)
    # Check if stale
    blocked_duration = now - state["first_failure_epoch"]
    if blocked_duration >= blocked_threshold_hours * 3600:
        # Check if we need to notify (cooldown)
        last_notified = state.get("last_notified_epoch", 0)
        if now - last_notified >= notify_reminder_sec:
            state["last_notified_epoch"] = now
            state["last_notified_signature"] = f"blocked_governance_gate_{now}"
            save_state(state)
            print("BLOCKER_JSON:" + json.dumps({
                "step": "memory_consolidation_governance_gate_blocked",
                "reason": "stale_blocked",
                "summary": f"memory consolidation governance gate blocked for {blocked_duration // 3600}h (threshold {blocked_threshold_hours}h)",
            }))
    raise SystemExit(0)

# Any other failure statuses
if not payload.get("ok", False):
    state["consecutive_failures"] += 1
    state["last_failure_reason"] = f"failure_status_{status}"
    if state["first_failure_epoch"] == 0:
        state["first_failure_epoch"] = now
    state["last_failure_epoch"] = now
    save_state(state)
    print("BLOCKER_JSON:" + json.dumps({
        "step": "memory_consolidation_dry_run",
        "reason": "failure",
        "summary": f"memory consolidation dry-run failed with status {status}",
    }))
    raise SystemExit(0)

# Success: clear consecutive failures
state["consecutive_failures"] = 0
state["first_failure_epoch"] = 0
state["last_failure_epoch"] = 0
state["last_failure_reason"] = ""
save_state(state)
print("OK")
PY
)"
py_rc=$?
set -e

if [[ "$py_rc" -ne 0 ]]; then
  openclaw_watchdog_route_blocker "memory_consolidation_watchdog_script_error" "task=memory_consolidation_watchdog; reason=script_error_${py_rc}" "$STATE_FILE"
  exit 0
fi

if [[ -z "$py_output" ]]; then
  exit 0
fi

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  if [[ "$line" == BLOCKER_JSON:* ]]; then
    openclaw_watchdog_route_blocker \
      "memory_consolidation_governance_gate_blocked" \
      "$(openclaw_blocker_summary_from_line "$line")" \
      "$STATE_FILE" \
      "warning"
  else
    printf '%s\n' "$line"
  fi
done <<< "$py_output"

exit 0