#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
TICK_SCRIPT="$ROOT/ops/autopilot/bin/hl_autopilot_tick.sh"
STATE_FILE="$ROOT/ops/autopilot/state/hl_terminal_v1.json"
LOCK_WRAP="$ROOT/ops/openclaw/cron_wrappers/openclaw_cron_lock_timeout.sh"
HASH_WRAP="$ROOT/ops/openclaw/cron_wrappers/openclaw_cron_state_hash.sh"
VERIFY_SCRIPT="$ROOT/ops/openclaw/continuity/verify_then_resume.sh"
RECONCILE_SCRIPT="$ROOT/ops/openclaw/continuity/reconcile.sh"
QUEUE_SYNC_SCRIPT="$ROOT/ops/openclaw/continuity/queue_sync_from_autopilot_json.sh"
EVENT_ROUTER="$ROOT/ops/openclaw/event_router.sh"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
VERIFY_GATE_LIB="$ROOT/ops/openclaw/lib/verify_gate.sh"

TELEMETRY_DIR="$ROOT/ops/telemetry/textfile"
STATE_DIR="$ROOT/state/cron_watchdog"

TIMEOUT_SEC="${OPENCLAW_AUTOPILOT_TICK_TIMEOUT_SEC:-300}"
GRACE_SEC="${OPENCLAW_AUTOPILOT_TICK_TIMEOUT_GRACE_SEC:-30}"
EVENT_COOLDOWN_SEC="${OPENCLAW_AUTOPILOT_EVENT_COOLDOWN_SEC:-1800}"
ENFORCE_VERIFY_THEN_RESUME="${OPENCLAW_ENFORCE_VERIFY_THEN_RESUME:-1}"
AUTO_RECONCILE_DRIFT="${OPENCLAW_AUTO_RECONCILE_DRIFT:-1}"

mkdir -p "$TELEMETRY_DIR" "$STATE_DIR"

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"
# shellcheck source=ops/openclaw/lib/verify_gate.sh
source "$VERIFY_GATE_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.hl_autopilot_tick"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$ROOT/state/continuity/latest/verify_last.json"

sync_queue_state() {
  if [[ -x "$QUEUE_SYNC_SCRIPT" ]]; then
    OPENCLAW_INTERNAL_MUTATION=1 \
    OPENCLAW_INTERNAL_MUTATION_CALLSITE="run_hl_autopilot_tick_watchdog.sh:sync_queue_state" \
      "$QUEUE_SYNC_SCRIPT" --json "$STATE_FILE" >/dev/null 2>&1 || true
  fi
}

run_verify_gate() {
  if [[ "$ENFORCE_VERIFY_THEN_RESUME" != "1" ]]; then
    return 0
  fi

  openclaw_verify_then_resume_gate \
    --task "hl_autopilot_tick" \
    --verify-script "$VERIFY_SCRIPT" \
    --verify-report "$ROOT/state/continuity/latest/verify_last.json" \
    --strict-autonomy-regressions \
    --stdout-file "/tmp/hl_autopilot_verify.out" \
    --stderr-file "/tmp/hl_autopilot_verify.err"
}

if [[ ! -x "$LOCK_WRAP" || ! -x "$HASH_WRAP" ]]; then
  openclaw_watchdog_route_blocker "wrapper_dependency_missing" "task=hl_autopilot_tick; reason=wrapper_dependency_missing"
  sync_queue_state
  exit 0
fi

if [[ ! -x "$TICK_SCRIPT" ]]; then
  openclaw_watchdog_route_blocker "tick_script_missing" "task=hl_autopilot_tick; reason=tick_script_missing"
  sync_queue_state
  exit 0
fi

state_sig="$(python3 - "$STATE_FILE" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print('state:missing')
    raise SystemExit(0)
raw = p.read_bytes()
print(f"state_hash={hashlib.sha256(raw).hexdigest()}|bytes={len(raw)}")
PY
)"

"$HASH_WRAP" \
  --state-file "$STATE_DIR/hl_autopilot_tick_input_hash.json" \
  --input "$state_sig" >/dev/null

openclaw_run_drift_reconcile_best_effort \
  --reconcile-script "$RECONCILE_SCRIPT" \
  --enabled "$AUTO_RECONCILE_DRIFT" \
  --stdout-file "/tmp/hl_autopilot_reconcile.out" \
  --stderr-file "/tmp/hl_autopilot_reconcile.err"

if ! run_verify_gate; then
  sync_queue_state
  exit 0
fi

set +e
output="$($LOCK_WRAP \
  --lock-name "hl_autopilot_tick" \
  --lock-dir "$STATE_DIR/locks" \
  --timeout-sec "$TIMEOUT_SEC" \
  --grace-sec "$GRACE_SEC" \
  --busy-exit-code 75 \
  --emit-blocker \
  --soft-timeout \
  -- bash "$TICK_SCRIPT" 2>&1)"
rc=$?
set -e

if [[ -n "$output" && "${OPENCLAW_WATCHDOG_VERBOSE_STDOUT:-0}" == "1" ]]; then
  filtered_output="$(printf '%s\n' "$output" | sed '/^BLOCKER:/d')"
  if [[ -n "$filtered_output" ]]; then
    # Keep raw child output off stdout so cron/chat first-line relays cannot leak internals.
    printf '%s\n' "$filtered_output" >&2
  fi
fi

if [[ "$rc" -eq 75 ]]; then
  sync_queue_state
  exit 0
fi

first_line="$(printf '%s\n' "$output" | awk 'NF{print; exit}')"

if [[ "$rc" -ne 0 ]]; then
  if [[ "$first_line" != BLOCKER:* ]]; then
    openclaw_watchdog_route_blocker "wrapper_exit" "task=hl_autopilot_tick; reason=wrapper_exit_${rc}"
  else
    openclaw_watchdog_route_blocker "task_blocker" "$(openclaw_blocker_summary_from_line "$first_line")"
  fi
  sync_queue_state
  exit 0
fi

if [[ "$first_line" == BLOCKER:* ]]; then
  openclaw_watchdog_route_blocker "task_blocker" "$(openclaw_blocker_summary_from_line "$first_line")"
  sync_queue_state
  exit 0
fi

now_epoch="$(date +%s)"
printf 'openclaw_autopilot_tick_last_success_epoch %s\n' "$now_epoch" > "$TELEMETRY_DIR/openclaw_autopilot_tick_last_success_epoch.prom"

sync_queue_state
exit 0
