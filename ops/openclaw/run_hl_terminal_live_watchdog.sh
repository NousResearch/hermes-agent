#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
REPO="/home/yeqiuqiu/projects/hl-terminal-gemini-canonical"
LOCK_WRAP="$ROOT/ops/openclaw/cron_wrappers/openclaw_cron_lock_timeout.sh"
HASH_WRAP="$ROOT/ops/openclaw/cron_wrappers/openclaw_cron_state_hash.sh"
VERIFY_SCRIPT="$ROOT/ops/openclaw/continuity/verify_then_resume.sh"
RECONCILE_SCRIPT="$ROOT/ops/openclaw/continuity/reconcile.sh"
EVENT_ROUTER="$ROOT/ops/openclaw/event_router.sh"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
VERIFY_GATE_LIB="$ROOT/ops/openclaw/lib/verify_gate.sh"

TELEMETRY_DIR="$ROOT/ops/telemetry/textfile"
STATE_DIR="$ROOT/state/cron_watchdog"

HOST="${HL_TERMINAL_HOST:-127.0.0.1}"
PORT="${HL_TERMINAL_PORT:-3033}"
PROFILE="${HL_TERMINAL_PROFILE:-full}"

CHECK_URL="http://${HOST}:${PORT}/system"
TIMEOUT_SEC="${HL_TERMINAL_LIVE_TIMEOUT_SEC:-600}"
GRACE_SEC="${HL_TERMINAL_LIVE_TIMEOUT_GRACE_SEC:-30}"
EVENT_COOLDOWN_SEC="${OPENCLAW_HL_LIVE_EVENT_COOLDOWN_SEC:-1800}"
ENFORCE_VERIFY_THEN_RESUME="${OPENCLAW_ENFORCE_VERIFY_THEN_RESUME:-1}"
AUTO_RECONCILE_DRIFT="${OPENCLAW_AUTO_RECONCILE_DRIFT:-1}"

mkdir -p "$TELEMETRY_DIR" "$STATE_DIR"

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"
# shellcheck source=ops/openclaw/lib/verify_gate.sh
source "$VERIFY_GATE_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.hl_terminal_live"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$CHECK_URL"

run_verify_gate() {
  if [[ "$ENFORCE_VERIFY_THEN_RESUME" != "1" ]]; then
    return 0
  fi

  openclaw_verify_then_resume_gate \
    --task "hl_terminal_live" \
    --verify-script "$VERIFY_SCRIPT" \
    --verify-report "$ROOT/state/continuity/latest/verify_last.json" \
    --strict-autonomy-regressions \
    --evidence-ref "$ROOT/state/continuity/latest/verify_last.json" \
    --stdout-file "/tmp/hl_terminal_live_verify.out" \
    --stderr-file "/tmp/hl_terminal_live_verify.err"
}

if [[ ! -x "$LOCK_WRAP" || ! -x "$HASH_WRAP" ]]; then
  openclaw_watchdog_route_blocker "wrapper_dependency_missing" "task=hl_terminal_live; reason=wrapper_dependency_missing"
  exit 0
fi

if [[ ! -d "$REPO" ]]; then
  openclaw_watchdog_route_blocker "repo_missing" "task=hl_terminal_live; reason=missing_repo_dir"
  exit 0
fi

# If already up, record success and exit silently.
if curl -fsS --max-time 3 "$CHECK_URL" >/dev/null 2>&1; then
  now_epoch="$(date +%s)"
  printf 'openclaw_hl_terminal_live_last_success_epoch %s\n' "$now_epoch" > "$TELEMETRY_DIR/openclaw_hl_terminal_live_last_success_epoch.prom"
  exit 0
fi

# Dedup input signature: host/port/profile only (deterministic)
"$HASH_WRAP" \
  --state-file "$STATE_DIR/hl_terminal_live_input_hash.json" \
  --input "host=${HOST}|port=${PORT}|profile=${PROFILE}" >/dev/null

openclaw_run_drift_reconcile_best_effort \
  --reconcile-script "$RECONCILE_SCRIPT" \
  --enabled "$AUTO_RECONCILE_DRIFT" \
  --stdout-file "/tmp/hl_terminal_live_reconcile.out" \
  --stderr-file "/tmp/hl_terminal_live_reconcile.err"

if ! run_verify_gate; then
  exit 0
fi

set +e
output="$($LOCK_WRAP \
  --lock-name "hl_terminal_live" \
  --lock-dir "$STATE_DIR/locks" \
  --timeout-sec "$TIMEOUT_SEC" \
  --grace-sec "$GRACE_SEC" \
  --busy-exit-code 75 \
  --emit-blocker \
  --soft-timeout \
  -- bash -lc "cd '$REPO' && node scripts/prod-stack.mjs up --profile '$PROFILE' --host '$HOST' --port '$PORT'" 2>&1)"
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
  # lock held by another run; silent skip
  exit 0
fi

first_line="$(printf '%s\n' "$output" | awk 'NF{print; exit}')"

if [[ "$rc" -ne 0 ]]; then
  if [[ "$first_line" != BLOCKER:* ]]; then
    openclaw_watchdog_route_blocker "wrapper_exit" "task=hl_terminal_live; reason=wrapper_exit_${rc}" "$CHECK_URL"
  else
    openclaw_watchdog_route_blocker "task_blocker" "$(openclaw_blocker_summary_from_line "$first_line")" "$CHECK_URL"
  fi
  exit 0
fi

# Re-check after attempted bring-up.
if curl -fsS --max-time 6 "$CHECK_URL" >/dev/null 2>&1; then
  now_epoch="$(date +%s)"
  printf 'openclaw_hl_terminal_live_last_success_epoch %s\n' "$now_epoch" > "$TELEMETRY_DIR/openclaw_hl_terminal_live_last_success_epoch.prom"
  exit 0
fi

openclaw_watchdog_route_blocker "not_responding_after_up" "task=hl_terminal_live; reason=not_responding_after_up; url=${CHECK_URL}" "$CHECK_URL"
exit 0
