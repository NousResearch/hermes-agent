#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
RUNNER_SCRIPT="$ROOT/ops/openclaw/architecture/run_competitive_parity_harness.sh"
LOCK_WRAP="$ROOT/ops/openclaw/cron_wrappers/openclaw_cron_lock_timeout.sh"
EVENT_ROUTER="$ROOT/ops/openclaw/event_router.sh"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"

STATE_DIR="$ROOT/state/cron_watchdog"
TELEMETRY_DIR="$ROOT/ops/telemetry/textfile"
LAST_JSON="$STATE_DIR/competitive_parity_last.json"
SCHEDULE_STATE_JSON="$STATE_DIR/competitive_parity_schedule_state.json"

TIMEOUT_SEC="${OPENCLAW_COMPETITIVE_PARITY_TIMEOUT_SEC:-900}"
GRACE_SEC="${OPENCLAW_COMPETITIVE_PARITY_TIMEOUT_GRACE_SEC:-30}"
EVENT_COOLDOWN_SEC="${OPENCLAW_COMPETITIVE_PARITY_EVENT_COOLDOWN_SEC:-21600}"
MIN_INTERVAL_SEC="${OPENCLAW_COMPETITIVE_PARITY_MIN_INTERVAL_SEC:-604800}"

FORCE_RUN=0
DRY_RUN=0
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: run_competitive_parity_harness.sh [options]

Periodic parity wrapper (silent-by-default, blocker-only events).
Default cadence gate: run at most once every 7 days.

Options:
  --force                  Bypass cadence gate
  --dry-run                Print run/skip decision and exit without executing runner
  --json                   With --dry-run, print machine JSON
  --min-interval-sec <n>   Override cadence minimum interval (default: 604800)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_RUN=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    --min-interval-sec)
      MIN_INTERVAL_SEC="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if ! [[ "$MIN_INTERVAL_SEC" =~ ^[0-9]+$ ]]; then
  echo "invalid --min-interval-sec: $MIN_INTERVAL_SEC (expected integer >= 0)" >&2
  exit 2
fi

mkdir -p "$STATE_DIR" "$TELEMETRY_DIR"

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.competitive_parity"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$LAST_JSON"

load_last_attempt_epoch() {
  python3 - "$SCHEDULE_STATE_JSON" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
if not p.exists():
    print(0)
    raise SystemExit(0)

try:
    obj = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)

val = obj.get("last_attempt_epoch")
try:
    n = int(val)
except Exception:
    n = 0
print(max(0, n))
PY
}

extract_run_id() {
  python3 - "$LAST_JSON" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("")
    raise SystemExit(0)

try:
    obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
except Exception:
    print("")
    raise SystemExit(0)

run_id = obj.get("run_id") or ((obj.get("dashboard") or {}).get("run_id")) or ""
print(str(run_id).strip())
PY
}

write_schedule_state() {
  local rc="$1"
  local status="$2"
  local run_id="$3"
  python3 - "$SCHEDULE_STATE_JSON" "$rc" "$status" "$run_id" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
status = sys.argv[3]
run_id = sys.argv[4]

now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
payload = {
    "schema_version": "openclaw.competitive_parity.schedule.v1",
    "last_attempt_epoch": int(now.timestamp()),
    "last_attempt_iso": now.isoformat().replace("+00:00", "Z"),
    "last_exit_code": rc,
    "last_status": status,
    "last_run_id": run_id or None,
}

path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

now_epoch="$(date +%s)"
last_attempt_epoch="$(load_last_attempt_epoch)"

run_allowed=1
skip_reason=""
age_sec=0
if [[ "$FORCE_RUN" -ne 1 && "$MIN_INTERVAL_SEC" -gt 0 && "$last_attempt_epoch" -gt 0 ]]; then
  age_sec=$(( now_epoch - last_attempt_epoch ))
  if (( age_sec < MIN_INTERVAL_SEC )); then
    run_allowed=0
    skip_reason="min_interval_not_elapsed"
  fi
fi

if [[ "$DRY_RUN" -eq 1 || "$run_allowed" -eq 0 ]]; then
  if [[ "$run_allowed" -eq 0 ]]; then
    if [[ "$DRY_RUN" -eq 1 || "$JSON_OUT" -eq 1 ]]; then
      if [[ "$JSON_OUT" -eq 1 ]]; then
        printf '{"ok":true,"run_allowed":false,"reason":"%s","age_sec":%s,"min_interval_sec":%s,"last_attempt_epoch":%s}\n' \
          "$skip_reason" "$age_sec" "$MIN_INTERVAL_SEC" "$last_attempt_epoch"
      else
        printf 'SKIP: competitive parity cadence gate (%s; age=%ss < min=%ss)\n' "$skip_reason" "$age_sec" "$MIN_INTERVAL_SEC"
      fi
    fi
    exit 0
  fi

  if [[ "$JSON_OUT" -eq 1 ]]; then
    printf '{"ok":true,"run_allowed":true,"reason":"dry_run","min_interval_sec":%s,"force":%s}\n' "$MIN_INTERVAL_SEC" "$FORCE_RUN"
  else
    printf 'READY: competitive parity runner would execute now (dry-run)\n'
  fi
  exit 0
fi

if [[ ! -x "$RUNNER_SCRIPT" ]]; then
  openclaw_watchdog_route_blocker "runner_missing" "task=competitive_parity_harness; reason=runner_missing"
  exit 0
fi

if [[ ! -x "$LOCK_WRAP" ]]; then
  openclaw_watchdog_route_blocker "wrapper_dependency_missing" "task=competitive_parity_harness; reason=wrapper_dependency_missing"
  exit 0
fi

set +e
output="$($LOCK_WRAP \
  --lock-name "competitive_parity_harness" \
  --lock-dir "$STATE_DIR/locks" \
  --timeout-sec "$TIMEOUT_SEC" \
  --grace-sec "$GRACE_SEC" \
  --busy-exit-code 75 \
  --emit-blocker \
  --soft-timeout \
  -- bash "$RUNNER_SCRIPT" --json --no-events 2>&1)"
rc=$?
set -e

if [[ -n "$output" ]]; then
  printf '%s\n' "$output" > "$LAST_JSON"
fi

if [[ "$rc" -eq 75 ]]; then
  exit 0
fi

run_id="$(extract_run_id)"
status="wrapper_exit"
if [[ "$rc" -eq 0 ]]; then
  status="done"
elif [[ "$rc" -eq 2 ]]; then
  status="blocked"
fi
write_schedule_state "$rc" "$status" "$run_id"

if [[ "$rc" -eq 0 ]]; then
  now_epoch="$(date +%s)"
  printf 'openclaw_competitive_parity_last_success_epoch %s\n' "$now_epoch" > "$TELEMETRY_DIR/openclaw_competitive_parity_last_success_epoch.prom"
  exit 0
fi

if [[ "$rc" -eq 2 ]]; then
  summary="$(python3 - "$LAST_JSON" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("task=competitive_parity_harness; reason=blocked")
    raise SystemExit(0)

raw = p.read_text(encoding="utf-8", errors="replace").strip()
obj = {}
try:
    obj = json.loads(raw)
except Exception:
    print("task=competitive_parity_harness; reason=blocked_unparseable_output")
    raise SystemExit(0)

run_id = obj.get("run_id") or ((obj.get("dashboard") or {}).get("run_id")) or "unknown"
dashboard = obj.get("dashboard") if isinstance(obj.get("dashboard"), dict) else {}
summary = dashboard.get("summary") if isinstance(dashboard.get("summary"), dict) else {}
blockers = summary.get("blocker_count")
regressions = summary.get("regression_count")
artifacts = dashboard.get("artifacts") if isinstance(dashboard.get("artifacts"), dict) else {}
evidence = artifacts.get("dashboard_latest") or artifacts.get("scorecard_summary") or "state/architecture/competitive_parity/dashboard/latest.json"
print(f"task=competitive_parity_harness; reason=blocked; run_id={run_id}; blockers={blockers}; regressions={regressions}; evidence={evidence}")
PY
)"
  openclaw_watchdog_route_blocker "harness_blocked" "$summary" "$LAST_JSON"
  exit 0
fi

openclaw_watchdog_route_blocker "wrapper_exit" "task=competitive_parity_harness; reason=wrapper_exit_${rc}" "$LAST_JSON"
exit 0
