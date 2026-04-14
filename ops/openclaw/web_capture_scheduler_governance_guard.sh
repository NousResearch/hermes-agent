#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
STATE_PATH="${OPENCLAW_WEB_CAPTURE_SCHEDULER_STATE_PATH:-$ROOT/state/continuity/latest/web_capture_scheduler_state.json}"
RUNNER_SCRIPT="${OPENCLAW_WEB_CAPTURE_SCHEDULER_RUNNER:-$ROOT/ops/openclaw/run_web_capture_scheduler.sh}"
MAX_AGE_SEC="${OPENCLAW_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC:-21600}"
PREEMPTIVE_PROBE_SEC="${OPENCLAW_WEB_CAPTURE_SCHEDULER_PREEMPTIVE_PROBE_SEC:-900}"
AUTO_PROBE="${OPENCLAW_WEB_CAPTURE_SCHEDULER_GOVERNANCE_AUTO_PROBE:-1}"
JSON_OUT=0
STRICT_EXIT=0

usage() {
  cat <<'EOF'
Usage: web_capture_scheduler_governance_guard.sh [options]

Fail-close scheduler-governance contract guard.
- Reads governed scheduler state contract from continuity latest state.
- Verifies schema/contract validity + freshness.
- Optional self-heal probe: run scheduler --dry-run --json once when unhealthy,
  then re-evaluate state.
- Emits exactly one protocol line first:
  - READY: ...
  - BLOCKER: ...

Options:
  --state-path <path>     Scheduler state path override
  --runner <path>         Scheduler runner path override
  --max-age-sec <n>       Freshness max age seconds (default: env or 21600)
  --preemptive-probe-sec <n>
                          Probe early when state age reaches max_age-sec minus this margin
                          (default: env or 900; 0 disables preemptive probe)
  --no-probe              Disable auto probe/self-heal attempt
  --json                  Emit structured JSON after first protocol line
  --strict                Exit non-zero when blocker state persists
  -h, --help
EOF
}

sanitize_inline() {
  printf '%s' "${1:-}" | tr '\r\n\t' '   ' | sed -e 's/[[:space:]]\+/ /g' -e 's/^ *//' -e 's/ *$//'
}

is_truthy() {
  local raw="${1:-}"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

evaluate_state_json() {
  python3 - "$STATE_PATH" "$MAX_AGE_SEC" <<'PY'
import datetime as dt
import json
import pathlib
import sys

state_path = pathlib.Path(sys.argv[1])
try:
    max_age_sec = max(0, int(sys.argv[2]))
except Exception:
    max_age_sec = 21600

EXPECTED_SCHEMA = "openclaw.web_capture.scheduler_state.v1"
ALLOWED_SELECTION_STATUS = {"executed", "selected_dry_run", "idle_no_eligible_macro"}


def parse_iso(raw: str):
    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(txt)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


result = {
    "ok": False,
    "state_path": str(state_path),
    "state_exists": state_path.exists(),
    "schema_version": None,
    "selection_status": None,
    "updated_at": None,
    "state_age_sec": None,
    "contract_state_valid": None,
    "eligible_macros": None,
    "total_macros": None,
    "max_age_sec": max_age_sec,
    "reasons": [],
}

if not state_path.exists():
    result["reasons"].append("scheduler_state_missing")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)

try:
    raw_obj = json.loads(state_path.read_text(encoding="utf-8"))
except Exception as exc:
    result["reasons"].append("scheduler_state_unreadable")
    result["parse_error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)

if not isinstance(raw_obj, dict):
    result["reasons"].append("scheduler_state_not_object")
    result["raw_type"] = type(raw_obj).__name__
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)

result["schema_version"] = raw_obj.get("schema_version")
result["selection_status"] = raw_obj.get("selection_status")
result["updated_at"] = raw_obj.get("updated_at")

summary = raw_obj.get("summary") if isinstance(raw_obj.get("summary"), dict) else {}
result["eligible_macros"] = summary.get("eligible_macros")
result["total_macros"] = summary.get("total_macros")

contract = raw_obj.get("contract") if isinstance(raw_obj.get("contract"), dict) else None
if contract is None:
    result["reasons"].append("scheduler_contract_missing")
else:
    contract_valid = contract.get("state_valid")
    result["contract_state_valid"] = bool(contract_valid) if contract_valid is not None else None
    if contract_valid is not True:
        result["reasons"].append("scheduler_contract_invalid")

if str(result["schema_version"] or "") != EXPECTED_SCHEMA:
    result["reasons"].append("scheduler_schema_version_invalid")

selection_status = str(result["selection_status"] or "").strip()
if not selection_status:
    result["reasons"].append("scheduler_selection_status_missing")
elif selection_status not in ALLOWED_SELECTION_STATUS:
    result["reasons"].append("scheduler_selection_status_invalid")

updated_dt = parse_iso(result.get("updated_at"))
if result.get("updated_at") is None or str(result.get("updated_at") or "").strip() == "":
    result["reasons"].append("scheduler_updated_at_missing")
elif updated_dt is None:
    result["reasons"].append("scheduler_updated_at_invalid")
else:
    age_sec = max(0, int((dt.datetime.now(dt.timezone.utc) - updated_dt).total_seconds()))
    result["state_age_sec"] = age_sec
    if max_age_sec > 0 and age_sec > max_age_sec:
        result["reasons"].append("scheduler_state_stale")

if result.get("total_macros") is not None:
    try:
        total_macros = int(result.get("total_macros"))
        if total_macros < 0:
            raise ValueError("negative")
    except Exception:
        result["reasons"].append("scheduler_total_macros_invalid")

if result.get("eligible_macros") is not None:
    try:
        eligible_macros = int(result.get("eligible_macros"))
        if eligible_macros < 0:
            raise ValueError("negative")
    except Exception:
        result["reasons"].append("scheduler_eligible_macros_invalid")

result["ok"] = len(result["reasons"]) == 0
print(json.dumps(result, ensure_ascii=False))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-path)
      STATE_PATH="${2:-}"
      shift 2
      ;;
    --runner)
      RUNNER_SCRIPT="${2:-}"
      shift 2
      ;;
    --max-age-sec)
      MAX_AGE_SEC="${2:-}"
      shift 2
      ;;
    --preemptive-probe-sec)
      PREEMPTIVE_PROBE_SEC="${2:-}"
      shift 2
      ;;
    --no-probe)
      AUTO_PROBE=0
      shift
      ;;
    --json)
      JSON_OUT=1
      shift
      ;;
    --strict)
      STRICT_EXIT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$MAX_AGE_SEC" =~ ^[0-9]+$ ]]; then
  echo "--max-age-sec must be an integer" >&2
  exit 2
fi
if ! [[ "$PREEMPTIVE_PROBE_SEC" =~ ^[0-9]+$ ]]; then
  echo "--preemptive-probe-sec must be an integer" >&2
  exit 2
fi

initial_eval="$(evaluate_state_json)"
final_eval="$initial_eval"
probe_attempted=0
probe_ok=0
probe_rc=0
probe_first_line=""
probe_err=""
probe_skipped_reason=""

readarray -t initial_fields < <(python3 - "$initial_eval" <<'PY'
import json,sys
obj=json.loads(sys.argv[1])
print('1' if obj.get('ok') else '0')
print(str(obj.get('state_age_sec') if obj.get('state_age_sec') is not None else ''))
PY
)

initial_ok="${initial_fields[0]:-0}"
initial_state_age_sec="${initial_fields[1]:-}"
preemptive_probe_due=0
preemptive_probe_trigger_age=""

if is_truthy "$AUTO_PROBE" && [[ "$initial_ok" == "1" ]] && [[ "$PREEMPTIVE_PROBE_SEC" -gt 0 ]] && [[ "$MAX_AGE_SEC" -gt 0 ]] && [[ "$initial_state_age_sec" =~ ^[0-9]+$ ]]; then
  probe_margin="$PREEMPTIVE_PROBE_SEC"
  if [[ "$probe_margin" -gt "$MAX_AGE_SEC" ]]; then
    probe_margin="$MAX_AGE_SEC"
  fi
  preemptive_probe_trigger_age="$((MAX_AGE_SEC - probe_margin))"
  if [[ "$initial_state_age_sec" -ge "$preemptive_probe_trigger_age" ]]; then
    preemptive_probe_due=1
  fi
fi

if is_truthy "$AUTO_PROBE" && ([[ "$initial_ok" != "1" ]] || [[ "$preemptive_probe_due" -eq 1 ]]); then
  probe_attempted=1
  if [[ ! -x "$RUNNER_SCRIPT" ]]; then
    probe_rc=127
    probe_skipped_reason="runner_missing"
    probe_err="runner_missing:$RUNNER_SCRIPT"
  else
    set +e
    probe_stdout="$($RUNNER_SCRIPT --dry-run --json 2>/tmp/web_capture_scheduler_governance_guard_probe.err)"
    probe_rc=$?
    set -e
    probe_first_line="$(printf '%s\n' "$probe_stdout" | awk 'NF{print; exit}')"
    probe_err="$(cat /tmp/web_capture_scheduler_governance_guard_probe.err 2>/dev/null || true)"

    if [[ "$probe_rc" -eq 0 ]]; then
      probe_ok=1
      final_eval="$(evaluate_state_json)"
    fi
  fi
fi

readarray -t final_fields < <(python3 - "$final_eval" <<'PY'
import json,sys
obj=json.loads(sys.argv[1])
reasons=list(obj.get('reasons') or [])
print('1' if obj.get('ok') else '0')
print(str(reasons[0] if reasons else 'ok'))
print(str(obj.get('state_age_sec') if obj.get('state_age_sec') is not None else ''))
print(str(obj.get('selection_status') or ''))
print(str(obj.get('contract_state_valid') if obj.get('contract_state_valid') is not None else ''))
print(str(obj.get('state_exists') if obj.get('state_exists') is not None else ''))
print(str(obj.get('schema_version') or ''))
print('|'.join(str(r) for r in reasons[:6]))
PY
)

final_ok="${final_fields[0]:-0}"
primary_reason="${final_fields[1]:-unknown}"
state_age_sec="${final_fields[2]:-}"
selection_status="${final_fields[3]:-unknown}"
contract_state_valid="${final_fields[4]:-unknown}"
state_exists="${final_fields[5]:-unknown}"
schema_version="${final_fields[6]:-}"
reason_csv="${final_fields[7]:-}"

if [[ "$final_ok" == "1" ]]; then
  age_segment="state_age_sec=${state_age_sec:-unknown}"
  if [[ "$probe_attempted" -eq 1 && "$probe_ok" -eq 1 && "$initial_ok" != "1" ]]; then
    echo "READY: web_capture_scheduler_governance_ok; ${age_segment}; selection_status=${selection_status}; recovered_via_probe=1"
  elif [[ "$probe_attempted" -eq 1 && "$probe_ok" -eq 1 && "$preemptive_probe_due" -eq 1 ]]; then
    echo "READY: web_capture_scheduler_governance_ok; ${age_segment}; selection_status=${selection_status}; refreshed_via_preemptive_probe=1"
  else
    echo "READY: web_capture_scheduler_governance_ok; ${age_segment}; selection_status=${selection_status}"
  fi
else
  detail="reason=${primary_reason}; state_exists=${state_exists}; contract_state_valid=${contract_state_valid}; schema_version=${schema_version:-unknown}; state_age_sec=${state_age_sec:-unknown}; selection_status=${selection_status}; reasons=${reason_csv:-none}"
  if [[ "$preemptive_probe_due" -eq 1 ]]; then
    detail+="; preemptive_probe_due=1; preemptive_probe_trigger_age_sec=${preemptive_probe_trigger_age:-unknown}; initial_state_age_sec=${initial_state_age_sec:-unknown}"
  fi
  if [[ "$probe_attempted" -eq 1 ]]; then
    detail+="; probe_attempted=1; probe_rc=${probe_rc}"
    if [[ -n "$probe_skipped_reason" ]]; then
      detail+="; probe_skipped_reason=${probe_skipped_reason}"
    fi
    if [[ -n "$probe_first_line" ]]; then
      detail+="; probe_first_line=$(sanitize_inline "$probe_first_line")"
    fi
    if [[ -n "$probe_err" ]]; then
      detail+="; probe_err=$(sanitize_inline "$probe_err")"
    fi
  fi
  echo "BLOCKER: web_capture_scheduler_governance_failed; ${detail:0:340}"
fi

if [[ "$JSON_OUT" -eq 1 ]]; then
  python3 - "$initial_eval" "$final_eval" "$probe_attempted" "$probe_ok" "$probe_rc" "$probe_first_line" "$probe_err" "$probe_skipped_reason" "$RUNNER_SCRIPT" "$STATE_PATH" "$MAX_AGE_SEC" "$PREEMPTIVE_PROBE_SEC" "$preemptive_probe_due" "$preemptive_probe_trigger_age" "$initial_state_age_sec" <<'PY'
import datetime as dt
import json
import sys

initial_eval = json.loads(sys.argv[1])
final_eval = json.loads(sys.argv[2])
probe_attempted = str(sys.argv[3]) == '1'
probe_ok = str(sys.argv[4]) == '1'
probe_rc = int(sys.argv[5])
probe_first_line = str(sys.argv[6] or '')
probe_err = str(sys.argv[7] or '')
probe_skipped_reason = str(sys.argv[8] or '')
runner_script = str(sys.argv[9] or '')
state_path = str(sys.argv[10] or '')
max_age_sec = int(sys.argv[11])
preemptive_probe_sec = int(sys.argv[12])
preemptive_probe_due = str(sys.argv[13]) == '1'
preemptive_probe_trigger_age_raw = str(sys.argv[14] or '').strip()
initial_state_age_raw = str(sys.argv[15] or '').strip()

preemptive_probe_trigger_age = int(preemptive_probe_trigger_age_raw) if preemptive_probe_trigger_age_raw.isdigit() else None
initial_state_age = int(initial_state_age_raw) if initial_state_age_raw.isdigit() else None

payload = {
    "schema_version": "openclaw.web_capture.scheduler_governance_guard.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
    "ok": bool(final_eval.get("ok")),
    "state_path": state_path,
    "max_age_sec": max_age_sec,
    "preemptive_probe_sec": preemptive_probe_sec,
    "preemptive_probe_due": preemptive_probe_due,
    "preemptive_probe_trigger_age_sec": preemptive_probe_trigger_age,
    "initial_state_age_sec": initial_state_age,
    "initial_eval": initial_eval,
    "final_eval": final_eval,
    "probe": {
        "attempted": probe_attempted,
        "ok": probe_ok,
        "rc": probe_rc,
        "runner": runner_script,
        "skipped_reason": probe_skipped_reason or None,
        "first_line": probe_first_line or None,
        "stderr": (probe_err[:400] if probe_err else None),
    },
}

print(json.dumps(payload, ensure_ascii=False))
PY
fi

if [[ "$STRICT_EXIT" -eq 1 && "$final_ok" != "1" ]]; then
  exit 1
fi

exit 0
