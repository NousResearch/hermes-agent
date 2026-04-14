#!/usr/bin/env bash
# shellcheck shell=bash

# Shared verify-before-mutate gate helper for wrappers/watchdogs.
# Requires blocker_routing.sh to be sourced first when routed blocker emission is desired.
#
# Main entrypoint:
#   openclaw_verify_then_resume_gate --task <task> --verify-script <path> [options]
#
# Returns:
#   0 -> verify gate passed
#   1 -> verify gate failed/missing and blocker was emitted

openclaw_truthy() {
  local raw="${1:-}"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    1|true|yes|y|on)
      return 0 ;;
    *)
      return 1 ;;
  esac
}

openclaw_verify_gate_resolve_root() {
  local root="${OPENCLAW_ROOT:-}"
  if [[ -n "$root" ]]; then
    printf '%s' "$root"
    return 0
  fi

  local src="${BASH_SOURCE[0]:-}"
  if [[ -n "$src" ]]; then
    local inferred_root
    inferred_root="$(cd "$(dirname "$src")/../../.." 2>/dev/null && pwd -P || true)"
    if [[ -n "$inferred_root" ]]; then
      printf '%s' "$inferred_root"
      return 0
    fi
  fi

  printf '%s' "/home/yeqiuqiu/clawd-architect"
}

openclaw_verify_gate_resolve_strict_autonomy() {
  local strict_autonomy_override="${1:-}"
  local strict_autonomy_policy_raw="${OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REGRESSIONS-}"
  local strict_autonomy_legacy_raw="${OPENCLAW_STRICT_AUTONOMY_REGRESSIONS-}"
  local strict_autonomy_required_raw="${OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REQUIRED:-0}"
  local strict_autonomy_policy_set="0"
  local strict_autonomy_legacy_set="0"

  if [[ -n "${OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REGRESSIONS+x}" ]]; then
    strict_autonomy_policy_set="1"
  fi
  if [[ -n "${OPENCLAW_STRICT_AUTONOMY_REGRESSIONS+x}" ]]; then
    strict_autonomy_legacy_set="1"
  fi

  local strict_autonomy="1"
  local strict_autonomy_effective="0"
  local strict_autonomy_required_effective="0"
  local strict_autonomy_source="default_on"
  local strict_autonomy_override_denied="0"

  # Shared policy toggle for wrapper-level strict autonomy regression gating.
  # Global default is strict-on. Explicit env policy (or legacy env for
  # backward compatibility) can still disable it by setting a falsey value.
  # Optional fail-closed policy: OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REQUIRED=1.
  if [[ "$strict_autonomy_policy_set" == "1" ]]; then
    strict_autonomy="$strict_autonomy_policy_raw"
    strict_autonomy_source="verify_gate_policy_env"
  elif [[ "$strict_autonomy_legacy_set" == "1" ]]; then
    strict_autonomy="$strict_autonomy_legacy_raw"
    strict_autonomy_source="legacy_env"
  fi

  if [[ "$strict_autonomy_override" == "1" ]]; then
    strict_autonomy="1"
    strict_autonomy_source="wrapper_flag_enable"
  elif [[ "$strict_autonomy_override" == "0" ]]; then
    strict_autonomy="0"
    strict_autonomy_source="wrapper_flag_disable"
  fi

  if openclaw_truthy "$strict_autonomy_required_raw"; then
    strict_autonomy_required_effective="1"
    strict_autonomy="1"
    strict_autonomy_source="verify_gate_required_env"
    if [[ "$strict_autonomy_override" == "0" ]]; then
      strict_autonomy_override_denied="1"
    fi
  fi

  if openclaw_truthy "$strict_autonomy"; then
    strict_autonomy_effective="1"
  fi

  OPENCLAW_VERIFY_GATE_RESOLVED_ENABLED="$strict_autonomy_effective"
  OPENCLAW_VERIFY_GATE_RESOLVED_REQUIRED="$strict_autonomy_required_effective"
  OPENCLAW_VERIFY_GATE_RESOLVED_SOURCE="$strict_autonomy_source"
  OPENCLAW_VERIFY_GATE_RESOLVED_OVERRIDE="$strict_autonomy_override"
  OPENCLAW_VERIFY_GATE_RESOLVED_OVERRIDE_DENIED="$strict_autonomy_override_denied"
  OPENCLAW_VERIFY_GATE_RESOLVED_POLICY_RAW="$strict_autonomy_policy_raw"
  OPENCLAW_VERIFY_GATE_RESOLVED_LEGACY_RAW="$strict_autonomy_legacy_raw"
  OPENCLAW_VERIFY_GATE_RESOLVED_REQUIRED_RAW="$strict_autonomy_required_raw"
}

openclaw_verify_reason_from_report() {
  local report_path="${1:-}"
  python3 - "$report_path" <<'PY'
import json
import pathlib
import sys

raw = str(sys.argv[1] or "").strip()
if not raw:
    print("verify_report_missing")
    raise SystemExit(0)

p = pathlib.Path(raw)
if not p.exists():
    print("verify_report_missing")
    raise SystemExit(0)

try:
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        print(obj.get("reason") or obj.get("status") or "verify_failed")
    else:
        print("verify_failed")
except Exception:
    print("verify_report_unreadable")
PY
}

openclaw_verify_emit_blocker() {
  local key="${1:-verify_then_resume_blocker}"
  local summary="${2:-task=unknown; reason=verify_failed}"
  local evidence_ref="${3:-}"

  if declare -F openclaw_watchdog_route_blocker >/dev/null 2>&1; then
    openclaw_watchdog_route_blocker "$key" "$summary" "$evidence_ref" >/dev/null
  else
    printf 'BLOCKER: %s\n' "$summary"
  fi
}

openclaw_verify_gate_storm_guard_decide() {
  local task="${1:-unknown_task}"
  local verify_report="${2:-}"
  local default_root="${3:-}"

  OPENCLAW_VERIFY_GATE_STORM_GUARD_DECISION="run"
  OPENCLAW_VERIFY_GATE_STORM_GUARD_REASON="under_budget"
  OPENCLAW_VERIFY_GATE_STORM_GUARD_WINDOW_COUNT="0"
  OPENCLAW_VERIFY_GATE_STORM_GUARD_EFFECTIVE_WINDOW_SEC="0"
  OPENCLAW_VERIFY_GATE_STORM_GUARD_EFFECTIVE_MAX_RUNS="0"
  OPENCLAW_VERIFY_GATE_STORM_GUARD_REPORT_STATUS="unknown"
  OPENCLAW_VERIFY_GATE_STORM_GUARD_REPORT_AGE_SEC=""
  local state_path_override="${OPENCLAW_VERIFY_GATE_STORM_GUARD_STATE_PATH:-}"
  OPENCLAW_VERIFY_GATE_STORM_GUARD_STATE_PATH=""

  local enabled_raw="${OPENCLAW_VERIFY_GATE_STORM_GUARD_ENABLED:-1}"
  if ! openclaw_truthy "$enabled_raw"; then
    OPENCLAW_VERIFY_GATE_STORM_GUARD_REASON="guard_disabled"
    return 0
  fi

  local window_sec_raw="${OPENCLAW_VERIFY_GATE_STORM_GUARD_WINDOW_SEC:-60}"
  local max_runs_raw="${OPENCLAW_VERIFY_GATE_STORM_GUARD_MAX_RUNS:-4}"
  local reuse_ready_max_age_sec_raw="${OPENCLAW_VERIFY_GATE_STORM_GUARD_REUSE_READY_MAX_AGE_SEC:-120}"
  local over_budget_missing_report_grace_runs_raw="${OPENCLAW_VERIFY_GATE_STORM_GUARD_OVER_BUDGET_MISSING_REPORT_GRACE_RUNS:-1}"
  local state_path="${state_path_override:-$default_root/state/continuity/latest/verify_gate_storm_guard_state.json}"

  local payload
  payload="$(python3 - "$task" "$verify_report" "$state_path" "$window_sec_raw" "$max_runs_raw" "$reuse_ready_max_age_sec_raw" "$over_budget_missing_report_grace_runs_raw" <<'PY'
import datetime as dt
import fcntl
import json
import os
import pathlib
import sys
import time


def parse_iso(value: object) -> dt.datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_non_negative_int(raw: str, default: int) -> int:
    try:
        value = int(float(str(raw).strip()))
    except Exception:
        value = int(default)
    return max(0, value)


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def emit(**fields: object) -> None:
    for key, value in fields.items():
        text = str(value if value is not None else "")
        text = text.replace("\n", " ").replace("\r", " ")
        print(f"{key}={text}")


task = str(sys.argv[1] or "unknown_task").strip() or "unknown_task"
verify_report = pathlib.Path(str(sys.argv[2] or "").strip())
state_path = pathlib.Path(str(sys.argv[3] or "").strip())
window_sec = parse_non_negative_int(sys.argv[4], 60)
max_runs = parse_non_negative_int(sys.argv[5], 4)
reuse_ready_max_age_sec = parse_non_negative_int(sys.argv[6], 120)
over_budget_missing_report_grace_runs = parse_non_negative_int(sys.argv[7], 1)

now_ts = int(time.time())
decision = "run"
reason = "under_budget"
report_status = "UNKNOWN"
report_age_sec = ""
window_count = 0

try:
    if not str(state_path):
        raise RuntimeError("missing_state_path")

    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = pathlib.Path(str(state_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)

        previous: dict[str, object] = {}
        if state_path.exists():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    previous = loaded
            except Exception:
                previous = {}

        consecutive_over_budget_unverified_runs = 0
        total_over_budget_unverified_grace_runs = 0
        if isinstance(previous, dict):
            try:
                consecutive_over_budget_unverified_runs = max(
                    0,
                    int(previous.get("consecutive_over_budget_unverified_runs") or 0),
                )
            except Exception:
                consecutive_over_budget_unverified_runs = 0
            try:
                total_over_budget_unverified_grace_runs = max(
                    0,
                    int(previous.get("total_over_budget_unverified_grace_runs") or 0),
                )
            except Exception:
                total_over_budget_unverified_grace_runs = 0

        history = previous.get("window_runs") if isinstance(previous, dict) else []
        runs: list[int] = []
        if isinstance(history, list):
            for item in history:
                try:
                    ts = int(item)
                except Exception:
                    continue
                if window_sec <= 0 or now_ts - ts <= window_sec:
                    runs.append(ts)

        prospective_window_count = len(runs) + 1
        window_count = len(runs)

        if window_sec <= 0 or max_runs <= 0:
            decision = "run"
            reason = "guard_bypassed_nonpositive_budget"
            consecutive_over_budget_unverified_runs = 0
            runs.append(now_ts)
        elif prospective_window_count > max_runs:
            report_payload: dict[str, object] = {}
            if verify_report.exists():
                try:
                    loaded = json.loads(verify_report.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        report_payload = loaded
                except Exception:
                    report_payload = {}

            if not verify_report.exists():
                report_status = "missing"
                if consecutive_over_budget_unverified_runs < over_budget_missing_report_grace_runs:
                    decision = "run"
                    reason = "over_budget_report_missing_grace_run"
                    consecutive_over_budget_unverified_runs += 1
                    total_over_budget_unverified_grace_runs += 1
                else:
                    decision = "block"
                    reason = "over_budget_report_missing"
            else:
                status_raw = str(report_payload.get("status") or "").strip()
                report_status = status_raw.upper() if status_raw else "UNKNOWN"
                ts_value = (
                    report_payload.get("timestamp")
                    or report_payload.get("generated_at")
                    or report_payload.get("updated_at")
                )
                ts_dt = parse_iso(ts_value)
                age = None
                if ts_dt is not None:
                    age = max(0, int(now_ts - int(ts_dt.timestamp())))
                    report_age_sec = str(age)

                if report_status == "READY" and age is not None and age <= reuse_ready_max_age_sec:
                    decision = "reuse_ready"
                    reason = "over_budget_reuse_fresh_ready_report"
                    consecutive_over_budget_unverified_runs = 0
                elif report_status == "READY" and age is None:
                    decision = "block"
                    reason = "over_budget_ready_report_missing_timestamp"
                    consecutive_over_budget_unverified_runs = 0
                elif report_status == "READY":
                    decision = "block"
                    reason = "over_budget_ready_report_stale"
                    consecutive_over_budget_unverified_runs = 0
                elif report_status in {"", "UNKNOWN"}:
                    if consecutive_over_budget_unverified_runs < over_budget_missing_report_grace_runs:
                        decision = "run"
                        reason = "over_budget_report_unknown_grace_run"
                        consecutive_over_budget_unverified_runs += 1
                        total_over_budget_unverified_grace_runs += 1
                    else:
                        decision = "block"
                        reason = "over_budget_report_unknown"
                else:
                    decision = "block"
                    reason = "over_budget_report_not_ready"
                    consecutive_over_budget_unverified_runs = 0
            if decision == "run":
                runs.append(now_ts)
        else:
            consecutive_over_budget_unverified_runs = 0
            runs.append(now_ts)

        window_count = len(runs)

        total_invocations = 0
        total_reuse_ready = 0
        total_blocked = 0
        if isinstance(previous, dict):
            try:
                total_invocations = int(previous.get("total_invocations") or 0)
            except Exception:
                total_invocations = 0
            try:
                total_reuse_ready = int(previous.get("total_reuse_ready") or 0)
            except Exception:
                total_reuse_ready = 0
            try:
                total_blocked = int(previous.get("total_blocked") or 0)
            except Exception:
                total_blocked = 0

        total_invocations += 1
        if decision == "reuse_ready":
            total_reuse_ready += 1
        if decision == "block":
            total_blocked += 1

        next_state = {
            "schema_version": "openclaw.verify_gate_storm_guard.v1",
            "updated_at": iso_now(),
            "task": task,
            "decision": decision,
            "reason": reason,
            "window_sec": int(window_sec),
            "max_runs": int(max_runs),
            "reuse_ready_max_age_sec": int(reuse_ready_max_age_sec),
            "over_budget_missing_report_grace_runs": int(over_budget_missing_report_grace_runs),
            "window_count": int(window_count),
            "window_runs": runs,
            "report_status": report_status,
            "report_age_sec": int(report_age_sec) if str(report_age_sec).strip() else None,
            "consecutive_over_budget_unverified_runs": int(consecutive_over_budget_unverified_runs),
            "total_invocations": int(total_invocations),
            "total_reuse_ready": int(total_reuse_ready),
            "total_blocked": int(total_blocked),
            "total_over_budget_unverified_grace_runs": int(total_over_budget_unverified_grace_runs),
        }

        tmp_path = state_path.with_name(state_path.name + ".tmp")
        tmp_path.write_text(json.dumps(next_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, state_path)
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)

except Exception as exc:
    decision = "run"
    reason = f"guard_eval_error:{type(exc).__name__}"

emit(
    decision=decision,
    reason=reason,
    window_count=window_count,
    window_sec=window_sec,
    max_runs=max_runs,
    report_status=report_status,
    report_age_sec=report_age_sec,
    state_path=str(state_path),
)
PY
)"

  if [[ -z "$payload" ]]; then
    OPENCLAW_VERIFY_GATE_STORM_GUARD_REASON="guard_eval_empty_payload"
    return 0
  fi

  while IFS='=' read -r key value; do
    case "$key" in
      decision)
        OPENCLAW_VERIFY_GATE_STORM_GUARD_DECISION="$value" ;;
      reason)
        OPENCLAW_VERIFY_GATE_STORM_GUARD_REASON="$value" ;;
      window_count)
        OPENCLAW_VERIFY_GATE_STORM_GUARD_WINDOW_COUNT="$value" ;;
      window_sec)
        OPENCLAW_VERIFY_GATE_STORM_GUARD_EFFECTIVE_WINDOW_SEC="$value" ;;
      max_runs)
        OPENCLAW_VERIFY_GATE_STORM_GUARD_EFFECTIVE_MAX_RUNS="$value" ;;
      report_status)
        OPENCLAW_VERIFY_GATE_STORM_GUARD_REPORT_STATUS="$value" ;;
      report_age_sec)
        OPENCLAW_VERIFY_GATE_STORM_GUARD_REPORT_AGE_SEC="$value" ;;
      state_path)
        OPENCLAW_VERIFY_GATE_STORM_GUARD_STATE_PATH="$value" ;;
    esac
  done <<< "$payload"

  return 0
}

openclaw_run_drift_reconcile_best_effort() {
  local reconcile_script=""
  local enabled="${OPENCLAW_AUTO_RECONCILE_DRIFT:-1}"
  local stdout_file="/tmp/openclaw_reconcile_best_effort.out"
  local stderr_file="/tmp/openclaw_reconcile_best_effort.err"
  local continuity_dispatcher="${OPENCLAW_CONTINUITY_DISPATCHER:-}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --reconcile-script)
        reconcile_script="${2:-}"; shift 2 ;;
      --enabled)
        enabled="${2:-}"; shift 2 ;;
      --stdout-file)
        stdout_file="${2:-}"; shift 2 ;;
      --stderr-file)
        stderr_file="${2:-}"; shift 2 ;;
      --continuity-dispatcher)
        continuity_dispatcher="${2:-}"; shift 2 ;;
      *)
        shift ;;
    esac
  done

  if [[ "$enabled" != "1" ]]; then
    return 0
  fi

  if [[ -z "$reconcile_script" || ! -x "$reconcile_script" ]]; then
    return 0
  fi

  if [[ -z "$continuity_dispatcher" ]]; then
    if [[ "$reconcile_script" == */ops/openclaw/continuity/reconcile.sh ]]; then
      local inferred_root="${reconcile_script%/ops/openclaw/continuity/reconcile.sh}"
      continuity_dispatcher="$inferred_root/ops/openclaw/continuity.sh"
    else
      local default_root
      default_root="$(openclaw_verify_gate_resolve_root)"
      continuity_dispatcher="$default_root/ops/openclaw/continuity.sh"
    fi
  fi

  local had_errexit=0
  if [[ "$-" == *e* ]]; then
    had_errexit=1
    set +e
  fi

  : >"$stdout_file"
  : >"$stderr_file"

  if [[ -x "$continuity_dispatcher" ]]; then
    local current_payload
    current_payload="$($continuity_dispatcher current --refresh --json 2>>"$stderr_file")"

    local action_token
    action_token="$(python3 - "$current_payload" <<'PY'
import json
import sys

raw = str(sys.argv[1] or "").strip()
token = ""
if raw:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            token = str(obj.get("action_token") or "").strip()
    except Exception:
        token = ""
print(token)
PY
)"

    if [[ -n "$action_token" ]]; then
      "$continuity_dispatcher" --action-token "$action_token" reconcile --json >>"$stdout_file" 2>>"$stderr_file"
    else
      printf 'INFO: skipped drift reconcile (missing continuity action_token)\n' >>"$stderr_file"
    fi
  else
    OPENCLAW_INTERNAL_MUTATION=1 \
    OPENCLAW_INTERNAL_MUTATION_CALLSITE="verify_gate.sh:reconcile_fallback" \
      "$reconcile_script" >>"$stdout_file" 2>>"$stderr_file"
  fi

  if [[ "$had_errexit" -eq 1 ]]; then
    set -e
  fi

  return 0
}

openclaw_verify_then_resume_gate() {
  local task=""
  local verify_script=""
  local verify_report=""
  local missing_key="verify_then_resume_missing"
  local blocker_key="verify_then_resume_blocker"
  local strict_autonomy_override=""
  local strict_autonomy_effective="0"
  local strict_autonomy_required_effective="0"
  local strict_autonomy_source="disabled"
  local strict_autonomy_override_denied="0"
  local strict_autonomy_context=""
  local summary_extra=""
  local evidence_ref=""
  local stdout_file=""
  local stderr_file=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --task)
        task="${2:-}"; shift 2 ;;
      --verify-script)
        verify_script="${2:-}"; shift 2 ;;
      --verify-report)
        verify_report="${2:-}"; shift 2 ;;
      --missing-key)
        missing_key="${2:-}"; shift 2 ;;
      --blocker-key)
        blocker_key="${2:-}"; shift 2 ;;
      --strict-autonomy-regressions)
        strict_autonomy_override="1"; shift ;;
      --no-strict-autonomy-regressions)
        strict_autonomy_override="0"; shift ;;
      --summary-extra)
        summary_extra="${2:-}"; shift 2 ;;
      --evidence-ref)
        evidence_ref="${2:-}"; shift 2 ;;
      --stdout-file)
        stdout_file="${2:-}"; shift 2 ;;
      --stderr-file)
        stderr_file="${2:-}"; shift 2 ;;
      *)
        shift ;;
    esac
  done

  if [[ -z "$task" ]]; then
    task="unknown_task"
  fi

  local default_root
  default_root="$(openclaw_verify_gate_resolve_root)"

  if [[ -z "$verify_script" ]]; then
    verify_script="$default_root/ops/openclaw/continuity/verify_then_resume.sh"
  fi

  if [[ -z "$verify_report" ]]; then
    verify_report="$default_root/state/continuity/latest/verify_last.json"
  fi

  openclaw_verify_gate_resolve_strict_autonomy "$strict_autonomy_override"
  strict_autonomy_effective="$OPENCLAW_VERIFY_GATE_RESOLVED_ENABLED"
  strict_autonomy_required_effective="$OPENCLAW_VERIFY_GATE_RESOLVED_REQUIRED"
  strict_autonomy_source="$OPENCLAW_VERIFY_GATE_RESOLVED_SOURCE"
  strict_autonomy_override_denied="$OPENCLAW_VERIFY_GATE_RESOLVED_OVERRIDE_DENIED"

  if [[ -z "$evidence_ref" ]]; then
    evidence_ref="$verify_report"
  fi

  strict_autonomy_context="strict_autonomy=${strict_autonomy_effective}; strict_source=${strict_autonomy_source}; strict_required=${strict_autonomy_required_effective}"
  if [[ -n "$strict_autonomy_override" ]]; then
    strict_autonomy_context="${strict_autonomy_context}; strict_override=${strict_autonomy_override}"
  fi

  local extra="; ${strict_autonomy_context}"
  if [[ -n "$summary_extra" ]]; then
    extra="; ${summary_extra}${extra}"
  fi

  # Fail-closed policy toggle: when strict autonomy regressions are required,
  # wrappers cannot explicitly disable the strict gate for this invocation.
  if [[ "$strict_autonomy_override_denied" == "1" ]]; then
    openclaw_verify_emit_blocker \
      "$blocker_key" \
      "task=${task}; reason=strict_autonomy_required_override_denied${extra}" \
      "$evidence_ref"
    return 1
  fi

  if [[ ! -x "$verify_script" ]]; then
    openclaw_verify_emit_blocker \
      "$missing_key" \
      "task=${task}; reason=verify_then_resume_missing${extra}" \
      "$evidence_ref"
    return 1
  fi

  if [[ -z "$stdout_file" ]]; then
    stdout_file="/tmp/openclaw_verify_gate_${task}.out"
  fi
  if [[ -z "$stderr_file" ]]; then
    stderr_file="/tmp/openclaw_verify_gate_${task}.err"
  fi

  openclaw_verify_gate_storm_guard_decide "$task" "$verify_report" "$default_root"
  local storm_guard_decision="${OPENCLAW_VERIFY_GATE_STORM_GUARD_DECISION:-run}"
  local storm_guard_reason="${OPENCLAW_VERIFY_GATE_STORM_GUARD_REASON:-under_budget}"
  local storm_guard_window_count="${OPENCLAW_VERIFY_GATE_STORM_GUARD_WINDOW_COUNT:-0}"
  local storm_guard_window_sec="${OPENCLAW_VERIFY_GATE_STORM_GUARD_EFFECTIVE_WINDOW_SEC:-0}"
  local storm_guard_max_runs="${OPENCLAW_VERIFY_GATE_STORM_GUARD_EFFECTIVE_MAX_RUNS:-0}"
  local storm_guard_report_status="${OPENCLAW_VERIFY_GATE_STORM_GUARD_REPORT_STATUS:-unknown}"
  local storm_guard_report_age_sec="${OPENCLAW_VERIFY_GATE_STORM_GUARD_REPORT_AGE_SEC:-}"
  local storm_guard_state_path="${OPENCLAW_VERIFY_GATE_STORM_GUARD_STATE_PATH:-}"

  if [[ "$storm_guard_decision" == "reuse_ready" ]]; then
    printf 'INFO: verify_gate_storm_guard task=%s decision=%s reason=%s window_count=%s max_runs=%s window_sec=%s report_status=%s report_age_sec=%s state=%s\n' \
      "$task" "$storm_guard_decision" "$storm_guard_reason" "$storm_guard_window_count" "$storm_guard_max_runs" "$storm_guard_window_sec" "$storm_guard_report_status" "$storm_guard_report_age_sec" "$storm_guard_state_path" >>"$stderr_file" 2>/dev/null || true
    return 0
  fi

  if [[ "$storm_guard_decision" == "block" ]]; then
    printf 'INFO: verify_gate_storm_guard task=%s decision=%s reason=%s window_count=%s max_runs=%s window_sec=%s report_status=%s report_age_sec=%s state=%s\n' \
      "$task" "$storm_guard_decision" "$storm_guard_reason" "$storm_guard_window_count" "$storm_guard_max_runs" "$storm_guard_window_sec" "$storm_guard_report_status" "$storm_guard_report_age_sec" "$storm_guard_state_path" >>"$stderr_file" 2>/dev/null || true
    openclaw_verify_emit_blocker \
      "$blocker_key" \
      "task=${task}; reason=verify_gate_storm_guard_${storm_guard_reason}; window_count=${storm_guard_window_count}; max_runs=${storm_guard_max_runs}; window_sec=${storm_guard_window_sec}; report_status=${storm_guard_report_status}${extra}" \
      "$evidence_ref"
    return 1
  fi

  local had_errexit=0
  if [[ "$-" == *e* ]]; then
    had_errexit=1
    set +e
  fi

  local -a verify_cmd
  verify_cmd=("$verify_script")
  if [[ "$strict_autonomy_effective" == "1" ]]; then
    verify_cmd+=("--strict-autonomy-regressions")
  fi

  OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_ENABLED="$strict_autonomy_effective" \
  OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_REQUIRED="$strict_autonomy_required_effective" \
  OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_SOURCE="$strict_autonomy_source" \
    "${verify_cmd[@]}" >"$stdout_file" 2>"$stderr_file"
  local verify_rc=$?

  if [[ "$had_errexit" -eq 1 ]]; then
    set -e
  fi

  if [[ "$verify_rc" -ne 0 ]]; then
    local reason
    reason="$(openclaw_verify_reason_from_report "$verify_report")"
    openclaw_verify_emit_blocker \
      "$blocker_key" \
      "task=${task}; reason=${reason}${extra}" \
      "$evidence_ref"
    return 1
  fi

  return 0
}
