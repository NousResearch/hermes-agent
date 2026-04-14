#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
VERIFY_GATE_LIB="${OPENCLAW_VERIFY_GATE_LIB:-$ROOT/ops/openclaw/lib/verify_gate.sh}"

usage() {
  cat <<'EOF'
Usage: verify_gate_status.sh [options]

Show preflight verify-gate strict-autonomy effective mode/source before running verify.

Options:
  --task <name>                      Logical wrapper task label (default: verify_gate_status)
  --verify-script <path>             verify_then_resume script path
  --verify-report <path>             verify report path (default: state/continuity/latest/verify_last.json)
  --dispatch-qualification <path>    dispatch qualification projection path
  --dispatch-max-age-sec <sec>       max age for launch-readiness severity projection freshness (default: 21600)
  --routing-decisions <path>         session routing decisions JSONL path
  --routing-max-age-sec <sec>        max age for live routing decision visibility (default: 21600)
  --strict-autonomy-regressions      Simulate wrapper override enable for this status call
  --no-strict-autonomy-regressions   Simulate wrapper override disable for this status call
  --json                             Emit structured JSON payload
  -h, --help                         Show help
EOF
}

json_mode=0
task="verify_gate_status"
verify_script="$ROOT/ops/openclaw/continuity/verify_then_resume.sh"
verify_report="$ROOT/state/continuity/latest/verify_last.json"
dispatch_qualification_path="$ROOT/state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json"
dispatch_qualification_max_age_sec="${OPENCLAW_VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_MAX_AGE_SEC:-21600}"
probe_overdue_blocker_min="${OPENCLAW_VERIFY_GATE_STATUS_PROBE_OVERDUE_BLOCKER_MIN:-1}"
routing_decisions="$ROOT/state/continuity/session_topology_router/decisions.jsonl"
routing_max_age_sec="${OPENCLAW_VERIFY_GATE_STATUS_ROUTING_MAX_AGE_SEC:-21600}"
strict_autonomy_override=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      task="${2:-}"
      shift 2 ;;
    --verify-script)
      verify_script="${2:-}"
      shift 2 ;;
    --verify-report)
      verify_report="${2:-}"
      shift 2 ;;
    --dispatch-qualification)
      dispatch_qualification_path="${2:-}"
      shift 2 ;;
    --dispatch-max-age-sec)
      dispatch_qualification_max_age_sec="${2:-}"
      shift 2 ;;
    --routing-decisions)
      routing_decisions="${2:-}"
      shift 2 ;;
    --routing-max-age-sec)
      routing_max_age_sec="${2:-}"
      shift 2 ;;
    --strict-autonomy-regressions)
      strict_autonomy_override="1"
      shift ;;
    --no-strict-autonomy-regressions)
      strict_autonomy_override="0"
      shift ;;
    --json)
      json_mode=1
      shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ ! -f "$VERIFY_GATE_LIB" ]]; then
  echo "verify gate library missing: $VERIFY_GATE_LIB" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VERIFY_GATE_LIB"

openclaw_verify_gate_resolve_strict_autonomy "$strict_autonomy_override"

strict_autonomy_effective="${OPENCLAW_VERIFY_GATE_RESOLVED_ENABLED:-0}"
strict_autonomy_required_effective="${OPENCLAW_VERIFY_GATE_RESOLVED_REQUIRED:-0}"
strict_autonomy_source="${OPENCLAW_VERIFY_GATE_RESOLVED_SOURCE:-disabled}"
strict_autonomy_override_denied="${OPENCLAW_VERIFY_GATE_RESOLVED_OVERRIDE_DENIED:-0}"
strict_autonomy_policy_raw="${OPENCLAW_VERIFY_GATE_RESOLVED_POLICY_RAW:-}"
strict_autonomy_legacy_raw="${OPENCLAW_VERIFY_GATE_RESOLVED_LEGACY_RAW:-}"
strict_autonomy_required_raw="${OPENCLAW_VERIFY_GATE_RESOLVED_REQUIRED_RAW:-}"

verify_script_exists="0"
verify_script_executable="0"
if [[ -e "$verify_script" ]]; then
  verify_script_exists="1"
fi
if [[ -x "$verify_script" ]]; then
  verify_script_executable="1"
fi

verify_report_exists="0"
if [[ -f "$verify_report" ]]; then
  verify_report_exists="1"
fi

verify_report_max_age_sec="${OPENCLAW_VERIFY_GATE_STATUS_VERIFY_MAX_AGE_SEC:-1800}"
run_verify_command="bash $ROOT/ops/openclaw/continuity.sh verify --json"
run_route_policy_lint_command="bash $ROOT/ops/openclaw/continuity.sh model-route-policy-lint --json"
run_layered_health_command="bash $ROOT/ops/openclaw/continuity/layered_health_snapshot.sh"
run_slo_snapshot_command="bash $ROOT/ops/openclaw/continuity/slo_evaluator_snapshot.sh"
run_dispatch_qualification_refresh_command="bash $ROOT/ops/openclaw/continuity/continuity_current.sh --json"
run_worker_health_canary_refresh_command="bash $ROOT/ops/openclaw/continuity.sh worker-health-canary --json"
run_failover_stress_runtime_refresh_command="bash $ROOT/ops/openclaw/continuity.sh handover --refresh --json && bash $ROOT/ops/openclaw/continuity.sh failover-stress-runtime-evidence --json"

failover_stress_runtime_evidence_path="${OPENCLAW_VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_EVIDENCE_PATH:-$ROOT/state/continuity/latest/failover_stress_runtime_evidence.json}"
failover_stress_runtime_max_age_sec="${OPENCLAW_VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_MAX_AGE_SEC:-21600}"
failover_stress_runtime_auto_refresh="${OPENCLAW_VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH:-0}"
failover_stress_runtime_auto_refresh_timeout_sec="${OPENCLAW_VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_TIMEOUT_SEC:-300}"
failover_stress_runtime_auto_refresh_cooldown_sec="${OPENCLAW_VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_COOLDOWN_SEC:-900}"
failover_stress_runtime_auto_refresh_state_path="${OPENCLAW_VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_STATE_PATH:-$ROOT/state/continuity/latest/failover_stress_runtime_evidence_autorefresh_state.json}"

layered_health_snapshot_path="${OPENCLAW_VERIFY_GATE_STATUS_LAYERED_HEALTH_PATH:-$ROOT/state/continuity/latest/layered_health_snapshot.json}"
slo_snapshot_path="${OPENCLAW_VERIFY_GATE_STATUS_SLO_SNAPSHOT_PATH:-$ROOT/state/continuity/latest/slo_snapshot.json}"
layered_health_required_lanes_csv="${OPENCLAW_VERIFY_GATE_STATUS_HEALTH_REQUIRED_LANES:-A1_CONTROL_PLANE,A2_RUNTIME_CONTINUITY,A3_MODEL_ROUTING,A6_OPS_OBSERVABILITY,C1_OPERATOR_SURFACE,C2_RELEASE_SUBSTRATE}"
layered_health_min_layer="${OPENCLAW_VERIFY_GATE_STATUS_HEALTH_MIN_LAYER:-truthful}"

internal_bypass_audit_path="${OPENCLAW_CONTINUITY_MUTATOR_INGRESS_AUDIT_PATH:-$ROOT/state/continuity/latest/mutator_ingress_audit.jsonl}"
internal_bypass_window_sec="${OPENCLAW_INTERNAL_BYPASS_STAGE_B_WINDOW_SEC:-21600}"
internal_bypass_unknown_threshold="${OPENCLAW_INTERNAL_BYPASS_STAGE_B_UNKNOWN_THRESHOLD:-0}"
internal_bypass_top_callsites="${OPENCLAW_INTERNAL_BYPASS_STAGE_B_TOP_CALLSITES:-5}"

predicted_blocker_reason=""
if [[ "$strict_autonomy_override_denied" == "1" ]]; then
  predicted_blocker_reason="strict_autonomy_required_override_denied"
elif [[ "$verify_script_executable" != "1" ]]; then
  predicted_blocker_reason="verify_then_resume_missing"
fi

verify_args=""
if [[ "$strict_autonomy_effective" == "1" ]]; then
  verify_args="--strict-autonomy-regressions"
fi

VERIFY_GATE_STATUS_JSON_MODE="$json_mode" \
VERIFY_GATE_STATUS_TASK="$task" \
VERIFY_GATE_STATUS_ROOT="$ROOT" \
VERIFY_GATE_STATUS_VERIFY_SCRIPT="$verify_script" \
VERIFY_GATE_STATUS_VERIFY_SCRIPT_EXISTS="$verify_script_exists" \
VERIFY_GATE_STATUS_VERIFY_SCRIPT_EXECUTABLE="$verify_script_executable" \
VERIFY_GATE_STATUS_VERIFY_REPORT="$verify_report" \
VERIFY_GATE_STATUS_VERIFY_REPORT_EXISTS="$verify_report_exists" \
VERIFY_GATE_STATUS_STRICT_ENABLED="$strict_autonomy_effective" \
VERIFY_GATE_STATUS_STRICT_SOURCE="$strict_autonomy_source" \
VERIFY_GATE_STATUS_STRICT_REQUIRED="$strict_autonomy_required_effective" \
VERIFY_GATE_STATUS_STRICT_OVERRIDE="$strict_autonomy_override" \
VERIFY_GATE_STATUS_STRICT_OVERRIDE_DENIED="$strict_autonomy_override_denied" \
VERIFY_GATE_STATUS_POLICY_RAW="$strict_autonomy_policy_raw" \
VERIFY_GATE_STATUS_LEGACY_RAW="$strict_autonomy_legacy_raw" \
VERIFY_GATE_STATUS_REQUIRED_RAW="$strict_autonomy_required_raw" \
VERIFY_GATE_STATUS_PREDICTED_BLOCKER_REASON="$predicted_blocker_reason" \
VERIFY_GATE_STATUS_VERIFY_ARGS="$verify_args" \
VERIFY_GATE_STATUS_VERIFY_MAX_AGE_SEC="$verify_report_max_age_sec" \
VERIFY_GATE_STATUS_RUN_VERIFY_COMMAND="$run_verify_command" \
VERIFY_GATE_STATUS_RUN_LAYERED_HEALTH_COMMAND="$run_layered_health_command" \
VERIFY_GATE_STATUS_RUN_SLO_SNAPSHOT_COMMAND="$run_slo_snapshot_command" \
VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_PATH="$dispatch_qualification_path" \
VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_MAX_AGE_SEC="$dispatch_qualification_max_age_sec" \
VERIFY_GATE_STATUS_PROBE_OVERDUE_BLOCKER_MIN="$probe_overdue_blocker_min" \
VERIFY_GATE_STATUS_RUN_DISPATCH_QUALIFICATION_REFRESH_COMMAND="$run_dispatch_qualification_refresh_command" \
VERIFY_GATE_STATUS_RUN_WORKER_HEALTH_CANARY_REFRESH_COMMAND="$run_worker_health_canary_refresh_command" \
VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_EVIDENCE_PATH="$failover_stress_runtime_evidence_path" \
VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_MAX_AGE_SEC="$failover_stress_runtime_max_age_sec" \
VERIFY_GATE_STATUS_RUN_FAILOVER_STRESS_RUNTIME_REFRESH_COMMAND="$run_failover_stress_runtime_refresh_command" \
VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH="$failover_stress_runtime_auto_refresh" \
VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_TIMEOUT_SEC="$failover_stress_runtime_auto_refresh_timeout_sec" \
VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_COOLDOWN_SEC="$failover_stress_runtime_auto_refresh_cooldown_sec" \
VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_STATE_PATH="$failover_stress_runtime_auto_refresh_state_path" \
VERIFY_GATE_STATUS_ROUTING_DECISIONS="$routing_decisions" \
VERIFY_GATE_STATUS_ROUTING_MAX_AGE_SEC="$routing_max_age_sec" \
VERIFY_GATE_STATUS_RUN_ROUTE_POLICY_LINT_COMMAND="$run_route_policy_lint_command" \
VERIFY_GATE_STATUS_LAYERED_HEALTH_PATH="$layered_health_snapshot_path" \
VERIFY_GATE_STATUS_SLO_SNAPSHOT_PATH="$slo_snapshot_path" \
VERIFY_GATE_STATUS_HEALTH_REQUIRED_LANES="$layered_health_required_lanes_csv" \
VERIFY_GATE_STATUS_HEALTH_MIN_LAYER="$layered_health_min_layer" \
VERIFY_GATE_STATUS_INTERNAL_BYPASS_AUDIT_PATH="$internal_bypass_audit_path" \
VERIFY_GATE_STATUS_INTERNAL_BYPASS_WINDOW_SEC="$internal_bypass_window_sec" \
VERIFY_GATE_STATUS_INTERNAL_BYPASS_UNKNOWN_THRESHOLD="$internal_bypass_unknown_threshold" \
VERIFY_GATE_STATUS_INTERNAL_BYPASS_TOP_CALLSITES="$internal_bypass_top_callsites" \
VERIFY_GATE_STATUS_ROOT="$ROOT" \
  python3 - <<'PY'
import datetime as dt
import json
import os
import pathlib
import subprocess
from collections import Counter


def truthy(raw: str) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_iso(value: str) -> dt.datetime | None:
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


def safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def dedupe_nonempty_strings(values: object) -> list[str]:
    rows = values if isinstance(values, list) else []
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        txt = str(row or "").strip()
        if not txt or txt in seen:
            continue
        out.append(txt)
        seen.add(txt)
    return out


REPEATABILITY_IDENTITY_DRIFT_FIELDS = {
    "publish_reason",
    "active_top_blocker",
    "effective_top_blocker",
}


def repeatability_identity_drift_tolerated(
    *,
    repeatability: dict,
    overall_verdict: str,
    publish_chain_verdict: str,
    publish_assertions_failed: int,
) -> tuple[bool, list[str]]:
    mismatch_fields = dedupe_nonempty_strings(repeatability.get("mismatch_fields"))
    if not mismatch_fields:
        mismatch_fields = dedupe_nonempty_strings(repeatability.get("tolerated_mismatch_fields"))
    if not mismatch_fields:
        return False, []
    if overall_verdict != "PASS" or publish_chain_verdict != "PASS" or publish_assertions_failed > 0:
        return False, mismatch_fields
    return set(mismatch_fields).issubset(REPEATABILITY_IDENTITY_DRIFT_FIELDS), mismatch_fields


def iso_z(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_path(raw: object, root_path: pathlib.Path | None = None) -> str | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    candidate = pathlib.Path(txt).expanduser()
    if not candidate.is_absolute() and root_path is not None:
        candidate = (root_path / candidate)
    try:
        return str(candidate.resolve())
    except Exception:
        try:
            return str(candidate.absolute())
        except Exception:
            return str(candidate)


def policy_path_matches_expected(raw_path: object, *, expected_path: str | None, root_path: pathlib.Path) -> bool:
    normalized = normalize_path(raw_path, root_path)
    if normalized is None:
        return True
    if expected_path is None:
        return True
    return normalized == expected_path


def routing_row_matches_expected_policy_paths(
    row: dict,
    *,
    expected_routing_policy_path: str | None,
    expected_pool_policy_path: str | None,
    root_path: pathlib.Path,
) -> tuple[bool, list[str], str | None, str | None]:
    routing_policy = row.get("routing_policy") if isinstance(row.get("routing_policy"), dict) else {}
    pool_policy = row.get("pool_policy") if isinstance(row.get("pool_policy"), dict) else {}

    routing_policy_path = normalize_path(routing_policy.get("policy_path"), root_path)
    pool_policy_path = normalize_path(pool_policy.get("policy_path"), root_path)

    mismatches: list[str] = []
    if not policy_path_matches_expected(
        routing_policy.get("policy_path"),
        expected_path=expected_routing_policy_path,
        root_path=root_path,
    ):
        mismatches.append("routing_policy_path_mismatch")
    if not policy_path_matches_expected(
        pool_policy.get("policy_path"),
        expected_path=expected_pool_policy_path,
        root_path=root_path,
    ):
        mismatches.append("pool_policy_path_mismatch")

    return len(mismatches) == 0, mismatches, routing_policy_path, pool_policy_path


def summarize_internal_bypass_stage_b(now_dt: dt.datetime) -> dict:
    audit_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_INTERNAL_BYPASS_AUDIT_PATH", ""))
    window_sec = max(0, safe_int(os.environ.get("VERIFY_GATE_STATUS_INTERNAL_BYPASS_WINDOW_SEC", "21600"), 21600))
    unknown_threshold = max(0, safe_int(os.environ.get("VERIFY_GATE_STATUS_INTERNAL_BYPASS_UNKNOWN_THRESHOLD", "0"), 0))
    top_callsites = max(1, safe_int(os.environ.get("VERIFY_GATE_STATUS_INTERNAL_BYPASS_TOP_CALLSITES", "5"), 5))

    window_start_dt = now_dt - dt.timedelta(seconds=window_sec) if window_sec > 0 else None

    summary = {
        "audit_path": str(audit_path),
        "audit_present": audit_path.exists(),
        "window_sec": window_sec,
        "window_start": iso_z(window_start_dt),
        "window_end": iso_z(now_dt),
        "rows_in_window": 0,
        "internal_bypass_rows_in_window": 0,
        "allowlisted_count": 0,
        "allowlisted_authority_enforced_count": 0,
        "allowlist_disabled_count": 0,
        "unknown_callsite_soft_allow_count": 0,
        "break_glass_allow_count": 0,
        "break_glass_authority_enforced_count": 0,
        "break_glass_denied_count": 0,
        "unknown_callsite_denied_count": 0,
        "unknown_callsite_allowed_total": 0,
        "unknown_callsite_total": 0,
        "unknown_callsites_top": [],
        "parse_error_count": 0,
        "invalid_timestamp_count": 0,
        "closeout_unknown_threshold": unknown_threshold,
        "evidence_sufficient": False,
        "closeout_ready": False,
        "closeout_failure_reason": None,
        "inspect_audit_command": f"tail -n 200 {audit_path}",
    }

    if not audit_path.exists():
        summary["closeout_failure_reason"] = "audit_missing"
        return summary
    if not audit_path.is_file():
        summary["closeout_failure_reason"] = "audit_not_regular_file"
        return summary

    unknown_counter: Counter[str] = Counter()

    try:
        with audit_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    summary["parse_error_count"] += 1
                    continue
                if not isinstance(row, dict):
                    summary["parse_error_count"] += 1
                    continue

                ts = parse_iso(str(row.get("ts") or ""))
                if ts is None:
                    summary["invalid_timestamp_count"] += 1
                    continue
                if window_start_dt is not None and ts < window_start_dt:
                    continue

                summary["rows_in_window"] += 1

                mode = str(row.get("mode") or "").strip()
                detail = str(row.get("detail") or "").strip()
                callsite = str(row.get("callsite") or "").strip() or "(missing_callsite)"

                if mode == "internal_bypass":
                    summary["internal_bypass_rows_in_window"] += 1

                if detail == "internal_env_allowlisted":
                    summary["allowlisted_count"] += 1
                elif detail == "internal_env_allowlisted_authority_enforced":
                    summary["allowlisted_authority_enforced_count"] += 1
                elif detail == "internal_env_allowlist_disabled":
                    summary["allowlist_disabled_count"] += 1
                elif detail == "internal_env_unknown_callsite_soft_allow":
                    summary["unknown_callsite_soft_allow_count"] += 1
                    summary["break_glass_allow_count"] += 1
                    unknown_counter[callsite] += 1
                elif detail == "internal_env_unknown_callsite_break_glass_allow":
                    summary["break_glass_allow_count"] += 1
                    unknown_counter[callsite] += 1
                elif detail == "internal_env_unknown_callsite_break_glass_authority_enforced":
                    summary["break_glass_allow_count"] += 1
                    summary["break_glass_authority_enforced_count"] += 1
                    unknown_counter[callsite] += 1
                elif detail in {"internal_callsite_not_allowlisted", "internal_unknown_callsite_break_glass_authority_block"}:
                    summary["unknown_callsite_denied_count"] += 1
                    summary["break_glass_denied_count"] += 1
                    unknown_counter[callsite] += 1
    except Exception:
        summary["closeout_failure_reason"] = "audit_unreadable"
        return summary

    summary["unknown_callsite_allowed_total"] = int(summary.get("break_glass_allow_count") or 0)
    summary["unknown_callsite_total"] = (
        int(summary.get("unknown_callsite_allowed_total") or 0)
        + int(summary.get("unknown_callsite_denied_count") or 0)
    )
    summary["unknown_callsites_top"] = [
        {"callsite": callsite, "count": count}
        for callsite, count in unknown_counter.most_common(top_callsites)
    ]

    summary["evidence_sufficient"] = int(summary.get("internal_bypass_rows_in_window") or 0) > 0
    if not summary["evidence_sufficient"]:
        summary["closeout_failure_reason"] = "insufficient_window_activity"
    elif int(summary.get("unknown_callsite_allowed_total") or 0) > unknown_threshold:
        summary["closeout_failure_reason"] = "unknown_callsites_present"
    else:
        summary["closeout_failure_reason"] = None

    summary["closeout_ready"] = summary.get("closeout_failure_reason") is None
    return summary


def summarize_routing_preflight(now_dt: dt.datetime) -> dict:
    decisions_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_ROUTING_DECISIONS", ""))
    root_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_ROOT", "")).expanduser()
    max_age_sec = max(0, safe_int(os.environ.get("VERIFY_GATE_STATUS_ROUTING_MAX_AGE_SEC", "21600"), 21600))
    lint_command = str(os.environ.get("VERIFY_GATE_STATUS_RUN_ROUTE_POLICY_LINT_COMMAND", "")).strip() or None
    expected_routing_policy_path = normalize_path(
        os.environ.get(
            "VERIFY_GATE_STATUS_EXPECTED_ROUTING_POLICY_PATH",
            str(root_path / "docs" / "ops" / "session_topology_routing_policy_v1.json"),
        ),
        root_path,
    )
    expected_pool_policy_path = normalize_path(
        os.environ.get(
            "VERIFY_GATE_STATUS_EXPECTED_POOL_POLICY_PATH",
            str(root_path / "docs" / "ops" / "model_pool_policy_v1.json"),
        ),
        root_path,
    )

    summary = {
        "decision_log_path": str(decisions_path),
        "decision_log_present": decisions_path.exists(),
        "max_age_sec": max_age_sec,
        "policy_scope": {
            "expected_routing_policy_path": expected_routing_policy_path,
            "expected_pool_policy_path": expected_pool_policy_path,
            "rows_with_expected_policy_path": 0,
            "rows_with_unexpected_policy_path": 0,
            "latest_unexpected": None,
        },
        "rows_scanned": 0,
        "decision_rows_seen": 0,
        "parse_error_count": 0,
        "latest": {
            "decision": None,
            "evaluated_at": None,
            "age_sec": None,
            "fresh": None,
            "route_class": None,
            "selected_model": None,
            "required_rollout_stage": None,
            "selected_rule_id": None,
            "block_gate": None,
            "block_reason": None,
            "policy_path_match_expected": None,
            "routing_policy_path": None,
            "pool_policy_path": None,
            "actionable_failure": {
                "gate": None,
                "reason": None,
                "hint": None,
                "commands": [],
            },
        },
        "effective": {
            "blocked": None,
            "blocked_fresh": None,
            "route_class": None,
            "selected_model": None,
            "required_rollout_stage": None,
            "block_gate": None,
            "block_reason": None,
            "decision_source": None,
            "actionable_hint": None,
            "first_actionable_command": None,
            "inspect_decisions_command": f"tail -n 60 {decisions_path}",
            "recheck_policy_command": lint_command,
        },
        "failure_reason": None,
    }

    if not decisions_path.exists():
        summary["failure_reason"] = "routing_decisions_missing"
        return summary
    if not decisions_path.is_file():
        summary["failure_reason"] = "routing_decisions_not_regular_file"
        return summary

    latest_observed_row = None
    latest_observed_ts = None
    latest_expected_row = None
    latest_expected_ts = None

    try:
        with decisions_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                summary["rows_scanned"] += 1
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    summary["parse_error_count"] += 1
                    continue
                if not isinstance(row, dict):
                    summary["parse_error_count"] += 1
                    continue
                if str(row.get("schema") or "").strip() != "clawd.session_topology_routing.decision.v1":
                    continue

                summary["decision_rows_seen"] += 1
                row_ts = parse_iso(str(row.get("evaluated_at") or ""))
                row_matches_expected, mismatch_reasons, routing_policy_path, pool_policy_path = (
                    routing_row_matches_expected_policy_paths(
                        row,
                        expected_routing_policy_path=expected_routing_policy_path,
                        expected_pool_policy_path=expected_pool_policy_path,
                        root_path=root_path,
                    )
                )

                if row_matches_expected:
                    summary["policy_scope"]["rows_with_expected_policy_path"] += 1
                else:
                    summary["policy_scope"]["rows_with_unexpected_policy_path"] += 1

                if latest_observed_row is None:
                    latest_observed_row = row
                    latest_observed_ts = row_ts
                elif row_ts is not None:
                    if latest_observed_ts is None or row_ts >= latest_observed_ts:
                        latest_observed_row = row
                        latest_observed_ts = row_ts
                elif latest_observed_ts is None:
                    latest_observed_row = row

                if row_matches_expected:
                    if latest_expected_row is None:
                        latest_expected_row = row
                        latest_expected_ts = row_ts
                    elif row_ts is not None:
                        if latest_expected_ts is None or row_ts >= latest_expected_ts:
                            latest_expected_row = row
                            latest_expected_ts = row_ts
                    elif latest_expected_ts is None:
                        latest_expected_row = row
                else:
                    latest_unexpected = summary["policy_scope"].get("latest_unexpected")
                    latest_unexpected_ts = parse_iso(str((latest_unexpected or {}).get("evaluated_at") or "")) if isinstance(latest_unexpected, dict) else None
                    should_replace_unexpected = False
                    if not isinstance(latest_unexpected, dict):
                        should_replace_unexpected = True
                    elif row_ts is not None and (latest_unexpected_ts is None or row_ts >= latest_unexpected_ts):
                        should_replace_unexpected = True
                    elif row_ts is None and latest_unexpected_ts is None:
                        should_replace_unexpected = True
                    if should_replace_unexpected:
                        summary["policy_scope"]["latest_unexpected"] = {
                            "evaluated_at": row.get("evaluated_at"),
                            "decision": str(row.get("decision") or "").strip().upper() or None,
                            "block_gate": str(row.get("block_gate") or "").strip() or None,
                            "block_reason": str(row.get("block_reason") or "").strip() or None,
                            "routing_policy_path": routing_policy_path,
                            "pool_policy_path": pool_policy_path,
                            "mismatch_reasons": mismatch_reasons,
                        }
    except Exception:
        summary["failure_reason"] = "routing_decisions_unreadable"
        return summary

    latest_row = latest_expected_row or latest_observed_row
    latest_ts = latest_expected_ts if latest_expected_row is not None else latest_observed_ts
    effective_source = "latest_expected_policy_path_row" if latest_expected_row is not None else "latest_observed_row"

    if latest_row is None:
        summary["failure_reason"] = "routing_decisions_no_valid_rows"
        return summary

    route = latest_row.get("route") if isinstance(latest_row.get("route"), dict) else {}
    actionable = latest_row.get("actionable_failure") if isinstance(latest_row.get("actionable_failure"), dict) else {}
    actionable_commands = [
        str(cmd).strip()
        for cmd in (actionable.get("commands") if isinstance(actionable.get("commands"), list) else [])
        if str(cmd).strip()
    ]

    evaluated_at = latest_row.get("evaluated_at")
    age_sec = None
    fresh = None
    if latest_ts is not None:
        age_sec = max(0, int((now_dt - latest_ts).total_seconds()))
        fresh = True if max_age_sec <= 0 else age_sec <= max_age_sec

    decision = str(latest_row.get("decision") or "").strip().upper() or None
    route_class = str(route.get("route_class") or "").strip() or None
    selected_model = str(route.get("selected_model") or "").strip() or None
    required_rollout_stage = str(route.get("required_rollout_stage") or "").strip() or None
    selected_rule_id = str(route.get("selected_rule_id") or "").strip() or None
    block_gate = str(latest_row.get("block_gate") or "").strip() or None
    block_reason = str(latest_row.get("block_reason") or "").strip() or None
    latest_matches_expected, _, routing_policy_path, pool_policy_path = routing_row_matches_expected_policy_paths(
        latest_row,
        expected_routing_policy_path=expected_routing_policy_path,
        expected_pool_policy_path=expected_pool_policy_path,
        root_path=root_path,
    )

    blocked = decision == "BLOCK"
    blocked_fresh = bool(blocked and (fresh is not False))
    effective_route_class = route_class if fresh is not False else None
    effective_selected_model = selected_model if fresh is not False else None
    effective_required_rollout_stage = required_rollout_stage if fresh is not False else None
    effective_block_gate = block_gate if blocked_fresh else None
    effective_block_reason = block_reason if blocked_fresh else None
    effective_actionable_hint = actionable.get("hint") if blocked_fresh else None
    effective_first_actionable_command = actionable_commands[0] if (blocked_fresh and actionable_commands) else None

    summary["latest"] = {
        "decision": decision,
        "evaluated_at": evaluated_at,
        "age_sec": age_sec,
        "fresh": fresh,
        "route_class": route_class,
        "selected_model": selected_model,
        "required_rollout_stage": required_rollout_stage,
        "selected_rule_id": selected_rule_id,
        "block_gate": block_gate,
        "block_reason": block_reason,
        "policy_path_match_expected": latest_matches_expected,
        "routing_policy_path": routing_policy_path,
        "pool_policy_path": pool_policy_path,
        "actionable_failure": {
            "gate": actionable.get("gate"),
            "reason": actionable.get("reason"),
            "hint": actionable.get("hint"),
            "commands": actionable_commands,
        },
    }
    summary["effective"] = {
        "blocked": blocked_fresh,
        "blocked_fresh": blocked_fresh,
        "route_class": effective_route_class,
        "selected_model": effective_selected_model,
        "required_rollout_stage": effective_required_rollout_stage,
        "block_gate": effective_block_gate,
        "block_reason": effective_block_reason,
        "decision_source": effective_source,
        "actionable_hint": effective_actionable_hint,
        "first_actionable_command": effective_first_actionable_command,
        "inspect_decisions_command": f"tail -n 60 {decisions_path}",
        "recheck_policy_command": lint_command,
    }

    if blocked_fresh:
        summary["failure_reason"] = "routing_blocked"
    elif fresh is False:
        summary["failure_reason"] = "routing_decision_stale"
    else:
        summary["failure_reason"] = None

    return summary


def summarize_launch_readiness_severity_gate(now_dt: dt.datetime) -> dict:
    dispatch_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_PATH", ""))
    max_age_sec = max(
        0,
        safe_int(os.environ.get("VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_MAX_AGE_SEC", "21600"), 21600),
    )
    refresh_command = (
        str(os.environ.get("VERIFY_GATE_STATUS_RUN_DISPATCH_QUALIFICATION_REFRESH_COMMAND", "")).strip()
        or None
    )

    summary = {
        "dispatch_qualification_path": str(dispatch_path),
        "dispatch_qualification_present": dispatch_path.exists(),
        "max_age_sec": max_age_sec,
        "generated_at": None,
        "age_sec": None,
        "fresh": None,
        "launch_readiness_state": None,
        "launch_readiness_reason": None,
        "severity_state": None,
        "severity_reason": None,
        "severity_active": False,
        "threshold_ticks": None,
        "non_ready_ticks_consecutive": None,
        "cohort_worker_count": None,
        "active_blocker": False,
        "blocker_reason": None,
        "failure_reason": None,
        "inspect_dispatch_qualification_command": f"cat {dispatch_path}",
        "refresh_dispatch_qualification_command": refresh_command,
    }

    if not dispatch_path.exists():
        summary["failure_reason"] = "dispatch_qualification_missing"
        return summary
    if not dispatch_path.is_file():
        summary["failure_reason"] = "dispatch_qualification_not_regular_file"
        return summary

    payload = None
    try:
        loaded = json.loads(dispatch_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
        else:
            summary["failure_reason"] = "dispatch_qualification_invalid"
            return summary
    except Exception:
        summary["failure_reason"] = "dispatch_qualification_unreadable"
        return summary

    generated_at_raw = str(payload.get("generated_at") or "").strip()
    summary["generated_at"] = generated_at_raw or None

    generated_at_dt = parse_iso(generated_at_raw) if generated_at_raw else None
    if generated_at_raw and generated_at_dt is None:
        summary["failure_reason"] = "dispatch_qualification_generated_at_invalid"
    elif generated_at_dt is not None:
        age_sec = max(0, int((now_dt - generated_at_dt).total_seconds()))
        summary["age_sec"] = age_sec
        summary["fresh"] = True if max_age_sec <= 0 else age_sec <= max_age_sec
        if summary["fresh"] is False:
            summary["failure_reason"] = "dispatch_qualification_stale"

    launch_readiness = payload.get("launch_readiness") if isinstance(payload.get("launch_readiness"), dict) else {}
    severity_gate = launch_readiness.get("severity_gate") if isinstance(launch_readiness.get("severity_gate"), dict) else {}

    summary["launch_readiness_state"] = str(launch_readiness.get("state") or "").strip() or None
    summary["launch_readiness_reason"] = str(launch_readiness.get("reason") or "").strip() or None
    summary["severity_state"] = str(severity_gate.get("state") or "").strip() or None
    summary["severity_reason"] = str(severity_gate.get("reason") or "").strip() or None
    summary["severity_active"] = bool(severity_gate.get("active") is True)
    summary["threshold_ticks"] = (
        int(severity_gate.get("threshold_ticks"))
        if isinstance(severity_gate.get("threshold_ticks"), int)
        else None
    )
    summary["non_ready_ticks_consecutive"] = (
        int(severity_gate.get("non_ready_ticks_consecutive"))
        if isinstance(severity_gate.get("non_ready_ticks_consecutive"), int)
        else None
    )
    summary["cohort_worker_count"] = (
        int(severity_gate.get("cohort_worker_count"))
        if isinstance(severity_gate.get("cohort_worker_count"), int)
        else None
    )

    if not severity_gate and summary.get("failure_reason") is None:
        summary["failure_reason"] = "launch_readiness_severity_gate_missing"

    if bool(summary.get("severity_active")) and summary.get("fresh") is not False:
        state_for_reason = str(summary.get("severity_state") or "severity_unknown").strip() or "severity_unknown"
        gate_reason = (
            str(summary.get("severity_reason") or "").strip()
            or f"launch_readiness_severity_{state_for_reason}_active"
        )
        summary["active_blocker"] = True
        summary["blocker_reason"] = gate_reason
        summary["failure_reason"] = "launch_readiness_severity_gate_active"

    return summary


def summarize_launch_readiness_worker_health_canary_gate(now_dt: dt.datetime) -> dict:
    dispatch_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_PATH", ""))
    root_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_ROOT", "")).expanduser()
    dispatch_max_age_sec = max(
        0,
        safe_int(os.environ.get("VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_MAX_AGE_SEC", "21600"), 21600),
    )
    refresh_dispatch_command = (
        str(os.environ.get("VERIFY_GATE_STATUS_RUN_DISPATCH_QUALIFICATION_REFRESH_COMMAND", "")).strip()
        or None
    )
    refresh_canary_command = (
        str(os.environ.get("VERIFY_GATE_STATUS_RUN_WORKER_HEALTH_CANARY_REFRESH_COMMAND", "")).strip()
        or None
    )
    default_canary_path = dispatch_path.with_name("execution_supervisor_worker_health_canary_latest.json")

    canary_max_age_default = max(
        0,
        safe_int(os.environ.get("OPENCLAW_VERIFY_GATE_STATUS_WORKER_HEALTH_CANARY_MAX_AGE_SEC", "21600"), 21600),
    )
    canary_future_skew_default = max(
        0,
        safe_int(os.environ.get("OPENCLAW_VERIFY_GATE_STATUS_WORKER_HEALTH_CANARY_FUTURE_SKEW_SEC", "120"), 120),
    )

    canary_failure_reasons = {
        "worker_health_canary_missing",
        "worker_health_canary_path_missing",
        "worker_health_canary_not_regular_file",
        "worker_health_canary_unreadable",
        "worker_health_canary_invalid",
        "worker_health_canary_generated_at_missing",
        "worker_health_canary_generated_at_invalid",
        "worker_health_canary_future",
        "worker_health_canary_stale",
    }

    dispatch_failure_reasons = {
        "dispatch_qualification_missing",
        "dispatch_qualification_not_regular_file",
        "dispatch_qualification_unreadable",
        "dispatch_qualification_invalid",
        "dispatch_qualification_generated_at_invalid",
        "dispatch_qualification_stale",
    }

    def _resolve_artifact_path(path_raw: str) -> pathlib.Path | None:
        cleaned = str(path_raw or "").strip()
        if not cleaned:
            return None
        candidate = pathlib.Path(cleaned)
        if not candidate.is_absolute() and str(root_path):
            candidate = root_path / candidate
        try:
            return candidate.resolve()
        except Exception:
            return candidate

    def _evaluate_canary_artifact(
        *,
        path_obj: pathlib.Path | None,
        declared_present: bool | None,
        declared_schema: str | None,
        canary_max_age_sec: int,
        canary_future_skew_sec: int,
    ) -> dict:
        result = {
            "present": bool(declared_present) if isinstance(declared_present, bool) else False,
            "path": str(path_obj) if path_obj is not None else None,
            "schema": declared_schema,
            "generated_at": None,
            "age_sec": None,
            "fresh": None,
            "failure_reason": None,
        }

        if path_obj is None:
            result["failure_reason"] = "worker_health_canary_path_missing"
            return result

        path_exists = path_obj.exists()
        path_is_file = path_exists and path_obj.is_file()

        if isinstance(declared_present, bool):
            result["present"] = declared_present
        else:
            result["present"] = bool(path_exists and path_is_file)

        if declared_present is False:
            result["failure_reason"] = "worker_health_canary_missing"
            return result
        if not path_exists:
            result["failure_reason"] = "worker_health_canary_missing"
            return result
        if not path_is_file:
            result["failure_reason"] = "worker_health_canary_not_regular_file"
            return result

        canary_payload = None
        try:
            loaded_canary = json.loads(path_obj.read_text(encoding="utf-8"))
            if isinstance(loaded_canary, dict):
                canary_payload = loaded_canary
            else:
                result["failure_reason"] = "worker_health_canary_invalid"
                return result
        except Exception:
            result["failure_reason"] = "worker_health_canary_unreadable"
            return result

        if result.get("schema") is None:
            result["schema"] = str(canary_payload.get("schema") or "").strip() or None

        canary_generated_at_raw = str(canary_payload.get("generated_at") or "").strip()
        result["generated_at"] = canary_generated_at_raw or None
        if not canary_generated_at_raw:
            result["failure_reason"] = "worker_health_canary_generated_at_missing"
            result["fresh"] = False
            return result

        canary_generated_at_dt = parse_iso(canary_generated_at_raw)
        if canary_generated_at_dt is None:
            result["failure_reason"] = "worker_health_canary_generated_at_invalid"
            result["fresh"] = False
            return result

        if canary_generated_at_dt > now_dt + dt.timedelta(seconds=canary_future_skew_sec):
            result["failure_reason"] = "worker_health_canary_future"
            result["fresh"] = False
            result["age_sec"] = 0
            return result

        canary_age_sec = max(0, int((now_dt - canary_generated_at_dt).total_seconds()))
        result["age_sec"] = canary_age_sec
        canary_fresh = True if canary_max_age_sec <= 0 else canary_age_sec <= canary_max_age_sec
        result["fresh"] = canary_fresh
        if not canary_fresh:
            result["failure_reason"] = "worker_health_canary_stale"
        return result

    summary = {
        "dispatch_qualification_path": str(dispatch_path),
        "dispatch_qualification_present": dispatch_path.exists(),
        "dispatch_qualification_failure_reason": None,
        "dispatch_qualification_max_age_sec": dispatch_max_age_sec,
        "dispatch_qualification_generated_at": None,
        "dispatch_qualification_age_sec": None,
        "dispatch_qualification_fresh": None,
        "gate_required": None,
        "artifact_required": None,
        "resource_preflight_required": None,
        "resource_preflight_status": None,
        "resource_preflight_reason": None,
        "resource_preflight_blocking_candidate_count": 0,
        "resource_preflight_blocking_task_ids": [],
        "resource_preflight_telemetry_complete": None,
        "resource_preflight_lowest_headroom_pct": None,
        "uncertainty_confidence_score": None,
        "uncertainty_confidence_label": None,
        "uncertainty_confidence_quantiles": {},
        "uncertainty_reasons": [],
        "uncertainty_requires_operator_review": None,
        "worker_health_canary_present": False,
        "worker_health_canary_path": str(default_canary_path),
        "worker_health_canary_schema": None,
        "worker_health_canary_source": None,
        "worker_health_canary_generated_at": None,
        "worker_health_canary_age_sec": None,
        "worker_health_canary_fresh": None,
        "worker_health_canary_max_age_sec": canary_max_age_default,
        "worker_health_canary_future_skew_sec": canary_future_skew_default,
        "active_blocker": False,
        "blocker_reason": None,
        "failure_reason": None,
        "inspect_dispatch_qualification_command": f"cat {dispatch_path}",
        "inspect_worker_health_canary_command": f"cat {default_canary_path}",
        "refresh_worker_health_canary_command": refresh_canary_command,
        "refresh_dispatch_qualification_command": refresh_dispatch_command,
        "first_actionable_command": None,
        "action_priority": None,
    }

    dispatch_payload: dict = {}
    dispatch_failure_reason: str | None = None
    dispatch_fresh: bool | None = None

    if not dispatch_path.exists():
        dispatch_failure_reason = "dispatch_qualification_missing"
    elif not dispatch_path.is_file():
        dispatch_failure_reason = "dispatch_qualification_not_regular_file"
    else:
        try:
            loaded = json.loads(dispatch_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                dispatch_payload = loaded
            else:
                dispatch_failure_reason = "dispatch_qualification_invalid"
        except Exception:
            dispatch_failure_reason = "dispatch_qualification_unreadable"

    if dispatch_payload:
        dispatch_generated_at_raw = str(dispatch_payload.get("generated_at") or "").strip()
        summary["dispatch_qualification_generated_at"] = dispatch_generated_at_raw or None
        if dispatch_generated_at_raw:
            dispatch_generated_at_dt = parse_iso(dispatch_generated_at_raw)
            if dispatch_generated_at_dt is None:
                dispatch_failure_reason = dispatch_failure_reason or "dispatch_qualification_generated_at_invalid"
            else:
                dispatch_age_sec = max(0, int((now_dt - dispatch_generated_at_dt).total_seconds()))
                summary["dispatch_qualification_age_sec"] = dispatch_age_sec
                dispatch_fresh = True if dispatch_max_age_sec <= 0 else dispatch_age_sec <= dispatch_max_age_sec
                summary["dispatch_qualification_fresh"] = dispatch_fresh
                if dispatch_fresh is False:
                    dispatch_failure_reason = "dispatch_qualification_stale"

    summary["dispatch_qualification_failure_reason"] = dispatch_failure_reason

    launch_readiness = (
        dispatch_payload.get("launch_readiness")
        if isinstance(dispatch_payload.get("launch_readiness"), dict)
        else {}
    )
    source_obj = dispatch_payload.get("source") if isinstance(dispatch_payload.get("source"), dict) else {}

    artifact_required = bool(source_obj.get("worker_health_canary_artifact_required") is True)
    gate_required = bool(
        source_obj.get("worker_health_gate_required") is True
        or launch_readiness.get("worker_health_gate_required") is True
        or artifact_required
    )
    summary["artifact_required"] = artifact_required
    summary["gate_required"] = gate_required

    canary_max_age_sec = max(
        0,
        safe_int(source_obj.get("worker_health_canary_max_age_sec"), canary_max_age_default),
    )
    canary_future_skew_sec = max(
        0,
        safe_int(source_obj.get("worker_health_canary_future_skew_sec"), canary_future_skew_default),
    )
    summary["worker_health_canary_max_age_sec"] = canary_max_age_sec
    summary["worker_health_canary_future_skew_sec"] = canary_future_skew_sec

    dispatch_resource_preflight = (
        dispatch_payload.get("resource_preflight")
        if isinstance(dispatch_payload.get("resource_preflight"), dict)
        else {}
    )
    dispatch_uncertainty_signal = (
        dispatch_payload.get("uncertainty_signal")
        if isinstance(dispatch_payload.get("uncertainty_signal"), dict)
        else {}
    )

    summary["resource_preflight_required"] = bool(
        source_obj.get("resource_preflight_required") is True
        or dispatch_resource_preflight
    )
    summary["resource_preflight_status"] = str(
        dispatch_resource_preflight.get("status") or ""
    ).strip() or None
    summary["resource_preflight_reason"] = str(
        dispatch_resource_preflight.get("reason") or ""
    ).strip() or None
    summary["resource_preflight_blocking_candidate_count"] = max(
        0,
        safe_int(dispatch_resource_preflight.get("blocking_candidate_count"), 0),
    )
    summary["resource_preflight_blocking_task_ids"] = [
        str(token).strip()
        for token in (dispatch_resource_preflight.get("blocking_task_ids") or [])
        if str(token).strip()
    ]
    telemetry_complete = dispatch_resource_preflight.get("telemetry_complete")
    summary["resource_preflight_telemetry_complete"] = (
        telemetry_complete if isinstance(telemetry_complete, bool) else None
    )
    summary["resource_preflight_lowest_headroom_pct"] = (
        safe_int(dispatch_resource_preflight.get("lowest_headroom_pct"), 0)
        if dispatch_resource_preflight.get("lowest_headroom_pct") is not None
        else None
    )

    uncertainty_score = dispatch_uncertainty_signal.get("confidence_score")
    summary["uncertainty_confidence_score"] = (
        float(uncertainty_score)
        if isinstance(uncertainty_score, (int, float))
        else None
    )
    summary["uncertainty_confidence_label"] = str(
        dispatch_uncertainty_signal.get("confidence_label") or ""
    ).strip() or None
    summary["uncertainty_confidence_quantiles"] = (
        dict(dispatch_uncertainty_signal.get("confidence_quantiles") or {})
        if isinstance(dispatch_uncertainty_signal.get("confidence_quantiles"), dict)
        else {}
    )
    summary["uncertainty_reasons"] = [
        str(token).strip()
        for token in (dispatch_uncertainty_signal.get("uncertainty_reasons") or [])
        if str(token).strip()
    ]
    uncertainty_requires_operator_review = dispatch_uncertainty_signal.get("requires_operator_review")
    summary["uncertainty_requires_operator_review"] = (
        uncertainty_requires_operator_review
        if isinstance(uncertainty_requires_operator_review, bool)
        else None
    )

    dispatch_canary_present = bool(source_obj.get("worker_health_canary_present") is True)
    canary_path_raw = str(source_obj.get("worker_health_canary_path") or "").strip()
    canary_schema = str(source_obj.get("worker_health_canary_schema") or "").strip() or None
    canary_path_obj = _resolve_artifact_path(canary_path_raw) if canary_path_raw else None
    if canary_path_obj is None:
        canary_path_obj = _resolve_artifact_path(str(default_canary_path))
    if canary_path_obj is not None:
        summary["worker_health_canary_path"] = str(canary_path_obj)
        summary["inspect_worker_health_canary_command"] = f"cat {canary_path_obj}"

    dispatch_canary_eval: dict | None = None
    if gate_required and artifact_required:
        dispatch_canary_eval = _evaluate_canary_artifact(
            path_obj=canary_path_obj,
            declared_present=dispatch_canary_present,
            declared_schema=canary_schema,
            canary_max_age_sec=canary_max_age_sec,
            canary_future_skew_sec=canary_future_skew_sec,
        )

    fallback_canary_eval: dict | None = None
    fallback_checked = False
    should_attempt_fallback = bool((dispatch_canary_eval is None) or (dispatch_fresh is False))
    if should_attempt_fallback and canary_path_obj is not None:
        fallback_checked = bool(canary_path_obj.exists() and canary_path_obj.is_file())
        if fallback_checked:
            fallback_canary_eval = _evaluate_canary_artifact(
                path_obj=canary_path_obj,
                declared_present=None,
                declared_schema=None,
                canary_max_age_sec=canary_max_age_sec,
                canary_future_skew_sec=canary_future_skew_sec,
            )

    using_standalone_fallback = bool(
        fallback_canary_eval is not None and ((dispatch_canary_eval is None) or (dispatch_fresh is False))
    )
    source_canary_eval = fallback_canary_eval if using_standalone_fallback else dispatch_canary_eval

    if isinstance(source_canary_eval, dict):
        summary["worker_health_canary_present"] = bool(source_canary_eval.get("present") is True)
        summary["worker_health_canary_path"] = source_canary_eval.get("path")
        summary["worker_health_canary_schema"] = source_canary_eval.get("schema")
        summary["worker_health_canary_generated_at"] = source_canary_eval.get("generated_at")
        summary["worker_health_canary_age_sec"] = source_canary_eval.get("age_sec")
        summary["worker_health_canary_fresh"] = source_canary_eval.get("fresh")
        if source_canary_eval.get("path"):
            summary["inspect_worker_health_canary_command"] = f"cat {source_canary_eval.get('path')}"

    summary["worker_health_canary_source"] = (
        "standalone_worker_health_canary"
        if using_standalone_fallback
        else ("dispatch_qualification" if isinstance(dispatch_canary_eval, dict) else None)
    )

    canary_failure_reason = (
        str(source_canary_eval.get("failure_reason") or "").strip()
        if isinstance(source_canary_eval, dict)
        else ""
    )

    if canary_failure_reason:
        summary["failure_reason"] = canary_failure_reason
    elif dispatch_failure_reason:
        summary["failure_reason"] = dispatch_failure_reason
    elif not gate_required:
        summary["failure_reason"] = "worker_health_canary_gate_not_required"
    elif gate_required and artifact_required and not isinstance(source_canary_eval, dict):
        summary["failure_reason"] = "worker_health_canary_missing"

    resource_preflight_status = str(summary.get("resource_preflight_status") or "").strip().lower()
    resource_preflight_reason = str(summary.get("resource_preflight_reason") or "").strip()
    resource_preflight_idle_no_dispatch_candidate = bool(
        summary.get("resource_preflight_required") is True
        and resource_preflight_status == "idle"
        and resource_preflight_reason == "no_dispatch_candidate_to_validate"
        and max(0, safe_int(summary.get("resource_preflight_blocking_candidate_count"), 0)) == 0
    )
    if (
        summary.get("resource_preflight_required") is True
        and resource_preflight_status == "blocked"
        and summary.get("dispatch_qualification_fresh") is not False
    ):
        summary["active_blocker"] = True
        summary["blocker_reason"] = "dispatch_resource_preflight_blocked"
        if not summary.get("failure_reason") or summary.get("failure_reason") == "worker_health_canary_gate_not_required":
            summary["failure_reason"] = "dispatch_resource_preflight_blocked"
    elif (
        summary.get("resource_preflight_required") is True
        and resource_preflight_status in {"degraded", "unknown"}
        and summary.get("failure_reason") in {None, "", "worker_health_canary_gate_not_required"}
    ):
        summary["failure_reason"] = "dispatch_resource_preflight_degraded"

    if (
        summary.get("failure_reason") in canary_failure_reasons
        and (summary.get("dispatch_qualification_fresh") is not False or using_standalone_fallback)
    ):
        if resource_preflight_idle_no_dispatch_candidate:
            summary["active_blocker"] = False
            summary["blocker_reason"] = None
        else:
            summary["active_blocker"] = True
            summary["blocker_reason"] = summary.get("failure_reason")

    uncertainty_review_required = bool(summary.get("uncertainty_requires_operator_review") is True)
    if (
        summary.get("active_blocker") is not True
        and uncertainty_review_required
        and summary.get("failure_reason") in {None, "", "worker_health_canary_gate_not_required"}
    ):
        summary["failure_reason"] = "dispatch_uncertainty_operator_review_required"

    if summary.get("active_blocker"):
        summary["action_priority"] = "p1"
        if (
            summary.get("failure_reason") in canary_failure_reasons
            and summary.get("worker_health_canary_source") == "standalone_worker_health_canary"
        ):
            summary["first_actionable_command"] = (
                summary.get("inspect_worker_health_canary_command")
                or summary.get("refresh_worker_health_canary_command")
                or summary.get("refresh_dispatch_qualification_command")
                or summary.get("inspect_dispatch_qualification_command")
            )
        else:
            summary["first_actionable_command"] = (
                summary.get("inspect_dispatch_qualification_command")
                or summary.get("inspect_worker_health_canary_command")
                or summary.get("refresh_worker_health_canary_command")
                or summary.get("refresh_dispatch_qualification_command")
            )
    elif summary.get("failure_reason") in canary_failure_reasons:
        summary["action_priority"] = "p2"
        summary["first_actionable_command"] = (
            summary.get("inspect_worker_health_canary_command")
            or summary.get("refresh_worker_health_canary_command")
            or summary.get("refresh_dispatch_qualification_command")
            or summary.get("inspect_dispatch_qualification_command")
        )
    elif summary.get("failure_reason") in dispatch_failure_reasons:
        summary["action_priority"] = "p2"
        summary["first_actionable_command"] = (
            summary.get("refresh_dispatch_qualification_command")
            or summary.get("inspect_dispatch_qualification_command")
        )
    elif str(summary.get("failure_reason") or "").strip() in {
        "dispatch_resource_preflight_blocked",
        "dispatch_resource_preflight_degraded",
        "dispatch_uncertainty_operator_review_required",
    }:
        summary["action_priority"] = "p2" if summary.get("active_blocker") is not True else "p1"
        summary["first_actionable_command"] = (
            summary.get("inspect_dispatch_qualification_command")
            or summary.get("refresh_dispatch_qualification_command")
        )
    elif uncertainty_review_required:
        summary["action_priority"] = "p2"
        summary["first_actionable_command"] = (
            summary.get("inspect_dispatch_qualification_command")
            or summary.get("refresh_dispatch_qualification_command")
        )

    if (
        summary.get("failure_reason") == "dispatch_qualification_missing"
        and not fallback_checked
        and summary.get("first_actionable_command") is None
    ):
        summary["first_actionable_command"] = (
            summary.get("refresh_dispatch_qualification_command")
            or summary.get("inspect_dispatch_qualification_command")
        )

    return summary


def summarize_launch_readiness_probe_execution_gate(now_dt: dt.datetime) -> dict:
    dispatch_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_PATH", ""))
    root_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_ROOT", "")).expanduser()
    max_age_sec = max(
        0,
        safe_int(os.environ.get("VERIFY_GATE_STATUS_DISPATCH_QUALIFICATION_MAX_AGE_SEC", "21600"), 21600),
    )
    overdue_blocker_min = max(
        1,
        safe_int(os.environ.get("VERIFY_GATE_STATUS_PROBE_OVERDUE_BLOCKER_MIN", "1"), 1),
    )
    refresh_command = (
        str(os.environ.get("VERIFY_GATE_STATUS_RUN_DISPATCH_QUALIFICATION_REFRESH_COMMAND", "")).strip()
        or None
    )

    summary = {
        "dispatch_qualification_path": str(dispatch_path),
        "dispatch_qualification_present": dispatch_path.exists(),
        "dispatch_qualification_failure_reason": None,
        "max_age_sec": max_age_sec,
        "generated_at": None,
        "age_sec": None,
        "fresh": None,
        "probe_execution_source": None,
        "probe_execution_status": None,
        "probe_execution_reason": None,
        "pending_worker_count": 0,
        "due_now_worker_count": 0,
        "overdue_worker_count": 0,
        "due_now_cohort_workers": [],
        "due_now_cohort_signature": None,
        "due_now_cohort_signature_first_seen_at": None,
        "due_now_cohort_signature_consecutive_ticks": 0,
        "overdue_cohort_workers": [],
        "overdue_cohort_signature": None,
        "overdue_cohort_signature_first_seen_at": None,
        "overdue_cohort_signature_consecutive_ticks": 0,
        "oldest_due_now_started_at": None,
        "oldest_due_now_worker": None,
        "oldest_due_now_age_sec": None,
        "oldest_overdue_started_at": None,
        "oldest_overdue_worker": None,
        "oldest_overdue_age_sec": None,
        "overdue_blocker_min": overdue_blocker_min,
        "due_now_active": False,
        "overdue_active": False,
        "launch_readiness_state": None,
        "launch_readiness_reason": None,
        "due_now_idle_no_dispatch_candidate": False,
        "active_blocker": False,
        "blocker_reason": None,
        "failure_reason": None,
        "inspect_dispatch_qualification_command": f"cat {dispatch_path}",
        "refresh_dispatch_qualification_command": refresh_command,
        "probe_execution_plan_path": None,
        "probe_execution_plan_present": None,
        "inspect_probe_execution_plan_command": None,
        "first_actionable_command": None,
        "action_priority": None,
        "action_priority_source": None,
        "demotion_restore_pending_worker_count": 0,
        "demotion_demoted_worker_count": 0,
        "demotion_restored_worker_count": 0,
        "demotion_overdue_probe_worker_count": 0,
        "demotion_oldest_restore_pending_since": None,
        "demotion_oldest_restore_pending_worker": None,
        "demotion_oldest_restore_pending_age_sec": None,
        "demotion_oldest_demoted_at": None,
        "demotion_oldest_demoted_worker": None,
        "demotion_oldest_demoted_age_sec": None,
        "demotion_latest_restored_at": None,
        "demotion_latest_restored_worker": None,
        "demotion_latest_restored_age_sec": None,
        "demotion_action_priority": None,
    }

    default_probe_plan_path = dispatch_path.with_name("execution_supervisor_probe_execution_plan_latest.json")
    summary["probe_execution_plan_path"] = str(default_probe_plan_path)
    summary["probe_execution_plan_present"] = bool(
        default_probe_plan_path.exists() and default_probe_plan_path.is_file()
    )
    summary["inspect_probe_execution_plan_command"] = f"cat {default_probe_plan_path}"

    dispatch_failure_reason: str | None = None
    dispatch_payload: dict = {}
    dispatch_probe_execution_plan: dict = {}
    dispatch_generated_at_raw = ""
    dispatch_age_sec: int | None = None
    dispatch_fresh: bool | None = None

    if not dispatch_path.exists():
        dispatch_failure_reason = "dispatch_qualification_missing"
    elif not dispatch_path.is_file():
        dispatch_failure_reason = "dispatch_qualification_not_regular_file"
    else:
        try:
            loaded = json.loads(dispatch_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                dispatch_payload = loaded
            else:
                dispatch_failure_reason = "dispatch_qualification_invalid"
        except Exception:
            dispatch_failure_reason = "dispatch_qualification_unreadable"

    if dispatch_payload:
        dispatch_generated_at_raw = str(dispatch_payload.get("generated_at") or "").strip()
        if dispatch_generated_at_raw:
            dispatch_generated_at_dt = parse_iso(dispatch_generated_at_raw)
            if dispatch_generated_at_dt is None:
                dispatch_failure_reason = (
                    dispatch_failure_reason or "dispatch_qualification_generated_at_invalid"
                )
            else:
                dispatch_age_sec = max(0, int((now_dt - dispatch_generated_at_dt).total_seconds()))
                dispatch_fresh = True if max_age_sec <= 0 else dispatch_age_sec <= max_age_sec
                if dispatch_fresh is False:
                    dispatch_failure_reason = "dispatch_qualification_stale"

        launch_readiness = (
            dispatch_payload.get("launch_readiness")
            if isinstance(dispatch_payload.get("launch_readiness"), dict)
            else {}
        )
        summary["launch_readiness_state"] = str(launch_readiness.get("state") or "").strip() or None
        summary["launch_readiness_reason"] = str(launch_readiness.get("reason") or "").strip() or None
        launch_readiness_demotion_posture = (
            launch_readiness.get("demotion_restore_posture")
            if isinstance(launch_readiness.get("demotion_restore_posture"), dict)
            else {}
        )
        launch_readiness_canary_schedule = (
            launch_readiness.get("canary_probe_schedule")
            if isinstance(launch_readiness.get("canary_probe_schedule"), dict)
            else {}
        )

        summary["demotion_restore_pending_worker_count"] = max(
            0,
            safe_int(launch_readiness_demotion_posture.get("restore_pending_worker_count"), 0),
        )
        summary["demotion_demoted_worker_count"] = max(
            0,
            safe_int(launch_readiness_demotion_posture.get("demoted_worker_count"), 0),
        )
        summary["demotion_restored_worker_count"] = max(
            0,
            safe_int(launch_readiness_demotion_posture.get("restored_worker_count"), 0),
        )
        summary["demotion_overdue_probe_worker_count"] = max(
            0,
            safe_int(launch_readiness_canary_schedule.get("overdue_probe_worker_count"), 0),
        )
        summary["demotion_oldest_restore_pending_since"] = (
            str(launch_readiness_demotion_posture.get("oldest_restore_pending_since") or "").strip() or None
        )
        summary["demotion_oldest_restore_pending_worker"] = (
            str(launch_readiness_demotion_posture.get("oldest_restore_pending_worker") or "").strip() or None
        )
        if launch_readiness_demotion_posture.get("oldest_restore_pending_age_sec") is not None:
            summary["demotion_oldest_restore_pending_age_sec"] = max(
                0,
                safe_int(launch_readiness_demotion_posture.get("oldest_restore_pending_age_sec"), 0),
            )
        summary["demotion_oldest_demoted_at"] = (
            str(launch_readiness_demotion_posture.get("oldest_demoted_at") or "").strip() or None
        )
        summary["demotion_oldest_demoted_worker"] = (
            str(launch_readiness_demotion_posture.get("oldest_demoted_worker") or "").strip() or None
        )
        if launch_readiness_demotion_posture.get("oldest_demoted_age_sec") is not None:
            summary["demotion_oldest_demoted_age_sec"] = max(
                0,
                safe_int(launch_readiness_demotion_posture.get("oldest_demoted_age_sec"), 0),
            )
        summary["demotion_latest_restored_at"] = (
            str(launch_readiness_demotion_posture.get("latest_restored_at") or "").strip() or None
        )
        summary["demotion_latest_restored_worker"] = (
            str(launch_readiness_demotion_posture.get("latest_restored_worker") or "").strip() or None
        )
        if launch_readiness_demotion_posture.get("latest_restored_age_sec") is not None:
            summary["demotion_latest_restored_age_sec"] = max(
                0,
                safe_int(launch_readiness_demotion_posture.get("latest_restored_age_sec"), 0),
            )

        demotion_action_priority = str(
            launch_readiness_demotion_posture.get("action_priority") or ""
        ).strip().lower()
        if demotion_action_priority not in {"p1", "p2"}:
            if summary["demotion_overdue_probe_worker_count"] > 0:
                demotion_action_priority = "p1"
            elif summary["demotion_restore_pending_worker_count"] > 0:
                demotion_action_priority = "p2"
            else:
                demotion_action_priority = ""
        if demotion_action_priority in {"p1", "p2"}:
            summary["demotion_action_priority"] = demotion_action_priority

        dispatch_probe_execution_plan = (
            launch_readiness.get("probe_execution_plan")
            if isinstance(launch_readiness.get("probe_execution_plan"), dict)
            else (
                dispatch_payload.get("probe_execution_plan")
                if isinstance(dispatch_payload.get("probe_execution_plan"), dict)
                else {}
            )
        )

        probe_plan_path_raw = str(
            dispatch_probe_execution_plan.get("source_path")
            or dispatch_probe_execution_plan.get("plan_path")
            or ""
        ).strip()
        if probe_plan_path_raw:
            probe_plan_candidate = pathlib.Path(probe_plan_path_raw)
            if not probe_plan_candidate.is_absolute() and str(root_path):
                probe_plan_candidate = root_path / probe_plan_candidate
            try:
                probe_plan_candidate = probe_plan_candidate.resolve()
            except Exception:
                pass
            summary["probe_execution_plan_path"] = str(probe_plan_candidate)
            summary["probe_execution_plan_present"] = bool(
                probe_plan_candidate.exists() and probe_plan_candidate.is_file()
            )
            summary["inspect_probe_execution_plan_command"] = f"cat {probe_plan_candidate}"

    fallback_probe_execution_plan: dict = {}
    fallback_failure_reason: str | None = None
    fallback_generated_at_raw = ""
    fallback_age_sec: int | None = None
    fallback_fresh: bool | None = None

    should_attempt_fallback = bool((not dispatch_probe_execution_plan) or (dispatch_fresh is False))
    if should_attempt_fallback:
        probe_plan_path_obj = pathlib.Path(str(summary.get("probe_execution_plan_path") or "").strip())
        if probe_plan_path_obj.exists() and probe_plan_path_obj.is_file():
            try:
                loaded = json.loads(probe_plan_path_obj.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    fallback_probe_execution_plan = loaded
                else:
                    fallback_failure_reason = "launch_readiness_probe_execution_plan_invalid"
            except Exception:
                fallback_failure_reason = "launch_readiness_probe_execution_plan_unreadable"
        elif not dispatch_probe_execution_plan:
            fallback_failure_reason = "launch_readiness_probe_execution_plan_missing"

        if fallback_probe_execution_plan:
            fallback_generated_at_raw = str(fallback_probe_execution_plan.get("generated_at") or "").strip()
            if fallback_generated_at_raw:
                fallback_generated_at_dt = parse_iso(fallback_generated_at_raw)
                if fallback_generated_at_dt is None:
                    fallback_failure_reason = "launch_readiness_probe_execution_plan_generated_at_invalid"
                else:
                    fallback_age_sec = max(0, int((now_dt - fallback_generated_at_dt).total_seconds()))
                    fallback_fresh = True if max_age_sec <= 0 else fallback_age_sec <= max_age_sec
                    if fallback_fresh is False:
                        fallback_failure_reason = "launch_readiness_probe_execution_plan_stale"

    source_probe_execution_plan = dispatch_probe_execution_plan
    source_generated_at_raw = dispatch_generated_at_raw
    source_age_sec = dispatch_age_sec
    source_fresh = dispatch_fresh
    source_failure_reason = dispatch_failure_reason
    using_standalone_fallback = False

    if fallback_probe_execution_plan and ((not dispatch_probe_execution_plan) or (dispatch_fresh is False)):
        source_probe_execution_plan = fallback_probe_execution_plan
        source_generated_at_raw = fallback_generated_at_raw
        source_age_sec = fallback_age_sec
        source_fresh = fallback_fresh
        source_failure_reason = dispatch_failure_reason or fallback_failure_reason
        using_standalone_fallback = True

    summary["dispatch_qualification_failure_reason"] = dispatch_failure_reason
    summary["probe_execution_source"] = (
        "standalone_probe_execution_plan"
        if using_standalone_fallback
        else ("dispatch_qualification" if dispatch_probe_execution_plan else None)
    )
    summary["generated_at"] = source_generated_at_raw or None
    summary["age_sec"] = source_age_sec
    summary["fresh"] = source_fresh

    summary["probe_execution_status"] = str(source_probe_execution_plan.get("status") or "").strip() or None
    summary["probe_execution_reason"] = str(source_probe_execution_plan.get("reason") or "").strip() or None
    summary["pending_worker_count"] = max(0, safe_int(source_probe_execution_plan.get("pending_worker_count"), 0))
    summary["due_now_worker_count"] = max(0, safe_int(source_probe_execution_plan.get("due_now_worker_count"), 0))
    summary["overdue_worker_count"] = max(0, safe_int(source_probe_execution_plan.get("overdue_worker_count"), 0))
    summary["due_now_cohort_workers"] = sorted(
        dedupe_nonempty_strings(source_probe_execution_plan.get("due_now_cohort_workers") or [])
    )
    summary["due_now_cohort_signature"] = (
        str(source_probe_execution_plan.get("due_now_cohort_signature") or "").strip() or None
    )
    summary["due_now_cohort_signature_first_seen_at"] = (
        str(source_probe_execution_plan.get("due_now_cohort_signature_first_seen_at") or "").strip() or None
    )
    summary["due_now_cohort_signature_consecutive_ticks"] = max(
        0,
        safe_int(source_probe_execution_plan.get("due_now_cohort_signature_consecutive_ticks"), 0),
    )
    summary["overdue_cohort_workers"] = sorted(
        dedupe_nonempty_strings(source_probe_execution_plan.get("overdue_cohort_workers") or [])
    )
    summary["overdue_cohort_signature"] = (
        str(source_probe_execution_plan.get("overdue_cohort_signature") or "").strip() or None
    )
    summary["overdue_cohort_signature_first_seen_at"] = (
        str(source_probe_execution_plan.get("overdue_cohort_signature_first_seen_at") or "").strip() or None
    )
    summary["overdue_cohort_signature_consecutive_ticks"] = max(
        0,
        safe_int(source_probe_execution_plan.get("overdue_cohort_signature_consecutive_ticks"), 0),
    )
    summary["oldest_due_now_started_at"] = str(source_probe_execution_plan.get("oldest_due_now_started_at") or "").strip() or None
    summary["oldest_due_now_worker"] = str(source_probe_execution_plan.get("oldest_due_now_worker") or "").strip() or None
    summary["oldest_due_now_age_sec"] = max(0, safe_int(source_probe_execution_plan.get("oldest_due_now_age_sec"), 0)) if source_probe_execution_plan.get("oldest_due_now_age_sec") is not None else None
    summary["oldest_overdue_started_at"] = str(source_probe_execution_plan.get("oldest_overdue_started_at") or "").strip() or None
    summary["oldest_overdue_worker"] = str(source_probe_execution_plan.get("oldest_overdue_worker") or "").strip() or None
    summary["oldest_overdue_age_sec"] = max(0, safe_int(source_probe_execution_plan.get("oldest_overdue_age_sec"), 0)) if source_probe_execution_plan.get("oldest_overdue_age_sec") is not None else None
    summary["due_now_active"] = bool(summary["due_now_worker_count"] > 0)
    summary["overdue_active"] = bool(summary["overdue_worker_count"] > 0)
    plan_action_priority = str(source_probe_execution_plan.get("action_priority") or "").strip().lower()
    if plan_action_priority in {"p1", "p2"}:
        summary["action_priority"] = plan_action_priority
        summary["action_priority_source"] = "probe_execution_plan"

    if not source_probe_execution_plan and source_failure_reason is None:
        source_failure_reason = fallback_failure_reason or "launch_readiness_probe_execution_plan_missing"

    due_now_idle_no_dispatch_candidate = bool(
        summary["due_now_worker_count"] > 0
        and summary["overdue_worker_count"] <= 0
        and str(summary.get("launch_readiness_state") or "").strip() == "idle"
        and str(summary.get("launch_readiness_reason") or "").strip() == "no_dispatch_candidate_to_qualify"
    )
    summary["due_now_idle_no_dispatch_candidate"] = due_now_idle_no_dispatch_candidate

    if (
        summary["overdue_worker_count"] >= overdue_blocker_min
        and summary.get("fresh") is not False
    ):
        summary["active_blocker"] = True
        summary["blocker_reason"] = "probe_execution_overdue"
        summary["failure_reason"] = "launch_readiness_probe_execution_overdue_blocker_active"
    elif (
        summary["due_now_worker_count"] > 0
        and (source_failure_reason is None or (using_standalone_fallback and summary.get("fresh") is not False))
    ):
        summary["failure_reason"] = (
            "launch_readiness_probe_execution_due_now_idle_no_dispatch_candidate"
            if due_now_idle_no_dispatch_candidate
            else "launch_readiness_probe_execution_due_now"
        )

    if summary.get("failure_reason") is None:
        summary["failure_reason"] = source_failure_reason

    if summary.get("action_priority") is None and bool(summary.get("active_blocker")):
        summary["action_priority"] = "p1"
        summary["action_priority_source"] = "probe_execution_gate"
    elif summary.get("action_priority") is None and bool(summary.get("due_now_active")):
        summary["action_priority"] = "p2"
        summary["action_priority_source"] = "probe_execution_gate"

    if summary.get("action_priority") is None and summary.get("demotion_action_priority") in {"p1", "p2"}:
        summary["action_priority"] = summary.get("demotion_action_priority")
        summary["action_priority_source"] = "demotion_restore_posture"

    if (
        bool(summary.get("active_blocker"))
        or bool(summary.get("due_now_active"))
        or summary.get("action_priority") in {"p1", "p2"}
    ):
        if summary.get("action_priority_source") == "demotion_restore_posture":
            summary["first_actionable_command"] = (
                summary.get("inspect_dispatch_qualification_command")
                or summary.get("inspect_probe_execution_plan_command")
            )
        else:
            summary["first_actionable_command"] = (
                summary.get("inspect_probe_execution_plan_command")
                or summary.get("inspect_dispatch_qualification_command")
            )
    elif summary.get("failure_reason") in {
        "dispatch_qualification_missing",
        "dispatch_qualification_not_regular_file",
        "dispatch_qualification_unreadable",
        "dispatch_qualification_invalid",
        "dispatch_qualification_stale",
        "launch_readiness_probe_execution_plan_missing",
        "launch_readiness_probe_execution_plan_unreadable",
        "launch_readiness_probe_execution_plan_invalid",
        "launch_readiness_probe_execution_plan_generated_at_invalid",
        "launch_readiness_probe_execution_plan_stale",
    }:
        summary["first_actionable_command"] = (
            summary.get("refresh_dispatch_qualification_command")
            or summary.get("inspect_dispatch_qualification_command")
        )

    return summary


def _evaluate_failover_stress_runtime_evidence_gate(
    now_dt: dt.datetime,
    evidence_path: pathlib.Path,
    max_age_sec: int,
    refresh_command: str | None,
) -> dict:
    summary = {
        "failover_stress_runtime_evidence_path": str(evidence_path),
        "present": evidence_path.exists(),
        "max_age_sec": max_age_sec,
        "generated_at": None,
        "age_sec": None,
        "fresh": None,
        "object_type": None,
        "overall_verdict": None,
        "publish_chain_verdict": None,
        "publish_assertions_failed": None,
        "repeatability_status": None,
        "repeatability_match": None,
        "repeatability_mismatch_fields": [],
        "repeatability_identity_transition_tolerated": False,
        "repeatability_tolerated_mismatch_fields": [],
        "active_top_blocker": None,
        "effective_top_blocker": None,
        "active_blocker": False,
        "blocker_reason": None,
        "failure_reason": None,
        "inspect_failover_stress_runtime_evidence_command": f"cat {evidence_path}",
        "refresh_failover_stress_runtime_evidence_command": refresh_command,
    }

    blocker_reasons = {
        "failover_stress_runtime_evidence_missing",
        "failover_stress_runtime_evidence_not_regular_file",
        "failover_stress_runtime_evidence_unreadable",
        "failover_stress_runtime_evidence_invalid",
        "failover_stress_runtime_generated_at_missing",
        "failover_stress_runtime_generated_at_invalid",
        "failover_stress_runtime_stale",
        "failover_stress_runtime_overall_verdict_nonpass",
        "failover_stress_runtime_publish_chain_nonpass",
        "failover_stress_runtime_publish_assertions_failed",
        "failover_stress_runtime_repeatability_mismatch",
    }

    if not evidence_path.exists():
        summary["failure_reason"] = "failover_stress_runtime_evidence_missing"
    elif not evidence_path.is_file():
        summary["failure_reason"] = "failover_stress_runtime_evidence_not_regular_file"
    else:
        payload = None
        try:
            loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
            else:
                summary["failure_reason"] = "failover_stress_runtime_evidence_invalid"
        except Exception:
            summary["failure_reason"] = "failover_stress_runtime_evidence_unreadable"

        if isinstance(payload, dict):
            summary["object_type"] = str(payload.get("object_type") or "").strip() or None
            generated_at_raw = str(payload.get("generated_at") or "").strip()
            summary["generated_at"] = generated_at_raw or None

            evidence_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            repeatability = (
                evidence_summary.get("repeatability")
                if isinstance(evidence_summary.get("repeatability"), dict)
                else {}
            )

            overall_verdict = str(evidence_summary.get("overall_verdict") or "").strip().upper() or "MISSING"
            publish_chain_verdict = (
                str(evidence_summary.get("publish_chain_verdict") or "").strip().upper() or "MISSING"
            )
            publish_assertions_failed = max(0, safe_int(evidence_summary.get("publish_assertions_failed"), 0))
            repeatability_status = str(repeatability.get("status") or "missing").strip().lower() or "missing"

            summary["overall_verdict"] = overall_verdict
            summary["publish_chain_verdict"] = publish_chain_verdict
            summary["publish_assertions_failed"] = publish_assertions_failed
            summary["repeatability_status"] = repeatability_status
            summary["repeatability_match"] = repeatability.get("match")
            summary["repeatability_mismatch_fields"] = [
                str(x).strip()
                for x in (repeatability.get("mismatch_fields") or [])
                if str(x).strip()
            ]
            summary["repeatability_tolerated_mismatch_fields"] = [
                str(x).strip()
                for x in (repeatability.get("tolerated_mismatch_fields") or [])
                if str(x).strip()
            ]
            summary["repeatability_identity_transition_tolerated"] = bool(
                repeatability.get("identity_transition_tolerated") is True
            )
            summary["active_top_blocker"] = (
                str(evidence_summary.get("active_top_blocker") or "").strip() or None
            )
            summary["effective_top_blocker"] = (
                str(evidence_summary.get("effective_top_blocker") or "").strip() or None
            )

            if repeatability_status != "match":
                tolerated, tolerated_fields = repeatability_identity_drift_tolerated(
                    repeatability=repeatability,
                    overall_verdict=overall_verdict,
                    publish_chain_verdict=publish_chain_verdict,
                    publish_assertions_failed=publish_assertions_failed,
                )
                if tolerated:
                    summary["repeatability_status"] = "match"
                    summary["repeatability_match"] = True
                    summary["repeatability_mismatch_fields"] = []
                    summary["repeatability_identity_transition_tolerated"] = True
                    summary["repeatability_tolerated_mismatch_fields"] = tolerated_fields
                    repeatability_status = "match"

            if not generated_at_raw:
                summary["failure_reason"] = "failover_stress_runtime_generated_at_missing"
            else:
                generated_at_dt = parse_iso(generated_at_raw)
                if generated_at_dt is None:
                    summary["failure_reason"] = "failover_stress_runtime_generated_at_invalid"
                else:
                    age_sec = max(0, int((now_dt - generated_at_dt).total_seconds()))
                    summary["age_sec"] = age_sec
                    summary["fresh"] = True if max_age_sec <= 0 else age_sec <= max_age_sec
                    if summary["fresh"] is False:
                        summary["failure_reason"] = "failover_stress_runtime_stale"

            if summary.get("failure_reason") is None:
                if overall_verdict != "PASS":
                    summary["failure_reason"] = "failover_stress_runtime_overall_verdict_nonpass"
                elif publish_chain_verdict != "PASS":
                    summary["failure_reason"] = "failover_stress_runtime_publish_chain_nonpass"
                elif publish_assertions_failed > 0:
                    summary["failure_reason"] = "failover_stress_runtime_publish_assertions_failed"
                elif repeatability_status != "match":
                    summary["failure_reason"] = "failover_stress_runtime_repeatability_mismatch"

    if summary.get("failure_reason") in blocker_reasons:
        summary["active_blocker"] = True
        summary["blocker_reason"] = summary.get("failure_reason")

    return summary


def summarize_failover_stress_runtime_evidence_gate(now_dt: dt.datetime) -> dict:
    evidence_path = pathlib.Path(
        str(
            os.environ.get(
                "VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_EVIDENCE_PATH",
                "",
            )
        ).strip()
    )
    max_age_sec = max(
        0,
        safe_int(
            os.environ.get("VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_MAX_AGE_SEC", "21600"),
            21600,
        ),
    )
    refresh_command = (
        str(os.environ.get("VERIFY_GATE_STATUS_RUN_FAILOVER_STRESS_RUNTIME_REFRESH_COMMAND", "")).strip()
        or None
    )
    auto_refresh_enabled = truthy(
        os.environ.get("VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH", "0")
    )
    auto_refresh_timeout_sec = max(
        1,
        safe_int(
            os.environ.get("VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_TIMEOUT_SEC", "300"),
            300,
        ),
    )
    auto_refresh_cooldown_sec = max(
        0,
        safe_int(
            os.environ.get("VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_COOLDOWN_SEC", "900"),
            900,
        ),
    )
    auto_refresh_state_path = pathlib.Path(
        str(
            os.environ.get(
                "VERIFY_GATE_STATUS_FAILOVER_STRESS_RUNTIME_AUTO_REFRESH_STATE_PATH",
                "",
            )
        ).strip()
    )

    summary = _evaluate_failover_stress_runtime_evidence_gate(
        now_dt=now_dt,
        evidence_path=evidence_path,
        max_age_sec=max_age_sec,
        refresh_command=refresh_command,
    )

    auto_refresh = {
        "enabled": auto_refresh_enabled,
        "attempted": False,
        "status": "disabled" if not auto_refresh_enabled else "not_needed",
        "trigger_reason": None,
        "skipped_reason": None,
        "timeout_sec": auto_refresh_timeout_sec,
        "cooldown_sec": auto_refresh_cooldown_sec,
        "state_path": str(auto_refresh_state_path) if str(auto_refresh_state_path).strip() else None,
        "command": refresh_command,
        "command_rc": None,
        "timed_out": False,
        "last_attempt_at": None,
        "next_allowed_at": None,
        "cooldown_remaining_sec": 0,
    }

    refreshable_reasons = {
        "failover_stress_runtime_evidence_missing",
        "failover_stress_runtime_evidence_not_regular_file",
        "failover_stress_runtime_evidence_unreadable",
        "failover_stress_runtime_evidence_invalid",
        "failover_stress_runtime_generated_at_missing",
        "failover_stress_runtime_generated_at_invalid",
        "failover_stress_runtime_stale",
        "failover_stress_runtime_repeatability_mismatch",
    }

    def _load_auto_refresh_state() -> dict:
        if not str(auto_refresh_state_path).strip() or not auto_refresh_state_path.exists() or not auto_refresh_state_path.is_file():
            return {}
        try:
            loaded = json.loads(auto_refresh_state_path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def _write_auto_refresh_state(payload: dict) -> None:
        if not str(auto_refresh_state_path).strip():
            return
        try:
            auto_refresh_state_path.parent.mkdir(parents=True, exist_ok=True)
            auto_refresh_state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    failure_reason = str(summary.get("failure_reason") or "").strip()
    if not auto_refresh_enabled:
        auto_refresh["skipped_reason"] = "disabled"
    elif not refresh_command:
        auto_refresh["status"] = "skipped"
        auto_refresh["skipped_reason"] = "refresh_command_missing"
    elif not failure_reason:
        auto_refresh["status"] = "not_needed"
    elif failure_reason not in refreshable_reasons:
        auto_refresh["status"] = "skipped"
        auto_refresh["skipped_reason"] = "failure_reason_not_refreshable"
        auto_refresh["trigger_reason"] = failure_reason
    else:
        auto_refresh["trigger_reason"] = failure_reason
        state = _load_auto_refresh_state()
        now_epoch = int(now_dt.timestamp())
        next_allowed_epoch = max(0, safe_int(state.get("next_allowed_epoch"), 0)) if isinstance(state, dict) else 0
        if next_allowed_epoch > 0:
            next_allowed_dt = dt.datetime.fromtimestamp(next_allowed_epoch, tz=dt.timezone.utc)
            auto_refresh["next_allowed_at"] = next_allowed_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if next_allowed_epoch > now_epoch:
            auto_refresh["status"] = "cooldown_active"
            auto_refresh["skipped_reason"] = "cooldown_active"
            auto_refresh["cooldown_remaining_sec"] = int(next_allowed_epoch - now_epoch)
        else:
            auto_refresh["attempted"] = True
            attempt_dt = dt.datetime.now(dt.timezone.utc)
            attempt_iso = attempt_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            auto_refresh["last_attempt_at"] = attempt_iso

            command_rc = None
            timed_out = False
            stdout_tail = ""
            stderr_tail = ""
            try:
                cp = subprocess.run(
                    ["bash", "-lc", str(refresh_command)],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=auto_refresh_timeout_sec,
                    env=os.environ.copy(),
                )
                command_rc = int(cp.returncode)
                stdout_tail = str(cp.stdout or "").strip()[-240:]
                stderr_tail = str(cp.stderr or "").strip()[-240:]
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                command_rc = 124
                stdout_tail = str(getattr(exc, "stdout", "") or "").strip()[-240:]
                stderr_tail = str(getattr(exc, "stderr", "") or "").strip()[-240:]

            auto_refresh["command_rc"] = command_rc
            auto_refresh["timed_out"] = timed_out

            next_allowed_epoch = now_epoch + auto_refresh_cooldown_sec if auto_refresh_cooldown_sec > 0 else 0
            next_allowed_iso = None
            if next_allowed_epoch > 0:
                next_allowed_iso = (
                    dt.datetime.fromtimestamp(next_allowed_epoch, tz=dt.timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                auto_refresh["next_allowed_at"] = next_allowed_iso

            state_payload = {
                "last_attempt_at": attempt_iso,
                "last_trigger_reason": failure_reason,
                "status": "timeout" if timed_out else ("command_ok" if command_rc == 0 else "command_failed"),
                "command_rc": command_rc,
                "timed_out": timed_out,
                "timeout_sec": auto_refresh_timeout_sec,
                "cooldown_sec": auto_refresh_cooldown_sec,
                "next_allowed_epoch": next_allowed_epoch,
                "next_allowed_at": next_allowed_iso,
                "stdout_tail": stdout_tail or None,
                "stderr_tail": stderr_tail or None,
            }
            _write_auto_refresh_state(state_payload)

            if timed_out:
                auto_refresh["status"] = "timeout"
            elif command_rc != 0:
                auto_refresh["status"] = "command_failed"
            else:
                auto_refresh["status"] = "refreshed"
                refreshed_summary = _evaluate_failover_stress_runtime_evidence_gate(
                    now_dt=dt.datetime.now(dt.timezone.utc),
                    evidence_path=evidence_path,
                    max_age_sec=max_age_sec,
                    refresh_command=refresh_command,
                )
                refreshed_summary["auto_refresh"] = auto_refresh
                return refreshed_summary

    summary["auto_refresh"] = auto_refresh
    return summary


def summarize_layered_health_gate(now_dt: dt.datetime) -> dict:
    layered_health_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_LAYERED_HEALTH_PATH", ""))
    slo_snapshot_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_SLO_SNAPSHOT_PATH", ""))
    run_layered_health_command = str(os.environ.get("VERIFY_GATE_STATUS_RUN_LAYERED_HEALTH_COMMAND", "")).strip() or None
    run_slo_snapshot_command = str(os.environ.get("VERIFY_GATE_STATUS_RUN_SLO_SNAPSHOT_COMMAND", "")).strip() or None

    required_lanes = [
        token.strip()
        for token in str(os.environ.get("VERIFY_GATE_STATUS_HEALTH_REQUIRED_LANES", "")).split(",")
        if token.strip()
    ]
    if not required_lanes:
        required_lanes = [
            "A1_CONTROL_PLANE",
            "A2_RUNTIME_CONTINUITY",
            "A3_MODEL_ROUTING",
            "A6_OPS_OBSERVABILITY",
            "C1_OPERATOR_SURFACE",
            "C2_RELEASE_SUBSTRATE",
        ]

    min_health_layer = str(os.environ.get("VERIFY_GATE_STATUS_HEALTH_MIN_LAYER", "truthful") or "truthful").strip().lower() or "truthful"
    layer_rank = {
        "alive": 1,
        "ready": 2,
        "safe-to-act": 3,
        "truthful": 4,
    }
    if min_health_layer not in layer_rank:
        min_health_layer = "truthful"

    summary = {
        "layered_health_snapshot_path": str(layered_health_path),
        "layered_health_snapshot_present": layered_health_path.exists(),
        "slo_snapshot_path": str(slo_snapshot_path),
        "slo_snapshot_present": slo_snapshot_path.exists(),
        "required_lanes": required_lanes,
        "min_health_layer": min_health_layer,
        "health_status": None,
        "health_layer": None,
        "lane_count": 0,
        "pass_count": 0,
        "degraded_count": 0,
        "failing_count": 0,
        "missing_required_lanes": [],
        "failing_required_lanes": [],
        "layer_insufficient_required_lanes": [],
        "restore_slo_status": None,
        "restore_slo_detail": None,
        "failure_reason": None,
        "closeout_ready": False,
        "inspect_layered_health_command": f"cat {layered_health_path}",
        "inspect_slo_snapshot_command": f"cat {slo_snapshot_path}",
        "run_layered_health_command": run_layered_health_command,
        "run_slo_snapshot_command": run_slo_snapshot_command,
    }

    layered_payload = None
    if not layered_health_path.exists():
        summary["failure_reason"] = "layered_health_snapshot_missing"
    elif not layered_health_path.is_file():
        summary["failure_reason"] = "layered_health_snapshot_not_regular_file"
    else:
        try:
            loaded = json.loads(layered_health_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                layered_payload = loaded
            else:
                summary["failure_reason"] = "layered_health_snapshot_invalid"
        except Exception:
            summary["failure_reason"] = "layered_health_snapshot_unreadable"

    if isinstance(layered_payload, dict):
        health_status = str(layered_payload.get("status") or "").strip().lower() or None
        health_layer = str(layered_payload.get("health_layer") or "").strip().lower() or None
        summary["health_status"] = health_status
        summary["health_layer"] = health_layer

        lane_rows = layered_payload.get("lanes") if isinstance(layered_payload.get("lanes"), list) else []
        lane_index = {
            str((row or {}).get("lane") or "").strip(): row
            for row in lane_rows
            if isinstance(row, dict) and str((row or {}).get("lane") or "").strip()
        }

        summary["lane_count"] = len(lane_rows)
        summary["pass_count"] = sum(1 for row in lane_rows if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "pass")
        summary["degraded_count"] = sum(1 for row in lane_rows if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "degraded")
        summary["failing_count"] = sum(1 for row in lane_rows if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "failing")

        missing_required = [lane for lane in required_lanes if lane not in lane_index]
        failing_required = []
        layer_insufficient_required = []
        for lane in required_lanes:
            row = lane_index.get(lane)
            if not isinstance(row, dict):
                continue
            lane_status = str(row.get("status") or "").strip().lower()
            lane_layer = str(row.get("health_layer") or "").strip().lower()
            if lane_status != "pass":
                failing_required.append(lane)
            if layer_rank.get(lane_layer, 0) < layer_rank[min_health_layer]:
                layer_insufficient_required.append(lane)

        summary["missing_required_lanes"] = missing_required
        summary["failing_required_lanes"] = failing_required
        summary["layer_insufficient_required_lanes"] = layer_insufficient_required

        if summary.get("failure_reason") is None:
            if health_status != "pass":
                summary["failure_reason"] = "layered_health_not_pass"
            elif layer_rank.get(str(health_layer or ""), 0) < layer_rank[min_health_layer]:
                summary["failure_reason"] = "layered_health_layer_insufficient"
            elif missing_required:
                summary["failure_reason"] = "layered_health_required_lanes_missing"
            elif failing_required:
                summary["failure_reason"] = "layered_health_required_lanes_not_pass"
            elif layer_insufficient_required:
                summary["failure_reason"] = "layered_health_required_lanes_layer_insufficient"

    slo_payload = None
    if not slo_snapshot_path.exists():
        if summary.get("failure_reason") is None:
            summary["failure_reason"] = "slo_snapshot_missing"
    elif not slo_snapshot_path.is_file():
        if summary.get("failure_reason") is None:
            summary["failure_reason"] = "slo_snapshot_not_regular_file"
    else:
        try:
            loaded = json.loads(slo_snapshot_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                slo_payload = loaded
            elif summary.get("failure_reason") is None:
                summary["failure_reason"] = "slo_snapshot_invalid"
        except Exception:
            if summary.get("failure_reason") is None:
                summary["failure_reason"] = "slo_snapshot_unreadable"

    if isinstance(slo_payload, dict):
        evaluations = slo_payload.get("evaluations") if isinstance(slo_payload.get("evaluations"), list) else []
        restore_eval = next(
            (
                row for row in evaluations
                if isinstance(row, dict) and str(row.get("id") or "").strip() == "SLO-4_RESTORE_DRILL_FRESHNESS"
            ),
            None,
        )
        restore_status = str((restore_eval or {}).get("status") or "").strip().lower() or None
        restore_detail = str((restore_eval or {}).get("detail") or "").strip() or None
        summary["restore_slo_status"] = restore_status
        summary["restore_slo_detail"] = restore_detail

        if restore_status != "pass" and summary.get("failure_reason") is None:
            summary["failure_reason"] = "restore_slo_not_pass"

    summary["closeout_ready"] = summary.get("failure_reason") is None
    return summary


json_mode = truthy(os.environ.get("VERIFY_GATE_STATUS_JSON_MODE", "0"))
verify_report_path = pathlib.Path(os.environ.get("VERIFY_GATE_STATUS_VERIFY_REPORT", ""))
now_dt = dt.datetime.now(dt.timezone.utc)

last_verify = {
    "status": None,
    "reason": None,
    "timestamp": None,
    "checkpoint_id": None,
}
if verify_report_path.exists():
    try:
        report_obj = json.loads(verify_report_path.read_text(encoding="utf-8"))
        if isinstance(report_obj, dict):
            last_verify = {
                "status": report_obj.get("status"),
                "reason": report_obj.get("reason"),
                "timestamp": report_obj.get("timestamp"),
                "checkpoint_id": report_obj.get("checkpoint_id"),
            }
    except Exception:
        last_verify = {
            "status": None,
            "reason": "verify_report_unreadable",
            "timestamp": None,
            "checkpoint_id": None,
        }

override_raw = str(os.environ.get("VERIFY_GATE_STATUS_STRICT_OVERRIDE", "")).strip()
override_value = None
if override_raw == "1":
    override_value = "enable"
elif override_raw == "0":
    override_value = "disable"

predicted_blocker_reason = str(os.environ.get("VERIFY_GATE_STATUS_PREDICTED_BLOCKER_REASON", "")).strip() or None
verify_args = [x for x in str(os.environ.get("VERIFY_GATE_STATUS_VERIFY_ARGS", "")).split() if x]
verify_max_age_sec = max(0, safe_int(os.environ.get("VERIFY_GATE_STATUS_VERIFY_MAX_AGE_SEC", "1800"), 1800))
run_verify_command = str(os.environ.get("VERIFY_GATE_STATUS_RUN_VERIFY_COMMAND", "")).strip() or None

verify_status_upper = str(last_verify.get("status") or "").strip().upper()
verify_ts_raw = str(last_verify.get("timestamp") or "").strip()
verify_ts_present = bool(verify_ts_raw)
verify_ts_parsed = parse_iso(verify_ts_raw) if verify_ts_present else None
verify_ts_valid = verify_ts_parsed is not None if verify_ts_present else None
verify_age_sec = None
if verify_ts_parsed is not None:
    verify_age_sec = max(0, int((now_dt - verify_ts_parsed).total_seconds()))

verify_fresh = None
if verify_age_sec is not None:
    if verify_max_age_sec > 0:
        verify_fresh = verify_age_sec <= verify_max_age_sec
    else:
        verify_fresh = True

status_evidence_failure_reason = None
if not verify_report_path.exists():
    status_evidence_failure_reason = "verify_report_missing"
elif str(last_verify.get("reason") or "").strip() == "verify_report_unreadable":
    status_evidence_failure_reason = "verify_report_unreadable"
elif not verify_ts_present:
    status_evidence_failure_reason = "verify_report_timestamp_missing"
elif verify_ts_parsed is None:
    status_evidence_failure_reason = "verify_report_timestamp_invalid"
elif verify_max_age_sec > 0 and (verify_age_sec is None or verify_age_sec > verify_max_age_sec):
    status_evidence_failure_reason = "verify_report_stale"

ready_claim_supported = None
if verify_status_upper == "READY":
    ready_claim_supported = status_evidence_failure_reason is None

status_evidence_gate = {
    "verify_max_age_sec": verify_max_age_sec,
    "verify_max_age_enforced": verify_max_age_sec > 0,
    "failure_reason": status_evidence_failure_reason,
    "report_exists": verify_report_path.exists(),
    "report_unreadable": str(last_verify.get("reason") or "").strip() == "verify_report_unreadable",
    "timestamp_present": verify_ts_present,
    "timestamp_valid": verify_ts_valid,
    "age_sec": verify_age_sec,
    "fresh": verify_fresh,
    "ready_claim_supported": ready_claim_supported,
    "run_verify_command": run_verify_command,
}

internal_bypass_stage_b = summarize_internal_bypass_stage_b(now_dt)
routing_preflight = summarize_routing_preflight(now_dt)
layered_health_gate = summarize_layered_health_gate(now_dt)
failover_stress_runtime_evidence_gate = summarize_failover_stress_runtime_evidence_gate(now_dt)
launch_readiness_severity_gate = summarize_launch_readiness_severity_gate(now_dt)
launch_readiness_worker_health_canary_gate = summarize_launch_readiness_worker_health_canary_gate(now_dt)
launch_readiness_probe_execution_gate = summarize_launch_readiness_probe_execution_gate(now_dt)

effective_predicted_blocker_reason = predicted_blocker_reason
if effective_predicted_blocker_reason is None and layered_health_gate.get("failure_reason"):
    effective_predicted_blocker_reason = f"layered_health_gate:{layered_health_gate.get('failure_reason')}"
if effective_predicted_blocker_reason is None and failover_stress_runtime_evidence_gate.get("active_blocker"):
    failover_runtime_block_reason = (
        str(failover_stress_runtime_evidence_gate.get("blocker_reason") or "").strip()
        or "failover_stress_runtime_evidence_gate_active"
    )
    effective_predicted_blocker_reason = (
        f"failover_stress_runtime_evidence_gate:{failover_runtime_block_reason}"
    )
if effective_predicted_blocker_reason is None and launch_readiness_worker_health_canary_gate.get("active_blocker"):
    canary_block_reason = (
        str(launch_readiness_worker_health_canary_gate.get("blocker_reason") or "").strip()
        or "worker_health_canary_gate_active"
    )
    effective_predicted_blocker_reason = f"execution_supervisor_worker_health_canary_gate:{canary_block_reason}"
if effective_predicted_blocker_reason is None and launch_readiness_severity_gate.get("active_blocker"):
    severity_block_reason = str(launch_readiness_severity_gate.get("blocker_reason") or "").strip() or "launch_readiness_severity_gate_active"
    effective_predicted_blocker_reason = f"execution_supervisor_launch_readiness_severity_gate:{severity_block_reason}"
if effective_predicted_blocker_reason is None and launch_readiness_probe_execution_gate.get("active_blocker"):
    probe_block_reason = str(launch_readiness_probe_execution_gate.get("blocker_reason") or "").strip() or "probe_execution_overdue"
    effective_predicted_blocker_reason = f"execution_supervisor_probe_execution_gate:{probe_block_reason}"

predicted_ready_to_run = effective_predicted_blocker_reason is None

payload = {
    "schema_version": "continuity.verify_gate_status.v1",
    "generated_at": now_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "task": os.environ.get("VERIFY_GATE_STATUS_TASK", "verify_gate_status"),
    "verify_script": {
        "path": os.environ.get("VERIFY_GATE_STATUS_VERIFY_SCRIPT", ""),
        "exists": truthy(os.environ.get("VERIFY_GATE_STATUS_VERIFY_SCRIPT_EXISTS", "0")),
        "executable": truthy(os.environ.get("VERIFY_GATE_STATUS_VERIFY_SCRIPT_EXECUTABLE", "0")),
    },
    "verify_report": {
        "path": os.environ.get("VERIFY_GATE_STATUS_VERIFY_REPORT", ""),
        "exists": truthy(os.environ.get("VERIFY_GATE_STATUS_VERIFY_REPORT_EXISTS", "0")),
        "last": last_verify,
    },
    "status_evidence_gate": status_evidence_gate,
    "internal_bypass_stage_b": internal_bypass_stage_b,
    "routing_preflight": routing_preflight,
    "layered_health_gate": layered_health_gate,
    "failover_stress_runtime_evidence_gate": failover_stress_runtime_evidence_gate,
    "launch_readiness_severity_gate": launch_readiness_severity_gate,
    "launch_readiness_worker_health_canary_gate": launch_readiness_worker_health_canary_gate,
    "launch_readiness_probe_execution_gate": launch_readiness_probe_execution_gate,
    "strict_autonomy_regressions": {
        "enabled": truthy(os.environ.get("VERIFY_GATE_STATUS_STRICT_ENABLED", "0")),
        "source": os.environ.get("VERIFY_GATE_STATUS_STRICT_SOURCE", "disabled"),
        "required": truthy(os.environ.get("VERIFY_GATE_STATUS_STRICT_REQUIRED", "0")),
        "override": override_value,
        "override_denied_if_run": truthy(os.environ.get("VERIFY_GATE_STATUS_STRICT_OVERRIDE_DENIED", "0")),
        "inputs": {
            "verify_gate_policy_env": (os.environ.get("VERIFY_GATE_STATUS_POLICY_RAW", "") or None),
            "legacy_env": (os.environ.get("VERIFY_GATE_STATUS_LEGACY_RAW", "") or None),
            "required_env": (os.environ.get("VERIFY_GATE_STATUS_REQUIRED_RAW", "") or None),
        },
        "verify_args": verify_args,
    },
    "predicted_gate": {
        "ready_to_run": predicted_ready_to_run,
        "predicted_blocker_reason": effective_predicted_blocker_reason,
    },
}

if json_mode:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    strict = payload["strict_autonomy_regressions"]
    script = payload["verify_script"]
    report = payload["verify_report"]
    predicted = payload["predicted_gate"]
    status_evidence = payload.get("status_evidence_gate") or {}
    stage_b = payload.get("internal_bypass_stage_b") or {}
    routing = payload.get("routing_preflight") or {}
    layered_health = payload.get("layered_health_gate") or {}
    failover_stress_runtime = payload.get("failover_stress_runtime_evidence_gate") or {}
    launch_readiness_severity = payload.get("launch_readiness_severity_gate") or {}
    launch_readiness_worker_health_canary = payload.get("launch_readiness_worker_health_canary_gate") or {}
    launch_readiness_probe_execution = payload.get("launch_readiness_probe_execution_gate") or {}
    last = report["last"]

    print("VERIFY GATE STATUS")
    print(f"task={payload['task']}")
    print(
        "verify_script="
        f"{script['path']}; exists={int(script['exists'])}; executable={int(script['executable'])}"
    )
    print(
        "strict_autonomy="
        f"{int(strict['enabled'])}; source={strict['source']}; required={int(strict['required'])}; "
        f"override={strict['override'] or 'none'}; override_denied_if_run={int(strict['override_denied_if_run'])}"
    )
    print(
        "strict_inputs="
        f"verify_gate_policy_env={strict['inputs']['verify_gate_policy_env'] or 'unset'}; "
        f"legacy_env={strict['inputs']['legacy_env'] or 'unset'}; "
        f"required_env={strict['inputs']['required_env'] or 'unset'}"
    )
    print("verify_args=" + (" ".join(strict["verify_args"]) if strict["verify_args"] else "(none)"))
    print(
        "predicted_gate="
        f"ready_to_run={int(predicted['ready_to_run'])}; "
        f"predicted_blocker_reason={predicted['predicted_blocker_reason'] or 'none'}"
    )
    print(
        "status_evidence_gate="
        f"failure_reason={status_evidence.get('failure_reason') or 'none'}; "
        f"fresh={status_evidence.get('fresh') if status_evidence.get('fresh') is not None else 'n/a'}; "
        f"age_sec={status_evidence.get('age_sec') if status_evidence.get('age_sec') is not None else 'n/a'}; "
        f"max_age_sec={status_evidence.get('verify_max_age_sec')}; "
        f"ready_claim_supported={status_evidence.get('ready_claim_supported') if status_evidence.get('ready_claim_supported') is not None else 'n/a'}"
    )
    print(
        "internal_bypass_stage_b="
        f"failure_reason={stage_b.get('closeout_failure_reason') or 'none'}; "
        f"closeout_ready={stage_b.get('closeout_ready')}; "
        f"unknown_total={stage_b.get('unknown_callsite_total')}; "
        f"break_glass_allow={stage_b.get('break_glass_allow_count')}; "
        f"break_glass_denied={stage_b.get('break_glass_denied_count')}; "
        f"window_sec={stage_b.get('window_sec')}; "
        f"audit_present={stage_b.get('audit_present')}"
    )
    routing_latest = routing.get("latest") if isinstance(routing.get("latest"), dict) else {}
    routing_effective = routing.get("effective") if isinstance(routing.get("effective"), dict) else {}
    print(
        "routing_preflight="
        f"failure_reason={routing.get('failure_reason') or 'none'}; "
        f"decision={routing_latest.get('decision') or 'unknown'}; "
        f"fresh={routing_latest.get('fresh') if routing_latest.get('fresh') is not None else 'n/a'}; "
        f"route_class={routing_latest.get('route_class') or 'none'}; "
        f"selected_model={routing_latest.get('selected_model') or 'none'}; "
        f"block_reason={routing_latest.get('block_reason') or 'none'}"
    )
    print(
        "layered_health_gate="
        f"failure_reason={layered_health.get('failure_reason') or 'none'}; "
        f"closeout_ready={layered_health.get('closeout_ready')}; "
        f"health_status={layered_health.get('health_status') or 'unknown'}; "
        f"health_layer={layered_health.get('health_layer') or 'unknown'}; "
        f"restore_slo_status={layered_health.get('restore_slo_status') or 'unknown'}"
    )
    print(
        "failover_stress_runtime_evidence_gate="
        f"failure_reason={failover_stress_runtime.get('failure_reason') or 'none'}; "
        f"active_blocker={failover_stress_runtime.get('active_blocker')}; "
        f"overall_verdict={failover_stress_runtime.get('overall_verdict') or 'unknown'}; "
        f"publish_chain_verdict={failover_stress_runtime.get('publish_chain_verdict') or 'unknown'}; "
        f"repeatability_status={failover_stress_runtime.get('repeatability_status') or 'unknown'}; "
        f"fresh={failover_stress_runtime.get('fresh') if failover_stress_runtime.get('fresh') is not None else 'n/a'}"
    )
    print(
        "launch_readiness_severity_gate="
        f"failure_reason={launch_readiness_severity.get('failure_reason') or 'none'}; "
        f"active_blocker={launch_readiness_severity.get('active_blocker')}; "
        f"severity_state={launch_readiness_severity.get('severity_state') or 'none'}; "
        f"severity_reason={launch_readiness_severity.get('severity_reason') or 'none'}; "
        f"fresh={launch_readiness_severity.get('fresh') if launch_readiness_severity.get('fresh') is not None else 'n/a'}"
    )
    print(
        "launch_readiness_worker_health_canary_gate="
        f"failure_reason={launch_readiness_worker_health_canary.get('failure_reason') or 'none'}; "
        f"active_blocker={launch_readiness_worker_health_canary.get('active_blocker')}; "
        f"gate_required={launch_readiness_worker_health_canary.get('gate_required')}; "
        f"artifact_required={launch_readiness_worker_health_canary.get('artifact_required')}; "
        f"canary_fresh={launch_readiness_worker_health_canary.get('worker_health_canary_fresh') if launch_readiness_worker_health_canary.get('worker_health_canary_fresh') is not None else 'n/a'}"
    )
    print(
        "launch_readiness_probe_execution_gate="
        f"failure_reason={launch_readiness_probe_execution.get('failure_reason') or 'none'}; "
        f"active_blocker={launch_readiness_probe_execution.get('active_blocker')}; "
        f"due_now={launch_readiness_probe_execution.get('due_now_worker_count')}; "
        f"overdue={launch_readiness_probe_execution.get('overdue_worker_count')}; "
        f"fresh={launch_readiness_probe_execution.get('fresh') if launch_readiness_probe_execution.get('fresh') is not None else 'n/a'}"
    )
    if stage_b.get("unknown_callsites_top"):
        top = stage_b.get("unknown_callsites_top") or []
        first = top[0] if isinstance(top, list) and top else {}
        print(
            "internal_bypass_stage_b_top_unknown="
            f"{first.get('callsite') or 'n/a'}:{first.get('count') or 0}"
        )
    if status_evidence.get("failure_reason") and status_evidence.get("run_verify_command"):
        print("status_evidence_recovery=" + str(status_evidence.get("run_verify_command")))
    if stage_b.get("closeout_failure_reason") and stage_b.get("inspect_audit_command"):
        print("internal_bypass_stage_b_recovery=" + str(stage_b.get("inspect_audit_command")))
    if routing.get("failure_reason") and routing_effective.get("inspect_decisions_command"):
        print("routing_preflight_inspect=" + str(routing_effective.get("inspect_decisions_command")))
    if routing.get("failure_reason") in {"routing_blocked", "routing_decision_stale"} and routing_effective.get("recheck_policy_command"):
        print("routing_preflight_recheck=" + str(routing_effective.get("recheck_policy_command")))
    if routing.get("failure_reason") == "routing_blocked" and routing_effective.get("first_actionable_command"):
        print("routing_preflight_action=" + str(routing_effective.get("first_actionable_command")))
    if layered_health.get("failure_reason"):
        if layered_health.get("inspect_layered_health_command"):
            print("layered_health_inspect=" + str(layered_health.get("inspect_layered_health_command")))
        if layered_health.get("run_layered_health_command"):
            print("layered_health_recompute=" + str(layered_health.get("run_layered_health_command")))
        if layered_health.get("run_slo_snapshot_command"):
            print("layered_health_recompute_slo=" + str(layered_health.get("run_slo_snapshot_command")))
    if failover_stress_runtime.get("failure_reason") and failover_stress_runtime.get("inspect_failover_stress_runtime_evidence_command"):
        print(
            "failover_stress_runtime_evidence_inspect="
            + str(failover_stress_runtime.get("inspect_failover_stress_runtime_evidence_command"))
        )
    if failover_stress_runtime.get("active_blocker") and failover_stress_runtime.get("refresh_failover_stress_runtime_evidence_command"):
        print(
            "failover_stress_runtime_evidence_refresh="
            + str(failover_stress_runtime.get("refresh_failover_stress_runtime_evidence_command"))
        )
    if launch_readiness_severity.get("failure_reason") and launch_readiness_severity.get("inspect_dispatch_qualification_command"):
        print(
            "launch_readiness_severity_inspect="
            + str(launch_readiness_severity.get("inspect_dispatch_qualification_command"))
        )
    if launch_readiness_severity.get("active_blocker") and launch_readiness_severity.get("refresh_dispatch_qualification_command"):
        print(
            "launch_readiness_severity_refresh="
            + str(launch_readiness_severity.get("refresh_dispatch_qualification_command"))
        )
    if launch_readiness_worker_health_canary.get("failure_reason") and launch_readiness_worker_health_canary.get("inspect_dispatch_qualification_command"):
        print(
            "launch_readiness_worker_health_canary_inspect="
            + str(launch_readiness_worker_health_canary.get("inspect_dispatch_qualification_command"))
        )
    if launch_readiness_worker_health_canary.get("failure_reason") and launch_readiness_worker_health_canary.get("inspect_worker_health_canary_command"):
        print(
            "launch_readiness_worker_health_canary_artifact_inspect="
            + str(launch_readiness_worker_health_canary.get("inspect_worker_health_canary_command"))
        )
    if launch_readiness_worker_health_canary.get("failure_reason") and launch_readiness_worker_health_canary.get("refresh_worker_health_canary_command"):
        print(
            "launch_readiness_worker_health_canary_refresh="
            + str(launch_readiness_worker_health_canary.get("refresh_worker_health_canary_command"))
        )
    if launch_readiness_probe_execution.get("failure_reason") and launch_readiness_probe_execution.get("inspect_dispatch_qualification_command"):
        print(
            "launch_readiness_probe_execution_inspect="
            + str(launch_readiness_probe_execution.get("inspect_dispatch_qualification_command"))
        )
    if launch_readiness_probe_execution.get("failure_reason") and launch_readiness_probe_execution.get("inspect_probe_execution_plan_command"):
        print(
            "launch_readiness_probe_execution_plan_inspect="
            + str(launch_readiness_probe_execution.get("inspect_probe_execution_plan_command"))
        )
    if (
        launch_readiness_probe_execution.get("due_now_active")
        and launch_readiness_probe_execution.get("refresh_dispatch_qualification_command")
    ):
        print(
            "launch_readiness_probe_execution_refresh="
            + str(launch_readiness_probe_execution.get("refresh_dispatch_qualification_command"))
        )
    if launch_readiness_probe_execution.get("first_actionable_command"):
        print(
            "launch_readiness_probe_execution_action="
            + str(launch_readiness_probe_execution.get("first_actionable_command"))
        )
    print(
        "last_verify="
        f"status={last.get('status') or 'unknown'}; "
        f"reason={last.get('reason') or 'unknown'}; "
        f"checkpoint_id={last.get('checkpoint_id') or 'unknown'}; "
        f"timestamp={last.get('timestamp') or 'unknown'}"
    )
PY
