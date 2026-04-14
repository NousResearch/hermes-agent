#!/usr/bin/env bash
# Layered Health Snapshot Runner (A6 Ops Lane)
# Generates a snapshot of multi-lane layered health contract compliance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LATEST_JSON="${WORKSPACE_DIR}/state/continuity/latest/layered_health_snapshot.json"
SLO_JSON="${WORKSPACE_DIR}/state/continuity/latest/slo_snapshot.json"
SLO_EVALUATOR_SCRIPT="${WORKSPACE_DIR}/ops/openclaw/continuity/slo_evaluator_snapshot.sh"
NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

mkdir -p "$(dirname "$LATEST_JSON")"

ALIVE="true"
if ! pgrep -f "openclaw-gateway" > /dev/null 2>&1 && ! pgrep -f "openclaw" > /dev/null 2>&1; then
  ALIVE="false"
fi

READY="true"
if [ "$ALIVE" != "true" ] || [ ! -d "${WORKSPACE_DIR}/state" ]; then
  READY="false"
fi

GATE_JSON=""
GATE_RC=0
if GATE_JSON=$(bash "${WORKSPACE_DIR}/ops/openclaw/continuity.sh" verify-gate-status --json 2>/dev/null); then
  GATE_RC=0
else
  GATE_RC=$?
  GATE_JSON=""
fi

SLO_EVAL_RC=0
if bash "$SLO_EVALUATOR_SCRIPT" > /dev/null 2>&1; then
  SLO_EVAL_RC=0
else
  SLO_EVAL_RC=$?
fi

CONTINUITY_NOW_JSON=""
CONTINUITY_NOW_RC=0
set +e
CONTINUITY_NOW_JSON="$(OPENCLAW_LAYERED_HEALTH_SNAPSHOT_ACTIVE=1 bash "${WORKSPACE_DIR}/ops/openclaw/continuity/continuity_now.sh" --strict --json 2>/dev/null)"
CONTINUITY_NOW_RC=$?
set -e

LAYERED_HEALTH_GATE_JSON="$(mktemp "${WORKSPACE_DIR}/state/continuity/latest/layered_health_gate_tmp.XXXXXX.json")"
LAYERED_HEALTH_CONTINUITY_JSON="$(mktemp "${WORKSPACE_DIR}/state/continuity/latest/layered_health_continuity_tmp.XXXXXX.json")"
trap 'rm -f "$LAYERED_HEALTH_GATE_JSON" "$LAYERED_HEALTH_CONTINUITY_JSON"' EXIT
printf '%s' "$GATE_JSON" > "$LAYERED_HEALTH_GATE_JSON"
printf '%s' "$CONTINUITY_NOW_JSON" > "$LAYERED_HEALTH_CONTINUITY_JSON"

LAYERED_HEALTH_ALIVE="$ALIVE" \
LAYERED_HEALTH_READY="$READY" \
LAYERED_HEALTH_NOW="$NOW" \
LAYERED_HEALTH_ROOT="$WORKSPACE_DIR" \
LAYERED_HEALTH_SLO_JSON="$SLO_JSON" \
LAYERED_HEALTH_GATE_JSON="$LAYERED_HEALTH_GATE_JSON" \
LAYERED_HEALTH_CONTINUITY_NOW_JSON="$LAYERED_HEALTH_CONTINUITY_JSON" \
LAYERED_HEALTH_SLO_EVAL_RC="$SLO_EVAL_RC" \
LAYERED_HEALTH_GATE_RC="$GATE_RC" \
LAYERED_HEALTH_CONTINUITY_NOW_RC="$CONTINUITY_NOW_RC" \
LAYERED_HEALTH_OUT="$LATEST_JSON" \
  python3 - <<'PY'
import json
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(os.environ["LAYERED_HEALTH_ROOT"]).resolve()
OUT = pathlib.Path(os.environ["LAYERED_HEALTH_OUT"]).resolve()
NOW = str(os.environ.get("LAYERED_HEALTH_NOW") or "")

LAYER_RANK = {
    "alive": 1,
    "ready": 2,
    "safe-to-act": 3,
    "truthful": 4,
}

TARGET_LANES = [
    "A1_CONTROL_PLANE",
    "A2_RUNTIME_CONTINUITY",
    "A3_MODEL_ROUTING",
    "A6_OPS_OBSERVABILITY",
    "C1_OPERATOR_SURFACE",
    "C2_RELEASE_SUBSTRATE",
]


def _truthy(raw: str) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _dedupe_nonempty(items: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _load_json(path: pathlib.Path) -> Any:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _lane_state(alive: bool, ready: bool, safe: bool, truthful: bool) -> Dict[str, str]:
    if truthful:
        return {"health_layer": "truthful", "status": "pass"}
    if safe:
        return {"health_layer": "safe-to-act", "status": "degraded"}
    if ready:
        return {"health_layer": "ready", "status": "failing"}
    return {"health_layer": "alive", "status": "failing"}


def _issue(lane: str, severity: str, layer: str, message: str, hint: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "lane": lane,
        "severity": severity,
        "message": message,
        "layer_failed": layer,
    }
    if hint:
        payload["remediation_hint"] = hint
    return payload


alive = _truthy(os.environ.get("LAYERED_HEALTH_ALIVE", "false"))
ready = _truthy(os.environ.get("LAYERED_HEALTH_READY", "false"))
gate_rc = _safe_int(os.environ.get("LAYERED_HEALTH_GATE_RC"), 1)
slo_eval_rc = _safe_int(os.environ.get("LAYERED_HEALTH_SLO_EVAL_RC"), 1)
continuity_now_rc = _safe_int(os.environ.get("LAYERED_HEALTH_CONTINUITY_NOW_RC"), 1)
verify_then_resume_active = str(os.environ.get("OPENCLAW_VERIFY_THEN_RESUME_ACTIVE", "0")).strip().lower() in {"1", "true", "yes", "on"}


gate_payload = _load_json(pathlib.Path(os.environ.get("LAYERED_HEALTH_GATE_JSON", "")))
if not isinstance(gate_payload, dict):
    gate_payload = {}

def _extract_gate_views(payload: Dict[str, Any]) -> Dict[str, Any]:
    predicted = payload.get("predicted_gate") if isinstance(payload.get("predicted_gate"), dict) else {}
    verify_last = ((payload.get("verify_report") or {}).get("last") or {}) if isinstance((payload.get("verify_report") or {}).get("last"), dict) else {}
    status = payload.get("status_evidence_gate") if isinstance(payload.get("status_evidence_gate"), dict) else {}
    bypass = payload.get("internal_bypass_stage_b") if isinstance(payload.get("internal_bypass_stage_b"), dict) else {}
    routing = payload.get("routing_preflight") if isinstance(payload.get("routing_preflight"), dict) else {}
    routing_effective_local = routing.get("effective") if isinstance(routing.get("effective"), dict) else {}
    routing_latest_local = routing.get("latest") if isinstance(routing.get("latest"), dict) else {}
    return {
        "gate_predicted": predicted,
        "gate_verify_last": verify_last,
        "status_evidence": status,
        "internal_bypass": bypass,
        "routing_preflight": routing,
        "routing_effective": routing_effective_local,
        "routing_latest": routing_latest_local,
    }


gate_views = _extract_gate_views(gate_payload)
gate_predicted = gate_views["gate_predicted"]
gate_verify_last = gate_views["gate_verify_last"]
status_evidence = gate_views["status_evidence"]
internal_bypass = gate_views["internal_bypass"]
routing_preflight = gate_views["routing_preflight"]
routing_effective = gate_views["routing_effective"]
routing_latest = gate_views["routing_latest"]

routing_preflight_refresh_enabled = _truthy(
    os.environ.get("OPENCLAW_LAYERED_HEALTH_ROUTING_PREFLIGHT_AUTO_REFRESH", "1")
)
routing_preflight_refresh_timeout_sec = max(
    30,
    _safe_int(os.environ.get("OPENCLAW_LAYERED_HEALTH_ROUTING_PREFLIGHT_REFRESH_TIMEOUT_SEC", "600"), 600),
)
default_routing_preflight_refresh_command = (
    f"bash {ROOT / 'ops' / 'openclaw' / 'continuity' / 'routing_preflight_refresh.sh'} --json"
)
routing_preflight_refresh_command = (
    str(
        os.environ.get(
            "OPENCLAW_LAYERED_HEALTH_ROUTING_PREFLIGHT_REFRESH_COMMAND",
            default_routing_preflight_refresh_command,
        )
        or ""
    ).strip()
    or str(routing_effective.get("recheck_policy_command") or "").strip()
)

routing_preflight_refresh = {
    "enabled": routing_preflight_refresh_enabled,
    "attempted": False,
    "status": "disabled" if not routing_preflight_refresh_enabled else "not_needed",
    "trigger_reason": None,
    "command": routing_preflight_refresh_command or None,
    "timeout_sec": routing_preflight_refresh_timeout_sec,
    "command_rc": None,
    "timed_out": False,
    "skipped_reason": None,
    "stderr_tail": None,
}

routing_failure_reason_initial = str(routing_preflight.get("failure_reason") or "").strip()
routing_refreshable_reasons = {
    "routing_decisions_missing",
    "routing_decisions_not_regular_file",
    "routing_decisions_unreadable",
    "routing_decisions_no_valid_rows",
    "routing_decision_stale",
}

if not routing_preflight_refresh_enabled:
    routing_preflight_refresh["skipped_reason"] = "disabled"
elif not routing_failure_reason_initial:
    routing_preflight_refresh["status"] = "not_needed"
elif routing_failure_reason_initial not in routing_refreshable_reasons:
    routing_preflight_refresh["status"] = "skipped"
    routing_preflight_refresh["trigger_reason"] = routing_failure_reason_initial
    routing_preflight_refresh["skipped_reason"] = "failure_reason_not_refreshable"
elif not routing_preflight_refresh_command:
    routing_preflight_refresh["status"] = "skipped"
    routing_preflight_refresh["trigger_reason"] = routing_failure_reason_initial
    routing_preflight_refresh["skipped_reason"] = "refresh_command_missing"
else:
    routing_preflight_refresh["attempted"] = True
    routing_preflight_refresh["trigger_reason"] = routing_failure_reason_initial
    try:
        refresh_cp = subprocess.run(
            ["bash", "-lc", routing_preflight_refresh_command],
            text=True,
            capture_output=True,
            check=False,
            timeout=routing_preflight_refresh_timeout_sec,
            env=os.environ.copy(),
        )
        routing_preflight_refresh["command_rc"] = int(refresh_cp.returncode)
        stderr_tail = str(refresh_cp.stderr or "").strip()[-400:]
        if stderr_tail:
            routing_preflight_refresh["stderr_tail"] = stderr_tail
        if refresh_cp.returncode != 0:
            routing_preflight_refresh["status"] = "command_failed"
        else:
            gate_reload_cp = subprocess.run(
                ["bash", str(ROOT / "ops" / "openclaw" / "continuity.sh"), "verify-gate-status", "--json"],
                text=True,
                capture_output=True,
                check=False,
                timeout=120,
                env=os.environ.copy(),
            )
            if gate_reload_cp.returncode == 0:
                try:
                    refreshed = json.loads(gate_reload_cp.stdout or "{}")
                    if isinstance(refreshed, dict):
                        gate_payload = refreshed
                except Exception:
                    pass
            gate_views = _extract_gate_views(gate_payload)
            gate_predicted = gate_views["gate_predicted"]
            gate_verify_last = gate_views["gate_verify_last"]
            status_evidence = gate_views["status_evidence"]
            internal_bypass = gate_views["internal_bypass"]
            routing_preflight = gate_views["routing_preflight"]
            routing_effective = gate_views["routing_effective"]
            routing_latest = gate_views["routing_latest"]
            refreshed_failure_reason = str(routing_preflight.get("failure_reason") or "").strip()
            routing_preflight_refresh["status"] = "refreshed" if not refreshed_failure_reason else "refreshed_still_blocked"
    except subprocess.TimeoutExpired as exc:
        routing_preflight_refresh["timed_out"] = True
        routing_preflight_refresh["command_rc"] = 124
        stderr_tail = str(getattr(exc, "stderr", "") or "").strip()[-400:]
        if stderr_tail:
            routing_preflight_refresh["stderr_tail"] = stderr_tail
        routing_preflight_refresh["status"] = "timeout"

status_evidence_reason = str(status_evidence.get("failure_reason") or "").strip()
ready_claim_supported = status_evidence.get("ready_claim_supported")
last_verify_status = str(gate_verify_last.get("status") or "unknown").strip().upper()
gate_ready = gate_predicted.get("ready_to_run") is True
gate_predicted_blocker_reason = str(gate_predicted.get("predicted_blocker_reason") or "").strip()
gate_layered_health_self_reference_only = bool(
    gate_predicted_blocker_reason.startswith("layered_health_gate:")
    and last_verify_status == "READY"
    and ready_claim_supported is True
    and not status_evidence_reason
)
preflight_self_recoverable = bool(
    gate_ready
    and last_verify_status == "BLOCKER"
    and ready_claim_supported is not True
    and not status_evidence_reason
)

safe_gate_ok = False
safe_gate_issue = None
if not gate_payload or gate_rc != 0:
    safe_gate_issue = "verify_gate_status unavailable"
elif not gate_ready and not gate_layered_health_self_reference_only and not verify_then_resume_active:
    safe_gate_issue = (
        f"Verify gate lacks fresh READY evidence (last_status={last_verify_status}, "
        f"evidence_reason={status_evidence_reason or 'none'})"
    )
elif (
    (last_verify_status != "READY" or ready_claim_supported is not True)
    and not gate_layered_health_self_reference_only
    and not preflight_self_recoverable
    and not verify_then_resume_active
):
    safe_gate_issue = (
        f"Verify gate lacks fresh READY evidence (last_status={last_verify_status}, "
        f"evidence_reason={status_evidence_reason or 'none'})"
    )
else:
    safe_gate_ok = True

slo_payload = _load_json(pathlib.Path(os.environ.get("LAYERED_HEALTH_SLO_JSON", "")))
if not isinstance(slo_payload, dict):
    slo_payload = {}

slo_status = str(slo_payload.get("status") or "unknown").strip().lower()
slo_evaluations = slo_payload.get("evaluations") if isinstance(slo_payload.get("evaluations"), list) else []
slo_status_by_id: Dict[str, str] = {}
failing_slos: List[str] = []
for row in slo_evaluations:
    if not isinstance(row, dict):
        continue
    sid = str(row.get("id") or "").strip()
    if not sid:
        continue
    state = str(row.get("status") or "unknown").strip().lower()
    slo_status_by_id[sid] = state
    if state != "pass":
        failing_slos.append(sid)

slo_safe_ok = False
slo_safe_issue = None
if not slo_payload:
    slo_safe_issue = f"SLO evidence unavailable (evaluator_rc={slo_eval_rc})"
elif slo_eval_rc != 0 or slo_status != "pass" or failing_slos:
    failing_str = ",".join(failing_slos) if failing_slos else "none"
    slo_safe_issue = (
        f"SLO error budgets not intact (slo_status={slo_status or 'unknown'}, "
        f"failing_targets={failing_str}, evaluator_rc={slo_eval_rc})"
    )
else:
    slo_safe_ok = True

restore_slo_status = str(slo_status_by_id.get("SLO-4_RESTORE_DRILL_FRESHNESS") or "").strip().lower() or "missing"
restore_slo_pass = restore_slo_status == "pass"
restore_rollout_blocker_reason = None if restore_slo_pass else "a6_restore_evidence_not_pass"
routing_failure_reason = str(routing_preflight.get("failure_reason") or "").strip()
routing_blocked_fresh = routing_effective.get("blocked_fresh") is True
routing_fresh = routing_latest.get("fresh")
routing_block_reason = str(routing_effective.get("block_reason") or "").strip()
routing_guardrail_block_reasons = {
    "telegram_direct_heavy_offload_required",
    "telegram_direct_worker_offload_required",
    "telegram_direct_worker_target_evidence_missing",
    "telegram_direct_worker_target_evidence_invalid",
}
routing_guardrail_block_active = bool(
    routing_failure_reason == "routing_blocked"
    and routing_blocked_fresh
    and routing_block_reason in routing_guardrail_block_reasons
)
routing_ready = bool(
    routing_guardrail_block_active
    or (not routing_failure_reason and not routing_blocked_fresh and routing_fresh is not False)
)

internal_bypass_ready = internal_bypass.get("closeout_ready") is True
internal_bypass_failure = str(internal_bypass.get("closeout_failure_reason") or "").strip()

continuity_payload = _load_json(pathlib.Path(os.environ.get("LAYERED_HEALTH_CONTINUITY_NOW_JSON", "")))
if not isinstance(continuity_payload, dict):
    continuity_payload = {}

continuity_not_ready_reasons = [
    str(x).strip() for x in (continuity_payload.get("not_ready_reasons") or []) if str(x).strip()
]
continuity_blocker_reasons = [
    str(x).strip() for x in (continuity_payload.get("blocker_reasons") or []) if str(x).strip()
]
continuity_reconcile_only_reasons = [
    str(x).strip() for x in (continuity_payload.get("reconcile_only_reasons") or []) if str(x).strip()
]
continuity_warning_reasons = [
    str(x).strip() for x in (continuity_payload.get("warning_reasons") or []) if str(x).strip()
]
continuity_coherence = continuity_payload.get("coherence") if isinstance(continuity_payload.get("coherence"), dict) else {}
continuity_layered_health_derivative_suppressed = bool(
    continuity_coherence.get("layered_health_derivative_suppressed") is True
)
continuity_layered_health_derivative_suppression_reason = str(
    continuity_coherence.get("layered_health_derivative_suppression_reason") or ""
).strip()
if (
    continuity_layered_health_derivative_suppressed
    and continuity_layered_health_derivative_suppression_reason == "layered_health_snapshot_self_reference"
):
    continuity_warning_reasons = [
        reason
        for reason in continuity_warning_reasons
        if reason
        not in {
            "verify_gate_preflight_blocker_predicted",
            "layered_health_gate_unready",
            "layered_health_gate_unready_suppressed_derivative",
            "layered_health_gate_unready_suppressed_by_verify_blocker",
        }
    ]

if not continuity_blocker_reasons and continuity_not_ready_reasons:
    drift_reason_fallback = {
        "pointer_drift",
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }
    continuity_blocker_reasons = [
        reason for reason in continuity_not_ready_reasons if reason not in drift_reason_fallback
    ]
    continuity_reconcile_only_reasons = [
        reason for reason in continuity_not_ready_reasons if reason in drift_reason_fallback
    ]

continuity_derivative_suppressed_blockers = (
    [reason for reason in continuity_blocker_reasons if reason == "layered_health_gate_unready"]
    if continuity_layered_health_derivative_suppressed
    else []
)
continuity_effective_blocker_reasons = [
    reason for reason in continuity_blocker_reasons if reason not in set(continuity_derivative_suppressed_blockers)
]

rollout_blocker_reason_set = {
    "execution_supervisor_launch_readiness_severity_gate_active",
    "execution_supervisor_worker_health_canary_gate_active",
    "execution_supervisor_probe_execution_overdue_gate_active",
    "failover_stress_runtime_evidence_gate_active",
}
continuity_rollout_blocker_seed = [
    str(x).strip() for x in (continuity_payload.get("rollout_blocker_reasons") or []) if str(x).strip()
]
continuity_rollout_blocker_reasons = _dedupe_nonempty(
    [
        reason
        for reason in continuity_rollout_blocker_seed
        if reason in continuity_effective_blocker_reasons
    ]
    + [
        reason
        for reason in continuity_effective_blocker_reasons
        if reason in rollout_blocker_reason_set
    ]
    + ([restore_rollout_blocker_reason] if restore_rollout_blocker_reason else [])
)

rollout_warning_reason_prefixes = (
    "execution_supervisor_launch_readiness_",
    "execution_supervisor_worker_health_canary_",
    "execution_supervisor_probe_execution_",
    "execution_supervisor_dispatch_qualification_",
    "execution_supervisor_dispatch_resource_preflight_",
    "execution_supervisor_dispatch_uncertainty_",
    "failover_stress_runtime_evidence_",
)
continuity_rollout_warning_seed = [
    str(x).strip() for x in (continuity_payload.get("rollout_warning_reasons") or []) if str(x).strip()
]
if continuity_rollout_warning_seed:
    continuity_rollout_warning_reasons = [
        reason for reason in continuity_rollout_warning_seed if reason not in rollout_blocker_reason_set
    ]
else:
    continuity_rollout_warning_reasons = [
        reason
        for reason in continuity_warning_reasons
        if reason not in rollout_blocker_reason_set
        and any(reason.startswith(prefix) for prefix in rollout_warning_reason_prefixes)
    ]

continuity_payload_available = bool(continuity_payload)
continuity_reconcile_only_pending = bool(
    continuity_payload_available and not continuity_effective_blocker_reasons and continuity_reconcile_only_reasons
)
continuity_semantic_blocked = bool(continuity_payload_available and continuity_effective_blocker_reasons)

if continuity_payload_available:
    truthful_runtime_ok = not continuity_semantic_blocked
else:
    truthful_runtime_ok = continuity_now_rc == 0

lane_rows: List[Dict[str, Any]] = []
aggregate_issues: List[Dict[str, Any]] = []

for lane in TARGET_LANES:
    issues: List[Dict[str, Any]] = []

    lane_safe_ok = True
    lane_truth_ok = True

    if lane == "A1_CONTROL_PLANE":
        if not safe_gate_ok:
            lane_safe_ok = False
            issues.append(_issue(lane, "warning", "safe-to-act", safe_gate_issue or "verify gate not ready"))
        if not slo_safe_ok:
            lane_safe_ok = False
            issues.append(_issue(lane, "warning", "safe-to-act", slo_safe_issue or "SLO evidence unavailable"))
    elif lane == "A2_RUNTIME_CONTINUITY":
        required = ["SLO-1_VERIFY_FRESHNESS", "SLO-2_CONTINUITY_FRESHNESS"]
        missing_or_failing = [sid for sid in required if slo_status_by_id.get(sid) != "pass"]
        if missing_or_failing:
            lane_safe_ok = False
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "safe-to-act",
                    "Continuity freshness SLOs not green (targets=" + ",".join(missing_or_failing) + ")",
                )
            )
        if status_evidence_reason:
            lane_truth_ok = False
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "truthful",
                    f"Verify status evidence gate degraded (failure_reason={status_evidence_reason})",
                )
            )
        if continuity_semantic_blocked:
            lane_truth_ok = False
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "truthful",
                    "Continuity runtime has blocker-grade not-ready reasons "
                    f"(blockers={','.join(continuity_effective_blocker_reasons)})",
                )
            )
        elif continuity_reconcile_only_pending:
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "truthful",
                    "Continuity runtime has drift-only reconcile residue "
                    f"(reasons={','.join(continuity_reconcile_only_reasons)})",
                )
            )
    elif lane == "A3_MODEL_ROUTING":
        if not routing_ready:
            lane_safe_ok = False
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "safe-to-act",
                    "Routing preflight not ready "
                    f"(failure_reason={routing_failure_reason or 'none'}, blocked_fresh={routing_blocked_fresh}, fresh={routing_fresh})",
                )
            )
        elif routing_guardrail_block_active:
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "safe-to-act",
                    "Routing preflight blocked by active thin-lane guardrail "
                    f"(reason={routing_block_reason})",
                )
            )
    elif lane == "A6_OPS_OBSERVABILITY":
        if not slo_safe_ok:
            lane_safe_ok = False
            issues.append(_issue(lane, "warning", "safe-to-act", slo_safe_issue or "SLO evidence unavailable"))
        if not restore_slo_pass:
            lane_safe_ok = False
            issues.append(
                _issue(
                    lane,
                    "critical",
                    "safe-to-act",
                    "Restore drill SLO not green (SLO-4_RESTORE_DRILL_FRESHNESS)",
                    hint="Run continuity restore-drill refresh: bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh restore-drill-refresh --json",
                )
            )
    elif lane == "C1_OPERATOR_SURFACE":
        if not gate_payload:
            lane_safe_ok = False
            issues.append(_issue(lane, "warning", "safe-to-act", "verify_gate_status surface unavailable"))
        if continuity_semantic_blocked:
            lane_truth_ok = False
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "truthful",
                    "Continuity now reports blocker-grade readiness posture "
                    f"(blockers={','.join(continuity_effective_blocker_reasons)})",
                )
            )
        elif continuity_reconcile_only_pending:
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "truthful",
                    "Continuity now reports reconcile-only drift residue "
                    f"(reasons={','.join(continuity_reconcile_only_reasons)})",
                )
            )
        elif not truthful_runtime_ok:
            lane_truth_ok = False
            issues.append(_issue(lane, "warning", "truthful", "Continuity now reports drift or staleness"))
    elif lane == "C2_RELEASE_SUBSTRATE":
        if not safe_gate_ok:
            lane_safe_ok = False
            issues.append(_issue(lane, "warning", "safe-to-act", safe_gate_issue or "verify gate not ready"))
        if not internal_bypass_ready:
            lane_safe_ok = False
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "safe-to-act",
                    "Internal bypass Stage-B closeout not ready "
                    f"(reason={internal_bypass_failure or 'unknown'})",
                )
            )
        if routing_blocked_fresh and not routing_guardrail_block_active:
            lane_safe_ok = False
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "safe-to-act",
                    f"Routing preflight blocked rollout coupling (reason={routing_failure_reason or 'routing_blocked'})",
                )
            )
        elif routing_guardrail_block_active:
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "safe-to-act",
                    "Routing rollout coupling blocked by intentional thin-lane guardrail "
                    f"(reason={routing_block_reason})",
                )
            )
        if continuity_rollout_blocker_reasons:
            lane_safe_ok = False
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "safe-to-act",
                    "Continuity runtime still carries rollout-coupled blocker reasons "
                    f"(blockers={','.join(continuity_rollout_blocker_reasons)})",
                )
            )
        elif continuity_rollout_warning_reasons:
            issues.append(
                _issue(
                    lane,
                    "warning",
                    "truthful",
                    "Continuity runtime carries rollout-coupled warning residue "
                    f"(warnings={','.join(continuity_rollout_warning_reasons)})",
                )
            )

    if not alive:
        issues.insert(0, _issue(lane, "critical", "alive", "OpenClaw gateway process not found"))
    elif not ready:
        issues.insert(0, _issue(lane, "critical", "ready", "State directory missing"))

    safe = alive and ready and lane_safe_ok
    truthful = safe and lane_truth_ok and truthful_runtime_ok

    if safe and not truthful and truthful_runtime_ok is False and not any(i.get("layer_failed") == "truthful" for i in issues):
        issues.append(_issue(lane, "warning", "truthful", "Continuity now reports drift or staleness"))

    state = _lane_state(alive, ready, safe, truthful)
    row = {
        "lane": lane,
        "health_layer": state["health_layer"],
        "status": state["status"],
        "issues": issues,
    }
    lane_rows.append(row)
    aggregate_issues.extend(issues)

layers = [str((row or {}).get("health_layer") or "alive") for row in lane_rows]
min_layer = min(layers, key=lambda token: LAYER_RANK.get(token, 0)) if layers else "alive"

if lane_rows and all(str((row or {}).get("status") or "") == "pass" for row in lane_rows):
    overall_status = "pass"
elif lane_rows and all(str((row or {}).get("health_layer") or "") in {"safe-to-act", "truthful"} for row in lane_rows):
    overall_status = "degraded"
else:
    overall_status = "failing"

failing_lanes = [str((row or {}).get("lane") or "") for row in lane_rows if str((row or {}).get("status") or "") == "failing"]
degraded_lanes = [str((row or {}).get("lane") or "") for row in lane_rows if str((row or {}).get("status") or "") == "degraded"]

payload = {
    "timestamp": NOW,
    "lane": "A1_CONTROL_PLANE",
    "health_layer": min_layer,
    "status": overall_status,
    "issues": aggregate_issues,
    "lanes": lane_rows,
    "summary": {
        "target_lanes": TARGET_LANES,
        "lane_count": len(lane_rows),
        "pass_count": len([x for x in lane_rows if x.get("status") == "pass"]),
        "degraded_count": len(degraded_lanes),
        "failing_count": len(failing_lanes),
        "degraded_lanes": degraded_lanes,
        "failing_lanes": failing_lanes,
    },
    "metrics": {
        "gate_rc": gate_rc,
        "slo_evaluator_rc": slo_eval_rc,
        "continuity_now_rc": continuity_now_rc,
        "continuity_payload_available": continuity_payload_available,
        "continuity_not_ready_count": len(continuity_not_ready_reasons),
        "continuity_blocker_count": len(continuity_blocker_reasons),
        "continuity_effective_blocker_count": len(continuity_effective_blocker_reasons),
        "continuity_reconcile_only_count": len(continuity_reconcile_only_reasons),
        "continuity_warning_count": len(continuity_warning_reasons),
        "continuity_derivative_suppressed_blocker_count": len(continuity_derivative_suppressed_blockers),
        "continuity_rollout_blocker_count": len(continuity_rollout_blocker_reasons),
        "continuity_rollout_warning_count": len(continuity_rollout_warning_reasons),
        "failing_slo_count": len(failing_slos),
        "restore_slo_status": restore_slo_status,
        "restore_rollout_blocker_active": bool(restore_rollout_blocker_reason),
        "routing_failure_reason": routing_failure_reason or None,
        "routing_preflight_refresh_status": routing_preflight_refresh.get("status"),
        "routing_preflight_refresh_attempted": routing_preflight_refresh.get("attempted") is True,
        "internal_bypass_closeout_ready": internal_bypass_ready,
    },
    "routing_preflight_refresh": routing_preflight_refresh,
    "continuity_truth_coupling": {
        "payload_available": continuity_payload_available,
        "runtime_rc": continuity_now_rc,
        "truthful_runtime_ok": truthful_runtime_ok,
        "semantic_blocked": continuity_semantic_blocked,
        "reconcile_only_pending": continuity_reconcile_only_pending,
        "not_ready_reasons": continuity_not_ready_reasons,
        "blocker_reasons": continuity_blocker_reasons,
        "effective_blocker_reasons": continuity_effective_blocker_reasons,
        "reconcile_only_reasons": continuity_reconcile_only_reasons,
        "warning_reasons": continuity_warning_reasons,
        "derivative_suppressed_blockers": continuity_derivative_suppressed_blockers,
        "rollout_blocker_reasons": continuity_rollout_blocker_reasons,
        "rollout_warning_reasons": continuity_rollout_warning_reasons,
        "restore_slo_status": restore_slo_status,
        "restore_rollout_blocker_reason": restore_rollout_blocker_reason,
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Generated layered health snapshot at {OUT}")
print(json.dumps(payload, ensure_ascii=False, indent=2))

raise SystemExit(2 if overall_status == "failing" else 0)
PY
