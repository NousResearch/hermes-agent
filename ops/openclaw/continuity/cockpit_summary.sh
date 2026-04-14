#!/usr/bin/env bash
# Operator Cockpit Summary Generator
# Synthesizes layered health and SLO snapshots into a unified action card.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

HEALTH_JSON="${WORKSPACE_DIR}/state/continuity/latest/layered_health_snapshot.json"
SLO_JSON="${WORKSPACE_DIR}/state/continuity/latest/slo_snapshot.json"
VERIFY_JSON="${WORKSPACE_DIR}/state/continuity/latest/verify_last.json"
ROLLBACK_TRIGGER_JSON="${WORKSPACE_DIR}/state/continuity/latest/release_error_budget_rollback_trigger_latest.json"

python3 - "$HEALTH_JSON" "$SLO_JSON" "$VERIFY_JSON" "$ROLLBACK_TRIGGER_JSON" <<'PY'
import json
import sys
from pathlib import Path
from typing import Any

health_path = Path(sys.argv[1])
slo_path = Path(sys.argv[2])
verify_path = Path(sys.argv[3])
rollback_trigger_path = Path(sys.argv[4])


def load_json_object(path: Path, label: str, payload_errors: list[str]) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        payload_errors.append(f"- Missing snapshot file: {path.name}")
        return None
    except Exception as exc:
        payload_errors.append(f"- {label} unreadable: {path.name} ({exc.__class__.__name__})")
        return None

    try:
        payload = json.loads(raw)
    except Exception:
        payload_errors.append(f"- {label} invalid JSON: {path.name}")
        return None

    if not isinstance(payload, dict):
        payload_errors.append(f"- {label} invalid payload: expected JSON object in {path.name}")
        return None
    return payload


def validate_health(payload: dict[str, Any], payload_errors: list[str]) -> tuple[str | None, list[str]]:
    status = payload.get("status")
    if status not in {"pass", "degraded", "failing"}:
        payload_errors.append("- layered_health_snapshot status invalid or unknown")
        status = None

    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        payload_errors.append("- layered_health_snapshot issues invalid: expected array")
        return status, []

    issue_lines: list[str] = []
    for idx, issue in enumerate(raw_issues):
        if not isinstance(issue, dict):
            payload_errors.append(f"- layered_health_snapshot issues[{idx}] invalid: expected object")
            continue
        layer_failed = issue.get("layer_failed")
        message = issue.get("message")
        if not isinstance(layer_failed, str) or not layer_failed.strip():
            payload_errors.append(f"- layered_health_snapshot issues[{idx}] missing layer_failed")
            continue
        if not isinstance(message, str) or not message.strip():
            payload_errors.append(f"- layered_health_snapshot issues[{idx}] missing message")
            continue
        issue_lines.append(f"- {layer_failed} layer failed: {message}")

    if status in {"degraded", "failing"} and not issue_lines:
        payload_errors.append(f"- layered_health_snapshot status={status} provides no failing issue detail")
    if status == "pass" and issue_lines:
        payload_errors.append("- layered_health_snapshot status=pass conflicts with non-empty issues")
    return status, issue_lines


def validate_slo(payload: dict[str, Any], payload_errors: list[str]) -> tuple[str | None, list[str], int]:
    status = payload.get("status")
    if status not in {"pass", "fail"}:
        payload_errors.append("- slo_snapshot status invalid or unknown")
        status = None

    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        payload_errors.append("- slo_snapshot evaluations invalid: expected array")
        return status, [], 0

    issue_lines: list[str] = []
    fail_count = 0
    for idx, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, dict):
            payload_errors.append(f"- slo_snapshot evaluations[{idx}] invalid: expected object")
            continue
        slo_id = evaluation.get("id")
        slo_status = evaluation.get("status")
        detail = evaluation.get("detail")
        if not isinstance(slo_id, str) or not slo_id.strip():
            payload_errors.append(f"- slo_snapshot evaluations[{idx}] missing id")
            continue
        if slo_status not in {"pass", "fail"}:
            payload_errors.append(f"- slo_snapshot evaluations[{idx}] status invalid for {slo_id}")
            continue
        if not isinstance(detail, str) or not detail.strip():
            payload_errors.append(f"- slo_snapshot evaluations[{idx}] missing detail for {slo_id}")
            continue
        if slo_status != "pass":
            fail_count += 1
            issue_lines.append(f"- {slo_id}: {detail}")

    if status == "pass" and fail_count > 0:
        payload_errors.append("- slo_snapshot status=pass conflicts with failing evaluations")
    if status == "fail" and fail_count == 0:
        payload_errors.append("- slo_snapshot status=fail without failing evaluations")
    return status, issue_lines, fail_count


def validate_verify(payload: dict[str, Any], payload_errors: list[str]) -> str:
    checkpoint_id = payload.get("checkpoint_id")
    if checkpoint_id in {None, ""}:
        return "none"
    return str(checkpoint_id)


def load_optional_json_object(path: Path, label: str, payload_errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        payload_errors.append(f"- {label} unreadable: {path.name} ({exc.__class__.__name__})")
        return None

    try:
        payload = json.loads(raw)
    except Exception:
        payload_errors.append(f"- {label} invalid JSON: {path.name}")
        return None

    if not isinstance(payload, dict):
        payload_errors.append(f"- {label} invalid payload: expected JSON object in {path.name}")
        return None
    return payload


def summarize_release_rollback_trigger(payload: dict[str, Any] | None, payload_errors: list[str]) -> tuple[bool, list[str], str]:
    if payload is None:
        return False, [], "`bash ops/openclaw/continuity/verify_then_resume.sh --run-rollback`"

    trigger_active = payload.get("trigger_active") is True
    rollback_command = str(payload.get("rollback_command") or "").strip()
    if not rollback_command:
        rollback_command = "bash ops/openclaw/continuity/verify_then_resume.sh --run-rollback"

    if payload.get("schema") not in {None, "clawd.release_error_budget_rollback_trigger.v1"}:
        payload_errors.append("- release_error_budget_rollback_trigger schema unexpected")

    if not trigger_active:
        return False, [], f"`{rollback_command}`"

    trigger_reason = str(payload.get("block_reason") or "a6_observability_failed")
    release_id = str(payload.get("release_id") or "unknown")
    detail = f"- release rollback trigger active: release_id={release_id} reason={trigger_reason}"
    return True, [detail], f"`{rollback_command}`"


def remediation_for(headline: str, issue_lines: list[str], rollback_action: str) -> str:
    issue_blob = "\n".join(issue_lines)
    if headline == "BLOCKER: RELEASE ROLLBACK TRIGGERED":
        return rollback_action
    if headline == "BLOCKER: OBSERVABILITY MISSING":
        return "`bash ops/openclaw/continuity/layered_health_snapshot.sh && bash ops/openclaw/continuity/slo_evaluator_snapshot.sh`"
    if headline == "BLOCKER: OBSERVABILITY INVALID":
        return "`cat docs/ops/incident_playbooks/blindness_recovery.md`"
    if "OpenClaw gateway process not found" in issue_blob:
        return "`openclaw gateway start`"
    if "SLO-1_VERIFY_FRESHNESS" in issue_blob or "SLO-2_CONTINUITY_FRESHNESS" in issue_blob:
        return "`bash ops/openclaw/snapshot_ground_truth.sh && bash ops/openclaw/continuity/verify_then_resume.sh`"
    if headline == "DEGRADED OPERATIONS":
        return "`bash ops/openclaw/continuity/continuity_now.sh --strict`"
    return "`cat docs/ops/incident_playbooks/blindness_recovery.md`"


payload_errors: list[str] = []
health_payload = load_json_object(health_path, "layered_health_snapshot", payload_errors)
slo_payload = load_json_object(slo_path, "slo_snapshot", payload_errors)
verify_payload = load_json_object(verify_path, "verify_last", payload_errors)

health_status: str | None = None
health_issues: list[str] = []
if health_payload is not None:
    health_status, health_issues = validate_health(health_payload, payload_errors)

slo_status: str | None = None
slo_issues: list[str] = []
slo_fail_count = 0
if slo_payload is not None:
    slo_status, slo_issues, slo_fail_count = validate_slo(slo_payload, payload_errors)

checkpoint_id = "none"
if verify_payload is not None:
    checkpoint_id = validate_verify(verify_payload, payload_errors)

rollback_trigger_payload = load_optional_json_object(
    rollback_trigger_path,
    "release_error_budget_rollback_trigger",
    payload_errors,
)
rollback_trigger_active, rollback_trigger_issues, rollback_action = summarize_release_rollback_trigger(
    rollback_trigger_payload,
    payload_errors,
)

status_icon = "🟢"
headline = "SYSTEM OPTIMAL"
issues: list[str] = []

if any(not path.exists() for path in (health_path, slo_path, verify_path)):
    status_icon = "🔴"
    headline = "BLOCKER: OBSERVABILITY MISSING"
    issues = payload_errors
elif payload_errors:
    status_icon = "🔴"
    headline = "BLOCKER: OBSERVABILITY INVALID"
    issues = payload_errors + health_issues + slo_issues
elif rollback_trigger_active:
    status_icon = "🔴"
    headline = "BLOCKER: RELEASE ROLLBACK TRIGGERED"
    issues = rollback_trigger_issues + health_issues + slo_issues
elif health_status == "failing" or slo_fail_count > 0:
    status_icon = "🔴"
    headline = "BLOCKER: SYSTEM HALTED"
    issues = health_issues + slo_issues
elif health_status == "degraded":
    status_icon = "🟡"
    headline = "DEGRADED OPERATIONS"
    issues = health_issues + slo_issues
else:
    issues = []

remediation = remediation_for(headline, issues, rollback_action)

print(f"{status_icon} **{headline}**")
print()
if issues:
    print("**Failing Constraints:**")
    print("\n".join(issues))
    print()
print("**Immediate Action:**")
print(remediation)
print()
print("---")
print(f"`chk_latest: {checkpoint_id}`")
PY
