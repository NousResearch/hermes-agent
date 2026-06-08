"""Integrated closeout status reporting across Hermes execution lanes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

REPORT_SCHEMA: Final = "kanban_cross_lane_closeout_report.v1"
PASSING: Final = {"pass", "passed", "success", "complete", "clear", "approve", "approved"}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _blockers(closeout_result: Mapping[str, Any]) -> list[str]:
    items = list(_as_sequence(closeout_result.get("blockers")))
    items.extend(_as_sequence(closeout_result.get("reason_codes")))
    return sorted({str(item) for item in items if str(item).strip()})


def _has_key(items: Sequence[str], needles: tuple[str, ...]) -> bool:
    return any(any(needle in item for needle in needles) for item in items)


def _dimension(status: str, *, keys: Sequence[str] = ()) -> dict[str, Any]:
    return {"status": status, "keys": list(keys)}


def _lane(evidence: Mapping[str, Any]) -> str:
    explicit_lane = str(evidence.get("lane") or "").strip().lower()
    if explicit_lane:
        return explicit_lane
    schema = str(evidence.get("schema") or "")
    if _as_mapping(evidence.get("ultragoal")):
        return "ultragoal"
    if schema == "kanban_parent_review_package.v1" or _as_mapping(evidence.get("parent_child_matrix")):
        return "autopilot"
    return "unknown"


def _worker_status(evidence: Mapping[str, Any], blocker_keys: Sequence[str]) -> str:
    worker = _as_mapping(evidence.get("worker_evidence"))
    if not worker and evidence.get("schema") == "kanban_parent_review_package.v1":
        return "PASS" if _as_sequence(evidence.get("children")) else "MISSING"
    if not worker:
        return "MISSING"
    ledger = _as_mapping(evidence.get("done_criteria_ledger"))
    if ledger.get("criteria_hash") and worker.get("criteria_hash") != ledger.get("criteria_hash"):
        return "STALE"
    if _has_key(blocker_keys, ("stale", "superseded")):
        return "STALE"
    return "PASS"


def _verifier_status(evidence: Mapping[str, Any]) -> str:
    verification = _as_mapping(evidence.get("verification"))
    if verification:
        return "BLOCKED" if verification.get("allowed") is False else "PASS"
    verifier = _as_mapping(evidence.get("verifier_result"))
    if not verifier:
        parent = _as_mapping(evidence.get("parent_child_matrix")) or evidence
        remaining = parent.get("remainingChildren") or parent.get("remaining_children") or []
        if evidence.get("schema") == "kanban_parent_review_package.v1" and parent.get("parentRollupState") == "complete" and remaining == []:
            return "PASS"
        return "MISSING"
    verdict = _text(verifier.get("verdict") or verifier.get("status"))
    return "PASS" if verifier.get("passed") is True or verdict in PASSING else "BLOCKED"


def _parent_status(parent_report: Mapping[str, Any]) -> str:
    if not parent_report:
        return "PASS"
    if parent_report.get("parentScopeComplete") is True or parent_report.get("parentRollupState") == "complete":
        return "PASS"
    if parent_report.get("cannotFinalCloseoutParent") is True or parent_report.get("parentRollupState"):
        return "BLOCKED"
    return "MISSING"


def _cleanup_status(evidence: Mapping[str, Any]) -> str:
    cleanup = _as_mapping(evidence.get("cleanup"))
    if cleanup:
        status = _text(cleanup.get("status"))
        return "PASS" if cleanup.get("proof") or cleanup.get("worktree_clean") is True or status in PASSING else "BLOCKED"
    parent = _as_mapping(evidence.get("parent_child_matrix")) or evidence
    remaining = parent.get("remainingChildren") or parent.get("remaining_children") or []
    if evidence.get("schema") == "kanban_parent_review_package.v1" and parent.get("parentRollupState") == "complete" and remaining == []:
        return "PASS"
    side_effects = _as_mapping(evidence.get("side_effects"))
    if side_effects.get("gateway_restart_or_reload") is False:
        return "PASS"
    return "MISSING"


def _reviewer_status(evidence: Mapping[str, Any]) -> str:
    reviewer = _as_mapping(evidence.get("reviewer_result"))
    quality_gate = _as_mapping(evidence.get("quality_gate")) or _as_mapping(reviewer.get("quality_gate"))
    if not reviewer and not quality_gate:
        parent = _as_mapping(evidence.get("parent_child_matrix")) or evidence
        remaining = parent.get("remainingChildren") or parent.get("remaining_children") or []
        if evidence.get("schema") == "kanban_parent_review_package.v1" and parent.get("parentRollupState") == "complete" and remaining == []:
            return "PASS"
        return "MISSING"
    decision = _text(
        quality_gate.get("decision")
        or quality_gate.get("status")
        or reviewer.get("recommendation")
        or reviewer.get("decision")
    )
    return "PASS" if decision in PASSING else "BLOCKED"


def _completion_status(evidence: Mapping[str, Any], parent_status: str) -> str:
    if parent_status == "BLOCKED":
        return "BLOCKED"
    if _as_mapping(evidence.get("done_criteria_ledger")) or evidence.get("schema") == "kanban_parent_review_package.v1":
        return "PASS"
    return "MISSING"


def build_cross_lane_closeout_report(
    *,
    evidence: Mapping[str, Any],
    closeout_result: Mapping[str, Any] | None = None,
    parent_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable PASS/BLOCKED/MISSING/STALE closeout report."""
    result = _as_mapping(closeout_result)
    parent = _as_mapping(parent_report) or _as_mapping(evidence.get("parent_child_matrix"))
    blocker_keys = _blockers(result)
    parent_dimension = _parent_status(parent)
    worker_dimension = _worker_status(evidence, blocker_keys)
    stale = _has_key(blocker_keys, ("stale", "superseded"))
    dimensions = {
        "kanban_relation_drift": _dimension("BLOCKED" if _has_key(blocker_keys, ("relation_drift", "drift")) else "PASS", keys=blocker_keys),
        "kanban_ready_contract": _dimension("BLOCKED" if _has_key(blocker_keys, ("ready_contract", "dependency", "unresolved")) else "PASS", keys=blocker_keys),
        "lane_owned_evidence": _dimension("STALE" if stale else worker_dimension, keys=blocker_keys),
        "parent_child_coverage": _dimension(parent_dimension, keys=[str(item) for item in _as_sequence(parent.get("remainingChildren"))]),
        "worker_evidence": _dimension(worker_dimension, keys=blocker_keys),
        "verifier_result": _dimension(_verifier_status(evidence), keys=blocker_keys),
        "completion_audit": _dimension(_completion_status(evidence, parent_dimension), keys=blocker_keys),
        "cleanup": _dimension(_cleanup_status(evidence)),
        "cleanup_readiness": _dimension(_cleanup_status(evidence)),
        "reviewer_readiness": _dimension(_reviewer_status(evidence)),
    }
    status = "PASS" if all(item["status"] == "PASS" for item in dimensions.values()) else "BLOCKED"
    return {"schema": REPORT_SCHEMA, "lane": _lane(evidence), "status": status, "blocker_keys": blocker_keys, "dimensions": dimensions}


def format_cross_lane_closeout_report(report: Mapping[str, Any]) -> str:
    """Render a concise operator-facing report."""
    dimensions = _as_mapping(report.get("dimensions"))
    lines = ["Cross-lane closeout report", f"lane: {report.get('lane')}", f"status: {report.get('status')}"]
    blocker_keys = sorted(str(item) for item in _as_sequence(report.get("blocker_keys")) if str(item).strip())
    if blocker_keys:
        lines.append("blockers: " + ", ".join(blocker_keys))
    for name in sorted(dimensions):
        dimension = _as_mapping(dimensions[name])
        lines.append(f"{name}: {dimension.get('status')}")
    return "\n".join(lines)
