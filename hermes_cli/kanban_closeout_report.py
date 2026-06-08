"""Operator-facing closeout report synthesis for Kanban execution lanes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hermes_cli.kanban_cross_lane_report import build_cross_lane_closeout_report

_DIMENSION_KEYS = (
    "lane_identity",
    "kanban_relation_drift",
    "kanban_ready_contract",
    "lane_owned_evidence",
    "parent_child_coverage",
    "worker_evidence",
    "verifier_result",
    "completion_audit",
    "cleanup",
    "reviewer_readiness",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _status_from_blockers(blockers: list[str]) -> str:
    return "PASS" if not blockers else "BLOCKED"


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _lane(evidence: Mapping[str, Any]) -> str:
    lane = str(evidence.get("lane") or evidence.get("execution_lane") or "").strip().lower()
    if lane in {"ultragoal", "autopilot"}:
        return lane
    if evidence.get("goal_contract") or evidence.get("goalContract"):
        return "ultragoal"
    if evidence.get("parent_child_matrix") or evidence.get("parentRollupState"):
        return "autopilot"
    return "unknown"


def _is_parent_review_package(evidence: Mapping[str, Any]) -> bool:
    return evidence.get("schema") == "kanban_parent_review_package.v1"


def _parent_rollup(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(evidence.get("parent_child_matrix") or evidence.get("parentChildMatrix")) or evidence


def _parent_rollup_complete(evidence: Mapping[str, Any]) -> bool:
    parent = _parent_rollup(evidence)
    state = str(parent.get("parentRollupState") or parent.get("parent_rollup_state") or "").strip()
    remaining = parent.get("remainingChildren") or parent.get("remaining_children") or []
    return state == "complete" and remaining == []


def _string_list(value: Any, malformed_key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if _has_text(item)]
    return [malformed_key]


def _verification_blockers(verification: Mapping[str, Any], needles: tuple[str, ...] | None = None) -> list[str]:
    raw = _string_list(verification.get("blockers"), "verification_blockers:not_list")
    raw.extend(_string_list(verification.get("reason_codes"), "verification_reason_codes:not_list"))
    if needles is None:
        return raw
    return [item for item in raw if any(needle in item for needle in needles)]


def _relation_drift_blockers(evidence: Mapping[str, Any], verification: Mapping[str, Any]) -> list[str]:
    raw = evidence.get("relation_drift_blockers") or evidence.get("relationDriftBlockers")
    blockers = _string_list(raw, "relation_drift_blockers:not_list")
    blockers.extend(_verification_blockers(verification, ("relation_drift", "drift")))
    return list(dict.fromkeys(blockers))


def _ready_contract_blockers(evidence: Mapping[str, Any], verification: Mapping[str, Any]) -> list[str]:
    raw = evidence.get("task_ready_contract_blockers") or evidence.get("ready_contract_blockers")
    blockers = _string_list(raw, "ready_contract_blockers:not_list")
    blockers.extend(_verification_blockers(verification, ("ready_contract", "dependency", "unresolved")))
    return list(dict.fromkeys(blockers))


def _lane_owned_evidence_status(lane: str, evidence: Mapping[str, Any]) -> str:
    if lane == "ultragoal" and (evidence.get("goal_contract") or evidence.get("goalContract")):
        return "PASS"
    if lane == "autopilot" and (evidence.get("parent_child_matrix") or evidence.get("parentRollupState")):
        return "PASS"
    return "MISSING"


def _parent_child_status(lane: str, evidence: Mapping[str, Any]) -> str:
    matrix = _mapping(evidence.get("parent_child_matrix") or evidence.get("parentChildMatrix"))
    if not matrix:
        if _is_parent_review_package(evidence) and _parent_rollup_complete(evidence):
            return "PASS"
        return "PASS" if lane == "ultragoal" else "MISSING"
    if _parent_rollup_complete(evidence):
        return "PASS"
    state = str(matrix.get("parentRollupState") or matrix.get("parent_rollup_state") or "").strip()
    if state in {"partial", "review_blocked", "needs_user_decision"}:
        return "BLOCKED"
    return "MISSING"


def _worker_status(evidence: Mapping[str, Any]) -> str:
    worker = _mapping(evidence.get("worker_evidence") or evidence.get("workerEvidence"))
    if not worker:
        if _is_parent_review_package(evidence) and _parent_rollup_complete(evidence):
            return "PASS"
        return "MISSING"
    if worker.get("authority_boundary_confirmed") is True and isinstance(worker.get("per_criterion"), Mapping):
        return "PASS"
    return "STALE"


def _verifier_status(evidence: Mapping[str, Any], verification: Mapping[str, Any]) -> str:
    if verification.get("allowed") is False or _verification_blockers(verification):
        return "BLOCKED"
    verifier = _mapping(evidence.get("verifier_result") or evidence.get("verifierResult") or evidence.get("verifier_verdict"))
    verdict = str(verifier.get("verdict") or "").strip().lower()
    if not verifier:
        if evidence.get("schema") == "kanban_parent_review_package.v1":
            return "PASS"
        return "MISSING"
    if verdict in {"pass", "passed"}:
        return "PASS"
    return "BLOCKED"


def _audit_status(evidence: Mapping[str, Any], verification: Mapping[str, Any]) -> str:
    audit = _mapping(evidence.get("completion_audit") or evidence.get("completionAudit"))
    if not audit:
        if evidence.get("schema") == "kanban_parent_review_package.v1":
            return "PASS"
        return "MISSING"
    return "PASS" if str(audit.get("status") or "").strip().upper() == "PASS" else "BLOCKED"


def _cleanup_status(evidence: Mapping[str, Any], verification: Mapping[str, Any]) -> str:
    cleanup = _mapping(evidence.get("cleanup"))
    if not cleanup:
        if evidence.get("schema") == "kanban_parent_review_package.v1":
            return "PASS"
        return "MISSING"
    if cleanup.get("worktree_clean") is True or cleanup.get("readOnlyProof") is True or cleanup.get("proof"):
        return "PASS"
    return "STALE"


def _reviewer_status(evidence: Mapping[str, Any]) -> str:
    readiness = _mapping(evidence.get("reviewer_readiness") or evidence.get("reviewerReadiness"))
    if readiness:
        return "PASS" if str(readiness.get("status") or "").strip().upper() == "PASS" else "BLOCKED"
    verifier = _mapping(evidence.get("verifier_result") or evidence.get("verifier_verdict"))
    if not verifier and _is_parent_review_package(evidence) and _parent_rollup_complete(evidence):
        return "PASS"
    return "PASS" if str(verifier.get("verdict") or "").strip().lower() in {"pass", "passed"} else "MISSING"


def _dimension_blockers(dimensions: Mapping[str, str]) -> list[str]:
    blockers: list[str] = []
    for key in _DIMENSION_KEYS:
        status = dimensions.get(key, "MISSING")
        if status != "PASS":
            blockers.append(f"{key}:{status.lower()}")
    return blockers


def build_closeout_report(
    *,
    task_id: str,
    phase: str,
    evidence: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    lane = _lane(evidence)
    relation_blockers = _relation_drift_blockers(evidence, verification)
    ready_blockers = _ready_contract_blockers(evidence, verification)
    dimensions = {
        "lane_identity": "PASS" if lane in {"ultragoal", "autopilot"} else "MISSING",
        "kanban_relation_drift": _status_from_blockers(relation_blockers),
        "kanban_ready_contract": _status_from_blockers(ready_blockers),
        "lane_owned_evidence": _lane_owned_evidence_status(lane, evidence),
        "parent_child_coverage": _parent_child_status(lane, evidence),
        "worker_evidence": _worker_status(evidence),
        "verifier_result": _verifier_status(evidence, verification),
        "completion_audit": _audit_status(evidence, verification),
        "cleanup": _cleanup_status(evidence, verification),
        "reviewer_readiness": _reviewer_status(evidence),
    }
    verification_blockers = _verification_blockers(verification)
    blocker_keys = list(dict.fromkeys(
        verification_blockers
        + relation_blockers
        + ready_blockers
        + _dimension_blockers(dimensions)
    ))
    cross_lane_report = build_cross_lane_closeout_report(
        evidence=evidence,
        closeout_result={"allowed": not blocker_keys, "blockers": blocker_keys},
    )
    return {
        "schema": "kanban_integrated_closeout_report.v1",
        "task_id": task_id,
        "phase": phase,
        "lane": lane,
        "dimensions": dimensions,
        "blocker_keys": blocker_keys,
        "ready": not blocker_keys,
        "cross_lane_report": cross_lane_report,
    }
