from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_block_review_queue import (
    explain_review_queue_item,
    recommend_review_queue_action,
    validate_review_queue_item,
)


MEMORY_REVIEW_DECISION_GATE_VERSION = "0.1"
SUPPORTED_REVIEW_DECISIONS = (
    "approve_to_proposal",
    "reject",
    "request_more_evidence",
    "defer",
)
MEMORY_REVIEW_DECISION_GATE_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_decision_candidates_only": True,
    "applies_decisions": False,
    "creates_real_proposals": False,
}


def create_review_decision_candidate(
    queue_item: Mapping[str, Any],
    reviewer: str | None = None,
    decision: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only decision candidate for a review queue item."""
    item = deepcopy(dict(queue_item))
    queue_validation = validate_review_queue_item(item)
    chosen_decision, default_rationale = _decision_for_queue_item(item, queue_validation)
    if decision is not None:
        chosen_decision = str(decision)
        default_rationale = "Explicit decision override supplied for validation only; no decision is applied."

    candidate = {
        "decision_id": None,
        "queue_item_id": item.get("queue_item_id"),
        "block_id": item.get("block_id"),
        "block_type": item.get("block_type"),
        "project_scope": item.get("project_scope"),
        "reviewer": reviewer if reviewer is not None else item.get("reviewer"),
        "decision": chosen_decision,
        "rationale": rationale or default_rationale,
        "risk_level": item.get("risk_level"),
        "source_pattern_ids": deepcopy(list(item.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(item.get("source_fact_ids", []) or [])),
        "queue_item_validation": deepcopy(queue_validation),
        "decision_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "queue_item_snapshot": deepcopy(item),
        "policy": dict(MEMORY_REVIEW_DECISION_GATE_POLICY),
    }
    candidate["decision_id"] = _decision_id(candidate)
    candidate["decision_validation"] = validate_review_decision_candidate(candidate)
    candidate["next_step_recommendation"] = recommend_review_decision_action(candidate)
    return candidate


def evaluate_review_queue_item(queue_item: Mapping[str, Any], reviewer: str | None = None) -> dict[str, Any]:
    """Evaluate a queue item into a read-only decision candidate."""
    return create_review_decision_candidate(queue_item, reviewer=reviewer)


def validate_review_decision_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(candidate.get("policy"))
    queue_validation = _as_dict(candidate.get("queue_item_validation"))

    for key in (
        "decision_id",
        "queue_item_id",
        "block_id",
        "block_type",
        "project_scope",
        "reviewer",
        "decision",
        "rationale",
        "risk_level",
        "source_pattern_ids",
        "source_fact_ids",
        "queue_item_validation",
        "decision_validation",
        "next_step_recommendation",
        "queue_item_snapshot",
        "policy",
    ):
        if key not in candidate:
            errors.append(f"missing_{key}")
    if candidate.get("decision") not in SUPPORTED_REVIEW_DECISIONS:
        errors.append(f"unsupported_decision:{candidate.get('decision')}")
    if candidate.get("risk_level") not in {"low", "medium", "high"}:
        errors.append("risk_level_must_be_low_medium_or_high")
    if not isinstance(candidate.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(candidate.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if not isinstance(queue_validation.get("valid"), bool):
        errors.append("queue_item_validation_valid_must_be_bool")
    for key, expected in MEMORY_REVIEW_DECISION_GATE_POLICY.items():
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")

    return {"valid": not errors, "errors": errors}


def explain_review_decision_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_review_decision_candidate(candidate)
    snapshot = _as_dict(candidate.get("queue_item_snapshot"))
    return {
        "decision_id": candidate.get("decision_id"),
        "queue_item_id": candidate.get("queue_item_id"),
        "block_id": candidate.get("block_id"),
        "block_type": candidate.get("block_type"),
        "project_scope": candidate.get("project_scope"),
        "reviewer": candidate.get("reviewer"),
        "decision": candidate.get("decision"),
        "rationale": candidate.get("rationale"),
        "risk_level": candidate.get("risk_level"),
        "source_pattern_count": len(candidate.get("source_pattern_ids", []) or []),
        "source_fact_count": len(candidate.get("source_fact_ids", []) or []),
        "validation": validation,
        "queue_item_explanation": explain_review_queue_item(snapshot) if snapshot else {},
        "applied": False,
        "created_real_proposal": False,
        "created_operation_event": False,
        "policy": dict(MEMORY_REVIEW_DECISION_GATE_POLICY),
    }


def recommend_review_decision_action(candidate: Mapping[str, Any]) -> dict[str, Any]:
    decision = candidate.get("decision")
    snapshot = _as_dict(candidate.get("queue_item_snapshot"))
    if decision == "approve_to_proposal":
        action = "prepare_governed_write_proposal"
        reason = "Decision candidate can proceed to a separate governed proposal flow, but this gate does not create it."
    elif decision == "reject":
        action = "record_rejection_outside_gate"
        reason = "Decision candidate indicates rejection; this gate does not persist or apply the rejection."
    elif decision == "request_more_evidence":
        action = "collect_more_source_evidence"
        reason = "Decision candidate requires stronger source evidence before any proposal path."
    else:
        action = "defer_for_manual_review"
        reason = "Decision candidate cannot determine a v0.1 route deterministically."
    return {
        "action": action,
        "reason": reason,
        "decision": decision,
        "queue_item_recommendation": recommend_review_queue_action(snapshot) if snapshot else {},
        "creates_decision_candidates_only": True,
        "applies_decisions": False,
        "creates_real_proposals": False,
        "policy": dict(MEMORY_REVIEW_DECISION_GATE_POLICY),
    }


def summarize_review_decisions(candidates: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    by_decision = {decision: 0 for decision in SUPPORTED_REVIEW_DECISIONS}
    by_risk = {"high": 0, "medium": 0, "low": 0}
    invalid_count = 0
    for candidate in candidates:
        decision = str(candidate.get("decision"))
        if decision in by_decision:
            by_decision[decision] += 1
        risk = str(candidate.get("risk_level"))
        if risk in by_risk:
            by_risk[risk] += 1
        validation = validate_review_decision_candidate(candidate)
        if validation["valid"] is not True:
            invalid_count += 1
    return {
        "total": len(candidates),
        "by_decision": by_decision,
        "by_risk": by_risk,
        "invalid_count": invalid_count,
        "policy": dict(MEMORY_REVIEW_DECISION_GATE_POLICY),
    }


def _decision_for_queue_item(item: Mapping[str, Any], queue_validation: Mapping[str, Any]) -> tuple[str, str]:
    block_validation = _as_dict(item.get("validation"))
    block_type = str(item.get("block_type"))
    risk_level = str(item.get("risk_level"))
    has_sources = bool(item.get("source_pattern_ids") or item.get("source_fact_ids"))

    if queue_validation.get("valid") is not True:
        return "reject", "Invalid review queue item violates the v0.1 queue contract."
    if block_validation.get("valid") is not True:
        return "reject", "Invalid memory block candidate inside the queue item must be rejected."
    if risk_level == "high" and block_type == "safety_policy":
        return "request_more_evidence", "High-risk safety policy blocks require more evidence before proposal handling."
    if risk_level == "high":
        return "reject", "High-risk unsupported or invalid queue item condition must be rejected."
    if risk_level == "medium" and block_type in {"procedural_rules", "methodology"}:
        if has_sources:
            return "approve_to_proposal", "Valid medium-risk rule or methodology block has source ids and may enter a governed proposal flow."
        return "request_more_evidence", "Medium-risk rule or methodology block needs source ids before proposal handling."
    if risk_level == "medium" and block_type in {"persona", "collaboration_style", "human_profile"}:
        if has_sources:
            return "approve_to_proposal", "Valid medium-risk profile block has source ids and may enter a governed proposal flow."
        return "request_more_evidence", "Medium-risk profile block needs source ids before proposal handling."
    if risk_level == "low" and block_type in {"project_context", "current_task_state"}:
        return "approve_to_proposal", "Valid low-risk project context or task state block may enter a governed proposal flow."
    return "defer", "No deterministic v0.1 decision rule matched this queue item."


def _decision_id(candidate: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_REVIEW_DECISION_GATE_VERSION,
        "queue_item_id": candidate.get("queue_item_id"),
        "block_id": candidate.get("block_id"),
        "block_type": candidate.get("block_type"),
        "project_scope": candidate.get("project_scope"),
        "reviewer": candidate.get("reviewer"),
        "decision": candidate.get("decision"),
        "rationale": candidate.get("rationale"),
        "risk_level": candidate.get("risk_level"),
        "source_pattern_ids": list(candidate.get("source_pattern_ids", [])),
        "source_fact_ids": list(candidate.get("source_fact_ids", [])),
        "queue_item_validation": candidate.get("queue_item_validation", {}),
        "policy": candidate.get("policy", {}),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"memory-review-decision:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}
