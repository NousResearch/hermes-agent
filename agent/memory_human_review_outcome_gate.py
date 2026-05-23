from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_governance_submission_packet import (
    explain_governance_submission_packet,
    recommend_governance_submission_packet_action,
    validate_governance_submission_packet,
)


MEMORY_HUMAN_REVIEW_OUTCOME_GATE_VERSION = "0.1"
MEMORY_HUMAN_REVIEW_OUTCOME_KIND = "memory_human_review_outcome_candidate"
MEMORY_HUMAN_REVIEW_OUTCOME_STATUS = "review_outcome_candidate"
MEMORY_HUMAN_REVIEW_OUTCOME_ROUTING = "manual_real_proposal_creation_still_required"
SUPPORTED_HUMAN_REVIEW_OUTCOMES = (
    "approve_real_proposal_creation",
    "request_changes",
    "reject",
    "defer",
)
MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_outcome_candidates_only": True,
    "creates_real_proposals": False,
    "applies_proposals": False,
    "persists_approvals": False,
    "submits_to_governance": False,
    "converts_to_real_proposal": False,
}

_FORBIDDEN_TRUE_KEYS = (
    "submitted",
    "applied",
    "persisted",
    "approved",
    "converted_to_real_proposal",
    "created_real_proposal",
    "created_operation_event",
    "created_proposal_record",
    "submitted_to_governance",
)


def create_human_review_outcome_candidate(
    packet: Mapping[str, Any],
    reviewer: str | None = None,
    outcome: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only human-review outcome candidate."""
    source = deepcopy(dict(packet))
    packet_validation = validate_governance_submission_packet(source)
    selected_outcome = outcome if outcome is not None else evaluate_human_review_packet(source, reviewer=reviewer)["outcome"]
    selected_rationale = rationale if rationale is not None else _outcome_rationale(selected_outcome, packet_validation, explicit=outcome is not None)

    candidate = {
        "outcome_id": None,
        "outcome_kind": MEMORY_HUMAN_REVIEW_OUTCOME_KIND,
        "outcome_status": MEMORY_HUMAN_REVIEW_OUTCOME_STATUS,
        "routing": MEMORY_HUMAN_REVIEW_OUTCOME_ROUTING,
        "source_packet_id": source.get("packet_id"),
        "source_submission_id": source.get("source_submission_id"),
        "source_draft_id": source.get("source_draft_id"),
        "source_decision_id": source.get("source_decision_id"),
        "source_queue_item_id": source.get("source_queue_item_id"),
        "block_id": source.get("block_id"),
        "block_type": source.get("block_type"),
        "project_scope": source.get("project_scope"),
        "reviewer": reviewer if reviewer is not None else source.get("reviewer"),
        "outcome": selected_outcome,
        "rationale": selected_rationale,
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "packet_validation": deepcopy(packet_validation),
        "outcome_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_packet_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY),
    }
    candidate["outcome_id"] = _outcome_id(candidate)
    candidate["outcome_validation"] = validate_human_review_outcome_candidate(candidate)
    candidate["next_step_recommendation"] = recommend_human_review_outcome_action(candidate)
    return candidate


def evaluate_human_review_packet(packet: Mapping[str, Any], reviewer: str | None = None) -> dict[str, Any]:
    validation = validate_governance_submission_packet(packet)
    errors = list(validation.get("errors", []) or [])
    if "missing_payload_preview" in errors:
        outcome = "request_changes"
    elif "missing_source_evidence" in errors:
        outcome = "request_changes"
    elif validation.get("valid") is not True:
        outcome = "reject"
    elif isinstance(packet.get("payload_preview"), Mapping) and (packet.get("source_pattern_ids") or packet.get("source_fact_ids")):
        outcome = "approve_real_proposal_creation"
    else:
        outcome = "defer"
    return {
        "outcome": outcome,
        "rationale": _outcome_rationale(outcome, validation),
        "reviewer": reviewer if reviewer is not None else packet.get("reviewer"),
        "packet_validation": validation,
        "packet_explanation": explain_governance_submission_packet(packet),
        "packet_recommendation": recommend_governance_submission_packet_action(packet),
        "policy": dict(MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY),
    }


def validate_human_review_outcome_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(candidate.get("policy"))
    packet_validation = _as_dict(candidate.get("packet_validation"))
    source_snapshot = _as_dict(candidate.get("source_packet_snapshot"))

    for key in (
        "outcome_id",
        "outcome_kind",
        "outcome_status",
        "routing",
        "source_packet_id",
        "source_submission_id",
        "source_draft_id",
        "source_decision_id",
        "source_queue_item_id",
        "block_id",
        "block_type",
        "project_scope",
        "reviewer",
        "outcome",
        "rationale",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "packet_validation",
        "outcome_validation",
        "next_step_recommendation",
        "source_packet_snapshot",
        "policy",
    ):
        if key not in candidate:
            errors.append(f"missing_{key}")
    if candidate.get("outcome_kind") != MEMORY_HUMAN_REVIEW_OUTCOME_KIND:
        errors.append("outcome_kind_must_be_memory_human_review_outcome_candidate")
    if candidate.get("outcome_status") != MEMORY_HUMAN_REVIEW_OUTCOME_STATUS:
        errors.append("outcome_status_must_be_review_outcome_candidate")
    if candidate.get("routing") != MEMORY_HUMAN_REVIEW_OUTCOME_ROUTING:
        errors.append("routing_must_require_manual_real_proposal_creation_still_required")
    if candidate.get("outcome") not in SUPPORTED_HUMAN_REVIEW_OUTCOMES:
        errors.append("unsupported_human_review_outcome")
    if not isinstance(candidate.get("rationale"), str) or not candidate.get("rationale"):
        errors.append("rationale_must_be_non_empty_string")
    if not isinstance(candidate.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(candidate.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if not isinstance(packet_validation.get("valid"), bool):
        errors.append("packet_validation_must_include_boolean_valid")
    if source_snapshot:
        snapshot_validation = validate_governance_submission_packet(source_snapshot)
        if snapshot_validation != packet_validation:
            errors.append("packet_validation_must_match_source_packet_snapshot")
    for forbidden_key in _FORBIDDEN_TRUE_KEYS:
        if candidate.get(forbidden_key) is True:
            errors.append(f"{forbidden_key}_must_be_false_or_absent")
    for key, expected in MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY.items():
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_review_outcome_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_review_outcome_candidate(candidate)
    source_snapshot = _as_dict(candidate.get("source_packet_snapshot"))
    return {
        "outcome_id": candidate.get("outcome_id"),
        "outcome_kind": candidate.get("outcome_kind"),
        "outcome_status": candidate.get("outcome_status"),
        "routing": candidate.get("routing"),
        "source_packet_id": candidate.get("source_packet_id"),
        "source_submission_id": candidate.get("source_submission_id"),
        "source_draft_id": candidate.get("source_draft_id"),
        "source_decision_id": candidate.get("source_decision_id"),
        "source_queue_item_id": candidate.get("source_queue_item_id"),
        "block_id": candidate.get("block_id"),
        "block_type": candidate.get("block_type"),
        "project_scope": candidate.get("project_scope"),
        "reviewer": candidate.get("reviewer"),
        "outcome": candidate.get("outcome"),
        "rationale": candidate.get("rationale"),
        "source_pattern_count": len(candidate.get("source_pattern_ids", []) or []),
        "source_fact_count": len(candidate.get("source_fact_ids", []) or []),
        "validation": validation,
        "packet_explanation": explain_governance_submission_packet(source_snapshot) if source_snapshot else {},
        "submitted": False,
        "applied": False,
        "persisted": False,
        "approved": False,
        "converted_to_real_proposal": False,
        "created_real_proposal": False,
        "created_operation_event": False,
        "submitted_to_governance": False,
        "policy": dict(MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY),
    }


def recommend_human_review_outcome_action(candidate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_review_outcome_candidate(candidate)
    outcome = candidate.get("outcome")
    if not validation["valid"]:
        action = "do_not_use_outcome_candidate"
        reason = _recommendation_reason(validation)
    elif outcome == "approve_real_proposal_creation":
        action = "manual_real_proposal_creation_required"
        reason = "Outcome candidate approves manual real proposal creation, but this gate does not create, submit, apply, or persist it."
    elif outcome == "request_changes":
        action = "return_packet_for_changes"
        reason = "Outcome candidate requests packet changes before any real proposal creation."
    elif outcome == "reject":
        action = "do_not_create_real_proposal"
        reason = "Outcome candidate rejects this packet for real proposal creation."
    else:
        action = "defer_manual_review_outcome"
        reason = "Outcome candidate defers until a human reviewer resolves the unknown condition."
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_REVIEW_OUTCOME_ROUTING,
        "validation": validation,
        "packet_recommendation": recommend_governance_submission_packet_action(_as_dict(candidate.get("source_packet_snapshot"))),
        "creates_outcome_candidates_only": True,
        "creates_real_proposals": False,
        "applies_proposals": False,
        "persists_approvals": False,
        "submits_to_governance": False,
        "converts_to_real_proposal": False,
        "policy": dict(MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY),
    }


def summarize_human_review_outcomes(
    candidates: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    by_outcome: dict[str, int] = {}
    by_block_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    valid_count = 0
    invalid_count = 0
    for candidate in candidates:
        outcome = str(candidate.get("outcome"))
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        block_type = str(candidate.get("block_type"))
        by_block_type[block_type] = by_block_type.get(block_type, 0) + 1
        status = str(candidate.get("outcome_status"))
        by_status[status] = by_status.get(status, 0) + 1
        validation = validate_human_review_outcome_candidate(candidate)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(candidates),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "by_outcome": dict(sorted(by_outcome.items())),
        "by_block_type": dict(sorted(by_block_type.items())),
        "by_status": dict(sorted(by_status.items())),
        "policy": dict(MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY),
    }


def _outcome_rationale(outcome: str, packet_validation: Mapping[str, Any], explicit: bool = False) -> str:
    if explicit and outcome in SUPPORTED_HUMAN_REVIEW_OUTCOMES:
        return "Explicit supported human review outcome override recorded as a read-only candidate only."
    errors = list(packet_validation.get("errors", []) or [])
    if outcome == "approve_real_proposal_creation":
        return "Packet is valid and includes payload preview plus source evidence."
    if outcome == "request_changes" and "missing_payload_preview" in errors:
        return "Packet is missing payload_preview required before real proposal creation review."
    if outcome == "request_changes" and "missing_source_evidence" in errors:
        return "Packet is missing source pattern or fact evidence required before real proposal creation review."
    if outcome == "reject":
        return "Packet is invalid for human review outcome approval."
    if outcome == "defer":
        return "Packet condition is unknown for v0.1 and must be deferred."
    return "Human review outcome is unsupported by v0.1."


def _recommendation_reason(validation: Mapping[str, Any]) -> str:
    errors = list(validation.get("errors", []) or [])
    if "unsupported_human_review_outcome" in errors:
        return "Outcome candidate uses an unsupported human review outcome label."
    if "packet_validation_must_match_source_packet_snapshot" in errors:
        return "Outcome candidate packet validation does not match its source packet snapshot."
    return "Outcome candidate violates the v0.1 human review outcome validation contract."


def _outcome_id(candidate: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_REVIEW_OUTCOME_GATE_VERSION,
        "outcome_kind": candidate.get("outcome_kind"),
        "outcome_status": candidate.get("outcome_status"),
        "routing": candidate.get("routing"),
        "source_packet_id": candidate.get("source_packet_id"),
        "source_submission_id": candidate.get("source_submission_id"),
        "source_draft_id": candidate.get("source_draft_id"),
        "source_decision_id": candidate.get("source_decision_id"),
        "source_queue_item_id": candidate.get("source_queue_item_id"),
        "block_id": candidate.get("block_id"),
        "block_type": candidate.get("block_type"),
        "project_scope": candidate.get("project_scope"),
        "reviewer": candidate.get("reviewer"),
        "outcome": candidate.get("outcome"),
        "rationale": candidate.get("rationale"),
        "payload_preview": candidate.get("payload_preview"),
        "source_pattern_ids": list(candidate.get("source_pattern_ids", [])),
        "source_fact_ids": list(candidate.get("source_fact_ids", [])),
        "packet_validation": candidate.get("packet_validation", {}),
        "policy": candidate.get("policy", {}),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"memory-human-review-outcome:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
