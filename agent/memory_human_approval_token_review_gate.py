from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_request import (
    MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED,
    explain_human_approval_token_request,
    recommend_human_approval_token_request_action,
    validate_human_approval_token_request,
)
from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_forbidden_true_keys_false_or_absent,
    validate_policy_flags,
)


MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_KIND = "memory_human_approval_token_review_outcome_candidate"
MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_STATUS = "token_review_outcome_candidate"
MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_ROUTING = "manual_token_issuance_still_required"
SUPPORTED_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOMES = (
    "approve_token_issuance",
    "request_changes",
    "reject",
    "defer",
)
MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_review_outcome_candidates_only": True,
    "issues_real_approval_tokens": False,
    "persists_approvals": False,
    "creates_real_proposals": False,
    "writes_proposal_files": False,
    "writes_operation_ledger": False,
    "applies_proposals": False,
    "submits_to_governance": False,
    "converts_to_real_proposal": False,
}

_FORBIDDEN_TRUE_KEYS = (
    "token_issued",
    "approved",
    "persisted",
    "submitted",
    "written",
    "created_real_proposal",
    "created_operation_event",
    "writes_proposal_files",
    "writes_operation_ledger",
    "converted_to_real_proposal",
)


def create_human_approval_token_review_outcome(
    request_candidate: Mapping[str, Any],
    reviewer: str | None = None,
    outcome: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only human approval token review outcome candidate."""
    source = deepcopy(dict(request_candidate))
    request_validation = validate_human_approval_token_request(source)
    evaluation = evaluate_human_approval_token_request(source, reviewer=reviewer)
    selected_outcome = outcome if outcome is not None else evaluation["outcome"]
    selected_rationale = (
        rationale
        if rationale is not None
        else _outcome_rationale(selected_outcome, source, request_validation, explicit=outcome is not None)
    )

    candidate = {
        "review_outcome_id": None,
        "review_outcome_kind": MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_KIND,
        "review_outcome_status": MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_STATUS,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_ROUTING,
        "source_request_id": source.get("request_id"),
        "source_gate_id": source.get("source_gate_id"),
        "source_dry_run_id": source.get("source_dry_run_id"),
        "source_plan_id": source.get("source_plan_id"),
        "source_outcome_id": source.get("source_outcome_id"),
        "source_packet_id": source.get("source_packet_id"),
        "source_submission_id": source.get("source_submission_id"),
        "source_draft_id": source.get("source_draft_id"),
        "source_decision_id": source.get("source_decision_id"),
        "source_queue_item_id": source.get("source_queue_item_id"),
        "block_id": source.get("block_id"),
        "block_type": source.get("block_type"),
        "project_scope": source.get("project_scope"),
        "reviewer": reviewer if reviewer is not None else source.get("requester"),
        "outcome": selected_outcome,
        "rationale": selected_rationale,
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "request_validation": deepcopy(request_validation),
        "review_outcome_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_request_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY),
    }
    candidate["review_outcome_id"] = _review_outcome_id(candidate)
    candidate["review_outcome_validation"] = validate_human_approval_token_review_outcome(candidate)
    candidate["next_step_recommendation"] = recommend_human_approval_token_review_action(candidate)
    return candidate


def evaluate_human_approval_token_request(
    request_candidate: Mapping[str, Any],
    reviewer: str | None = None,
) -> dict[str, Any]:
    validation = validate_human_approval_token_request(request_candidate)
    errors = list(validation.get("errors", []) or [])
    if validation.get("valid") is not True:
        if _contains_any(errors, ("missing_proposal_record_preview", "missing_operation_ledger_preview", "missing_target_paths_preview", "missing_source_evidence")):
            outcome = "request_changes"
        else:
            outcome = "reject"
    elif request_candidate.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED:
        outcome = "reject"
    elif not isinstance(request_candidate.get("proposal_record_preview"), Mapping):
        outcome = "request_changes"
    elif not isinstance(request_candidate.get("operation_ledger_preview"), Mapping):
        outcome = "request_changes"
    elif not isinstance(request_candidate.get("target_paths_preview"), Mapping):
        outcome = "request_changes"
    elif not (request_candidate.get("source_pattern_ids") or request_candidate.get("source_fact_ids")):
        outcome = "request_changes"
    elif request_candidate.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED:
        outcome = "approve_token_issuance"
    else:
        outcome = "defer"
    return {
        "outcome": outcome,
        "rationale": _outcome_rationale(outcome, request_candidate, validation),
        "reviewer": reviewer if reviewer is not None else request_candidate.get("requester"),
        "request_validation": validation,
        "request_explanation": explain_human_approval_token_request(request_candidate),
        "request_recommendation": recommend_human_approval_token_request_action(request_candidate),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY),
    }


def validate_human_approval_token_review_outcome(outcome_candidate: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(outcome_candidate.get("policy"))
    request_validation = _as_dict(outcome_candidate.get("request_validation"))
    source_snapshot = _as_dict(outcome_candidate.get("source_request_snapshot"))

    for key in (
        "review_outcome_id",
        "review_outcome_kind",
        "review_outcome_status",
        "routing",
        "source_request_id",
        "source_gate_id",
        "source_dry_run_id",
        "source_plan_id",
        "source_outcome_id",
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
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "request_validation",
        "review_outcome_validation",
        "next_step_recommendation",
        "source_request_snapshot",
        "policy",
    ):
        if key not in outcome_candidate:
            errors.append(f"missing_{key}")
    if outcome_candidate.get("review_outcome_kind") != MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_KIND:
        errors.append("review_outcome_kind_must_be_memory_human_approval_token_review_outcome_candidate")
    if outcome_candidate.get("review_outcome_status") != MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_STATUS:
        errors.append("review_outcome_status_must_be_token_review_outcome_candidate")
    if outcome_candidate.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_ROUTING:
        errors.append("routing_must_require_manual_token_issuance_still_required")
    if outcome_candidate.get("outcome") not in SUPPORTED_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOMES:
        errors.append("unsupported_human_approval_token_review_outcome")
    if not isinstance(outcome_candidate.get("rationale"), str) or not outcome_candidate.get("rationale"):
        errors.append("rationale_must_be_non_empty_string")
    if not isinstance(outcome_candidate.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(outcome_candidate.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if not isinstance(request_validation.get("valid"), bool):
        errors.append("request_validation_must_include_boolean_valid")
    if source_snapshot:
        snapshot_validation = validate_human_approval_token_request(source_snapshot)
        if snapshot_validation != request_validation:
            errors.append("request_validation_must_match_source_request_snapshot")
    errors.extend(
        validate_forbidden_true_keys_false_or_absent(outcome_candidate, _FORBIDDEN_TRUE_KEYS)
    )
    errors.extend(validate_policy_flags(policy, MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY))

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_review_outcome(outcome_candidate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_review_outcome(outcome_candidate)
    source_snapshot = _as_dict(outcome_candidate.get("source_request_snapshot"))
    return {
        "review_outcome_id": outcome_candidate.get("review_outcome_id"),
        "review_outcome_kind": outcome_candidate.get("review_outcome_kind"),
        "review_outcome_status": outcome_candidate.get("review_outcome_status"),
        "routing": outcome_candidate.get("routing"),
        "source_request_id": outcome_candidate.get("source_request_id"),
        "source_gate_id": outcome_candidate.get("source_gate_id"),
        "source_dry_run_id": outcome_candidate.get("source_dry_run_id"),
        "source_plan_id": outcome_candidate.get("source_plan_id"),
        "source_outcome_id": outcome_candidate.get("source_outcome_id"),
        "source_packet_id": outcome_candidate.get("source_packet_id"),
        "source_submission_id": outcome_candidate.get("source_submission_id"),
        "source_draft_id": outcome_candidate.get("source_draft_id"),
        "source_decision_id": outcome_candidate.get("source_decision_id"),
        "source_queue_item_id": outcome_candidate.get("source_queue_item_id"),
        "block_id": outcome_candidate.get("block_id"),
        "block_type": outcome_candidate.get("block_type"),
        "project_scope": outcome_candidate.get("project_scope"),
        "reviewer": outcome_candidate.get("reviewer"),
        "outcome": outcome_candidate.get("outcome"),
        "rationale": outcome_candidate.get("rationale"),
        "source_pattern_count": len(outcome_candidate.get("source_pattern_ids", []) or []),
        "source_fact_count": len(outcome_candidate.get("source_fact_ids", []) or []),
        "validation": validation,
        "request_explanation": explain_human_approval_token_request(source_snapshot) if source_snapshot else {},
        "token_issued": False,
        "approved": False,
        "persisted": False,
        "submitted": False,
        "written": False,
        "created_real_proposal": False,
        "created_operation_event": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "converted_to_real_proposal": False,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY),
    }


def recommend_human_approval_token_review_action(outcome_candidate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_review_outcome(outcome_candidate)
    outcome = outcome_candidate.get("outcome")
    if not validation["valid"]:
        action = "do_not_use_token_review_outcome_candidate"
        reason = _recommendation_reason(validation)
    elif outcome == "approve_token_issuance":
        action = "manual_token_issuance_still_required"
        reason = "Review outcome candidate approves a separate manual token issuance flow, but this gate does not issue, persist, submit, or apply a token."
    elif outcome == "request_changes":
        action = "return_approval_token_request_for_changes"
        reason = "Review outcome candidate requests changes before any separate token issuance flow."
    elif outcome == "reject":
        action = "do_not_issue_approval_token"
        reason = "Review outcome candidate rejects token issuance for this request."
    else:
        action = "defer_manual_token_review_outcome"
        reason = "Review outcome candidate defers until a human reviewer resolves the unknown condition."
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_ROUTING,
        "validation": validation,
        "request_recommendation": recommend_human_approval_token_request_action(_as_dict(outcome_candidate.get("source_request_snapshot"))),
        "creates_review_outcome_candidates_only": True,
        "issues_real_approval_tokens": False,
        "persists_approvals": False,
        "creates_real_proposals": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "applies_proposals": False,
        "submits_to_governance": False,
        "converts_to_real_proposal": False,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY),
    }


def summarize_human_approval_token_review_outcomes(
    outcomes: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(
        outcomes,
        "review_outcome_status",
        type_key="outcome",
    )
    valid_count = 0
    invalid_count = 0
    for outcome_candidate in outcomes:
        validation = validate_human_approval_token_review_outcome(outcome_candidate)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": candidate_summary["total"],
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "by_outcome": candidate_summary["by_type"],
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY),
    }


def _outcome_rationale(
    outcome: str,
    request_candidate: Mapping[str, Any],
    request_validation: Mapping[str, Any],
    explicit: bool = False,
) -> str:
    if explicit and outcome in SUPPORTED_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOMES:
        return "Explicit supported approval token review outcome override recorded as a read-only candidate only."
    errors = list(request_validation.get("errors", []) or [])
    if outcome == "approve_token_issuance":
        return "Request is valid, review-required, and includes preview artifacts plus source evidence."
    if outcome == "request_changes" and "missing_proposal_record_preview" in errors:
        return "Request is missing proposal_record_preview required before token issuance review."
    if outcome == "request_changes" and "missing_operation_ledger_preview" in errors:
        return "Request is missing operation_ledger_preview required before token issuance review."
    if outcome == "request_changes" and "missing_target_paths_preview" in errors:
        return "Request is missing target_paths_preview required before token issuance review."
    if outcome == "request_changes" and "missing_source_evidence" in errors:
        return "Request is missing source pattern or fact evidence required before token issuance review."
    if outcome == "reject" and request_candidate.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED:
        return "Request candidate is locked and cannot proceed to token issuance review."
    if outcome == "reject":
        return "Request candidate is invalid for human approval token review."
    if outcome == "defer":
        return "Request condition is unknown for v0.1 and must be deferred."
    return "Approval token review outcome is unsupported by v0.1."


def _recommendation_reason(validation: Mapping[str, Any]) -> str:
    errors = list(validation.get("errors", []) or [])
    if "unsupported_human_approval_token_review_outcome" in errors:
        return "Review outcome candidate uses an unsupported approval token review outcome label."
    if "request_validation_must_match_source_request_snapshot" in errors:
        return "Review outcome candidate request validation does not match its source request snapshot."
    return "Review outcome candidate violates the v0.1 approval token review validation contract."


def _review_outcome_id(candidate: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_VERSION,
        "review_outcome_kind": candidate.get("review_outcome_kind"),
        "review_outcome_status": candidate.get("review_outcome_status"),
        "routing": candidate.get("routing"),
        "source_request_id": candidate.get("source_request_id"),
        "source_gate_id": candidate.get("source_gate_id"),
        "source_dry_run_id": candidate.get("source_dry_run_id"),
        "source_plan_id": candidate.get("source_plan_id"),
        "source_outcome_id": candidate.get("source_outcome_id"),
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
        "proposal_record_preview": candidate.get("proposal_record_preview", {}),
        "operation_ledger_preview": candidate.get("operation_ledger_preview", {}),
        "target_paths_preview": candidate.get("target_paths_preview", {}),
        "payload_preview": candidate.get("payload_preview"),
        "source_pattern_ids": list(candidate.get("source_pattern_ids", [])),
        "source_fact_ids": list(candidate.get("source_fact_ids", [])),
        "request_validation": candidate.get("request_validation", {}),
        "policy": candidate.get("policy", {}),
    }
    return build_stable_digest("memory-human-approval-token-review-outcome:v0.1", identity)


def _contains_any(values: list[str], targets: tuple[str, ...]) -> bool:
    return any(target in values for target in targets)


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
