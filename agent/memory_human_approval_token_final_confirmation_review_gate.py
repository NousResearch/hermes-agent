from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_final_confirmation_request import (
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED,
    explain_human_approval_token_final_confirmation_request,
    recommend_human_approval_token_final_confirmation_request_action,
    validate_human_approval_token_final_confirmation_request,
)
from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_forbidden_true_keys_false_or_absent,
    validate_policy_flags,
)


MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_GATE_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_KIND = (
    "memory_human_approval_token_final_confirmation_review_outcome_candidate"
)
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_STATUS = (
    "final_confirmation_review_outcome_candidate"
)
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_ROUTING = (
    "manual_token_write_still_required_after_final_confirmation"
)
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_SUPPORTED_OUTCOMES = {
    "confirm_token_write",
    "request_changes",
    "reject",
    "defer",
}
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_final_confirmation_review_outcomes_only": True,
    "issues_real_approval_tokens": False,
    "persists_approvals": False,
    "creates_real_proposals": False,
    "writes_proposal_files": False,
    "writes_operation_ledger": False,
    "writes_token_files": False,
    "writes_approval_audit": False,
    "applies_proposals": False,
    "submits_to_governance": False,
    "converts_to_real_proposal": False,
}

_DEFAULT_CONFIRMER = "hermes_memory_human_approval_token_final_confirmation_review_gate_v0.1"
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
    "writes_token_files",
    "writes_approval_audit",
    "converted_to_real_proposal",
)
_PREVIEW_FIELDS = (
    "approval_token_record_preview",
    "approval_audit_record_preview",
    "token_target_paths_preview",
    "proposal_record_preview",
    "operation_ledger_preview",
    "target_paths_preview",
)


def create_human_approval_token_final_confirmation_review_outcome(
    request_candidate: Mapping[str, Any],
    confirmer: str | None = None,
    outcome: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only final human confirmation review outcome candidate."""
    source = deepcopy(dict(request_candidate))
    final_confirmation_request_validation = validate_human_approval_token_final_confirmation_request(source)
    evaluated = evaluate_human_approval_token_final_confirmation_request(source, confirmer=confirmer)
    selected_outcome = outcome if outcome is not None else evaluated["outcome"]
    selected_rationale = rationale if rationale is not None else evaluated["rationale"]

    outcome_candidate = {
        "review_outcome_id": None,
        "review_outcome_kind": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_KIND,
        "review_outcome_status": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_STATUS,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_ROUTING,
        "source_final_confirmation_request_id": source.get("request_id"),
        "source_token_write_lock_gate_id": source.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": source.get("source_token_issuance_dry_run_id"),
        "source_token_issuance_plan_id": source.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": source.get("source_review_outcome_id"),
        "source_request_id": source.get("source_request_id"),
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
        "confirmer": confirmer if confirmer is not None else _DEFAULT_CONFIRMER,
        "outcome": selected_outcome,
        "rationale": selected_rationale,
        "approval_token_record_preview": deepcopy(source.get("approval_token_record_preview")),
        "approval_audit_record_preview": deepcopy(source.get("approval_audit_record_preview")),
        "token_target_paths_preview": deepcopy(source.get("token_target_paths_preview")),
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "final_confirmation_request_validation": deepcopy(final_confirmation_request_validation),
        "review_outcome_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_final_confirmation_request_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY),
    }
    outcome_candidate["review_outcome_id"] = _review_outcome_id(outcome_candidate)
    outcome_candidate["review_outcome_validation"] = (
        validate_human_approval_token_final_confirmation_review_outcome(outcome_candidate)
    )
    outcome_candidate["next_step_recommendation"] = (
        recommend_human_approval_token_final_confirmation_review_action(outcome_candidate)
    )
    return outcome_candidate


def evaluate_human_approval_token_final_confirmation_request(
    request_candidate: Mapping[str, Any],
    confirmer: str | None = None,
) -> dict[str, Any]:
    validation = validate_human_approval_token_final_confirmation_request(request_candidate)
    outcome, rationale = _evaluated_outcome(request_candidate, validation)
    return {
        "confirmer": confirmer if confirmer is not None else _DEFAULT_CONFIRMER,
        "outcome": outcome,
        "rationale": rationale,
        "validation": validation,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_ROUTING,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY),
    }


def validate_human_approval_token_final_confirmation_review_outcome(
    outcome_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(outcome_candidate.get("policy"))
    source_snapshot = _as_dict(outcome_candidate.get("source_final_confirmation_request_snapshot"))
    expected_request_validation = validate_human_approval_token_final_confirmation_request(source_snapshot)

    for key in (
        "review_outcome_id",
        "review_outcome_kind",
        "review_outcome_status",
        "routing",
        "source_final_confirmation_request_id",
        "source_token_write_lock_gate_id",
        "source_token_issuance_dry_run_id",
        "source_token_issuance_plan_id",
        "source_review_outcome_id",
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
        "confirmer",
        "outcome",
        "rationale",
        "approval_token_record_preview",
        "approval_audit_record_preview",
        "token_target_paths_preview",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "final_confirmation_request_validation",
        "review_outcome_validation",
        "next_step_recommendation",
        "source_final_confirmation_request_snapshot",
        "policy",
    ):
        if key not in outcome_candidate:
            errors.append(f"missing_{key}")
    if outcome_candidate.get("review_outcome_kind") != MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_KIND:
        errors.append("review_outcome_kind_must_be_memory_human_approval_token_final_confirmation_review_outcome_candidate")
    if outcome_candidate.get("review_outcome_status") != MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_STATUS:
        errors.append("review_outcome_status_must_be_final_confirmation_review_outcome_candidate")
    if outcome_candidate.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_ROUTING:
        errors.append("routing_must_require_manual_token_write_still_required_after_final_confirmation")
    if outcome_candidate.get("outcome") not in MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_SUPPORTED_OUTCOMES:
        errors.append("outcome_must_be_supported")
    if not isinstance(outcome_candidate.get("confirmer"), str) or not outcome_candidate.get("confirmer"):
        errors.append("confirmer_must_be_non_empty_string")
    if not isinstance(outcome_candidate.get("rationale"), str) or not outcome_candidate.get("rationale"):
        errors.append("rationale_must_be_non_empty_string")
    if outcome_candidate.get("final_confirmation_request_validation") != expected_request_validation:
        errors.append("final_confirmation_request_validation_must_match_source_final_confirmation_request_snapshot")
    if outcome_candidate.get("source_final_confirmation_request_id") != source_snapshot.get("request_id"):
        errors.append("source_final_confirmation_request_id_must_match_source_snapshot")
    for source_key in (
        "source_token_write_lock_gate_id",
        "source_token_issuance_dry_run_id",
        "source_token_issuance_plan_id",
        "source_review_outcome_id",
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
    ):
        if outcome_candidate.get(source_key) != source_snapshot.get(source_key):
            errors.append(f"{source_key}_must_match_source_snapshot")
    for field in _PREVIEW_FIELDS + ("payload_preview",):
        if field in outcome_candidate and field in source_snapshot and outcome_candidate.get(field) != source_snapshot.get(field):
            errors.append(f"{field}_must_match_source_final_confirmation_request_snapshot")
    if outcome_candidate.get("source_pattern_ids") != list(source_snapshot.get("source_pattern_ids", []) or []):
        errors.append("source_pattern_ids_must_match_source_final_confirmation_request_snapshot")
    if outcome_candidate.get("source_fact_ids") != list(source_snapshot.get("source_fact_ids", []) or []):
        errors.append("source_fact_ids_must_match_source_final_confirmation_request_snapshot")
    errors.extend(validate_forbidden_true_keys_false_or_absent(outcome_candidate, _FORBIDDEN_TRUE_KEYS))
    errors.extend(validate_policy_flags(policy, MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY))

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_final_confirmation_review_outcome(
    outcome_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_final_confirmation_review_outcome(outcome_candidate)
    source_snapshot = _as_dict(outcome_candidate.get("source_final_confirmation_request_snapshot"))
    return {
        "review_outcome_id": outcome_candidate.get("review_outcome_id"),
        "review_outcome_kind": outcome_candidate.get("review_outcome_kind"),
        "review_outcome_status": outcome_candidate.get("review_outcome_status"),
        "routing": outcome_candidate.get("routing"),
        "source_final_confirmation_request_id": outcome_candidate.get("source_final_confirmation_request_id"),
        "block_id": outcome_candidate.get("block_id"),
        "block_type": outcome_candidate.get("block_type"),
        "project_scope": outcome_candidate.get("project_scope"),
        "confirmer": outcome_candidate.get("confirmer"),
        "outcome": outcome_candidate.get("outcome"),
        "rationale": outcome_candidate.get("rationale"),
        "source_pattern_count": len(outcome_candidate.get("source_pattern_ids", []) or []),
        "source_fact_count": len(outcome_candidate.get("source_fact_ids", []) or []),
        "validation": validation,
        "final_confirmation_request_explanation": (
            explain_human_approval_token_final_confirmation_request(source_snapshot) if source_snapshot else {}
        ),
        "token_issued": False,
        "approved": False,
        "persisted": False,
        "submitted": False,
        "written": False,
        "created_real_proposal": False,
        "created_operation_event": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "writes_token_files": False,
        "writes_approval_audit": False,
        "converted_to_real_proposal": False,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY),
    }


def recommend_human_approval_token_final_confirmation_review_action(
    outcome_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_final_confirmation_review_outcome(outcome_candidate)
    source_snapshot = _as_dict(outcome_candidate.get("source_final_confirmation_request_snapshot"))
    if validation["valid"] and outcome_candidate.get("outcome") == "confirm_token_write":
        action = "route_to_manual_token_write_without_issuing_token_in_review_gate"
        reason = "Final confirmation review outcome confirms manual token write can proceed separately; this candidate does not issue, persist, submit, or write anything."
    elif validation["valid"] and outcome_candidate.get("outcome") == "request_changes":
        action = "request_changes_before_manual_token_write"
        reason = "Final confirmation review found remediable request, preview, or source evidence gaps."
    elif validation["valid"] and outcome_candidate.get("outcome") == "reject":
        action = "reject_manual_token_write"
        reason = "Final confirmation review rejected the request candidate."
    else:
        action = "defer_manual_token_write"
        reason = "Final confirmation review outcome is deferred or invalid; no token write may proceed."
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_ROUTING,
        "validation": validation,
        "final_confirmation_request_recommendation": (
            recommend_human_approval_token_final_confirmation_request_action(source_snapshot)
            if source_snapshot
            else {}
        ),
        "creates_final_confirmation_review_outcomes_only": True,
        "issues_real_approval_tokens": False,
        "persists_approvals": False,
        "creates_real_proposals": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "writes_token_files": False,
        "writes_approval_audit": False,
        "applies_proposals": False,
        "submits_to_governance": False,
        "converts_to_real_proposal": False,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY),
    }


def summarize_human_approval_token_final_confirmation_review_outcomes(
    outcomes: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(
        outcomes,
        "review_outcome_status",
        type_key="outcome",
    )
    valid_count = 0
    invalid_count = 0
    for outcome in outcomes:
        validation = validate_human_approval_token_final_confirmation_review_outcome(outcome)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": candidate_summary["total"],
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "by_outcome": candidate_summary["by_type"],
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY),
    }


def _evaluated_outcome(request_candidate: Mapping[str, Any], validation: Mapping[str, Any]) -> tuple[str, str]:
    for field in _PREVIEW_FIELDS:
        if not isinstance(request_candidate.get(field), Mapping):
            return "request_changes", f"Final confirmation request is missing {field}."
    if not (request_candidate.get("source_pattern_ids") or request_candidate.get("source_fact_ids")):
        return "request_changes", "Final confirmation request is missing source evidence."
    if _preview_integrity_errors(request_candidate):
        return "request_changes", "Final confirmation request preview integrity failed."
    if request_candidate.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED:
        return "reject", "Final confirmation request candidate is locked."
    if validation.get("valid") is not True:
        return "reject", "Final confirmation request candidate is invalid."
    if request_candidate.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED:
        return "confirm_token_write", "Final confirmation request is valid, review-required, and preview/source evidence is intact."
    return "defer", "Final confirmation review gate reached an unknown condition."


def _preview_integrity_errors(container: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _PREVIEW_FIELDS:
        preview = container.get(field)
        if not isinstance(preview, Mapping):
            continue
        if preview.get("preview_only") is not True:
            errors.append(f"{field}_must_be_preview_only")
        for key, value in preview.items():
            if value is True and (
                key in _FORBIDDEN_TRUE_KEYS
                or key in {"created", "written", "token_issued", "approved", "persisted"}
                or "created" in key
                or "written" in key
                or key.startswith("writes_")
            ):
                errors.append(f"{field}_{key}_must_not_be_true")
    return errors


def _review_outcome_id(outcome_candidate: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_GATE_VERSION,
        "review_outcome_kind": outcome_candidate.get("review_outcome_kind"),
        "review_outcome_status": outcome_candidate.get("review_outcome_status"),
        "routing": outcome_candidate.get("routing"),
        "source_final_confirmation_request_id": outcome_candidate.get("source_final_confirmation_request_id"),
        "source_token_write_lock_gate_id": outcome_candidate.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": outcome_candidate.get("source_token_issuance_dry_run_id"),
        "source_token_issuance_plan_id": outcome_candidate.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": outcome_candidate.get("source_review_outcome_id"),
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
        "confirmer": outcome_candidate.get("confirmer"),
        "outcome": outcome_candidate.get("outcome"),
        "rationale": outcome_candidate.get("rationale"),
        "approval_token_record_preview": outcome_candidate.get("approval_token_record_preview", {}),
        "approval_audit_record_preview": outcome_candidate.get("approval_audit_record_preview", {}),
        "token_target_paths_preview": outcome_candidate.get("token_target_paths_preview", {}),
        "proposal_record_preview": outcome_candidate.get("proposal_record_preview", {}),
        "operation_ledger_preview": outcome_candidate.get("operation_ledger_preview", {}),
        "target_paths_preview": outcome_candidate.get("target_paths_preview", {}),
        "payload_preview": outcome_candidate.get("payload_preview"),
        "source_pattern_ids": list(outcome_candidate.get("source_pattern_ids", [])),
        "source_fact_ids": list(outcome_candidate.get("source_fact_ids", [])),
        "final_confirmation_request_validation": outcome_candidate.get("final_confirmation_request_validation", {}),
        "policy": outcome_candidate.get("policy", {}),
    }
    return build_stable_digest(
        "memory-human-approval-token-final-confirmation-review-outcome:v0.1",
        identity,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
