from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_forbidden_true_keys_false_or_absent,
    validate_policy_flags,
)
from agent.memory_real_proposal_write_lock_gate import (
    MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE,
    MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED,
    explain_real_proposal_write_lock_gate,
    recommend_real_proposal_write_lock_action,
    validate_real_proposal_write_lock_gate,
)


MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_KIND = "memory_human_approval_token_request_candidate"
MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED = "approval_token_review_required"
MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_ROUTING = "manual_human_approval_token_review_required"
MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_approval_token_requests_only": True,
    "issues_real_approval_tokens": False,
    "persists_approvals": False,
    "creates_real_proposals": False,
    "writes_proposal_files": False,
    "writes_operation_ledger": False,
    "applies_proposals": False,
    "submits_to_governance": False,
    "converts_to_real_proposal": False,
}

_DEFAULT_REQUESTER = "hermes_memory_human_approval_token_request_v0.1"
_PREVIEW_FIELDS = (
    "proposal_record_preview",
    "operation_ledger_preview",
    "target_paths_preview",
)
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
_LOCK_REASONS = {
    None,
    "invalid_write_lock_gate",
    "invalid_real_proposal_dry_run",
    "write_lock_gate_locked",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_source_evidence",
    "preview_integrity_failed",
}


def create_human_approval_token_request(
    write_lock_gate: Mapping[str, Any],
    requester: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only human approval token request candidate."""
    source = deepcopy(dict(write_lock_gate))
    write_lock_gate_validation = validate_real_proposal_write_lock_gate(source)
    lock_reason = _request_lock_reason(source, write_lock_gate_validation)
    request_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED
    )

    request = {
        "request_id": None,
        "request_kind": MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_KIND,
        "request_status": request_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_ROUTING,
        "lock_reason": lock_reason,
        "source_gate_id": source.get("gate_id"),
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
        "requester": requester if requester is not None else _DEFAULT_REQUESTER,
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "approval_request_checklist": _approval_request_checklist(),
        "approval_token_requirements": _approval_token_requirements(),
        "write_lock_gate_validation": deepcopy(write_lock_gate_validation),
        "request_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_write_lock_gate_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY),
    }
    request["request_id"] = _request_id(request)
    request["request_validation"] = validate_human_approval_token_request(request)
    request["next_step_recommendation"] = recommend_human_approval_token_request_action(request)
    return request


def validate_human_approval_token_request(request: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(request.get("policy"))
    source_snapshot = _as_dict(request.get("source_write_lock_gate_snapshot"))
    write_lock_gate_validation = _as_dict(request.get("write_lock_gate_validation"))
    expected_write_lock_gate_validation = validate_real_proposal_write_lock_gate(source_snapshot)
    expected_lock_reason = _request_lock_reason(source_snapshot, write_lock_gate_validation)
    preview_integrity_errors = _preview_integrity_errors(request)

    for key in (
        "request_id",
        "request_kind",
        "request_status",
        "routing",
        "lock_reason",
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
        "requester",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "approval_request_checklist",
        "approval_token_requirements",
        "write_lock_gate_validation",
        "request_validation",
        "next_step_recommendation",
        "source_write_lock_gate_snapshot",
        "policy",
    ):
        if key not in request:
            errors.append(f"missing_{key}")
    if request.get("request_kind") != MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_KIND:
        errors.append("request_kind_must_be_memory_human_approval_token_request_candidate")
    if request.get("request_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED,
    }:
        errors.append("request_status_must_be_supported")
    if request.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_ROUTING:
        errors.append("routing_must_require_manual_human_approval_token_review")
    if request.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if request.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_approval_token_request_checks")
    if expected_lock_reason is None and request.get("request_status") != MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED:
        errors.append("eligible_write_lock_gate_must_require_approval_token_review")
    if expected_lock_reason is not None and request.get("request_status") != MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED:
        errors.append("locked_or_invalid_write_lock_gate_must_lock_request")
    if not isinstance(request.get("requester"), str) or not request.get("requester"):
        errors.append("requester_must_be_non_empty_string")
    if write_lock_gate_validation != expected_write_lock_gate_validation:
        errors.append("write_lock_gate_validation_must_match_source_write_lock_gate_snapshot")
    if (
        write_lock_gate_validation.get("valid") is True
        and source_snapshot.get("gate_status") == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE
        and request.get("request_status") != MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED
    ):
        errors.append("eligible_write_lock_gate_must_not_lock_request")
    if source_snapshot.get("gate_status") == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED:
        expected_source_reason = source_snapshot.get("lock_reason") or "write_lock_gate_locked"
        if request.get("lock_reason") != expected_source_reason:
            errors.append("locked_write_lock_gate_reason_must_be_preserved")
    if not isinstance(request.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(request.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(request.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(request.get("source_pattern_ids"), list) and isinstance(request.get("source_fact_ids"), list):
        if not request.get("source_pattern_ids") and not request.get("source_fact_ids"):
            errors.append("missing_source_evidence")
    if not isinstance(request.get("approval_request_checklist"), list) or not request.get("approval_request_checklist"):
        errors.append("approval_request_checklist_must_be_non_empty_list")
    if not isinstance(request.get("approval_token_requirements"), list) or not request.get("approval_token_requirements"):
        errors.append("approval_token_requirements_must_be_non_empty_list")
    errors.extend(preview_integrity_errors)
    errors.extend(validate_forbidden_true_keys_false_or_absent(request, _FORBIDDEN_TRUE_KEYS))
    errors.extend(validate_policy_flags(policy, MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY))

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_request(request: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_request(request)
    source_snapshot = _as_dict(request.get("source_write_lock_gate_snapshot"))
    return {
        "request_id": request.get("request_id"),
        "request_kind": request.get("request_kind"),
        "request_status": request.get("request_status"),
        "routing": request.get("routing"),
        "lock_reason": request.get("lock_reason"),
        "source_gate_id": request.get("source_gate_id"),
        "source_dry_run_id": request.get("source_dry_run_id"),
        "source_plan_id": request.get("source_plan_id"),
        "source_outcome_id": request.get("source_outcome_id"),
        "source_packet_id": request.get("source_packet_id"),
        "source_submission_id": request.get("source_submission_id"),
        "source_draft_id": request.get("source_draft_id"),
        "source_decision_id": request.get("source_decision_id"),
        "source_queue_item_id": request.get("source_queue_item_id"),
        "block_id": request.get("block_id"),
        "block_type": request.get("block_type"),
        "project_scope": request.get("project_scope"),
        "requester": request.get("requester"),
        "source_pattern_count": len(request.get("source_pattern_ids", []) or []),
        "source_fact_count": len(request.get("source_fact_ids", []) or []),
        "approval_request_checklist": deepcopy(request.get("approval_request_checklist")),
        "approval_token_requirements": deepcopy(request.get("approval_token_requirements")),
        "validation": validation,
        "write_lock_gate_explanation": explain_real_proposal_write_lock_gate(source_snapshot) if source_snapshot else {},
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY),
    }


def recommend_human_approval_token_request_action(request: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_request(request)
    source_snapshot = _as_dict(request.get("source_write_lock_gate_snapshot"))
    if validation["valid"] and request.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED:
        action = "perform_manual_human_approval_token_review_without_issuing_token"
        reason = "Request candidate is ready for manual human review; it does not issue or persist an approval token."
    else:
        action = "keep_approval_token_request_locked"
        reason = _recommendation_reason(request.get("lock_reason"), validation)
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_ROUTING,
        "validation": validation,
        "write_lock_gate_recommendation": recommend_real_proposal_write_lock_action(source_snapshot) if source_snapshot else {},
        "creates_approval_token_requests_only": True,
        "issues_real_approval_tokens": False,
        "persists_approvals": False,
        "creates_real_proposals": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "applies_proposals": False,
        "submits_to_governance": False,
        "converts_to_real_proposal": False,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY),
    }


def summarize_human_approval_token_requests(
    requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(requests, "request_status")
    locked_count = 0
    review_required_count = 0
    valid_count = 0
    invalid_count = 0
    for request in requests:
        if request.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED:
            locked_count += 1
        if request.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED:
            review_required_count += 1
        validation = validate_human_approval_token_request(request)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(requests),
        "locked_count": locked_count,
        "review_required_count": review_required_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY),
    }


def _request_lock_reason(source: Mapping[str, Any], write_lock_gate_validation: Mapping[str, Any]) -> str | None:
    if not isinstance(source.get("proposal_record_preview"), Mapping):
        return "missing_proposal_record_preview"
    if not isinstance(source.get("operation_ledger_preview"), Mapping):
        return "missing_operation_ledger_preview"
    if not isinstance(source.get("target_paths_preview"), Mapping):
        return "missing_target_paths_preview"
    if not (source.get("source_pattern_ids") or source.get("source_fact_ids")):
        return "missing_source_evidence"
    if _preview_integrity_errors(source):
        return "preview_integrity_failed"
    if write_lock_gate_validation.get("valid") is not True:
        return "invalid_write_lock_gate"
    if source.get("gate_status") == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED:
        return source.get("lock_reason") or "write_lock_gate_locked"
    if source.get("gate_status") != MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE:
        return "invalid_write_lock_gate"
    return None


def _preview_integrity_errors(container: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _PREVIEW_FIELDS:
        preview = container.get(field)
        if not isinstance(preview, Mapping):
            continue
        if preview.get("preview_only") is not True:
            errors.append(f"{field}_must_be_preview_only")
        for key, value in preview.items():
            if value is True and ("written" in key or "created" in key):
                errors.append(f"{field}_{key}_must_not_be_true")
    return errors


def _approval_request_checklist() -> list[dict[str, str]]:
    return [
        {
            "id": "verify_write_lock_gate_validation",
            "description": "Confirm the source write-lock gate is valid and eligible for human approval token review.",
        },
        {
            "id": "verify_preview_artifacts_only",
            "description": "Confirm proposal, operation-ledger, and target-path artifacts are previews only.",
        },
        {
            "id": "verify_no_token_or_approval_persistence",
            "description": "Confirm this request does not issue tokens or persist approvals.",
        },
        {
            "id": "verify_no_real_proposal_or_ledger_write",
            "description": "Confirm this request creates no real proposal records, files, or operation-ledger events.",
        },
        {
            "id": "route_to_manual_human_approval_token_review",
            "description": "Route the request candidate to manual human approval token review.",
        },
    ]


def _approval_token_requirements() -> list[dict[str, str]]:
    return [
        {
            "id": "human_reviewer_required",
            "description": "A human reviewer must inspect the request before any separate approval token flow.",
        },
        {
            "id": "approval_token_must_be_external_to_request",
            "description": "Any real approval token must be issued outside this read-only request candidate.",
        },
        {
            "id": "real_write_requires_later_governed_step",
            "description": "Any real proposal write requires a later governed operation outside this module.",
        },
    ]


def _recommendation_reason(lock_reason: Any, validation: Mapping[str, Any]) -> str:
    if lock_reason == "invalid_write_lock_gate":
        return "Source write-lock gate is invalid."
    if lock_reason == "write_lock_gate_locked":
        return "Source write-lock gate remains locked."
    if lock_reason == "missing_proposal_record_preview":
        return "Approval token request is missing the proposal record preview."
    if lock_reason == "missing_operation_ledger_preview":
        return "Approval token request is missing the operation-ledger preview."
    if lock_reason == "missing_target_paths_preview":
        return "Approval token request is missing the target paths preview."
    if lock_reason == "missing_source_evidence":
        return "Approval token request is missing source pattern or fact evidence."
    if lock_reason == "preview_integrity_failed":
        return "Approval token request preview artifacts are not preview-only or include written/created flags."
    if validation.get("errors"):
        return "Approval token request violates the v0.1 validation contract."
    return "Approval token request remains locked until the source write-lock gate is eligible."


def _request_id(request: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_VERSION,
        "request_kind": request.get("request_kind"),
        "request_status": request.get("request_status"),
        "routing": request.get("routing"),
        "lock_reason": request.get("lock_reason"),
        "source_gate_id": request.get("source_gate_id"),
        "source_dry_run_id": request.get("source_dry_run_id"),
        "source_plan_id": request.get("source_plan_id"),
        "source_outcome_id": request.get("source_outcome_id"),
        "source_packet_id": request.get("source_packet_id"),
        "source_submission_id": request.get("source_submission_id"),
        "source_draft_id": request.get("source_draft_id"),
        "source_decision_id": request.get("source_decision_id"),
        "source_queue_item_id": request.get("source_queue_item_id"),
        "block_id": request.get("block_id"),
        "block_type": request.get("block_type"),
        "project_scope": request.get("project_scope"),
        "requester": request.get("requester"),
        "proposal_record_preview": request.get("proposal_record_preview", {}),
        "operation_ledger_preview": request.get("operation_ledger_preview", {}),
        "target_paths_preview": request.get("target_paths_preview", {}),
        "payload_preview": request.get("payload_preview"),
        "source_pattern_ids": list(request.get("source_pattern_ids", [])),
        "source_fact_ids": list(request.get("source_fact_ids", [])),
        "approval_request_checklist": request.get("approval_request_checklist", []),
        "approval_token_requirements": request.get("approval_token_requirements", []),
        "write_lock_gate_validation": request.get("write_lock_gate_validation", {}),
        "policy": request.get("policy", {}),
    }
    return build_stable_digest("memory-human-approval-token-request:v0.1", identity)


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
