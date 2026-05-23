from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_write_lock_gate import (
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE,
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED,
    explain_human_approval_token_write_lock_gate,
    recommend_human_approval_token_write_lock_action,
    validate_human_approval_token_write_lock_gate,
)


MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_KIND = (
    "memory_human_approval_token_final_confirmation_request_candidate"
)
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED = (
    "final_confirmation_review_required"
)
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_ROUTING = (
    "manual_final_confirmation_review_required_before_any_token_write"
)
MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_final_confirmation_requests_only": True,
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

_DEFAULT_REQUESTER = "hermes_memory_human_approval_token_final_confirmation_request_v0.1"
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
_LOCK_REASONS = {
    None,
    "token_write_lock_gate_locked",
    "invalid_token_write_lock_gate",
    "missing_approval_token_record_preview",
    "missing_approval_audit_record_preview",
    "missing_token_target_paths_preview",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_source_evidence",
    "preview_integrity_failed",
    "token_issuance_dry_run_locked",
    "invalid_token_issuance_dry_run",
    "token_issuance_plan_locked",
    "token_review_requested_changes",
    "token_review_rejected",
    "token_review_deferred",
    "invalid_token_review_outcome",
    "missing_token_plan_controls",
}


def create_human_approval_token_final_confirmation_request(
    token_write_lock_gate: Mapping[str, Any],
    requester: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only final human confirmation request candidate."""
    source = deepcopy(dict(token_write_lock_gate))
    token_write_lock_gate_validation = validate_human_approval_token_write_lock_gate(source)
    lock_reason = _request_lock_reason(source, token_write_lock_gate_validation)
    request_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    )

    request = {
        "request_id": None,
        "request_kind": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_KIND,
        "request_status": request_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_ROUTING,
        "lock_reason": lock_reason,
        "source_token_write_lock_gate_id": source.get("gate_id"),
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
        "requester": requester if requester is not None else _DEFAULT_REQUESTER,
        "approval_token_record_preview": deepcopy(source.get("approval_token_record_preview")),
        "approval_audit_record_preview": deepcopy(source.get("approval_audit_record_preview")),
        "token_target_paths_preview": deepcopy(source.get("token_target_paths_preview")),
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "final_confirmation_checklist": _final_confirmation_checklist(),
        "token_write_lock_gate_validation": deepcopy(token_write_lock_gate_validation),
        "request_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_token_write_lock_gate_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY),
    }
    request["request_id"] = _request_id(request)
    request["request_validation"] = validate_human_approval_token_final_confirmation_request(request)
    request["next_step_recommendation"] = recommend_human_approval_token_final_confirmation_request_action(request)
    return request


def validate_human_approval_token_final_confirmation_request(request: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(request.get("policy"))
    source_snapshot = _as_dict(request.get("source_token_write_lock_gate_snapshot"))
    token_write_lock_gate_validation = _as_dict(request.get("token_write_lock_gate_validation"))
    expected_token_write_lock_gate_validation = validate_human_approval_token_write_lock_gate(source_snapshot)
    expected_lock_reason = _request_lock_reason(source_snapshot, token_write_lock_gate_validation)
    preview_integrity_errors = _preview_integrity_errors(request)

    for key in (
        "request_id",
        "request_kind",
        "request_status",
        "routing",
        "lock_reason",
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
        "requester",
        "approval_token_record_preview",
        "approval_audit_record_preview",
        "token_target_paths_preview",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "final_confirmation_checklist",
        "token_write_lock_gate_validation",
        "request_validation",
        "next_step_recommendation",
        "source_token_write_lock_gate_snapshot",
        "policy",
    ):
        if key not in request:
            errors.append(f"missing_{key}")
    if request.get("request_kind") != MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_KIND:
        errors.append("request_kind_must_be_memory_human_approval_token_final_confirmation_request_candidate")
    if request.get("request_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED,
    }:
        errors.append("request_status_must_be_supported")
    if request.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_ROUTING:
        errors.append("routing_must_require_manual_final_confirmation_review_before_any_token_write")
    if request.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if request.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_final_confirmation_request_checks")
    if expected_lock_reason is None and request.get("request_status") != MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED:
        errors.append("valid_token_write_lock_gate_must_require_final_confirmation_review")
    if expected_lock_reason is not None and request.get("request_status") != MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED:
        errors.append("invalid_or_locked_token_write_lock_gate_must_lock_request")
    if not isinstance(request.get("requester"), str) or not request.get("requester"):
        errors.append("requester_must_be_non_empty_string")
    if token_write_lock_gate_validation != expected_token_write_lock_gate_validation:
        errors.append("token_write_lock_gate_validation_must_match_source_token_write_lock_gate_snapshot")
    if source_snapshot.get("gate_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE,
        MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED,
    }:
        errors.append("source_token_write_lock_gate_status_must_be_supported")
    if request.get("source_token_write_lock_gate_id") != source_snapshot.get("gate_id"):
        errors.append("source_token_write_lock_gate_id_must_match_source_snapshot")
    if not isinstance(request.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(request.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(request.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(request.get("source_pattern_ids"), list) and isinstance(request.get("source_fact_ids"), list):
        if not request.get("source_pattern_ids") and not request.get("source_fact_ids"):
            errors.append("missing_source_evidence")
    if request.get("final_confirmation_checklist") != _final_confirmation_checklist():
        errors.append("final_confirmation_checklist_must_match_v0_1_deterministic_checks")
    for field in _PREVIEW_FIELDS:
        if not isinstance(request.get(field), Mapping):
            errors.append(f"missing_{field}")
    errors.extend(preview_integrity_errors)
    for field in (
        "approval_token_record_preview",
        "approval_audit_record_preview",
        "token_target_paths_preview",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
    ):
        if field in request and field in source_snapshot and request.get(field) != source_snapshot.get(field):
            errors.append(f"{field}_must_match_source_token_write_lock_gate_snapshot")
    if request.get("source_pattern_ids") != list(source_snapshot.get("source_pattern_ids", []) or []):
        errors.append("source_pattern_ids_must_match_source_token_write_lock_gate_snapshot")
    if request.get("source_fact_ids") != list(source_snapshot.get("source_fact_ids", []) or []):
        errors.append("source_fact_ids_must_match_source_token_write_lock_gate_snapshot")
    for forbidden_key in _FORBIDDEN_TRUE_KEYS:
        if request.get(forbidden_key) is True:
            errors.append(f"{forbidden_key}_must_be_false_or_absent")
    for key, expected in MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY.items():
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_final_confirmation_request(request: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_final_confirmation_request(request)
    source_snapshot = _as_dict(request.get("source_token_write_lock_gate_snapshot"))
    return {
        "request_id": request.get("request_id"),
        "request_kind": request.get("request_kind"),
        "request_status": request.get("request_status"),
        "routing": request.get("routing"),
        "lock_reason": request.get("lock_reason"),
        "source_token_write_lock_gate_id": request.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": request.get("source_token_issuance_dry_run_id"),
        "source_token_issuance_plan_id": request.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": request.get("source_review_outcome_id"),
        "source_request_id": request.get("source_request_id"),
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
        "approval_token_record_preview": deepcopy(request.get("approval_token_record_preview")),
        "approval_audit_record_preview": deepcopy(request.get("approval_audit_record_preview")),
        "token_target_paths_preview": deepcopy(request.get("token_target_paths_preview")),
        "proposal_record_preview": deepcopy(request.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(request.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(request.get("target_paths_preview")),
        "final_confirmation_checklist": deepcopy(request.get("final_confirmation_checklist")),
        "validation": validation,
        "token_write_lock_gate_explanation": (
            explain_human_approval_token_write_lock_gate(source_snapshot) if source_snapshot else {}
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY),
    }


def recommend_human_approval_token_final_confirmation_request_action(request: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_final_confirmation_request(request)
    source_snapshot = _as_dict(request.get("source_token_write_lock_gate_snapshot"))
    if (
        validation["valid"]
        and request.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED
    ):
        action = "route_to_manual_final_confirmation_review_before_any_token_write"
        reason = "Final confirmation request is review-required only; it does not issue, approve, persist, submit, or write approval tokens."
    else:
        action = "keep_final_confirmation_request_locked"
        reason = _recommendation_reason(request.get("lock_reason"), validation)
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_ROUTING,
        "validation": validation,
        "token_write_lock_gate_recommendation": (
            recommend_human_approval_token_write_lock_action(source_snapshot) if source_snapshot else {}
        ),
        "creates_final_confirmation_requests_only": True,
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY),
    }


def summarize_human_approval_token_final_confirmation_requests(
    requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    by_block_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_lock_reason: dict[str, int] = {}
    locked_count = 0
    final_confirmation_review_required_count = 0
    valid_count = 0
    invalid_count = 0
    for request in requests:
        block_type = str(request.get("block_type"))
        by_block_type[block_type] = by_block_type.get(block_type, 0) + 1
        status = str(request.get("request_status"))
        by_status[status] = by_status.get(status, 0) + 1
        reason = str(request.get("lock_reason"))
        by_lock_reason[reason] = by_lock_reason.get(reason, 0) + 1
        if request.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED:
            locked_count += 1
        if request.get("request_status") == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED:
            final_confirmation_review_required_count += 1
        validation = validate_human_approval_token_final_confirmation_request(request)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(requests),
        "locked_count": locked_count,
        "final_confirmation_review_required_count": final_confirmation_review_required_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "by_block_type": dict(sorted(by_block_type.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_lock_reason": dict(sorted(by_lock_reason.items())),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY),
    }


def _request_lock_reason(
    source: Mapping[str, Any],
    token_write_lock_gate_validation: Mapping[str, Any],
) -> str | None:
    if source.get("gate_status") == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED:
        return source.get("lock_reason") or "token_write_lock_gate_locked"
    if not isinstance(source.get("approval_token_record_preview"), Mapping):
        return "missing_approval_token_record_preview"
    if not isinstance(source.get("approval_audit_record_preview"), Mapping):
        return "missing_approval_audit_record_preview"
    if not isinstance(source.get("token_target_paths_preview"), Mapping):
        return "missing_token_target_paths_preview"
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
    if token_write_lock_gate_validation.get("valid") is not True:
        return "invalid_token_write_lock_gate"
    if source.get("gate_status") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE:
        return "invalid_token_write_lock_gate"
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
            if value is True and (
                key in _FORBIDDEN_TRUE_KEYS
                or key in {"created", "written", "token_issued", "approved", "persisted"}
                or "created" in key
                or "written" in key
                or key.startswith("writes_")
            ):
                errors.append(f"{field}_{key}_must_not_be_true")
    return errors


def _final_confirmation_checklist() -> list[dict[str, str]]:
    return [
        {
            "id": "verify_token_write_lock_gate_validation",
            "description": "Confirm the source token write-lock gate is valid and eligible for final human confirmation.",
        },
        {
            "id": "verify_token_previews_only",
            "description": "Confirm approval token, approval audit, token-path, proposal, ledger, and target-path previews are preview-only.",
        },
        {
            "id": "verify_no_token_approval_or_write_flags",
            "description": "Confirm previews do not mark tokens as issued, approvals as persisted, records as created, or files as written.",
        },
        {
            "id": "verify_payload_and_source_evidence",
            "description": "Confirm payload preview and source pattern or fact evidence are present.",
        },
        {
            "id": "require_manual_final_confirmation_review",
            "description": "Route to manual final confirmation review before any approval token write.",
        },
    ]


def _recommendation_reason(lock_reason: Any, validation: Mapping[str, Any]) -> str:
    if lock_reason == "invalid_token_write_lock_gate":
        return "Source token write-lock gate candidate is invalid."
    if lock_reason in {
        "token_write_lock_gate_locked",
        "token_issuance_dry_run_locked",
        "invalid_token_issuance_dry_run",
        "token_issuance_plan_locked",
        "token_review_requested_changes",
        "token_review_rejected",
        "token_review_deferred",
        "invalid_token_review_outcome",
        "missing_token_plan_controls",
    }:
        return f"Final confirmation request inherits the source token write-lock gate lock: {lock_reason}."
    if lock_reason in {
        "missing_approval_token_record_preview",
        "missing_approval_audit_record_preview",
        "missing_token_target_paths_preview",
        "missing_proposal_record_preview",
        "missing_operation_ledger_preview",
        "missing_target_paths_preview",
        "missing_source_evidence",
    }:
        return f"Final confirmation request is locked because {lock_reason}."
    if lock_reason == "preview_integrity_failed":
        return "Final confirmation request preview artifacts are not preview-only or include issuance, approval, created, persisted, or written flags."
    if validation.get("errors"):
        return "Final confirmation request violates the v0.1 validation contract."
    return "Approval token writing remains locked until a separate final human confirmation review completes."


def _request_id(request: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_VERSION,
        "request_kind": request.get("request_kind"),
        "request_status": request.get("request_status"),
        "routing": request.get("routing"),
        "lock_reason": request.get("lock_reason"),
        "source_token_write_lock_gate_id": request.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": request.get("source_token_issuance_dry_run_id"),
        "source_token_issuance_plan_id": request.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": request.get("source_review_outcome_id"),
        "source_request_id": request.get("source_request_id"),
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
        "approval_token_record_preview": request.get("approval_token_record_preview", {}),
        "approval_audit_record_preview": request.get("approval_audit_record_preview", {}),
        "token_target_paths_preview": request.get("token_target_paths_preview", {}),
        "proposal_record_preview": request.get("proposal_record_preview", {}),
        "operation_ledger_preview": request.get("operation_ledger_preview", {}),
        "target_paths_preview": request.get("target_paths_preview", {}),
        "payload_preview": request.get("payload_preview"),
        "source_pattern_ids": list(request.get("source_pattern_ids", [])),
        "source_fact_ids": list(request.get("source_fact_ids", [])),
        "final_confirmation_checklist": request.get("final_confirmation_checklist", []),
        "token_write_lock_gate_validation": request.get("token_write_lock_gate_validation", {}),
        "policy": request.get("policy", {}),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"memory-human-approval-token-final-confirmation-request:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
