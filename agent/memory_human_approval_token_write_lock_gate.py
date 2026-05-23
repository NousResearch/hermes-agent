from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_issuance_dry_run import (
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED,
    explain_human_approval_token_issuance_dry_run,
    recommend_human_approval_token_issuance_dry_run_action,
    validate_human_approval_token_issuance_dry_run,
)


MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_KIND = "memory_human_approval_token_write_lock_gate_candidate"
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE = "eligible_for_final_human_confirmation"
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ROUTING = (
    "final_human_confirmation_required_before_any_token_write"
)
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_token_write_lock_candidates_only": True,
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

_DEFAULT_OPERATOR = "hermes_memory_human_approval_token_write_lock_gate_v0.1"
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
    "token_issuance_dry_run_locked",
    "invalid_token_issuance_dry_run",
    "missing_approval_token_record_preview",
    "missing_approval_audit_record_preview",
    "missing_token_target_paths_preview",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_source_evidence",
    "preview_integrity_failed",
    "token_issuance_plan_locked",
    "token_review_requested_changes",
    "token_review_rejected",
    "token_review_deferred",
    "invalid_token_review_outcome",
    "missing_token_plan_controls",
}


def create_human_approval_token_write_lock_gate(
    token_dry_run: Mapping[str, Any],
    operator: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only token write-lock gate candidate."""
    source = deepcopy(dict(token_dry_run))
    token_dry_run_validation = validate_human_approval_token_issuance_dry_run(source)
    lock_reason = _write_lock_reason(source, token_dry_run_validation)
    gate_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    )

    gate = {
        "gate_id": None,
        "gate_kind": MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_KIND,
        "gate_status": gate_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ROUTING,
        "lock_reason": lock_reason,
        "source_token_issuance_dry_run_id": source.get("dry_run_id"),
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
        "operator": operator if operator is not None else _DEFAULT_OPERATOR,
        "approval_token_record_preview": deepcopy(source.get("approval_token_record_preview")),
        "approval_audit_record_preview": deepcopy(source.get("approval_audit_record_preview")),
        "token_target_paths_preview": deepcopy(source.get("token_target_paths_preview")),
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "write_lock_checklist": _write_lock_checklist(),
        "token_dry_run_validation": deepcopy(token_dry_run_validation),
        "gate_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_token_issuance_dry_run_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY),
    }
    gate["gate_id"] = _gate_id(gate)
    gate["gate_validation"] = validate_human_approval_token_write_lock_gate(gate)
    gate["next_step_recommendation"] = recommend_human_approval_token_write_lock_action(gate)
    return gate


def validate_human_approval_token_write_lock_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(gate.get("policy"))
    source_snapshot = _as_dict(gate.get("source_token_issuance_dry_run_snapshot"))
    token_dry_run_validation = _as_dict(gate.get("token_dry_run_validation"))
    expected_token_dry_run_validation = validate_human_approval_token_issuance_dry_run(source_snapshot)
    expected_lock_reason = _write_lock_reason(source_snapshot, token_dry_run_validation)
    preview_integrity_errors = _preview_integrity_errors(gate)

    for key in (
        "gate_id",
        "gate_kind",
        "gate_status",
        "routing",
        "lock_reason",
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
        "operator",
        "approval_token_record_preview",
        "approval_audit_record_preview",
        "token_target_paths_preview",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "write_lock_checklist",
        "token_dry_run_validation",
        "gate_validation",
        "next_step_recommendation",
        "source_token_issuance_dry_run_snapshot",
        "policy",
    ):
        if key not in gate:
            errors.append(f"missing_{key}")
    if gate.get("gate_kind") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_KIND:
        errors.append("gate_kind_must_be_memory_human_approval_token_write_lock_gate_candidate")
    if gate.get("gate_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE,
        MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED,
    }:
        errors.append("gate_status_must_be_supported")
    if gate.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ROUTING:
        errors.append("routing_must_require_final_human_confirmation_before_any_token_write")
    if gate.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if gate.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_token_write_lock_checks")
    if expected_lock_reason is None and gate.get("gate_status") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE:
        errors.append("valid_token_dry_run_must_be_eligible_for_final_human_confirmation")
    if expected_lock_reason is not None and gate.get("gate_status") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED:
        errors.append("invalid_token_write_lock_check_must_be_locked")
    if not isinstance(gate.get("operator"), str) or not gate.get("operator"):
        errors.append("operator_must_be_non_empty_string")
    if token_dry_run_validation != expected_token_dry_run_validation:
        errors.append("token_dry_run_validation_must_match_source_token_issuance_dry_run_snapshot")
    if source_snapshot.get("dry_run_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED,
    }:
        errors.append("source_dry_run_status_must_be_supported")
    if gate.get("source_token_issuance_dry_run_id") != source_snapshot.get("dry_run_id"):
        errors.append("source_token_issuance_dry_run_id_must_match_source_snapshot")
    if not isinstance(gate.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(gate.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(gate.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(gate.get("source_pattern_ids"), list) and isinstance(gate.get("source_fact_ids"), list):
        if not gate.get("source_pattern_ids") and not gate.get("source_fact_ids"):
            errors.append("missing_source_evidence")
    if not isinstance(gate.get("write_lock_checklist"), list) or not gate.get("write_lock_checklist"):
        errors.append("write_lock_checklist_must_be_non_empty_list")
    if gate.get("write_lock_checklist") != _write_lock_checklist():
        errors.append("write_lock_checklist_must_match_v0_1_deterministic_checks")
    for field in _PREVIEW_FIELDS:
        if not isinstance(gate.get(field), Mapping):
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
        if field in gate and field in source_snapshot and gate.get(field) != source_snapshot.get(field):
            errors.append(f"{field}_must_match_source_token_issuance_dry_run_snapshot")
    if gate.get("source_pattern_ids") != list(source_snapshot.get("source_pattern_ids", []) or []):
        errors.append("source_pattern_ids_must_match_source_token_issuance_dry_run_snapshot")
    if gate.get("source_fact_ids") != list(source_snapshot.get("source_fact_ids", []) or []):
        errors.append("source_fact_ids_must_match_source_token_issuance_dry_run_snapshot")
    for forbidden_key in _FORBIDDEN_TRUE_KEYS:
        if gate.get(forbidden_key) is True:
            errors.append(f"{forbidden_key}_must_be_false_or_absent")
    for key, expected in MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY.items():
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_write_lock_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_write_lock_gate(gate)
    source_snapshot = _as_dict(gate.get("source_token_issuance_dry_run_snapshot"))
    return {
        "gate_id": gate.get("gate_id"),
        "gate_kind": gate.get("gate_kind"),
        "gate_status": gate.get("gate_status"),
        "routing": gate.get("routing"),
        "lock_reason": gate.get("lock_reason"),
        "source_token_issuance_dry_run_id": gate.get("source_token_issuance_dry_run_id"),
        "source_token_issuance_plan_id": gate.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": gate.get("source_review_outcome_id"),
        "source_request_id": gate.get("source_request_id"),
        "source_gate_id": gate.get("source_gate_id"),
        "source_dry_run_id": gate.get("source_dry_run_id"),
        "source_plan_id": gate.get("source_plan_id"),
        "source_outcome_id": gate.get("source_outcome_id"),
        "source_packet_id": gate.get("source_packet_id"),
        "source_submission_id": gate.get("source_submission_id"),
        "source_draft_id": gate.get("source_draft_id"),
        "source_decision_id": gate.get("source_decision_id"),
        "source_queue_item_id": gate.get("source_queue_item_id"),
        "block_id": gate.get("block_id"),
        "block_type": gate.get("block_type"),
        "project_scope": gate.get("project_scope"),
        "operator": gate.get("operator"),
        "source_pattern_count": len(gate.get("source_pattern_ids", []) or []),
        "source_fact_count": len(gate.get("source_fact_ids", []) or []),
        "approval_token_record_preview": deepcopy(gate.get("approval_token_record_preview")),
        "approval_audit_record_preview": deepcopy(gate.get("approval_audit_record_preview")),
        "token_target_paths_preview": deepcopy(gate.get("token_target_paths_preview")),
        "proposal_record_preview": deepcopy(gate.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(gate.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(gate.get("target_paths_preview")),
        "write_lock_checklist": deepcopy(gate.get("write_lock_checklist")),
        "validation": validation,
        "token_issuance_dry_run_explanation": (
            explain_human_approval_token_issuance_dry_run(source_snapshot) if source_snapshot else {}
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY),
    }


def recommend_human_approval_token_write_lock_action(gate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_write_lock_gate(gate)
    source_snapshot = _as_dict(gate.get("source_token_issuance_dry_run_snapshot"))
    if validation["valid"] and gate.get("gate_status") == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE:
        action = "request_separate_final_human_confirmation_before_token_write"
        reason = "Token write-lock gate is eligible only for a separate final human confirmation flow; it does not issue, persist, approve, submit, or write tokens."
    else:
        action = "keep_token_write_locked"
        reason = _recommendation_reason(gate.get("lock_reason"), validation)
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ROUTING,
        "validation": validation,
        "token_issuance_dry_run_recommendation": (
            recommend_human_approval_token_issuance_dry_run_action(source_snapshot) if source_snapshot else {}
        ),
        "creates_token_write_lock_candidates_only": True,
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY),
    }


def summarize_human_approval_token_write_lock_gates(
    gates: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    by_block_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_lock_reason: dict[str, int] = {}
    locked_count = 0
    eligible_count = 0
    valid_count = 0
    invalid_count = 0
    for gate in gates:
        block_type = str(gate.get("block_type"))
        by_block_type[block_type] = by_block_type.get(block_type, 0) + 1
        status = str(gate.get("gate_status"))
        by_status[status] = by_status.get(status, 0) + 1
        reason = str(gate.get("lock_reason"))
        by_lock_reason[reason] = by_lock_reason.get(reason, 0) + 1
        if gate.get("gate_status") == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED:
            locked_count += 1
        if gate.get("gate_status") == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE:
            eligible_count += 1
        validation = validate_human_approval_token_write_lock_gate(gate)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(gates),
        "locked_count": locked_count,
        "eligible_count": eligible_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "by_block_type": dict(sorted(by_block_type.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_lock_reason": dict(sorted(by_lock_reason.items())),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY),
    }


def _write_lock_reason(source: Mapping[str, Any], token_dry_run_validation: Mapping[str, Any]) -> str | None:
    if source.get("dry_run_status") == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED:
        return source.get("lock_reason") or "token_issuance_dry_run_locked"
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
    if token_dry_run_validation.get("valid") is not True:
        return "invalid_token_issuance_dry_run"
    if source.get("dry_run_status") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED:
        return "invalid_token_issuance_dry_run"
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


def _write_lock_checklist() -> list[dict[str, str]]:
    return [
        {
            "id": "verify_token_dry_run_validation",
            "description": "Confirm the source token issuance dry run is valid and still requires manual final preflight.",
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
            "id": "require_separate_final_human_confirmation",
            "description": "Require a separate final human confirmation flow before any token write.",
        },
    ]


def _recommendation_reason(lock_reason: Any, validation: Mapping[str, Any]) -> str:
    if lock_reason == "invalid_token_issuance_dry_run":
        return "Source token issuance dry-run candidate is invalid."
    if lock_reason in {
        "token_issuance_dry_run_locked",
        "token_issuance_plan_locked",
        "token_review_requested_changes",
        "token_review_rejected",
        "token_review_deferred",
        "invalid_token_review_outcome",
        "missing_token_plan_controls",
    }:
        return f"Token write-lock gate inherits the source dry-run lock: {lock_reason}."
    if lock_reason in {
        "missing_approval_token_record_preview",
        "missing_approval_audit_record_preview",
        "missing_token_target_paths_preview",
        "missing_proposal_record_preview",
        "missing_operation_ledger_preview",
        "missing_target_paths_preview",
        "missing_source_evidence",
    }:
        return f"Token write-lock gate is locked because {lock_reason}."
    if lock_reason == "preview_integrity_failed":
        return "Token write-lock gate preview artifacts are not preview-only or include issuance, approval, created, persisted, or written flags."
    if validation.get("errors"):
        return "Token write-lock gate violates the v0.1 validation contract."
    return "Token writing remains locked until a separate final human confirmation flow is completed."


def _gate_id(gate: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_VERSION,
        "gate_kind": gate.get("gate_kind"),
        "gate_status": gate.get("gate_status"),
        "routing": gate.get("routing"),
        "lock_reason": gate.get("lock_reason"),
        "source_token_issuance_dry_run_id": gate.get("source_token_issuance_dry_run_id"),
        "source_token_issuance_plan_id": gate.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": gate.get("source_review_outcome_id"),
        "source_request_id": gate.get("source_request_id"),
        "source_gate_id": gate.get("source_gate_id"),
        "source_dry_run_id": gate.get("source_dry_run_id"),
        "source_plan_id": gate.get("source_plan_id"),
        "source_outcome_id": gate.get("source_outcome_id"),
        "source_packet_id": gate.get("source_packet_id"),
        "source_submission_id": gate.get("source_submission_id"),
        "source_draft_id": gate.get("source_draft_id"),
        "source_decision_id": gate.get("source_decision_id"),
        "source_queue_item_id": gate.get("source_queue_item_id"),
        "block_id": gate.get("block_id"),
        "block_type": gate.get("block_type"),
        "project_scope": gate.get("project_scope"),
        "operator": gate.get("operator"),
        "approval_token_record_preview": gate.get("approval_token_record_preview", {}),
        "approval_audit_record_preview": gate.get("approval_audit_record_preview", {}),
        "token_target_paths_preview": gate.get("token_target_paths_preview", {}),
        "proposal_record_preview": gate.get("proposal_record_preview", {}),
        "operation_ledger_preview": gate.get("operation_ledger_preview", {}),
        "target_paths_preview": gate.get("target_paths_preview", {}),
        "payload_preview": gate.get("payload_preview"),
        "source_pattern_ids": list(gate.get("source_pattern_ids", [])),
        "source_fact_ids": list(gate.get("source_fact_ids", [])),
        "write_lock_checklist": gate.get("write_lock_checklist", []),
        "token_dry_run_validation": gate.get("token_dry_run_validation", {}),
        "policy": gate.get("policy", {}),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"memory-human-approval-token-write-lock-gate:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
