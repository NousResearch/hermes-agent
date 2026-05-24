from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_issuance_plan import (
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED,
    explain_human_approval_token_issuance_plan,
    recommend_human_approval_token_issuance_plan_action,
    validate_human_approval_token_issuance_plan,
)
from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_forbidden_true_keys_false_or_absent,
    validate_policy_flags,
)


MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_KIND = "memory_human_approval_token_issuance_dry_run_candidate"
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED = (
    "manual_token_issuance_final_preflight_required"
)
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_ROUTING = (
    "manual_final_human_confirmation_required_before_token_write"
)
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_token_issuance_dry_run_candidates_only": True,
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

_DEFAULT_OPERATOR = "hermes_memory_human_approval_token_issuance_dry_run_v0.1"
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
_LOCK_REASONS = {
    None,
    "token_issuance_plan_locked",
    "invalid_token_issuance_plan",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_source_evidence",
    "missing_token_plan_controls",
    "token_review_requested_changes",
    "token_review_rejected",
    "token_review_deferred",
    "invalid_token_review_outcome",
}


def create_human_approval_token_issuance_dry_run(
    plan: Mapping[str, Any],
    operator: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only human approval token issuance dry-run candidate."""
    source = deepcopy(dict(plan))
    plan_validation = validate_human_approval_token_issuance_plan(source)
    lock_reason = _dry_run_lock_reason(source, plan_validation)
    dry_run_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
    )

    dry_run = {
        "dry_run_id": None,
        "dry_run_kind": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_KIND,
        "dry_run_status": dry_run_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_ROUTING,
        "lock_reason": lock_reason,
        "source_token_issuance_plan_id": source.get("plan_id"),
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
        "approval_token_record_preview": _approval_token_record_preview(source, operator),
        "approval_audit_record_preview": _approval_audit_record_preview(source, operator),
        "token_target_paths_preview": _token_target_paths_preview(source, operator),
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "final_token_preflight_checklist": _final_token_preflight_checklist(),
        "token_issuance_plan_validation": deepcopy(plan_validation),
        "dry_run_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_token_issuance_plan_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY),
    }
    dry_run["dry_run_id"] = _dry_run_id(dry_run)
    dry_run["dry_run_validation"] = validate_human_approval_token_issuance_dry_run(dry_run)
    dry_run["next_step_recommendation"] = recommend_human_approval_token_issuance_dry_run_action(dry_run)
    return dry_run


def validate_human_approval_token_issuance_dry_run(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(dry_run.get("policy"))
    source_snapshot = _as_dict(dry_run.get("source_token_issuance_plan_snapshot"))
    token_plan_validation = _as_dict(dry_run.get("token_issuance_plan_validation"))
    expected_token_plan_validation = validate_human_approval_token_issuance_plan(source_snapshot)
    expected_lock_reason = _dry_run_lock_reason(source_snapshot, token_plan_validation)

    for key in (
        "dry_run_id",
        "dry_run_kind",
        "dry_run_status",
        "routing",
        "lock_reason",
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
        "final_token_preflight_checklist",
        "token_issuance_plan_validation",
        "dry_run_validation",
        "next_step_recommendation",
        "source_token_issuance_plan_snapshot",
        "policy",
    ):
        if key not in dry_run:
            errors.append(f"missing_{key}")
    if dry_run.get("dry_run_kind") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_KIND:
        errors.append("dry_run_kind_must_be_memory_human_approval_token_issuance_dry_run_candidate")
    if dry_run.get("dry_run_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED,
    }:
        errors.append("dry_run_status_must_be_supported")
    if dry_run.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_ROUTING:
        errors.append("routing_must_require_manual_final_human_confirmation_before_token_write")
    if dry_run.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if dry_run.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_token_issuance_dry_run_checks")
    if expected_lock_reason is None and dry_run.get("dry_run_status") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED:
        errors.append("valid_token_issuance_plan_must_require_final_preflight")
    if expected_lock_reason is not None and dry_run.get("dry_run_status") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED:
        errors.append("invalid_or_locked_token_issuance_plan_must_lock_dry_run")
    if not isinstance(dry_run.get("operator"), str) or not dry_run.get("operator"):
        errors.append("operator_must_be_non_empty_string")
    if token_plan_validation != expected_token_plan_validation:
        errors.append("token_issuance_plan_validation_must_match_source_token_issuance_plan_snapshot")
    if dry_run.get("source_token_issuance_plan_id") != source_snapshot.get("plan_id"):
        errors.append("source_token_issuance_plan_id_must_match_source_snapshot")
    if not isinstance(dry_run.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(dry_run.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(dry_run.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(dry_run.get("source_pattern_ids"), list) and isinstance(dry_run.get("source_fact_ids"), list):
        if not dry_run.get("source_pattern_ids") and not dry_run.get("source_fact_ids"):
            errors.append("missing_source_evidence")
    if not isinstance(dry_run.get("proposal_record_preview"), Mapping):
        errors.append("missing_proposal_record_preview")
    if not isinstance(dry_run.get("operation_ledger_preview"), Mapping):
        errors.append("missing_operation_ledger_preview")
    if not isinstance(dry_run.get("target_paths_preview"), Mapping):
        errors.append("missing_target_paths_preview")
    if not isinstance(dry_run.get("approval_token_record_preview"), Mapping):
        errors.append("approval_token_record_preview_must_be_mapping")
    else:
        errors.extend(_preview_record_errors("approval_token_record_preview", dry_run["approval_token_record_preview"]))
    if not isinstance(dry_run.get("approval_audit_record_preview"), Mapping):
        errors.append("approval_audit_record_preview_must_be_mapping")
    else:
        errors.extend(_preview_record_errors("approval_audit_record_preview", dry_run["approval_audit_record_preview"]))
    if not isinstance(dry_run.get("token_target_paths_preview"), Mapping):
        errors.append("token_target_paths_preview_must_be_mapping")
    else:
        errors.extend(_preview_record_errors("token_target_paths_preview", dry_run["token_target_paths_preview"]))
    if dry_run.get("final_token_preflight_checklist") != _final_token_preflight_checklist():
        errors.append("final_token_preflight_checklist_must_match_v0_1_deterministic_checks")
    for field in ("proposal_record_preview", "operation_ledger_preview", "target_paths_preview", "payload_preview"):
        if field in dry_run and field in source_snapshot and dry_run.get(field) != source_snapshot.get(field):
            errors.append(f"{field}_must_match_source_token_issuance_plan_snapshot")
    if dry_run.get("source_pattern_ids") != list(source_snapshot.get("source_pattern_ids", []) or []):
        errors.append("source_pattern_ids_must_match_source_token_issuance_plan_snapshot")
    if dry_run.get("source_fact_ids") != list(source_snapshot.get("source_fact_ids", []) or []):
        errors.append("source_fact_ids_must_match_source_token_issuance_plan_snapshot")
    errors.extend(validate_forbidden_true_keys_false_or_absent(dry_run, _FORBIDDEN_TRUE_KEYS))
    errors.extend(validate_policy_flags(policy, MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY))

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_issuance_dry_run(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_issuance_dry_run(dry_run)
    source_snapshot = _as_dict(dry_run.get("source_token_issuance_plan_snapshot"))
    return {
        "dry_run_id": dry_run.get("dry_run_id"),
        "dry_run_kind": dry_run.get("dry_run_kind"),
        "dry_run_status": dry_run.get("dry_run_status"),
        "routing": dry_run.get("routing"),
        "lock_reason": dry_run.get("lock_reason"),
        "source_token_issuance_plan_id": dry_run.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": dry_run.get("source_review_outcome_id"),
        "source_request_id": dry_run.get("source_request_id"),
        "source_gate_id": dry_run.get("source_gate_id"),
        "source_dry_run_id": dry_run.get("source_dry_run_id"),
        "source_plan_id": dry_run.get("source_plan_id"),
        "source_outcome_id": dry_run.get("source_outcome_id"),
        "source_packet_id": dry_run.get("source_packet_id"),
        "source_submission_id": dry_run.get("source_submission_id"),
        "source_draft_id": dry_run.get("source_draft_id"),
        "source_decision_id": dry_run.get("source_decision_id"),
        "source_queue_item_id": dry_run.get("source_queue_item_id"),
        "block_id": dry_run.get("block_id"),
        "block_type": dry_run.get("block_type"),
        "project_scope": dry_run.get("project_scope"),
        "operator": dry_run.get("operator"),
        "source_pattern_count": len(dry_run.get("source_pattern_ids", []) or []),
        "source_fact_count": len(dry_run.get("source_fact_ids", []) or []),
        "final_token_preflight_checklist": deepcopy(dry_run.get("final_token_preflight_checklist")),
        "validation": validation,
        "token_issuance_plan_explanation": explain_human_approval_token_issuance_plan(source_snapshot) if source_snapshot else {},
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY),
    }


def recommend_human_approval_token_issuance_dry_run_action(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_issuance_dry_run(dry_run)
    source_snapshot = _as_dict(dry_run.get("source_token_issuance_plan_snapshot"))
    if validation["valid"] and dry_run.get("dry_run_status") == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED:
        action = "request_manual_final_human_confirmation_before_token_write"
        reason = "Token issuance dry-run candidate is ready for final human confirmation only; it does not issue, persist, submit, or write tokens."
    else:
        action = "keep_token_issuance_dry_run_locked"
        reason = _recommendation_reason(dry_run.get("lock_reason"), validation)
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_ROUTING,
        "validation": validation,
        "token_issuance_plan_recommendation": recommend_human_approval_token_issuance_plan_action(source_snapshot) if source_snapshot else {},
        "creates_token_issuance_dry_run_candidates_only": True,
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY),
    }


def summarize_human_approval_token_issuance_dry_runs(
    dry_runs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(dry_runs, "dry_run_status")
    lock_reason_summary = summarize_candidates(dry_runs, "lock_reason")
    final_preflight_required_count = 0
    locked_count = 0
    valid_count = 0
    invalid_count = 0
    for dry_run in dry_runs:
        if dry_run.get("dry_run_status") == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED:
            final_preflight_required_count += 1
        if dry_run.get("dry_run_status") == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED:
            locked_count += 1
        validation = validate_human_approval_token_issuance_dry_run(dry_run)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(dry_runs),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "final_preflight_required_count": final_preflight_required_count,
        "locked_count": locked_count,
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "by_lock_reason": lock_reason_summary["by_status"],
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY),
    }


def _dry_run_lock_reason(plan: Mapping[str, Any], plan_validation: Mapping[str, Any]) -> str | None:
    if plan.get("plan_status") == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED:
        return plan.get("lock_reason") or "token_issuance_plan_locked"
    if not isinstance(plan.get("proposal_record_preview"), Mapping):
        return "missing_proposal_record_preview"
    if not isinstance(plan.get("operation_ledger_preview"), Mapping):
        return "missing_operation_ledger_preview"
    if not isinstance(plan.get("target_paths_preview"), Mapping):
        return "missing_target_paths_preview"
    if not (plan.get("source_pattern_ids") or plan.get("source_fact_ids")):
        return "missing_source_evidence"
    if not isinstance(plan.get("token_issuance_steps"), list) or not isinstance(plan.get("token_preflight_checks"), list):
        return "missing_token_plan_controls"
    if plan_validation.get("valid") is not True:
        return "invalid_token_issuance_plan"
    if plan.get("plan_status") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED:
        return "invalid_token_issuance_plan"
    return None


def _approval_token_record_preview(plan: Mapping[str, Any], operator: str | None) -> dict[str, Any]:
    token_id = _preview_id("approval-token", plan, operator)
    return {
        "approval_token_id_preview": token_id,
        "record_kind_preview": "human_approval_token_record_preview",
        "preview_only": True,
        "token_issued": False,
        "approved": False,
        "persisted": False,
        "written": False,
        "submitted": False,
        "block_id": plan.get("block_id"),
        "block_type": plan.get("block_type"),
        "project_scope": plan.get("project_scope"),
        "source_token_issuance_plan_id": plan.get("plan_id"),
        "source_review_outcome_id": plan.get("source_review_outcome_id"),
        "source_request_id": plan.get("source_request_id"),
        "source_gate_id": plan.get("source_gate_id"),
        "source_dry_run_id": plan.get("source_dry_run_id"),
        "source_plan_id": plan.get("source_plan_id"),
        "source_pattern_ids": deepcopy(list(plan.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(plan.get("source_fact_ids", []) or [])),
        "operator": operator if operator is not None else _DEFAULT_OPERATOR,
    }


def _approval_audit_record_preview(plan: Mapping[str, Any], operator: str | None) -> dict[str, Any]:
    return {
        "approval_audit_id_preview": _preview_id("approval-audit", plan, operator),
        "record_kind_preview": "human_approval_token_audit_record_preview",
        "operation_type_preview": "human_approval_token_issuance_final_preflight",
        "preview_only": True,
        "created_operation_event": False,
        "writes_approval_audit": False,
        "writes_operation_ledger": False,
        "written": False,
        "block_id": plan.get("block_id"),
        "block_type": plan.get("block_type"),
        "project_scope": plan.get("project_scope"),
        "source_token_issuance_plan_id": plan.get("plan_id"),
        "operator": operator if operator is not None else _DEFAULT_OPERATOR,
    }


def _token_target_paths_preview(plan: Mapping[str, Any], operator: str | None) -> dict[str, Any]:
    return {
        "base": "${HERMES_HOME}",
        "approval_token_file": "memory/approvals/human_approval_tokens.jsonl",
        "approval_audit_file": "memory/audit/human_approval_token_audit.jsonl",
        "approval_token_selector": _preview_id("approval-token", plan, operator),
        "approval_audit_selector": _preview_id("approval-audit", plan, operator),
        "preview_only": True,
        "token_issued": False,
        "persisted": False,
        "writes_token_files": False,
        "writes_approval_audit": False,
        "written": False,
    }


def _final_token_preflight_checklist() -> list[str]:
    return [
        "source_token_issuance_plan_is_valid_and_manual_plan_required",
        "approval_token_record_is_preview_only",
        "approval_audit_record_is_preview_only",
        "token_target_paths_are_preview_only",
        "no_real_approval_token_issued",
        "no_approval_persisted",
        "no_real_proposal_created",
        "no_operation_ledger_event_created",
        "no_proposal_file_written",
        "no_token_file_written",
        "no_approval_audit_written",
        "no_memory_or_graph_or_config_write",
        "manual_final_human_confirmation_required_before_token_write",
    ]


def _preview_record_errors(field: str, preview: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if preview.get("preview_only") is not True:
        errors.append(f"{field}_must_be_preview_only")
    for forbidden in _FORBIDDEN_TRUE_KEYS:
        if preview.get(forbidden) is True:
            errors.append(f"{field}_{forbidden}_must_be_false_or_absent")
    return errors


def _recommendation_reason(lock_reason: Any, validation: Mapping[str, Any]) -> str:
    if lock_reason in {
        "missing_proposal_record_preview",
        "missing_operation_ledger_preview",
        "missing_target_paths_preview",
        "missing_source_evidence",
        "missing_token_plan_controls",
    }:
        return f"Token issuance dry run is locked because {lock_reason}."
    if lock_reason in {
        "token_issuance_plan_locked",
        "token_review_requested_changes",
        "token_review_rejected",
        "token_review_deferred",
        "invalid_token_review_outcome",
    }:
        return f"Token issuance dry run inherits the source plan lock: {lock_reason}."
    if "token_issuance_plan_validation_must_match_source_token_issuance_plan_snapshot" in list(validation.get("errors", []) or []):
        return "Token issuance dry-run validation does not match its source plan snapshot."
    return "Token issuance dry-run candidate violates the v0.1 validation contract."


def _dry_run_id(dry_run: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_VERSION,
        "dry_run_kind": dry_run.get("dry_run_kind"),
        "dry_run_status": dry_run.get("dry_run_status"),
        "routing": dry_run.get("routing"),
        "lock_reason": dry_run.get("lock_reason"),
        "source_token_issuance_plan_id": dry_run.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": dry_run.get("source_review_outcome_id"),
        "source_request_id": dry_run.get("source_request_id"),
        "source_gate_id": dry_run.get("source_gate_id"),
        "source_dry_run_id": dry_run.get("source_dry_run_id"),
        "source_plan_id": dry_run.get("source_plan_id"),
        "source_outcome_id": dry_run.get("source_outcome_id"),
        "source_packet_id": dry_run.get("source_packet_id"),
        "source_submission_id": dry_run.get("source_submission_id"),
        "source_draft_id": dry_run.get("source_draft_id"),
        "source_decision_id": dry_run.get("source_decision_id"),
        "source_queue_item_id": dry_run.get("source_queue_item_id"),
        "block_id": dry_run.get("block_id"),
        "block_type": dry_run.get("block_type"),
        "project_scope": dry_run.get("project_scope"),
        "operator": dry_run.get("operator"),
        "approval_token_record_preview": dry_run.get("approval_token_record_preview", {}),
        "approval_audit_record_preview": dry_run.get("approval_audit_record_preview", {}),
        "token_target_paths_preview": dry_run.get("token_target_paths_preview", {}),
        "proposal_record_preview": dry_run.get("proposal_record_preview", {}),
        "operation_ledger_preview": dry_run.get("operation_ledger_preview", {}),
        "target_paths_preview": dry_run.get("target_paths_preview", {}),
        "payload_preview": dry_run.get("payload_preview"),
        "source_pattern_ids": list(dry_run.get("source_pattern_ids", [])),
        "source_fact_ids": list(dry_run.get("source_fact_ids", [])),
        "final_token_preflight_checklist": list(dry_run.get("final_token_preflight_checklist", [])),
        "token_issuance_plan_validation": dry_run.get("token_issuance_plan_validation", {}),
        "policy": dry_run.get("policy", {}),
    }
    return build_stable_digest("memory-human-approval-token-issuance-dry-run:v0.1", identity)


def _preview_id(kind: str, plan: Mapping[str, Any], operator: str | None) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_VERSION,
        "kind": kind,
        "source_token_issuance_plan_id": plan.get("plan_id"),
        "source_review_outcome_id": plan.get("source_review_outcome_id"),
        "source_request_id": plan.get("source_request_id"),
        "source_gate_id": plan.get("source_gate_id"),
        "block_id": plan.get("block_id"),
        "block_type": plan.get("block_type"),
        "project_scope": plan.get("project_scope"),
        "operator": operator if operator is not None else _DEFAULT_OPERATOR,
    }
    return build_stable_digest(f"preview:{kind}:v0.1", identity)


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
