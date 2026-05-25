from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_final_confirmation_review_gate import (
    explain_human_approval_token_final_confirmation_review_outcome,
    recommend_human_approval_token_final_confirmation_review_action,
    validate_human_approval_token_final_confirmation_review_outcome,
)
from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_forbidden_true_keys_false_or_absent,
    validate_policy_flags,
)


MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_KIND = (
    "memory_human_approval_token_write_execution_plan_candidate"
)
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_REQUIRED = (
    "manual_token_write_execution_plan_required"
)
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_ROUTING = (
    "manual_token_write_dry_run_required_before_any_token_write"
)
MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_token_write_execution_plan_candidates_only": True,
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

_DEFAULT_EXECUTOR = "hermes_memory_human_approval_token_write_execution_plan_v0.1"
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
    "final_confirmation_requested_changes",
    "final_confirmation_rejected",
    "final_confirmation_deferred",
    "invalid_final_confirmation_review_outcome",
    "missing_approval_token_record_preview",
    "missing_approval_audit_record_preview",
    "missing_token_target_paths_preview",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_source_evidence",
    "preview_integrity_failed",
}


def create_human_approval_token_write_execution_plan(
    final_confirmation_review_outcome: Mapping[str, Any],
    executor: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only token write execution plan candidate."""
    source = deepcopy(dict(final_confirmation_review_outcome))
    final_confirmation_review_outcome_validation = (
        validate_human_approval_token_final_confirmation_review_outcome(source)
    )
    lock_reason = _plan_lock_reason(source, final_confirmation_review_outcome_validation)
    plan_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_LOCKED
    )

    plan = {
        "plan_id": None,
        "plan_kind": MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_KIND,
        "plan_status": plan_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_ROUTING,
        "lock_reason": lock_reason,
        "source_final_confirmation_review_outcome_id": source.get("review_outcome_id"),
        "source_final_confirmation_request_id": source.get("source_final_confirmation_request_id"),
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
        "executor": executor if executor is not None else _DEFAULT_EXECUTOR,
        "outcome": source.get("outcome"),
        "approval_token_record_preview": deepcopy(source.get("approval_token_record_preview")),
        "approval_audit_record_preview": deepcopy(source.get("approval_audit_record_preview")),
        "token_target_paths_preview": deepcopy(source.get("token_target_paths_preview")),
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "token_write_execution_steps": _token_write_execution_steps(),
        "token_write_execution_preflight_checks": _token_write_execution_preflight_checks(),
        "final_confirmation_review_outcome_validation": deepcopy(
            final_confirmation_review_outcome_validation
        ),
        "plan_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_final_confirmation_review_outcome_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_POLICY),
    }
    plan["plan_id"] = _plan_id(plan)
    plan["plan_validation"] = validate_human_approval_token_write_execution_plan(plan)
    plan["next_step_recommendation"] = recommend_human_approval_token_write_execution_plan_action(plan)
    return plan


def validate_human_approval_token_write_execution_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(plan.get("policy"))
    source_snapshot = _as_dict(plan.get("source_final_confirmation_review_outcome_snapshot"))
    final_confirmation_review_outcome_validation = _as_dict(
        plan.get("final_confirmation_review_outcome_validation")
    )
    expected_final_confirmation_review_outcome_validation = (
        validate_human_approval_token_final_confirmation_review_outcome(source_snapshot)
    )
    expected_lock_reason = _plan_lock_reason(
        source_snapshot,
        final_confirmation_review_outcome_validation,
    )
    preview_integrity_errors = _preview_integrity_errors(plan)

    for key in (
        "plan_id",
        "plan_kind",
        "plan_status",
        "routing",
        "lock_reason",
        "source_final_confirmation_review_outcome_id",
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
        "executor",
        "outcome",
        "approval_token_record_preview",
        "approval_audit_record_preview",
        "token_target_paths_preview",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "token_write_execution_steps",
        "token_write_execution_preflight_checks",
        "final_confirmation_review_outcome_validation",
        "plan_validation",
        "next_step_recommendation",
        "source_final_confirmation_review_outcome_snapshot",
        "policy",
    ):
        if key not in plan:
            errors.append(f"missing_{key}")
    if plan.get("plan_kind") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_KIND:
        errors.append("plan_kind_must_be_memory_human_approval_token_write_execution_plan_candidate")
    if plan.get("plan_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_LOCKED,
    }:
        errors.append("plan_status_must_be_supported")
    if plan.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_ROUTING:
        errors.append("routing_must_require_manual_token_write_dry_run_before_any_token_write")
    if plan.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if plan.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_token_write_execution_plan_checks")
    if expected_lock_reason is None and plan.get("plan_status") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_REQUIRED:
        errors.append("confirmed_final_confirmation_review_outcome_must_require_manual_token_write_execution_plan")
    if expected_lock_reason is not None and plan.get("plan_status") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_LOCKED:
        errors.append("invalid_or_non_confirmed_final_confirmation_review_outcome_must_lock_plan")
    if not isinstance(plan.get("executor"), str) or not plan.get("executor"):
        errors.append("executor_must_be_non_empty_string")
    if final_confirmation_review_outcome_validation != expected_final_confirmation_review_outcome_validation:
        errors.append("final_confirmation_review_outcome_validation_must_match_source_final_confirmation_review_outcome_snapshot")
    if plan.get("source_final_confirmation_review_outcome_id") != source_snapshot.get("review_outcome_id"):
        errors.append("source_final_confirmation_review_outcome_id_must_match_source_snapshot")
    for source_key in (
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
        "outcome",
    ):
        if plan.get(source_key) != source_snapshot.get(source_key):
            errors.append(f"{source_key}_must_match_source_snapshot")
    if not isinstance(plan.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(plan.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(plan.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(plan.get("source_pattern_ids"), list) and isinstance(plan.get("source_fact_ids"), list):
        if (
            not plan.get("source_pattern_ids")
            and not plan.get("source_fact_ids")
            and plan.get("lock_reason") != "missing_source_evidence"
        ):
            errors.append("missing_source_evidence")
    if plan.get("token_write_execution_steps") != _token_write_execution_steps():
        errors.append("token_write_execution_steps_must_match_v0_1_deterministic_steps")
    if plan.get("token_write_execution_preflight_checks") != _token_write_execution_preflight_checks():
        errors.append("token_write_execution_preflight_checks_must_match_v0_1_deterministic_checks")
    for field in _PREVIEW_FIELDS:
        if not isinstance(plan.get(field), Mapping) and plan.get("lock_reason") != f"missing_{field}":
            errors.append(f"missing_{field}")
    if plan.get("lock_reason") != "preview_integrity_failed":
        errors.extend(preview_integrity_errors)
    for field in _PREVIEW_FIELDS + ("payload_preview",):
        if field in plan and field in source_snapshot and plan.get(field) != source_snapshot.get(field):
            errors.append(f"{field}_must_match_source_final_confirmation_review_outcome_snapshot")
    if plan.get("source_pattern_ids") != list(source_snapshot.get("source_pattern_ids", []) or []):
        errors.append("source_pattern_ids_must_match_source_final_confirmation_review_outcome_snapshot")
    if plan.get("source_fact_ids") != list(source_snapshot.get("source_fact_ids", []) or []):
        errors.append("source_fact_ids_must_match_source_final_confirmation_review_outcome_snapshot")
    errors.extend(validate_forbidden_true_keys_false_or_absent(plan, _FORBIDDEN_TRUE_KEYS))
    errors.extend(validate_policy_flags(policy, MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_POLICY))

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_write_execution_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_write_execution_plan(plan)
    source_snapshot = _as_dict(plan.get("source_final_confirmation_review_outcome_snapshot"))
    return {
        "plan_id": plan.get("plan_id"),
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
        "lock_reason": plan.get("lock_reason"),
        "source_final_confirmation_review_outcome_id": plan.get("source_final_confirmation_review_outcome_id"),
        "block_id": plan.get("block_id"),
        "block_type": plan.get("block_type"),
        "project_scope": plan.get("project_scope"),
        "executor": plan.get("executor"),
        "outcome": plan.get("outcome"),
        "source_pattern_count": len(plan.get("source_pattern_ids", []) or []),
        "source_fact_count": len(plan.get("source_fact_ids", []) or []),
        "token_write_execution_step_count": len(plan.get("token_write_execution_steps", []) or []),
        "token_write_execution_preflight_check_count": len(
            plan.get("token_write_execution_preflight_checks", []) or []
        ),
        "validation": validation,
        "final_confirmation_review_outcome_explanation": (
            explain_human_approval_token_final_confirmation_review_outcome(source_snapshot)
            if source_snapshot
            else {}
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_POLICY),
    }


def recommend_human_approval_token_write_execution_plan_action(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_write_execution_plan(plan)
    source_snapshot = _as_dict(plan.get("source_final_confirmation_review_outcome_snapshot"))
    if validation["valid"] and plan.get("plan_status") == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_REQUIRED:
        action = "route_to_manual_token_write_dry_run_without_issuing_token"
        reason = "Execution plan candidate is ready for a separate manual token write dry run; this candidate does not issue, persist, submit, or write anything."
    elif validation["valid"]:
        action = "keep_token_write_execution_plan_locked"
        reason = f"Execution plan candidate is locked by {plan.get('lock_reason')}."
    else:
        action = "repair_token_write_execution_plan_candidate"
        reason = "Execution plan candidate failed validation and cannot proceed."
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_ROUTING,
        "validation": validation,
        "final_confirmation_review_outcome_recommendation": (
            recommend_human_approval_token_final_confirmation_review_action(source_snapshot)
            if source_snapshot
            else {}
        ),
        "creates_token_write_execution_plan_candidates_only": True,
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_POLICY),
    }


def summarize_human_approval_token_write_execution_plans(
    plans: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(plans, "plan_status")
    lock_reason_summary = summarize_candidates(plans, "lock_reason")
    execution_plan_required_count = 0
    locked_count = 0
    valid_count = 0
    invalid_count = 0
    for plan in plans:
        if plan.get("plan_status") == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_REQUIRED:
            execution_plan_required_count += 1
        if plan.get("plan_status") == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_LOCKED:
            locked_count += 1
        validation = validate_human_approval_token_write_execution_plan(plan)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(plans),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "execution_plan_required_count": execution_plan_required_count,
        "locked_count": locked_count,
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "by_lock_reason": lock_reason_summary["by_status"],
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_POLICY),
    }


def _plan_lock_reason(
    outcome_candidate: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str | None:
    outcome = outcome_candidate.get("outcome")
    if outcome == "request_changes":
        return "final_confirmation_requested_changes"
    if outcome == "reject":
        return "final_confirmation_rejected"
    if outcome == "defer":
        return "final_confirmation_deferred"
    if outcome != "confirm_token_write":
        return "invalid_final_confirmation_review_outcome"
    for field in _PREVIEW_FIELDS:
        if not isinstance(outcome_candidate.get(field), Mapping):
            return f"missing_{field}"
    if not (outcome_candidate.get("source_pattern_ids") or outcome_candidate.get("source_fact_ids")):
        return "missing_source_evidence"
    if _preview_integrity_errors(outcome_candidate):
        return "preview_integrity_failed"
    if validation.get("valid") is not True:
        return "invalid_final_confirmation_review_outcome"
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


def _token_write_execution_steps() -> list[dict[str, Any]]:
    return [
        {
            "id": "verify_final_confirmation_review_outcome",
            "order": 1,
            "description": "Re-validate the final confirmation review outcome candidate.",
            "writes": False,
        },
        {
            "id": "verify_preview_integrity",
            "order": 2,
            "description": "Confirm token, audit, proposal, ledger, and path previews are preview-only.",
            "writes": False,
        },
        {
            "id": "verify_source_evidence",
            "order": 3,
            "description": "Confirm source pattern or fact evidence is present before any future write dry run.",
            "writes": False,
        },
        {
            "id": "prepare_manual_token_write_dry_run_packet",
            "order": 4,
            "description": "Prepare the next read-only dry-run input from existing previews.",
            "writes": False,
        },
    ]


def _token_write_execution_preflight_checks() -> list[dict[str, Any]]:
    return [
        {
            "id": "final_confirmation_outcome_is_confirm_token_write",
            "required": True,
            "blocks_token_write": True,
        },
        {
            "id": "approval_token_record_preview_is_preview_only",
            "required": True,
            "blocks_token_write": True,
        },
        {
            "id": "approval_audit_record_preview_is_preview_only",
            "required": True,
            "blocks_token_write": True,
        },
        {
            "id": "token_target_paths_preview_is_preview_only",
            "required": True,
            "blocks_token_write": True,
        },
        {
            "id": "proposal_and_operation_ledger_previews_are_not_written",
            "required": True,
            "blocks_token_write": True,
        },
        {
            "id": "source_evidence_is_present",
            "required": True,
            "blocks_token_write": True,
        },
        {
            "id": "no_memory_config_graph_proposal_ledger_token_or_approval_writes",
            "required": True,
            "blocks_token_write": True,
        },
    ]


def _plan_id(plan: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_EXECUTION_PLAN_VERSION,
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
        "lock_reason": plan.get("lock_reason"),
        "source_final_confirmation_review_outcome_id": plan.get("source_final_confirmation_review_outcome_id"),
        "source_final_confirmation_request_id": plan.get("source_final_confirmation_request_id"),
        "source_token_write_lock_gate_id": plan.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": plan.get("source_token_issuance_dry_run_id"),
        "source_token_issuance_plan_id": plan.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": plan.get("source_review_outcome_id"),
        "source_request_id": plan.get("source_request_id"),
        "source_gate_id": plan.get("source_gate_id"),
        "source_dry_run_id": plan.get("source_dry_run_id"),
        "source_plan_id": plan.get("source_plan_id"),
        "source_outcome_id": plan.get("source_outcome_id"),
        "source_packet_id": plan.get("source_packet_id"),
        "source_submission_id": plan.get("source_submission_id"),
        "source_draft_id": plan.get("source_draft_id"),
        "source_decision_id": plan.get("source_decision_id"),
        "source_queue_item_id": plan.get("source_queue_item_id"),
        "block_id": plan.get("block_id"),
        "block_type": plan.get("block_type"),
        "project_scope": plan.get("project_scope"),
        "executor": plan.get("executor"),
        "outcome": plan.get("outcome"),
        "approval_token_record_preview": plan.get("approval_token_record_preview", {}),
        "approval_audit_record_preview": plan.get("approval_audit_record_preview", {}),
        "token_target_paths_preview": plan.get("token_target_paths_preview", {}),
        "proposal_record_preview": plan.get("proposal_record_preview", {}),
        "operation_ledger_preview": plan.get("operation_ledger_preview", {}),
        "target_paths_preview": plan.get("target_paths_preview", {}),
        "payload_preview": plan.get("payload_preview"),
        "source_pattern_ids": list(plan.get("source_pattern_ids", [])),
        "source_fact_ids": list(plan.get("source_fact_ids", [])),
        "token_write_execution_steps": plan.get("token_write_execution_steps", []),
        "token_write_execution_preflight_checks": plan.get("token_write_execution_preflight_checks", []),
        "final_confirmation_review_outcome_validation": plan.get(
            "final_confirmation_review_outcome_validation", {}
        ),
        "policy": plan.get("policy", {}),
    }
    return build_stable_digest(
        "memory-human-approval-token-write-execution-plan:v0.1",
        identity,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
