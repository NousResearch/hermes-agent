from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_real_write_executor_contract_review_gate import (
    explain_human_approval_token_real_write_executor_contract_review_outcome,
    recommend_human_approval_token_real_write_executor_contract_review_action,
    validate_human_approval_token_real_write_executor_contract_review_outcome,
)
from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_forbidden_true_keys_false_or_absent,
    validate_policy_flags,
)


MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_KIND = (
    "memory_human_approval_token_real_write_executor_implementation_plan_candidate"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED = (
    "real_token_write_executor_implementation_plan_required"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_ROUTING = (
    "real_token_write_executor_implementation_dry_run_required_before_code_implementation"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_real_write_executor_implementation_plan_candidates_only": True,
    "invokes_real_token_write_executor": False,
    "implements_real_token_write_executor": False,
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

_DEFAULT_IMPLEMENTER = (
    "hermes_memory_human_approval_token_real_write_executor_implementation_plan_v0.1"
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
    "writes_token_files",
    "writes_approval_audit",
    "invokes_real_token_write_executor",
    "implements_real_token_write_executor",
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
_WRITE_PREVIEW_FIELDS = (
    "approval_token_write_payload_preview",
    "approval_audit_write_payload_preview",
    "token_write_target_paths_preview",
)
_EXECUTOR_CONTRACT_FIELD_TYPES = {
    "executor_contract_inputs": Mapping,
    "executor_hard_lock_checks": list,
    "executor_audit_fields": list,
    "executor_rollback_rules": list,
    "executor_forbidden_side_effects": list,
    "executor_contract_checklist": list,
}
_LOCK_REASONS = {
    None,
    "contract_review_requested_changes",
    "contract_review_rejected",
    "contract_review_deferred",
    "invalid_contract_review_outcome",
    "missing_approval_token_record_preview",
    "missing_approval_audit_record_preview",
    "missing_token_target_paths_preview",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_approval_token_write_payload_preview",
    "missing_approval_audit_write_payload_preview",
    "missing_token_write_target_paths_preview",
    "missing_executor_contract_inputs",
    "missing_executor_hard_lock_checks",
    "missing_executor_audit_fields",
    "missing_executor_rollback_rules",
    "missing_executor_forbidden_side_effects",
    "missing_executor_contract_checklist",
    "missing_contract_review_checklist",
    "missing_source_evidence",
    "preview_integrity_failed",
    "contract_review_integrity_failed",
}
_SOURCE_KEYS = (
    "source_contract_id",
    "source_write_final_gate_id",
    "source_write_execution_dry_run_id",
    "source_write_execution_plan_id",
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
    "outcome",
)
_REQUIRED_PLAN_KEYS = (
    "plan_id",
    "plan_kind",
    "plan_status",
    "routing",
    "lock_reason",
    "source_contract_review_outcome_id",
) + _SOURCE_KEYS + (
    "implementer",
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
    "token_write_execution_steps",
    "token_write_execution_preflight_checks",
    "approval_token_write_payload_preview",
    "approval_audit_write_payload_preview",
    "token_write_target_paths_preview",
    "final_token_write_preflight_checklist",
    "final_gate_checklist",
    "executor_contract_inputs",
    "executor_hard_lock_checks",
    "executor_audit_fields",
    "executor_rollback_rules",
    "executor_forbidden_side_effects",
    "executor_contract_checklist",
    "contract_review_checklist",
    "implementation_plan_interfaces",
    "implementation_plan_files",
    "implementation_plan_idempotency_strategy",
    "implementation_plan_filesystem_safety_model",
    "implementation_plan_audit_strategy",
    "implementation_plan_rollback_strategy",
    "implementation_plan_test_plan",
    "implementation_plan_forbidden_actions",
    "contract_review_outcome_validation",
    "plan_validation",
    "next_step_recommendation",
    "source_contract_review_outcome_snapshot",
    "policy",
)


def create_human_approval_token_real_write_executor_implementation_plan(
    contract_review_outcome: Mapping[str, Any],
    implementer: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only real token write executor implementation plan."""
    source = deepcopy(dict(contract_review_outcome))
    contract_review_outcome_validation = (
        validate_human_approval_token_real_write_executor_contract_review_outcome(source)
    )
    lock_reason = _plan_lock_reason(source, contract_review_outcome_validation)
    plan_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED
    )

    plan = {
        "plan_id": None,
        "plan_kind": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_KIND,
        "plan_status": plan_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_ROUTING,
        "lock_reason": lock_reason,
        "source_contract_review_outcome_id": source.get("review_outcome_id"),
        "source_contract_id": source.get("source_contract_id"),
        "source_write_final_gate_id": source.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": source.get(
            "source_write_execution_dry_run_id"
        ),
        "source_write_execution_plan_id": source.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": source.get(
            "source_final_confirmation_review_outcome_id"
        ),
        "source_final_confirmation_request_id": source.get(
            "source_final_confirmation_request_id"
        ),
        "source_token_write_lock_gate_id": source.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": source.get(
            "source_token_issuance_dry_run_id"
        ),
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
        "implementer": implementer if implementer is not None else _DEFAULT_IMPLEMENTER,
        "outcome": source.get("outcome"),
        "rationale": source.get("rationale"),
        "approval_token_record_preview": deepcopy(source.get("approval_token_record_preview")),
        "approval_audit_record_preview": deepcopy(source.get("approval_audit_record_preview")),
        "token_target_paths_preview": deepcopy(source.get("token_target_paths_preview")),
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "token_write_execution_steps": deepcopy(
            list(source.get("token_write_execution_steps", []) or [])
        ),
        "token_write_execution_preflight_checks": deepcopy(
            list(source.get("token_write_execution_preflight_checks", []) or [])
        ),
        "approval_token_write_payload_preview": deepcopy(
            source.get("approval_token_write_payload_preview")
        ),
        "approval_audit_write_payload_preview": deepcopy(
            source.get("approval_audit_write_payload_preview")
        ),
        "token_write_target_paths_preview": deepcopy(source.get("token_write_target_paths_preview")),
        "final_token_write_preflight_checklist": deepcopy(
            source.get("final_token_write_preflight_checklist")
        ),
        "final_gate_checklist": deepcopy(source.get("final_gate_checklist")),
        "executor_contract_inputs": deepcopy(source.get("executor_contract_inputs")),
        "executor_hard_lock_checks": deepcopy(source.get("executor_hard_lock_checks")),
        "executor_audit_fields": deepcopy(source.get("executor_audit_fields")),
        "executor_rollback_rules": deepcopy(source.get("executor_rollback_rules")),
        "executor_forbidden_side_effects": deepcopy(
            source.get("executor_forbidden_side_effects")
        ),
        "executor_contract_checklist": deepcopy(source.get("executor_contract_checklist")),
        "contract_review_checklist": deepcopy(source.get("contract_review_checklist")),
        "implementation_plan_interfaces": _implementation_plan_interfaces(),
        "implementation_plan_files": _implementation_plan_files(),
        "implementation_plan_idempotency_strategy": _implementation_plan_idempotency_strategy(),
        "implementation_plan_filesystem_safety_model": (
            _implementation_plan_filesystem_safety_model()
        ),
        "implementation_plan_audit_strategy": _implementation_plan_audit_strategy(),
        "implementation_plan_rollback_strategy": _implementation_plan_rollback_strategy(),
        "implementation_plan_test_plan": _implementation_plan_test_plan(),
        "implementation_plan_forbidden_actions": _implementation_plan_forbidden_actions(),
        "contract_review_outcome_validation": deepcopy(contract_review_outcome_validation),
        "plan_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_contract_review_outcome_snapshot": deepcopy(source),
        "policy": dict(
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY
        ),
    }
    plan["plan_id"] = _plan_id(plan)
    plan["plan_validation"] = (
        validate_human_approval_token_real_write_executor_implementation_plan(plan)
    )
    plan["next_step_recommendation"] = (
        recommend_human_approval_token_real_write_executor_implementation_plan_action(plan)
    )
    return plan


def validate_human_approval_token_real_write_executor_implementation_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(plan.get("policy"))
    source_snapshot = _as_dict(plan.get("source_contract_review_outcome_snapshot"))
    contract_review_outcome_validation = _as_dict(
        plan.get("contract_review_outcome_validation")
    )
    expected_contract_review_outcome_validation = (
        validate_human_approval_token_real_write_executor_contract_review_outcome(
            source_snapshot
        )
    )
    expected_lock_reason = _plan_lock_reason(
        source_snapshot,
        contract_review_outcome_validation,
    )

    for key in _REQUIRED_PLAN_KEYS:
        if key not in plan:
            errors.append(f"missing_{key}")
    if (
        plan.get("plan_kind")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_KIND
    ):
        errors.append(
            "plan_kind_must_be_memory_human_approval_token_real_write_executor_implementation_plan_candidate"
        )
    if plan.get("plan_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED,
    }:
        errors.append("plan_status_must_be_supported")
    if (
        plan.get("routing")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_ROUTING
    ):
        errors.append(
            "routing_must_require_implementation_dry_run_before_code_implementation"
        )
    if plan.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if plan.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_real_write_executor_implementation_plan_checks")
    if (
        expected_lock_reason is None
        and plan.get("plan_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED
    ):
        errors.append(
            "approved_contract_review_outcome_must_require_real_token_write_executor_implementation_plan"
        )
    if (
        expected_lock_reason is not None
        and plan.get("plan_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED
    ):
        errors.append("invalid_or_non_approved_contract_review_outcome_must_lock_plan")
    if not isinstance(plan.get("implementer"), str) or not plan.get("implementer"):
        errors.append("implementer_must_be_non_empty_string")
    if (
        contract_review_outcome_validation
        != expected_contract_review_outcome_validation
    ):
        errors.append(
            "contract_review_outcome_validation_must_match_source_contract_review_outcome_snapshot"
        )
    if plan.get("source_contract_review_outcome_id") != source_snapshot.get(
        "review_outcome_id"
    ):
        errors.append("source_contract_review_outcome_id_must_match_source_snapshot")
    for source_key in _SOURCE_KEYS:
        if plan.get(source_key) != source_snapshot.get(source_key):
            errors.append(f"{source_key}_must_match_source_snapshot")

    if (
        not isinstance(plan.get("payload_preview"), Mapping)
        and plan.get("lock_reason") != "invalid_contract_review_outcome"
    ):
        errors.append("missing_payload_preview")
    if not isinstance(plan.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(plan.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(plan.get("source_pattern_ids"), list) and isinstance(
        plan.get("source_fact_ids"), list
    ):
        if (
            not plan.get("source_pattern_ids")
            and not plan.get("source_fact_ids")
            and plan.get("lock_reason") != "missing_source_evidence"
        ):
            errors.append("missing_source_evidence")
    for field in _PREVIEW_FIELDS + _WRITE_PREVIEW_FIELDS:
        if (
            not isinstance(plan.get(field), Mapping)
            and plan.get("lock_reason") != f"missing_{field}"
        ):
            errors.append(f"missing_{field}")
    for field, expected_type in _EXECUTOR_CONTRACT_FIELD_TYPES.items():
        if (
            not isinstance(plan.get(field), expected_type)
            and plan.get("lock_reason") != f"missing_{field}"
        ):
            errors.append(f"missing_{field}")
    if (
        not isinstance(plan.get("contract_review_checklist"), list)
        and plan.get("lock_reason") != "missing_contract_review_checklist"
    ):
        errors.append("missing_contract_review_checklist")
    if plan.get("lock_reason") != "preview_integrity_failed":
        errors.extend(_preview_integrity_errors(plan))
    if plan.get("lock_reason") is None:
        errors.extend(_contract_review_integrity_errors(plan, contract_review_outcome_validation))

    for field in _PREVIEW_FIELDS + _WRITE_PREVIEW_FIELDS + (
        "payload_preview",
        "token_write_execution_steps",
        "token_write_execution_preflight_checks",
        "final_token_write_preflight_checklist",
        "final_gate_checklist",
        "executor_contract_inputs",
        "executor_hard_lock_checks",
        "executor_audit_fields",
        "executor_rollback_rules",
        "executor_forbidden_side_effects",
        "executor_contract_checklist",
        "contract_review_checklist",
    ):
        if (
            field in plan
            and field in source_snapshot
            and plan.get(field) != source_snapshot.get(field)
        ):
            errors.append(f"{field}_must_match_source_contract_review_outcome_snapshot")
    if plan.get("source_pattern_ids") != list(
        source_snapshot.get("source_pattern_ids", []) or []
    ):
        errors.append("source_pattern_ids_must_match_source_contract_review_outcome_snapshot")
    if plan.get("source_fact_ids") != list(
        source_snapshot.get("source_fact_ids", []) or []
    ):
        errors.append("source_fact_ids_must_match_source_contract_review_outcome_snapshot")

    if plan.get("implementation_plan_interfaces") != _implementation_plan_interfaces():
        errors.append("implementation_plan_interfaces_must_match_v0_1_deterministic_plan")
    if plan.get("implementation_plan_files") != _implementation_plan_files():
        errors.append("implementation_plan_files_must_match_v0_1_deterministic_plan")
    if (
        plan.get("implementation_plan_idempotency_strategy")
        != _implementation_plan_idempotency_strategy()
    ):
        errors.append(
            "implementation_plan_idempotency_strategy_must_match_v0_1_deterministic_plan"
        )
    if (
        plan.get("implementation_plan_filesystem_safety_model")
        != _implementation_plan_filesystem_safety_model()
    ):
        errors.append(
            "implementation_plan_filesystem_safety_model_must_match_v0_1_deterministic_plan"
        )
    if (
        plan.get("implementation_plan_audit_strategy")
        != _implementation_plan_audit_strategy()
    ):
        errors.append("implementation_plan_audit_strategy_must_match_v0_1_deterministic_plan")
    if (
        plan.get("implementation_plan_rollback_strategy")
        != _implementation_plan_rollback_strategy()
    ):
        errors.append(
            "implementation_plan_rollback_strategy_must_match_v0_1_deterministic_plan"
        )
    if plan.get("implementation_plan_test_plan") != _implementation_plan_test_plan():
        errors.append("implementation_plan_test_plan_must_match_v0_1_deterministic_plan")
    if (
        plan.get("implementation_plan_forbidden_actions")
        != _implementation_plan_forbidden_actions()
    ):
        errors.append(
            "implementation_plan_forbidden_actions_must_match_v0_1_deterministic_plan"
        )
    errors.extend(
        validate_forbidden_true_keys_false_or_absent(
            plan,
            _FORBIDDEN_TRUE_KEYS,
        )
    )
    errors.extend(
        validate_policy_flags(
            policy,
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY,
        )
    )

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_real_write_executor_implementation_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_real_write_executor_implementation_plan(
        plan
    )
    source_snapshot = _as_dict(plan.get("source_contract_review_outcome_snapshot"))
    return {
        "plan_id": plan.get("plan_id"),
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
        "lock_reason": plan.get("lock_reason"),
        "source_contract_review_outcome_id": plan.get(
            "source_contract_review_outcome_id"
        ),
        "source_contract_id": plan.get("source_contract_id"),
        "source_write_final_gate_id": plan.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": plan.get(
            "source_write_execution_dry_run_id"
        ),
        "source_write_execution_plan_id": plan.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": plan.get(
            "source_final_confirmation_review_outcome_id"
        ),
        "block_id": plan.get("block_id"),
        "block_type": plan.get("block_type"),
        "project_scope": plan.get("project_scope"),
        "implementer": plan.get("implementer"),
        "outcome": plan.get("outcome"),
        "rationale": plan.get("rationale"),
        "source_pattern_count": len(plan.get("source_pattern_ids", []) or []),
        "source_fact_count": len(plan.get("source_fact_ids", []) or []),
        "implementation_plan_interfaces": deepcopy(
            plan.get("implementation_plan_interfaces")
        ),
        "implementation_plan_files": deepcopy(plan.get("implementation_plan_files")),
        "implementation_plan_forbidden_actions": deepcopy(
            plan.get("implementation_plan_forbidden_actions")
        ),
        "validation": validation,
        "contract_review_outcome_explanation": (
            explain_human_approval_token_real_write_executor_contract_review_outcome(
                source_snapshot
            )
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
        "invokes_real_token_write_executor": False,
        "implements_real_token_write_executor": False,
        "policy": dict(
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY
        ),
    }


def recommend_human_approval_token_real_write_executor_implementation_plan_action(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_real_write_executor_implementation_plan(
        plan
    )
    source_snapshot = _as_dict(plan.get("source_contract_review_outcome_snapshot"))
    if (
        validation["valid"]
        and plan.get("plan_status")
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED
    ):
        action = (
            "route_to_real_token_write_executor_implementation_dry_run_without_code_or_invocation"
        )
        reason = "Implementation plan candidate is ready for a separate dry run; it does not implement or invoke the executor and does not write token, approval, proposal, ledger, memory, graph, or config state."
    elif validation["valid"]:
        action = "keep_real_token_write_executor_implementation_plan_locked"
        reason = f"Implementation plan candidate is locked by {plan.get('lock_reason')}."
    else:
        action = "repair_real_token_write_executor_implementation_plan_candidate"
        reason = "Implementation plan candidate failed validation and cannot proceed."
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_ROUTING,
        "validation": validation,
        "contract_review_outcome_recommendation": (
            recommend_human_approval_token_real_write_executor_contract_review_action(
                source_snapshot
            )
            if source_snapshot
            else {}
        ),
        "creates_real_write_executor_implementation_plan_candidates_only": True,
        "invokes_real_token_write_executor": False,
        "implements_real_token_write_executor": False,
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
        "policy": dict(
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY
        ),
    }


def summarize_human_approval_token_real_write_executor_implementation_plans(
    plans: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(plans, "plan_status")
    by_lock_reason: dict[str, int] = {}
    implementation_plan_required_count = 0
    locked_count = 0
    valid_count = 0
    invalid_count = 0
    for plan in plans:
        lock_reason = str(plan.get("lock_reason"))
        by_lock_reason[lock_reason] = by_lock_reason.get(lock_reason, 0) + 1
        if (
            plan.get("plan_status")
            == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED
        ):
            implementation_plan_required_count += 1
        if (
            plan.get("plan_status")
            == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED
        ):
            locked_count += 1
        validation = validate_human_approval_token_real_write_executor_implementation_plan(
            plan
        )
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": candidate_summary["total"],
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "real_token_write_executor_implementation_plan_required_count": (
            implementation_plan_required_count
        ),
        "locked_count": locked_count,
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "by_lock_reason": dict(sorted(by_lock_reason.items())),
        "policy": dict(
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY
        ),
    }


def _plan_lock_reason(
    outcome_candidate: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str | None:
    outcome = outcome_candidate.get("outcome")
    if outcome == "request_contract_changes":
        return "contract_review_requested_changes"
    if outcome == "reject_contract":
        return "contract_review_rejected"
    if outcome == "defer_contract_review":
        return "contract_review_deferred"
    if outcome != "approve_executor_contract":
        return "invalid_contract_review_outcome"
    for field in _PREVIEW_FIELDS:
        if not isinstance(outcome_candidate.get(field), Mapping):
            return f"missing_{field}"
    for field in _WRITE_PREVIEW_FIELDS:
        if not isinstance(outcome_candidate.get(field), Mapping):
            return f"missing_{field}"
    for field, expected_type in _EXECUTOR_CONTRACT_FIELD_TYPES.items():
        if not isinstance(outcome_candidate.get(field), expected_type):
            return f"missing_{field}"
    if not isinstance(outcome_candidate.get("contract_review_checklist"), list):
        return "missing_contract_review_checklist"
    if not (
        outcome_candidate.get("source_pattern_ids")
        or outcome_candidate.get("source_fact_ids")
    ):
        return "missing_source_evidence"
    if _preview_integrity_errors(outcome_candidate):
        return "preview_integrity_failed"
    if _contract_review_integrity_errors(outcome_candidate, validation):
        return "contract_review_integrity_failed"
    if validation.get("valid") is not True:
        return "invalid_contract_review_outcome"
    return None


def _preview_integrity_errors(container: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _PREVIEW_FIELDS + _WRITE_PREVIEW_FIELDS:
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


def _contract_review_integrity_errors(
    outcome_candidate: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    final_gate_checklist = outcome_candidate.get("final_gate_checklist")
    if isinstance(final_gate_checklist, list):
        required_final_gate_checks = {
            "no_real_approval_token_issued",
            "no_approval_persisted",
            "no_real_proposal_created",
            "no_operation_ledger_event_created",
            "no_token_file_written",
            "no_approval_audit_written",
            "real_token_write_executor_required_but_not_invoked",
        }
        if required_final_gate_checks.difference(final_gate_checklist):
            errors.append("final_gate_checklist_must_preserve_no_write_and_no_executor_checks")
    final_preflight = outcome_candidate.get("final_token_write_preflight_checklist")
    if isinstance(final_preflight, list) and "no_real_approval_token_issued" not in final_preflight:
        errors.append("final_token_write_preflight_checklist_must_preserve_no_token_write_checks")
    forbidden_side_effects = outcome_candidate.get("executor_forbidden_side_effects")
    if isinstance(forbidden_side_effects, list):
        required_side_effects = {
            "write_memory",
            "write_memory_graph",
            "write_proposal_files",
            "write_operation_ledger",
            "issue_real_approval_tokens",
            "write_token_files",
            "write_approval_audit_files",
            "persist_approvals",
            "invoke_real_token_write_executor",
            "implement_real_token_write_executor",
            "submit_to_governance",
            "convert_to_real_proposal",
        }
        if required_side_effects.difference(forbidden_side_effects):
            errors.append("executor_forbidden_side_effects_must_preserve_no_write_boundary")
    executor_checklist = outcome_candidate.get("executor_contract_checklist")
    if isinstance(executor_checklist, list):
        required_executor_checks = {
            "contract_is_read_only_candidate_only",
            "no_real_executor_is_invoked",
            "no_real_executor_is_implemented",
            "no_memory_or_graph_or_config_write",
        }
        if required_executor_checks.difference(executor_checklist):
            errors.append("executor_contract_checklist_must_preserve_no_executor_boundary")
    review_checklist = outcome_candidate.get("contract_review_checklist")
    if isinstance(review_checklist, list):
        required_review_checks = {
            "no_real_approval_token_issued",
            "no_approval_persisted",
            "no_real_proposal_created",
            "no_operation_ledger_event_created",
            "no_proposal_file_written",
            "no_operation_ledger_written",
            "no_token_file_written",
            "no_approval_audit_written",
            "no_memory_or_graph_or_config_write",
            "real_token_write_executor_not_implemented",
            "real_token_write_executor_not_invoked",
            "contract_review_outcome_only",
        }
        if required_review_checks.difference(review_checklist):
            errors.append("contract_review_checklist_must_preserve_no_write_boundary")
    for error in list(validation.get("errors", []) or []):
        if (
            error == "contract_review_checklist_must_match_v0_1_deterministic_checks"
            or error.endswith("_must_match_source_contract_snapshot")
            or error.endswith("_must_match_source_snapshot")
            or error.startswith("final_gate_checklist_must_preserve")
            or error.startswith("final_token_write_preflight_checklist_must_preserve")
            or error.startswith("executor_forbidden_side_effects_must_preserve")
            or error.startswith("executor_contract_checklist_must_preserve")
        ):
            errors.append("contract_review_validation_integrity_errors")
    return _dedupe(errors)


def _implementation_plan_interfaces() -> list[dict[str, Any]]:
    return [
        {
            "name": "validate_real_token_write_executor_inputs",
            "purpose": "Future executor input validation against approved contract review outcome, previews, hard locks, and policy.",
            "module_boundary": "future_executor_module_only",
            "implemented_in_v0_1": False,
            "invoked_in_v0_1": False,
        },
        {
            "name": "prepare_real_token_write_payloads",
            "purpose": "Future conversion from approval token and audit payload previews into write-ready in-memory payloads.",
            "module_boundary": "future_executor_module_only",
            "implemented_in_v0_1": False,
            "invoked_in_v0_1": False,
        },
        {
            "name": "commit_real_token_write_atomically",
            "purpose": "Future atomic token and approval-audit write boundary after all human approval gates pass.",
            "module_boundary": "future_executor_module_only",
            "implemented_in_v0_1": False,
            "invoked_in_v0_1": False,
        },
        {
            "name": "verify_real_token_write_audit_trail",
            "purpose": "Future post-write verification of token record, approval audit record, and idempotency metadata.",
            "module_boundary": "future_executor_module_only",
            "implemented_in_v0_1": False,
            "invoked_in_v0_1": False,
        },
    ]


def _implementation_plan_files() -> list[dict[str, Any]]:
    return [
        {
            "path": "agent/memory_human_approval_token_real_write_executor.py",
            "purpose": "Future executor module boundary; v0.1 plans the boundary only.",
            "create_in_v0_1": False,
            "contains_executor_code_in_v0_1": False,
            "writes_files_in_v0_1": False,
        },
        {
            "path": "tests/agent/test_memory_human_approval_token_real_write_executor.py",
            "purpose": "Future executor tests after a separate dry-run gate approves implementation.",
            "create_in_v0_1": False,
            "contains_executor_code_in_v0_1": False,
            "writes_files_in_v0_1": False,
        },
        {
            "path": "benchmarks/hermes_memory_bench/fixtures/smoke_cases.json",
            "purpose": "Future smoke coverage for real executor behavior only after executor implementation is separately approved.",
            "create_in_v0_1": False,
            "contains_executor_code_in_v0_1": False,
            "writes_files_in_v0_1": False,
        },
    ]


def _implementation_plan_idempotency_strategy() -> dict[str, Any]:
    return {
        "strategy": "future_hash_guarded_token_and_audit_pair",
        "v0_1_effect": "plan_only_no_idempotency_state_written",
        "identity_inputs": [
            "source_contract_review_outcome_id",
            "source_contract_id",
            "approval_token_write_payload_preview",
            "approval_audit_write_payload_preview",
            "token_write_target_paths_preview",
        ],
        "future_rules": [
            "derive_deterministic_write_fingerprint_before_any_future_write",
            "treat_existing_matching_token_and_audit_pair_as_idempotent_success_only_after_revalidation",
            "abort_on_existing_mismatched_token_or_audit_payload",
            "never_use_operation_ledger_creation_as_idempotency_evidence_in_v0_1",
        ],
    }


def _implementation_plan_filesystem_safety_model() -> dict[str, Any]:
    return {
        "model": "future_preview_path_confinement_and_atomic_pair_write",
        "v0_1_effect": "no_filesystem_write_or_directory_creation",
        "future_checks": [
            "resolve_paths_from_token_write_target_paths_preview_only",
            "reject_absolute_paths_outside_hermes_memory_approval_token_scope",
            "reject_parent_directory_traversal_and_symlink_targets",
            "require_atomic_temp_file_replace_in_same_directory",
            "require_token_and_audit_write_pair_to_commit_or_abort_together",
            "fsync_file_and_directory_before_success_if_future_executor_is_approved",
        ],
    }


def _implementation_plan_audit_strategy() -> dict[str, Any]:
    return {
        "strategy": "preview_only_audit_plan",
        "v0_1_effect": "no_approval_audit_file_write_and_no_operation_ledger_event",
        "future_audit_fields": [
            "source_contract_review_outcome_id",
            "source_contract_id",
            "source_write_final_gate_id",
            "block_id",
            "block_type",
            "project_scope",
            "write_fingerprint",
            "approval_token_target_path",
            "approval_audit_target_path",
        ],
        "future_rules": [
            "audit_payload_must_derive_from_approval_audit_write_payload_preview",
            "audit_record_must_not_create_operation_ledger_events",
            "audit_success_requires_token_and_audit_payload_hash_match",
        ],
    }


def _implementation_plan_rollback_strategy() -> dict[str, Any]:
    return {
        "strategy": "no_write_no_rollback_in_v0_1",
        "v0_1_effect": "no_rollback_action_because_no_write_is_performed",
        "future_rules": [
            "abort_before_first_write_when_any_preflight_check_fails",
            "remove_only_future_temp_files_created_by_same_executor_attempt",
            "never_delete_preexisting_token_or_audit_files",
            "treat_partial_commit_as_failed_and_require_manual_recovery_packet",
            "never_rollback_by_writing_memory_graph_proposal_or_operation_ledger_state",
        ],
    }


def _implementation_plan_test_plan() -> list[dict[str, Any]]:
    return [
        {
            "id": "approved_contract_review_outcome_requires_implementation_plan",
            "scope": "unit",
            "writes": False,
        },
        {
            "id": "non_approved_or_invalid_contract_review_outcome_locks_plan",
            "scope": "unit",
            "writes": False,
        },
        {
            "id": "missing_previews_executor_fields_or_source_evidence_lock_plan",
            "scope": "unit",
            "writes": False,
        },
        {
            "id": "preview_and_contract_review_integrity_failures_lock_plan",
            "scope": "unit",
            "writes": False,
        },
        {
            "id": "deterministic_interfaces_files_idempotency_filesystem_audit_rollback_and_tests",
            "scope": "unit",
            "writes": False,
        },
        {
            "id": "benchmark_smoke_case_stays_read_only_and_executor_free",
            "scope": "benchmark",
            "writes": False,
        },
    ]


def _implementation_plan_forbidden_actions() -> list[str]:
    return [
        "write_memory",
        "write_memory_graph",
        "modify_openclaw_config",
        "approve_allowlists",
        "create_proposal_events",
        "create_operation_ledger_events",
        "create_real_memory_proposals",
        "write_proposal_files",
        "write_operation_ledger",
        "issue_real_approval_tokens",
        "persist_approvals",
        "write_token_files",
        "write_approval_audit_files",
        "invoke_real_token_write_executor",
        "implement_real_token_write_executor",
        "submit_to_governance",
        "convert_to_real_proposal",
    ]


def _plan_id(plan: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_VERSION,
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
        "lock_reason": plan.get("lock_reason"),
        "source_contract_review_outcome_id": plan.get(
            "source_contract_review_outcome_id"
        ),
        "source_contract_id": plan.get("source_contract_id"),
        "source_write_final_gate_id": plan.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": plan.get(
            "source_write_execution_dry_run_id"
        ),
        "source_write_execution_plan_id": plan.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": plan.get(
            "source_final_confirmation_review_outcome_id"
        ),
        "source_final_confirmation_request_id": plan.get(
            "source_final_confirmation_request_id"
        ),
        "source_token_write_lock_gate_id": plan.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": plan.get(
            "source_token_issuance_dry_run_id"
        ),
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
        "implementer": plan.get("implementer"),
        "outcome": plan.get("outcome"),
        "rationale": plan.get("rationale"),
        "approval_token_record_preview": plan.get("approval_token_record_preview", {}),
        "approval_audit_record_preview": plan.get("approval_audit_record_preview", {}),
        "token_target_paths_preview": plan.get("token_target_paths_preview", {}),
        "proposal_record_preview": plan.get("proposal_record_preview", {}),
        "operation_ledger_preview": plan.get("operation_ledger_preview", {}),
        "target_paths_preview": plan.get("target_paths_preview", {}),
        "payload_preview": plan.get("payload_preview"),
        "source_pattern_ids": list(plan.get("source_pattern_ids", [])),
        "source_fact_ids": list(plan.get("source_fact_ids", [])),
        "token_write_execution_steps": list(
            plan.get("token_write_execution_steps", [])
        ),
        "token_write_execution_preflight_checks": list(
            plan.get("token_write_execution_preflight_checks", [])
        ),
        "approval_token_write_payload_preview": plan.get(
            "approval_token_write_payload_preview", {}
        ),
        "approval_audit_write_payload_preview": plan.get(
            "approval_audit_write_payload_preview", {}
        ),
        "token_write_target_paths_preview": plan.get(
            "token_write_target_paths_preview", {}
        ),
        "final_token_write_preflight_checklist": list(
            plan.get("final_token_write_preflight_checklist") or []
        ),
        "final_gate_checklist": list(plan.get("final_gate_checklist") or []),
        "executor_contract_inputs": plan.get("executor_contract_inputs", {}),
        "executor_hard_lock_checks": list(plan.get("executor_hard_lock_checks") or []),
        "executor_audit_fields": list(plan.get("executor_audit_fields") or []),
        "executor_rollback_rules": list(plan.get("executor_rollback_rules") or []),
        "executor_forbidden_side_effects": list(
            plan.get("executor_forbidden_side_effects") or []
        ),
        "executor_contract_checklist": list(
            plan.get("executor_contract_checklist") or []
        ),
        "contract_review_checklist": list(plan.get("contract_review_checklist") or []),
        "implementation_plan_interfaces": plan.get("implementation_plan_interfaces", []),
        "implementation_plan_files": plan.get("implementation_plan_files", []),
        "implementation_plan_idempotency_strategy": plan.get(
            "implementation_plan_idempotency_strategy", {}
        ),
        "implementation_plan_filesystem_safety_model": plan.get(
            "implementation_plan_filesystem_safety_model", {}
        ),
        "implementation_plan_audit_strategy": plan.get(
            "implementation_plan_audit_strategy", {}
        ),
        "implementation_plan_rollback_strategy": plan.get(
            "implementation_plan_rollback_strategy", {}
        ),
        "implementation_plan_test_plan": plan.get("implementation_plan_test_plan", []),
        "implementation_plan_forbidden_actions": list(
            plan.get("implementation_plan_forbidden_actions") or []
        ),
        "contract_review_outcome_validation": plan.get(
            "contract_review_outcome_validation", {}
        ),
        "policy": plan.get("policy", {}),
    }
    return build_stable_digest(
        "memory-human-approval-token-real-write-executor-implementation-plan:v0.1",
        identity,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
