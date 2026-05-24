from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_real_write_executor_implementation_dry_run import (
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED,
    explain_human_approval_token_real_write_executor_implementation_dry_run,
    recommend_human_approval_token_real_write_executor_implementation_dry_run_action,
    validate_human_approval_token_real_write_executor_implementation_dry_run,
)


MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_KIND = (
    "memory_human_approval_token_real_write_executor_code_review_plan_candidate"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_GATE_REQUIRED = (
    "real_token_write_executor_code_review_gate_required"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_ROUTING = (
    "real_token_write_executor_code_review_gate_required_before_executor_source_creation"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_real_write_executor_code_review_plan_candidates_only": True,
    "invokes_real_token_write_executor": False,
    "implements_real_token_write_executor": False,
    "creates_executor_source_files": False,
    "creates_executor_tests": False,
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

_DEFAULT_REVIEWER = (
    "hermes_memory_human_approval_token_real_write_executor_code_review_plan_v0.1"
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
    "creates_executor_source_files",
    "creates_executor_tests",
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
_IMPLEMENTATION_PLAN_FIELD_TYPES = {
    "implementation_plan_interfaces": list,
    "implementation_plan_files": list,
    "implementation_plan_idempotency_strategy": Mapping,
    "implementation_plan_filesystem_safety_model": Mapping,
    "implementation_plan_audit_strategy": Mapping,
    "implementation_plan_rollback_strategy": Mapping,
    "implementation_plan_test_plan": list,
    "implementation_plan_forbidden_actions": list,
}
_IMPLEMENTATION_DRY_RUN_FIELD_TYPES = {
    "implementation_dry_run_module_boundary_preview": Mapping,
    "implementation_dry_run_interface_preview": list,
    "implementation_dry_run_idempotency_check_preview": Mapping,
    "implementation_dry_run_filesystem_safety_check_preview": Mapping,
    "implementation_dry_run_audit_check_preview": Mapping,
    "implementation_dry_run_rollback_check_preview": Mapping,
    "implementation_dry_run_test_harness_preview": list,
    "implementation_dry_run_readiness_checklist": list,
}
_LOCK_REASONS = {
    None,
    "real_write_executor_implementation_dry_run_locked",
    "real_write_executor_implementation_plan_locked",
    "invalid_real_write_executor_implementation_plan",
    "invalid_real_write_executor_implementation_dry_run",
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
    "missing_implementation_plan_interfaces",
    "missing_implementation_plan_files",
    "missing_implementation_plan_idempotency_strategy",
    "missing_implementation_plan_filesystem_safety_model",
    "missing_implementation_plan_audit_strategy",
    "missing_implementation_plan_rollback_strategy",
    "missing_implementation_plan_test_plan",
    "missing_implementation_plan_forbidden_actions",
    "missing_implementation_dry_run_module_boundary_preview",
    "missing_implementation_dry_run_interface_preview",
    "missing_implementation_dry_run_idempotency_check_preview",
    "missing_implementation_dry_run_filesystem_safety_check_preview",
    "missing_implementation_dry_run_audit_check_preview",
    "missing_implementation_dry_run_rollback_check_preview",
    "missing_implementation_dry_run_test_harness_preview",
    "missing_implementation_dry_run_readiness_checklist",
    "missing_source_evidence",
    "preview_integrity_failed",
    "implementation_plan_integrity_failed",
    "implementation_dry_run_integrity_failed",
}
_SOURCE_KEYS = (
    "source_contract_review_outcome_id",
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
    "source_implementation_dry_run_id",
    "source_implementation_plan_id",
) + _SOURCE_KEYS + (
    "reviewer",
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
    "implementation_dry_run_module_boundary_preview",
    "implementation_dry_run_interface_preview",
    "implementation_dry_run_idempotency_check_preview",
    "implementation_dry_run_filesystem_safety_check_preview",
    "implementation_dry_run_audit_check_preview",
    "implementation_dry_run_rollback_check_preview",
    "implementation_dry_run_test_harness_preview",
    "implementation_dry_run_readiness_checklist",
    "code_review_scope",
    "code_review_required_files",
    "code_review_static_analysis_checks",
    "code_review_security_checks",
    "code_review_write_safety_checks",
    "code_review_test_matrix",
    "code_review_acceptance_criteria",
    "code_review_forbidden_actions",
    "code_review_plan_checklist",
    "implementation_dry_run_validation",
    "code_review_plan_validation",
    "next_step_recommendation",
    "source_implementation_dry_run_snapshot",
    "policy",
)


def create_human_approval_token_real_write_executor_code_review_plan(
    implementation_dry_run: Mapping[str, Any],
    reviewer: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only real token write executor code review plan."""
    source = deepcopy(dict(implementation_dry_run))
    implementation_dry_run_validation = (
        validate_human_approval_token_real_write_executor_implementation_dry_run(
            source
        )
    )
    lock_reason = _code_review_plan_lock_reason(
        source,
        implementation_dry_run_validation,
    )
    plan_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_GATE_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED
    )

    plan = {
        "plan_id": None,
        "plan_kind": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_KIND,
        "plan_status": plan_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_ROUTING,
        "lock_reason": lock_reason,
        "source_implementation_dry_run_id": source.get("dry_run_id"),
        "source_implementation_plan_id": source.get("source_implementation_plan_id"),
        "source_contract_review_outcome_id": source.get("source_contract_review_outcome_id"),
        "source_contract_id": source.get("source_contract_id"),
        "source_write_final_gate_id": source.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": source.get("source_write_execution_dry_run_id"),
        "source_write_execution_plan_id": source.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": source.get(
            "source_final_confirmation_review_outcome_id"
        ),
        "source_final_confirmation_request_id": source.get(
            "source_final_confirmation_request_id"
        ),
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
        "reviewer": reviewer if reviewer is not None else _DEFAULT_REVIEWER,
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
        "implementation_plan_interfaces": deepcopy(source.get("implementation_plan_interfaces")),
        "implementation_plan_files": deepcopy(source.get("implementation_plan_files")),
        "implementation_plan_idempotency_strategy": deepcopy(
            source.get("implementation_plan_idempotency_strategy")
        ),
        "implementation_plan_filesystem_safety_model": deepcopy(
            source.get("implementation_plan_filesystem_safety_model")
        ),
        "implementation_plan_audit_strategy": deepcopy(
            source.get("implementation_plan_audit_strategy")
        ),
        "implementation_plan_rollback_strategy": deepcopy(
            source.get("implementation_plan_rollback_strategy")
        ),
        "implementation_plan_test_plan": deepcopy(source.get("implementation_plan_test_plan")),
        "implementation_plan_forbidden_actions": deepcopy(
            source.get("implementation_plan_forbidden_actions")
        ),
        "implementation_dry_run_module_boundary_preview": deepcopy(
            source.get("implementation_dry_run_module_boundary_preview")
        ),
        "implementation_dry_run_interface_preview": deepcopy(
            source.get("implementation_dry_run_interface_preview")
        ),
        "implementation_dry_run_idempotency_check_preview": deepcopy(
            source.get("implementation_dry_run_idempotency_check_preview")
        ),
        "implementation_dry_run_filesystem_safety_check_preview": deepcopy(
            source.get("implementation_dry_run_filesystem_safety_check_preview")
        ),
        "implementation_dry_run_audit_check_preview": deepcopy(
            source.get("implementation_dry_run_audit_check_preview")
        ),
        "implementation_dry_run_rollback_check_preview": deepcopy(
            source.get("implementation_dry_run_rollback_check_preview")
        ),
        "implementation_dry_run_test_harness_preview": deepcopy(
            source.get("implementation_dry_run_test_harness_preview")
        ),
        "implementation_dry_run_readiness_checklist": deepcopy(
            source.get("implementation_dry_run_readiness_checklist")
        ),
        "code_review_scope": _code_review_scope(),
        "code_review_required_files": _code_review_required_files(),
        "code_review_static_analysis_checks": _code_review_static_analysis_checks(),
        "code_review_security_checks": _code_review_security_checks(),
        "code_review_write_safety_checks": _code_review_write_safety_checks(),
        "code_review_test_matrix": _code_review_test_matrix(),
        "code_review_acceptance_criteria": _code_review_acceptance_criteria(),
        "code_review_forbidden_actions": _code_review_forbidden_actions(),
        "code_review_plan_checklist": _code_review_plan_checklist(),
        "implementation_dry_run_validation": deepcopy(
            implementation_dry_run_validation
        ),
        "code_review_plan_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_implementation_dry_run_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY),
    }
    plan["plan_id"] = _plan_id(plan)
    plan["code_review_plan_validation"] = (
        validate_human_approval_token_real_write_executor_code_review_plan(plan)
    )
    plan["next_step_recommendation"] = (
        recommend_human_approval_token_real_write_executor_code_review_plan_action(
            plan
        )
    )
    return plan


def validate_human_approval_token_real_write_executor_code_review_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(plan.get("policy"))
    source_snapshot = _as_dict(plan.get("source_implementation_dry_run_snapshot"))
    implementation_dry_run_validation = _as_dict(
        plan.get("implementation_dry_run_validation")
    )
    expected_implementation_dry_run_validation = (
        validate_human_approval_token_real_write_executor_implementation_dry_run(
            source_snapshot
        )
    )
    expected_lock_reason = _code_review_plan_lock_reason(
        source_snapshot,
        implementation_dry_run_validation,
    )

    for key in _REQUIRED_PLAN_KEYS:
        if key not in plan:
            errors.append(f"missing_{key}")
    if (
        plan.get("plan_kind")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_KIND
    ):
        errors.append(
            "plan_kind_must_be_memory_human_approval_token_real_write_executor_code_review_plan_candidate"
        )
    if plan.get("plan_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_GATE_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED,
    }:
        errors.append("plan_status_must_be_supported")
    if (
        plan.get("routing")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_ROUTING
    ):
        errors.append(
            "routing_must_require_code_review_gate_before_executor_source_creation"
        )
    if plan.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if plan.get("lock_reason") != expected_lock_reason:
        errors.append(
            "lock_reason_must_match_real_write_executor_code_review_plan_checks"
        )
    if (
        expected_lock_reason is None
        and plan.get("plan_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_GATE_REQUIRED
    ):
        errors.append(
            "valid_implementation_dry_run_must_require_real_token_write_executor_code_review_gate"
        )
    if (
        expected_lock_reason is not None
        and plan.get("plan_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED
    ):
        errors.append("invalid_or_locked_implementation_dry_run_must_lock_plan")
    if not isinstance(plan.get("reviewer"), str) or not plan.get("reviewer"):
        errors.append("reviewer_must_be_non_empty_string")
    if (
        implementation_dry_run_validation
        != expected_implementation_dry_run_validation
    ):
        errors.append(
            "implementation_dry_run_validation_must_match_source_implementation_dry_run_snapshot"
        )
    if plan.get("source_implementation_dry_run_id") != source_snapshot.get("dry_run_id"):
        errors.append("source_implementation_dry_run_id_must_match_source_snapshot")
    if (
        plan.get("source_implementation_plan_id")
        != source_snapshot.get("source_implementation_plan_id")
    ):
        errors.append("source_implementation_plan_id_must_match_source_snapshot")
    for source_key in _SOURCE_KEYS:
        if plan.get(source_key) != source_snapshot.get(source_key):
            errors.append(f"{source_key}_must_match_source_snapshot")

    if (
        not isinstance(plan.get("payload_preview"), Mapping)
        and plan.get("lock_reason") != "invalid_real_write_executor_implementation_dry_run"
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
    for field, expected_type in _IMPLEMENTATION_PLAN_FIELD_TYPES.items():
        if (
            not isinstance(plan.get(field), expected_type)
            and plan.get("lock_reason") != f"missing_{field}"
        ):
            errors.append(f"missing_{field}")
    for field, expected_type in _IMPLEMENTATION_DRY_RUN_FIELD_TYPES.items():
        if (
            not isinstance(plan.get(field), expected_type)
            and plan.get("lock_reason") != f"missing_{field}"
        ):
            errors.append(f"missing_{field}")
    if plan.get("lock_reason") != "preview_integrity_failed":
        errors.extend(_preview_integrity_errors(plan))
    if plan.get("lock_reason") is None:
        errors.extend(
            _implementation_dry_run_integrity_errors(
                plan,
                implementation_dry_run_validation,
            )
        )

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
        "implementation_plan_interfaces",
        "implementation_plan_files",
        "implementation_plan_idempotency_strategy",
        "implementation_plan_filesystem_safety_model",
        "implementation_plan_audit_strategy",
        "implementation_plan_rollback_strategy",
        "implementation_plan_test_plan",
        "implementation_plan_forbidden_actions",
        "implementation_dry_run_module_boundary_preview",
        "implementation_dry_run_interface_preview",
        "implementation_dry_run_idempotency_check_preview",
        "implementation_dry_run_filesystem_safety_check_preview",
        "implementation_dry_run_audit_check_preview",
        "implementation_dry_run_rollback_check_preview",
        "implementation_dry_run_test_harness_preview",
        "implementation_dry_run_readiness_checklist",
    ):
        if (
            field in plan
            and field in source_snapshot
            and plan.get(field) != source_snapshot.get(field)
        ):
            errors.append(
                f"{field}_must_match_source_implementation_dry_run_snapshot"
            )
    if plan.get("source_pattern_ids") != list(
        source_snapshot.get("source_pattern_ids", []) or []
    ):
        errors.append(
            "source_pattern_ids_must_match_source_implementation_dry_run_snapshot"
        )
    if plan.get("source_fact_ids") != list(
        source_snapshot.get("source_fact_ids", []) or []
    ):
        errors.append(
            "source_fact_ids_must_match_source_implementation_dry_run_snapshot"
        )

    deterministic_fields = (
        ("code_review_scope", _code_review_scope()),
        ("code_review_required_files", _code_review_required_files()),
        ("code_review_static_analysis_checks", _code_review_static_analysis_checks()),
        ("code_review_security_checks", _code_review_security_checks()),
        ("code_review_write_safety_checks", _code_review_write_safety_checks()),
        ("code_review_test_matrix", _code_review_test_matrix()),
        ("code_review_acceptance_criteria", _code_review_acceptance_criteria()),
        ("code_review_forbidden_actions", _code_review_forbidden_actions()),
        ("code_review_plan_checklist", _code_review_plan_checklist()),
    )
    for field, expected in deterministic_fields:
        if plan.get(field) != expected:
            errors.append(f"{field}_must_match_v0_1_deterministic_plan")
    for forbidden_key in _FORBIDDEN_TRUE_KEYS:
        if plan.get(forbidden_key) is True:
            errors.append(f"{forbidden_key}_must_be_false_or_absent")
    for key, expected in (
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY.items()
    ):
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_real_write_executor_code_review_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_real_write_executor_code_review_plan(
        plan
    )
    source_snapshot = _as_dict(plan.get("source_implementation_dry_run_snapshot"))
    return {
        "plan_id": plan.get("plan_id"),
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
        "lock_reason": plan.get("lock_reason"),
        "source_implementation_dry_run_id": plan.get("source_implementation_dry_run_id"),
        "source_implementation_plan_id": plan.get("source_implementation_plan_id"),
        "source_contract_review_outcome_id": plan.get(
            "source_contract_review_outcome_id"
        ),
        "source_contract_id": plan.get("source_contract_id"),
        "source_write_final_gate_id": plan.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": plan.get(
            "source_write_execution_dry_run_id"
        ),
        "source_write_execution_plan_id": plan.get("source_write_execution_plan_id"),
        "block_id": plan.get("block_id"),
        "block_type": plan.get("block_type"),
        "project_scope": plan.get("project_scope"),
        "reviewer": plan.get("reviewer"),
        "outcome": plan.get("outcome"),
        "rationale": plan.get("rationale"),
        "source_pattern_count": len(plan.get("source_pattern_ids", []) or []),
        "source_fact_count": len(plan.get("source_fact_ids", []) or []),
        "code_review_scope": deepcopy(plan.get("code_review_scope")),
        "code_review_required_files": deepcopy(plan.get("code_review_required_files")),
        "code_review_plan_checklist": deepcopy(plan.get("code_review_plan_checklist")),
        "validation": validation,
        "implementation_dry_run_explanation": (
            explain_human_approval_token_real_write_executor_implementation_dry_run(
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
        "creates_executor_source_files": False,
        "creates_executor_tests": False,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY),
    }


def recommend_human_approval_token_real_write_executor_code_review_plan_action(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_real_write_executor_code_review_plan(
        plan
    )
    source_snapshot = _as_dict(plan.get("source_implementation_dry_run_snapshot"))
    if (
        validation["valid"]
        and plan.get("plan_status")
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_GATE_REQUIRED
    ):
        action = (
            "route_to_real_token_write_executor_code_review_gate_without_executor_source_creation"
        )
        reason = "Code review plan candidate is ready for a separate code review gate; it does not create executor source files, create executor tests, implement or invoke the executor, issue tokens, persist approvals, or write proposal, ledger, token, audit, memory, graph, or config state."
    elif validation["valid"]:
        action = "keep_real_token_write_executor_code_review_plan_locked"
        reason = f"Code review plan candidate is locked by {plan.get('lock_reason')}."
    else:
        action = "repair_real_token_write_executor_code_review_plan_candidate"
        reason = "Code review plan candidate failed validation and cannot proceed."
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_ROUTING,
        "validation": validation,
        "implementation_dry_run_recommendation": (
            recommend_human_approval_token_real_write_executor_implementation_dry_run_action(
                source_snapshot
            )
            if source_snapshot
            else {}
        ),
        "creates_real_write_executor_code_review_plan_candidates_only": True,
        "invokes_real_token_write_executor": False,
        "implements_real_token_write_executor": False,
        "creates_executor_source_files": False,
        "creates_executor_tests": False,
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY),
    }


def summarize_human_approval_token_real_write_executor_code_review_plans(
    plans: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    by_block_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_lock_reason: dict[str, int] = {}
    code_review_gate_required_count = 0
    locked_count = 0
    valid_count = 0
    invalid_count = 0
    for plan in plans:
        block_type = str(plan.get("block_type"))
        by_block_type[block_type] = by_block_type.get(block_type, 0) + 1
        status = str(plan.get("plan_status"))
        by_status[status] = by_status.get(status, 0) + 1
        lock_reason = str(plan.get("lock_reason"))
        by_lock_reason[lock_reason] = by_lock_reason.get(lock_reason, 0) + 1
        if (
            plan.get("plan_status")
            == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_GATE_REQUIRED
        ):
            code_review_gate_required_count += 1
        if (
            plan.get("plan_status")
            == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED
        ):
            locked_count += 1
        validation = validate_human_approval_token_real_write_executor_code_review_plan(
            plan
        )
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(plans),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "real_token_write_executor_code_review_gate_required_count": (
            code_review_gate_required_count
        ),
        "locked_count": locked_count,
        "by_block_type": dict(sorted(by_block_type.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_lock_reason": dict(sorted(by_lock_reason.items())),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY),
    }


def _code_review_plan_lock_reason(
    implementation_dry_run: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str | None:
    if (
        implementation_dry_run.get("dry_run_status")
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED
    ):
        return (
            implementation_dry_run.get("lock_reason")
            or "real_write_executor_implementation_dry_run_locked"
        )
    for field in _PREVIEW_FIELDS:
        if not isinstance(implementation_dry_run.get(field), Mapping):
            return f"missing_{field}"
    for field in _WRITE_PREVIEW_FIELDS:
        if not isinstance(implementation_dry_run.get(field), Mapping):
            return f"missing_{field}"
    for field, expected_type in _EXECUTOR_CONTRACT_FIELD_TYPES.items():
        if not isinstance(implementation_dry_run.get(field), expected_type):
            return f"missing_{field}"
    if not isinstance(implementation_dry_run.get("contract_review_checklist"), list):
        return "missing_contract_review_checklist"
    for field, expected_type in _IMPLEMENTATION_PLAN_FIELD_TYPES.items():
        if not isinstance(implementation_dry_run.get(field), expected_type):
            return f"missing_{field}"
    for field, expected_type in _IMPLEMENTATION_DRY_RUN_FIELD_TYPES.items():
        if not isinstance(implementation_dry_run.get(field), expected_type):
            return f"missing_{field}"
    if not (
        implementation_dry_run.get("source_pattern_ids")
        or implementation_dry_run.get("source_fact_ids")
    ):
        return "missing_source_evidence"
    if _preview_integrity_errors(implementation_dry_run):
        return "preview_integrity_failed"
    if _implementation_dry_run_integrity_errors(implementation_dry_run, validation):
        return "implementation_dry_run_integrity_failed"
    if validation.get("valid") is not True:
        return "invalid_real_write_executor_implementation_dry_run"
    if (
        implementation_dry_run.get("dry_run_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED
    ):
        return "invalid_real_write_executor_implementation_dry_run"
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


def _implementation_dry_run_integrity_errors(
    implementation_dry_run: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    module_boundary = implementation_dry_run.get(
        "implementation_dry_run_module_boundary_preview"
    )
    if isinstance(module_boundary, Mapping):
        expected_false_keys = (
            "module_created_in_v0_1",
            "creates_executor_source_files",
            "implements_real_token_write_executor",
            "invokes_real_token_write_executor",
            "writes_files",
        )
        if module_boundary.get("preview_only") is not True:
            errors.append("implementation_dry_run_module_boundary_must_be_preview_only")
        for key in expected_false_keys:
            if module_boundary.get(key) is not False:
                errors.append(f"implementation_dry_run_module_boundary_{key}_must_be_false")
    interfaces = implementation_dry_run.get("implementation_dry_run_interface_preview")
    if isinstance(interfaces, list):
        for interface in interfaces:
            if not isinstance(interface, Mapping):
                errors.append("implementation_dry_run_interface_preview_must_be_structured")
                continue
            if interface.get("preview_only") is not True:
                errors.append("implementation_dry_run_interface_preview_must_be_preview_only")
            if interface.get("implemented_in_v0_1") is not False:
                errors.append("implementation_dry_run_interfaces_must_not_implement_executor")
            if interface.get("invoked_in_v0_1") is not False:
                errors.append("implementation_dry_run_interfaces_must_not_invoke_executor")
            if interface.get("writes_files") is not False:
                errors.append("implementation_dry_run_interfaces_must_not_write_files")
    check_expectations = {
        "implementation_dry_run_idempotency_check_preview": (
            "writes_token_files",
            "writes_approval_audit",
            "writes_operation_ledger",
            "written",
        ),
        "implementation_dry_run_filesystem_safety_check_preview": (
            "creates_executor_source_files",
            "writes_token_files",
            "writes_approval_audit",
            "writes_proposal_files",
            "writes_operation_ledger",
            "written",
        ),
        "implementation_dry_run_audit_check_preview": (
            "created_operation_event",
            "writes_approval_audit",
            "writes_operation_ledger",
            "written",
        ),
        "implementation_dry_run_rollback_check_preview": (
            "writes_token_files",
            "writes_approval_audit",
            "writes_operation_ledger",
            "written",
        ),
    }
    for field, false_keys in check_expectations.items():
        preview = implementation_dry_run.get(field)
        if not isinstance(preview, Mapping):
            continue
        if preview.get("preview_only") is not True:
            errors.append(f"{field}_must_be_preview_only")
        for key in false_keys:
            if preview.get(key) is not False:
                errors.append(f"{field}_{key}_must_be_false")
    test_harness = implementation_dry_run.get("implementation_dry_run_test_harness_preview")
    if isinstance(test_harness, list):
        for test_case in test_harness:
            if not isinstance(test_case, Mapping):
                errors.append("implementation_dry_run_test_harness_preview_must_be_structured")
                continue
            if test_case.get("creates_files") is not False:
                errors.append("implementation_dry_run_test_harness_must_not_create_files")
            if test_case.get("writes") is not False:
                errors.append("implementation_dry_run_test_harness_must_not_write")
    readiness = implementation_dry_run.get("implementation_dry_run_readiness_checklist")
    if isinstance(readiness, list):
        required_readiness = {
            "no_real_approval_token_issued",
            "no_approval_persisted",
            "no_real_proposal_created",
            "no_operation_ledger_event_created",
            "no_proposal_file_written",
            "no_operation_ledger_written",
            "no_token_file_written",
            "no_approval_audit_written",
            "no_memory_or_graph_or_config_write",
            "real_token_write_executor_not_invoked",
            "real_token_write_executor_not_implemented",
            "executor_source_files_not_created",
            "code_review_plan_required_before_any_executor_code",
        }
        if required_readiness.difference(readiness):
            errors.append(
                "implementation_dry_run_readiness_checklist_must_preserve_no_write_boundary"
            )
    for error in list(validation.get("errors", []) or []):
        if (
            error.startswith("implementation_dry_run_")
            or error.endswith("_must_match_source_implementation_plan_snapshot")
        ):
            errors.append("implementation_dry_run_validation_integrity_errors")
    return _dedupe(errors)


def _code_review_scope() -> dict[str, Any]:
    return {
        "scope_kind": "real_token_write_executor_code_review_scope",
        "read_only": True,
        "review_only": True,
        "creates_executor_source_files": False,
        "creates_executor_tests": False,
        "implements_real_token_write_executor": False,
        "invokes_real_token_write_executor": False,
        "review_targets": [
            "future_executor_module_boundary",
            "future_executor_input_validation_contract",
            "future_token_and_audit_payload_preparation",
            "future_filesystem_atomicity_and_path_safety",
            "future_idempotency_and_rollback_controls",
            "future_no_proposal_or_operation_ledger_write_boundary",
        ],
    }


def _code_review_required_files() -> list[dict[str, Any]]:
    return [
        {
            "path": "agent/memory_human_approval_token_real_write_executor.py",
            "role": "future_executor_module",
            "required_before_creation": True,
            "create_in_v0_1": False,
            "contains_executor_code_in_v0_1": False,
            "writes_files_in_v0_1": False,
        },
        {
            "path": "tests/agent/test_memory_human_approval_token_real_write_executor.py",
            "role": "future_executor_test_module",
            "required_before_creation": True,
            "create_in_v0_1": False,
            "contains_executor_code_in_v0_1": False,
            "writes_files_in_v0_1": False,
        },
        {
            "path": "agent/memory_human_approval_token_real_write_executor_code_review_plan.py",
            "role": "current_read_only_code_review_plan_module",
            "required_before_creation": True,
            "create_in_v0_1": False,
            "contains_executor_code_in_v0_1": False,
            "writes_files_in_v0_1": False,
        },
    ]


def _code_review_static_analysis_checks() -> list[dict[str, Any]]:
    return [
        {
            "id": "no_executor_source_created_by_plan",
            "expectation": "code_review_plan_module_must_not_create_future_executor_module_or_tests",
            "writes": False,
        },
        {
            "id": "public_surface_is_read_only_candidate_builder",
            "expectation": "public_functions_create_validate_explain_recommend_and_summarize_only",
            "writes": False,
        },
        {
            "id": "no_dynamic_write_dispatch",
            "expectation": "no_open_write_append_touch_replace_unlink_mkdir_or_operation_ledger_calls_in_plan_path",
            "writes": False,
        },
        {
            "id": "deterministic_identifiers",
            "expectation": "plan_id_must_be_sha256_of_stable_json_identity",
            "writes": False,
        },
    ]


def _code_review_security_checks() -> list[dict[str, Any]]:
    return [
        {
            "id": "path_safety_review_required_before_future_executor",
            "expectation": "future_executor_must_reject_traversal_symlinks_and_unexpected_absolute_paths",
            "creates_executor_source_files": False,
        },
        {
            "id": "token_payload_integrity_review_required",
            "expectation": "future_executor_must_hash_and_compare_token_and_audit_payloads_before_any_real_write",
            "writes_token_files": False,
        },
        {
            "id": "audit_boundary_review_required",
            "expectation": "future_executor_must_not_create_operation_ledger_events_or_proposal_records",
            "writes_operation_ledger": False,
        },
        {
            "id": "approval_authority_review_required",
            "expectation": "future_executor_must_require_governed_human_approval_token_inputs_before_real_write_eligibility",
            "persists_approvals": False,
        },
    ]


def _code_review_write_safety_checks() -> list[dict[str, Any]]:
    return [
        {
            "id": "assert_no_token_file_write_in_v0_1",
            "assertion": "writes_token_files_false",
            "writes_token_files": False,
        },
        {
            "id": "assert_no_approval_audit_write_in_v0_1",
            "assertion": "writes_approval_audit_false",
            "writes_approval_audit": False,
        },
        {
            "id": "assert_no_proposal_file_write_in_v0_1",
            "assertion": "writes_proposal_files_false",
            "writes_proposal_files": False,
        },
        {
            "id": "assert_no_operation_ledger_write_in_v0_1",
            "assertion": "writes_operation_ledger_false",
            "writes_operation_ledger": False,
            "created_operation_event": False,
        },
        {
            "id": "assert_no_memory_graph_or_config_write_in_v0_1",
            "assertion": "memory_graph_and_config_writes_false",
            "would_write_memory": False,
            "would_write_graph": False,
            "would_modify_config": False,
        },
    ]


def _code_review_test_matrix() -> list[dict[str, Any]]:
    return [
        {
            "id": "valid_implementation_dry_run_creates_code_review_gate_required_plan",
            "scope": "unit",
            "writes": False,
            "creates_executor_source_files": False,
            "creates_executor_tests": False,
        },
        {
            "id": "locked_invalid_missing_and_integrity_failures_lock_plan",
            "scope": "unit",
            "writes": False,
            "creates_executor_source_files": False,
            "creates_executor_tests": False,
        },
        {
            "id": "deterministic_code_review_checks_do_not_create_executor_code",
            "scope": "unit",
            "writes": False,
            "creates_executor_source_files": False,
            "creates_executor_tests": False,
        },
        {
            "id": "benchmark_smoke_case_stays_read_only_executor_free_and_source_file_free",
            "scope": "benchmark",
            "writes": False,
            "creates_executor_source_files": False,
            "creates_executor_tests": False,
        },
    ]


def _code_review_acceptance_criteria() -> list[dict[str, Any]]:
    return [
        {
            "id": "valid_source_only",
            "requirement": "source_implementation_dry_run_validation_is_valid_and_status_is_code_review_plan_required",
            "required": True,
        },
        {
            "id": "no_executor_source_creation_in_v0_1",
            "requirement": "executor_source_files_and_executor_tests_must_not_be_created_in_v0_1",
            "required": True,
            "creates_executor_source_files": False,
            "creates_executor_tests": False,
        },
        {
            "id": "no_executor_implementation_or_invocation",
            "requirement": "plan_must_not_implement_or_invoke_real_token_write_executor",
            "required": True,
        },
        {
            "id": "no_token_audit_proposal_or_ledger_writes",
            "requirement": "plan_must_not_write_token_audit_proposal_or_operation_ledger_files",
            "required": True,
        },
    ]


def _code_review_forbidden_actions() -> list[str]:
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
        "create_executor_source_files",
        "create_executor_tests",
        "submit_to_governance",
        "convert_to_real_proposal",
    ]


def _code_review_plan_checklist() -> list[str]:
    return [
        "source_implementation_dry_run_is_valid_and_code_review_plan_required",
        "approval_token_record_preview_is_preview_only",
        "approval_audit_record_preview_is_preview_only",
        "token_target_paths_preview_is_preview_only",
        "proposal_record_preview_is_not_written",
        "operation_ledger_preview_is_not_written",
        "target_paths_preview_is_preview_only",
        "approval_token_write_payload_preview_is_preview_only",
        "approval_audit_write_payload_preview_is_preview_only",
        "token_write_target_paths_preview_is_preview_only",
        "executor_contract_inputs_are_present",
        "executor_contract_checklist_is_present",
        "implementation_plan_interfaces_are_present_and_not_implemented",
        "implementation_plan_files_are_present_and_not_created",
        "implementation_dry_run_previews_are_present_and_read_only",
        "code_review_scope_is_read_only",
        "code_review_required_files_are_not_created_in_v0_1",
        "code_review_static_analysis_checks_are_defined",
        "code_review_security_checks_are_defined",
        "code_review_write_safety_checks_assert_no_token_audit_proposal_or_ledger_writes",
        "code_review_test_matrix_has_no_write_cases",
        "code_review_acceptance_criteria_require_no_executor_source_creation",
        "no_real_approval_token_issued",
        "no_approval_persisted",
        "no_real_proposal_created",
        "no_operation_ledger_event_created",
        "no_proposal_file_written",
        "no_operation_ledger_written",
        "no_token_file_written",
        "no_approval_audit_written",
        "no_memory_or_graph_or_config_write",
        "real_token_write_executor_not_invoked",
        "real_token_write_executor_not_implemented",
        "executor_source_files_not_created",
        "executor_tests_not_created",
        "code_review_gate_required_before_executor_source_creation",
    ]


def _plan_id(plan: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_VERSION,
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
        "lock_reason": plan.get("lock_reason"),
        "source_implementation_dry_run_id": plan.get("source_implementation_dry_run_id"),
        "source_implementation_plan_id": plan.get("source_implementation_plan_id"),
        "source_contract_review_outcome_id": plan.get("source_contract_review_outcome_id"),
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
        "reviewer": plan.get("reviewer"),
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
        "implementation_dry_run_module_boundary_preview": plan.get(
            "implementation_dry_run_module_boundary_preview", {}
        ),
        "implementation_dry_run_interface_preview": plan.get(
            "implementation_dry_run_interface_preview", []
        ),
        "implementation_dry_run_idempotency_check_preview": plan.get(
            "implementation_dry_run_idempotency_check_preview", {}
        ),
        "implementation_dry_run_filesystem_safety_check_preview": plan.get(
            "implementation_dry_run_filesystem_safety_check_preview", {}
        ),
        "implementation_dry_run_audit_check_preview": plan.get(
            "implementation_dry_run_audit_check_preview", {}
        ),
        "implementation_dry_run_rollback_check_preview": plan.get(
            "implementation_dry_run_rollback_check_preview", {}
        ),
        "implementation_dry_run_test_harness_preview": plan.get(
            "implementation_dry_run_test_harness_preview", []
        ),
        "implementation_dry_run_readiness_checklist": list(
            plan.get("implementation_dry_run_readiness_checklist") or []
        ),
        "code_review_scope": plan.get("code_review_scope", {}),
        "code_review_required_files": plan.get("code_review_required_files", []),
        "code_review_static_analysis_checks": plan.get(
            "code_review_static_analysis_checks", []
        ),
        "code_review_security_checks": plan.get("code_review_security_checks", []),
        "code_review_write_safety_checks": plan.get(
            "code_review_write_safety_checks", []
        ),
        "code_review_test_matrix": plan.get("code_review_test_matrix", []),
        "code_review_acceptance_criteria": plan.get(
            "code_review_acceptance_criteria", []
        ),
        "code_review_forbidden_actions": list(
            plan.get("code_review_forbidden_actions") or []
        ),
        "code_review_plan_checklist": list(
            plan.get("code_review_plan_checklist") or []
        ),
        "implementation_dry_run_validation": plan.get(
            "implementation_dry_run_validation", {}
        ),
        "policy": plan.get("policy", {}),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return (
        "memory-human-approval-token-real-write-executor-code-review-plan:"
        f"v0.1:{digest}"
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
