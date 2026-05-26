from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_real_write_executor_implementation_plan import (
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED,
    explain_human_approval_token_real_write_executor_implementation_plan,
    recommend_human_approval_token_real_write_executor_implementation_plan_action,
    validate_human_approval_token_real_write_executor_implementation_plan,
)
from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_forbidden_true_keys_false_or_absent,
    validate_policy_flags,
)


MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_KIND = (
    "memory_human_approval_token_real_write_executor_implementation_dry_run_candidate"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED = (
    "real_token_write_executor_code_review_plan_required"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_ROUTING = (
    "real_token_write_executor_code_review_plan_required_before_any_executor_code"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_real_write_executor_implementation_dry_run_candidates_only": True,
    "invokes_real_token_write_executor": False,
    "implements_real_token_write_executor": False,
    "creates_executor_source_files": False,
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

_DEFAULT_OPERATOR = (
    "hermes_memory_human_approval_token_real_write_executor_implementation_dry_run_v0.1"
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
_LOCK_REASONS = {
    None,
    "real_write_executor_implementation_plan_locked",
    "invalid_real_write_executor_implementation_plan",
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
    "missing_source_evidence",
    "preview_integrity_failed",
    "implementation_plan_integrity_failed",
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
_REQUIRED_DRY_RUN_KEYS = (
    "dry_run_id",
    "dry_run_kind",
    "dry_run_status",
    "routing",
    "lock_reason",
    "source_implementation_plan_id",
) + _SOURCE_KEYS + (
    "operator",
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
    "implementation_plan_validation",
    "dry_run_validation",
    "next_step_recommendation",
    "source_implementation_plan_snapshot",
    "policy",
)


def create_human_approval_token_real_write_executor_implementation_dry_run(
    implementation_plan: Mapping[str, Any],
    operator: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only real token write executor implementation dry run."""
    source = deepcopy(dict(implementation_plan))
    implementation_plan_validation = (
        validate_human_approval_token_real_write_executor_implementation_plan(source)
    )
    lock_reason = _dry_run_lock_reason(source, implementation_plan_validation)
    dry_run_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED
    )

    dry_run = {
        "dry_run_id": None,
        "dry_run_kind": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_KIND,
        "dry_run_status": dry_run_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_ROUTING,
        "lock_reason": lock_reason,
        "source_implementation_plan_id": source.get("plan_id"),
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
        "operator": operator if operator is not None else _DEFAULT_OPERATOR,
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
        "implementation_dry_run_module_boundary_preview": (
            _implementation_dry_run_module_boundary_preview()
        ),
        "implementation_dry_run_interface_preview": _implementation_dry_run_interface_preview(),
        "implementation_dry_run_idempotency_check_preview": (
            _implementation_dry_run_idempotency_check_preview()
        ),
        "implementation_dry_run_filesystem_safety_check_preview": (
            _implementation_dry_run_filesystem_safety_check_preview()
        ),
        "implementation_dry_run_audit_check_preview": (
            _implementation_dry_run_audit_check_preview()
        ),
        "implementation_dry_run_rollback_check_preview": (
            _implementation_dry_run_rollback_check_preview()
        ),
        "implementation_dry_run_test_harness_preview": (
            _implementation_dry_run_test_harness_preview()
        ),
        "implementation_dry_run_readiness_checklist": (
            _implementation_dry_run_readiness_checklist()
        ),
        "implementation_plan_validation": deepcopy(implementation_plan_validation),
        "dry_run_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_implementation_plan_snapshot": deepcopy(source),
        "policy": dict(
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY
        ),
    }
    dry_run["dry_run_id"] = _dry_run_id(dry_run)
    dry_run["dry_run_validation"] = (
        validate_human_approval_token_real_write_executor_implementation_dry_run(
            dry_run
        )
    )
    dry_run["next_step_recommendation"] = (
        recommend_human_approval_token_real_write_executor_implementation_dry_run_action(
            dry_run
        )
    )
    return dry_run


def validate_human_approval_token_real_write_executor_implementation_dry_run(
    dry_run: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(dry_run.get("policy"))
    source_snapshot = _as_dict(dry_run.get("source_implementation_plan_snapshot"))
    implementation_plan_validation = _as_dict(dry_run.get("implementation_plan_validation"))
    expected_implementation_plan_validation = (
        validate_human_approval_token_real_write_executor_implementation_plan(
            source_snapshot
        )
    )
    expected_lock_reason = _dry_run_lock_reason(
        source_snapshot,
        implementation_plan_validation,
    )

    for key in _REQUIRED_DRY_RUN_KEYS:
        if key not in dry_run:
            errors.append(f"missing_{key}")
    if (
        dry_run.get("dry_run_kind")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_KIND
    ):
        errors.append(
            "dry_run_kind_must_be_memory_human_approval_token_real_write_executor_implementation_dry_run_candidate"
        )
    if dry_run.get("dry_run_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED,
    }:
        errors.append("dry_run_status_must_be_supported")
    if (
        dry_run.get("routing")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_ROUTING
    ):
        errors.append(
            "routing_must_require_code_review_plan_before_any_executor_code"
        )
    if dry_run.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if dry_run.get("lock_reason") != expected_lock_reason:
        errors.append(
            "lock_reason_must_match_real_write_executor_implementation_dry_run_checks"
        )
    if (
        expected_lock_reason is None
        and dry_run.get("dry_run_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED
    ):
        errors.append(
            "valid_implementation_plan_must_require_real_token_write_executor_code_review_plan"
        )
    if (
        expected_lock_reason is not None
        and dry_run.get("dry_run_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED
    ):
        errors.append("invalid_or_locked_implementation_plan_must_lock_dry_run")
    if not isinstance(dry_run.get("operator"), str) or not dry_run.get("operator"):
        errors.append("operator_must_be_non_empty_string")
    if (
        implementation_plan_validation
        != expected_implementation_plan_validation
    ):
        errors.append(
            "implementation_plan_validation_must_match_source_implementation_plan_snapshot"
        )
    if dry_run.get("source_implementation_plan_id") != source_snapshot.get("plan_id"):
        errors.append("source_implementation_plan_id_must_match_source_snapshot")
    for source_key in _SOURCE_KEYS:
        if dry_run.get(source_key) != source_snapshot.get(source_key):
            errors.append(f"{source_key}_must_match_source_snapshot")

    if (
        not isinstance(dry_run.get("payload_preview"), Mapping)
        and dry_run.get("lock_reason") != "invalid_real_write_executor_implementation_plan"
    ):
        errors.append("missing_payload_preview")
    if not isinstance(dry_run.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(dry_run.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(dry_run.get("source_pattern_ids"), list) and isinstance(
        dry_run.get("source_fact_ids"), list
    ):
        if (
            not dry_run.get("source_pattern_ids")
            and not dry_run.get("source_fact_ids")
            and dry_run.get("lock_reason") != "missing_source_evidence"
        ):
            errors.append("missing_source_evidence")
    for field in _PREVIEW_FIELDS + _WRITE_PREVIEW_FIELDS:
        if (
            not isinstance(dry_run.get(field), Mapping)
            and dry_run.get("lock_reason") != f"missing_{field}"
        ):
            errors.append(f"missing_{field}")
    for field, expected_type in _EXECUTOR_CONTRACT_FIELD_TYPES.items():
        if (
            not isinstance(dry_run.get(field), expected_type)
            and dry_run.get("lock_reason") != f"missing_{field}"
        ):
            errors.append(f"missing_{field}")
    if (
        not isinstance(dry_run.get("contract_review_checklist"), list)
        and dry_run.get("lock_reason") != "missing_contract_review_checklist"
    ):
        errors.append("missing_contract_review_checklist")
    for field, expected_type in _IMPLEMENTATION_PLAN_FIELD_TYPES.items():
        if (
            not isinstance(dry_run.get(field), expected_type)
            and dry_run.get("lock_reason") != f"missing_{field}"
        ):
            errors.append(f"missing_{field}")
    if dry_run.get("lock_reason") != "preview_integrity_failed":
        errors.extend(_preview_integrity_errors(dry_run))
    if dry_run.get("lock_reason") is None:
        errors.extend(
            _implementation_plan_integrity_errors(
                dry_run,
                implementation_plan_validation,
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
    ):
        if (
            field in dry_run
            and field in source_snapshot
            and dry_run.get(field) != source_snapshot.get(field)
        ):
            errors.append(f"{field}_must_match_source_implementation_plan_snapshot")
    if dry_run.get("source_pattern_ids") != list(
        source_snapshot.get("source_pattern_ids", []) or []
    ):
        errors.append("source_pattern_ids_must_match_source_implementation_plan_snapshot")
    if dry_run.get("source_fact_ids") != list(
        source_snapshot.get("source_fact_ids", []) or []
    ):
        errors.append("source_fact_ids_must_match_source_implementation_plan_snapshot")

    if (
        dry_run.get("implementation_dry_run_module_boundary_preview")
        != _implementation_dry_run_module_boundary_preview()
    ):
        errors.append(
            "implementation_dry_run_module_boundary_preview_must_match_v0_1_deterministic_preview"
        )
    if (
        dry_run.get("implementation_dry_run_interface_preview")
        != _implementation_dry_run_interface_preview()
    ):
        errors.append(
            "implementation_dry_run_interface_preview_must_match_v0_1_deterministic_preview"
        )
    if (
        dry_run.get("implementation_dry_run_idempotency_check_preview")
        != _implementation_dry_run_idempotency_check_preview()
    ):
        errors.append(
            "implementation_dry_run_idempotency_check_preview_must_match_v0_1_deterministic_preview"
        )
    if (
        dry_run.get("implementation_dry_run_filesystem_safety_check_preview")
        != _implementation_dry_run_filesystem_safety_check_preview()
    ):
        errors.append(
            "implementation_dry_run_filesystem_safety_check_preview_must_match_v0_1_deterministic_preview"
        )
    if (
        dry_run.get("implementation_dry_run_audit_check_preview")
        != _implementation_dry_run_audit_check_preview()
    ):
        errors.append(
            "implementation_dry_run_audit_check_preview_must_match_v0_1_deterministic_preview"
        )
    if (
        dry_run.get("implementation_dry_run_rollback_check_preview")
        != _implementation_dry_run_rollback_check_preview()
    ):
        errors.append(
            "implementation_dry_run_rollback_check_preview_must_match_v0_1_deterministic_preview"
        )
    if (
        dry_run.get("implementation_dry_run_test_harness_preview")
        != _implementation_dry_run_test_harness_preview()
    ):
        errors.append(
            "implementation_dry_run_test_harness_preview_must_match_v0_1_deterministic_preview"
        )
    if (
        dry_run.get("implementation_dry_run_readiness_checklist")
        != _implementation_dry_run_readiness_checklist()
    ):
        errors.append(
            "implementation_dry_run_readiness_checklist_must_match_v0_1_deterministic_preview"
        )
    errors.extend(
        validate_forbidden_true_keys_false_or_absent(
            dry_run,
            _FORBIDDEN_TRUE_KEYS,
        )
    )
    errors.extend(
        validate_policy_flags(
            policy,
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY,
        )
    )

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_real_write_executor_implementation_dry_run(
    dry_run: Mapping[str, Any],
) -> dict[str, Any]:
    validation = (
        validate_human_approval_token_real_write_executor_implementation_dry_run(
            dry_run
        )
    )
    source_snapshot = _as_dict(dry_run.get("source_implementation_plan_snapshot"))
    return {
        "dry_run_id": dry_run.get("dry_run_id"),
        "dry_run_kind": dry_run.get("dry_run_kind"),
        "dry_run_status": dry_run.get("dry_run_status"),
        "routing": dry_run.get("routing"),
        "lock_reason": dry_run.get("lock_reason"),
        "source_implementation_plan_id": dry_run.get("source_implementation_plan_id"),
        "source_contract_review_outcome_id": dry_run.get(
            "source_contract_review_outcome_id"
        ),
        "source_contract_id": dry_run.get("source_contract_id"),
        "source_write_final_gate_id": dry_run.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": dry_run.get(
            "source_write_execution_dry_run_id"
        ),
        "source_write_execution_plan_id": dry_run.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": dry_run.get(
            "source_final_confirmation_review_outcome_id"
        ),
        "block_id": dry_run.get("block_id"),
        "block_type": dry_run.get("block_type"),
        "project_scope": dry_run.get("project_scope"),
        "operator": dry_run.get("operator"),
        "outcome": dry_run.get("outcome"),
        "rationale": dry_run.get("rationale"),
        "source_pattern_count": len(dry_run.get("source_pattern_ids", []) or []),
        "source_fact_count": len(dry_run.get("source_fact_ids", []) or []),
        "implementation_dry_run_module_boundary_preview": deepcopy(
            dry_run.get("implementation_dry_run_module_boundary_preview")
        ),
        "implementation_dry_run_interface_preview": deepcopy(
            dry_run.get("implementation_dry_run_interface_preview")
        ),
        "implementation_dry_run_readiness_checklist": deepcopy(
            dry_run.get("implementation_dry_run_readiness_checklist")
        ),
        "validation": validation,
        "implementation_plan_explanation": (
            explain_human_approval_token_real_write_executor_implementation_plan(
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
        "policy": dict(
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY
        ),
    }


def recommend_human_approval_token_real_write_executor_implementation_dry_run_action(
    dry_run: Mapping[str, Any],
) -> dict[str, Any]:
    validation = (
        validate_human_approval_token_real_write_executor_implementation_dry_run(
            dry_run
        )
    )
    source_snapshot = _as_dict(dry_run.get("source_implementation_plan_snapshot"))
    if (
        validation["valid"]
        and dry_run.get("dry_run_status")
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED
    ):
        action = (
            "route_to_real_token_write_executor_code_review_plan_without_executor_code_or_invocation"
        )
        reason = "Implementation dry-run candidate is ready for a separate code review plan; it does not create executor source files, implement or invoke the executor, issue tokens, persist approvals, or write proposal, ledger, token, audit, memory, graph, or config state."
    elif validation["valid"]:
        action = "keep_real_token_write_executor_implementation_dry_run_locked"
        reason = f"Implementation dry-run candidate is locked by {dry_run.get('lock_reason')}."
    else:
        action = "repair_real_token_write_executor_implementation_dry_run_candidate"
        reason = "Implementation dry-run candidate failed validation and cannot proceed."
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_ROUTING,
        "validation": validation,
        "implementation_plan_recommendation": (
            recommend_human_approval_token_real_write_executor_implementation_plan_action(
                source_snapshot
            )
            if source_snapshot
            else {}
        ),
        "creates_real_write_executor_implementation_dry_run_candidates_only": True,
        "invokes_real_token_write_executor": False,
        "implements_real_token_write_executor": False,
        "creates_executor_source_files": False,
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
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY
        ),
    }


def summarize_human_approval_token_real_write_executor_implementation_dry_runs(
    dry_runs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(dry_runs, "dry_run_status")
    by_lock_reason: dict[str, int] = {}
    code_review_plan_required_count = 0
    locked_count = 0
    valid_count = 0
    invalid_count = 0
    for dry_run in dry_runs:
        lock_reason = str(dry_run.get("lock_reason"))
        by_lock_reason[lock_reason] = by_lock_reason.get(lock_reason, 0) + 1
        if (
            dry_run.get("dry_run_status")
            == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED
        ):
            code_review_plan_required_count += 1
        if (
            dry_run.get("dry_run_status")
            == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED
        ):
            locked_count += 1
        validation = (
            validate_human_approval_token_real_write_executor_implementation_dry_run(
                dry_run
            )
        )
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": candidate_summary["total"],
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "real_token_write_executor_code_review_plan_required_count": (
            code_review_plan_required_count
        ),
        "locked_count": locked_count,
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "by_lock_reason": dict(sorted(by_lock_reason.items())),
        "policy": dict(
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY
        ),
    }


def _dry_run_lock_reason(
    implementation_plan: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str | None:
    if (
        implementation_plan.get("plan_status")
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED
    ):
        return (
            implementation_plan.get("lock_reason")
            or "real_write_executor_implementation_plan_locked"
        )
    for field in _PREVIEW_FIELDS:
        if not isinstance(implementation_plan.get(field), Mapping):
            return f"missing_{field}"
    for field in _WRITE_PREVIEW_FIELDS:
        if not isinstance(implementation_plan.get(field), Mapping):
            return f"missing_{field}"
    for field, expected_type in _EXECUTOR_CONTRACT_FIELD_TYPES.items():
        if not isinstance(implementation_plan.get(field), expected_type):
            return f"missing_{field}"
    if not isinstance(implementation_plan.get("contract_review_checklist"), list):
        return "missing_contract_review_checklist"
    for field, expected_type in _IMPLEMENTATION_PLAN_FIELD_TYPES.items():
        if not isinstance(implementation_plan.get(field), expected_type):
            return f"missing_{field}"
    if not (
        implementation_plan.get("source_pattern_ids")
        or implementation_plan.get("source_fact_ids")
    ):
        return "missing_source_evidence"
    if _preview_integrity_errors(implementation_plan):
        return "preview_integrity_failed"
    if _implementation_plan_integrity_errors(implementation_plan, validation):
        return "implementation_plan_integrity_failed"
    if validation.get("valid") is not True:
        return "invalid_real_write_executor_implementation_plan"
    if (
        implementation_plan.get("plan_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED
    ):
        return "invalid_real_write_executor_implementation_plan"
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


def _implementation_plan_integrity_errors(
    implementation_plan: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    interfaces = implementation_plan.get("implementation_plan_interfaces")
    if isinstance(interfaces, list):
        for interface in interfaces:
            if not isinstance(interface, Mapping):
                errors.append("implementation_plan_interfaces_must_be_structured")
                continue
            if interface.get("implemented_in_v0_1") is not False:
                errors.append("implementation_plan_interfaces_must_not_implement_executor")
            if interface.get("invoked_in_v0_1") is not False:
                errors.append("implementation_plan_interfaces_must_not_invoke_executor")
    files = implementation_plan.get("implementation_plan_files")
    if isinstance(files, list):
        for file_plan in files:
            if not isinstance(file_plan, Mapping):
                errors.append("implementation_plan_files_must_be_structured")
                continue
            if file_plan.get("create_in_v0_1") is not False:
                errors.append("implementation_plan_files_must_not_create_executor_files")
            if file_plan.get("contains_executor_code_in_v0_1") is not False:
                errors.append("implementation_plan_files_must_not_contain_executor_code")
            if file_plan.get("writes_files_in_v0_1") is not False:
                errors.append("implementation_plan_files_must_not_write_files")
    strategy_expectations = (
        (
            "implementation_plan_idempotency_strategy",
            "v0_1_effect",
            "plan_only_no_idempotency_state_written",
        ),
        (
            "implementation_plan_filesystem_safety_model",
            "v0_1_effect",
            "no_filesystem_write_or_directory_creation",
        ),
        (
            "implementation_plan_audit_strategy",
            "v0_1_effect",
            "no_approval_audit_file_write_and_no_operation_ledger_event",
        ),
        (
            "implementation_plan_rollback_strategy",
            "v0_1_effect",
            "no_rollback_action_because_no_write_is_performed",
        ),
    )
    for field, key, expected in strategy_expectations:
        value = implementation_plan.get(field)
        if isinstance(value, Mapping) and value.get(key) != expected:
            errors.append(f"{field}_must_preserve_no_write_effect")
    test_plan = implementation_plan.get("implementation_plan_test_plan")
    if isinstance(test_plan, list):
        for test_case in test_plan:
            if not isinstance(test_case, Mapping):
                errors.append("implementation_plan_test_plan_must_be_structured")
                continue
            if test_case.get("writes") is not False:
                errors.append("implementation_plan_test_plan_must_not_write")
    forbidden_actions = implementation_plan.get("implementation_plan_forbidden_actions")
    if isinstance(forbidden_actions, list):
        required_forbidden_actions = {
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
        }
        if required_forbidden_actions.difference(forbidden_actions):
            errors.append("implementation_plan_forbidden_actions_must_preserve_no_write_boundary")
    for error in list(validation.get("errors", []) or []):
        if (
            error.startswith("implementation_plan_")
            or error.endswith("_must_match_source_contract_review_outcome_snapshot")
            or error.endswith("_must_match_source_snapshot")
            or error.startswith("contract_review_checklist_must_preserve")
            or error.startswith("executor_forbidden_side_effects_must_preserve")
            or error.startswith("executor_contract_checklist_must_preserve")
            or error.startswith("final_gate_checklist_must_preserve")
            or error.startswith("final_token_write_preflight_checklist_must_preserve")
            or error.startswith("policy_")
        ):
            errors.append("implementation_plan_validation_integrity_errors")
    return _dedupe(errors)


def _implementation_dry_run_module_boundary_preview() -> dict[str, Any]:
    return {
        "preview_kind": "real_token_write_executor_module_boundary_preview",
        "preview_only": True,
        "future_module": "agent/memory_human_approval_token_real_write_executor.py",
        "future_test_module": "tests/agent/test_memory_human_approval_token_real_write_executor.py",
        "module_created_in_v0_1": False,
        "creates_executor_source_files": False,
        "implements_real_token_write_executor": False,
        "invokes_real_token_write_executor": False,
        "writes_files": False,
        "boundary_rules": [
            "future_executor_code_requires_separate_code_review_plan",
            "future_executor_module_must_not_be_created_by_this_dry_run",
            "future_executor_invocation_must_remain_locked_until_after_code_review",
        ],
    }


def _implementation_dry_run_interface_preview() -> list[dict[str, Any]]:
    return [
        {
            "name": "validate_real_token_write_executor_inputs",
            "preview_only": True,
            "future_signature": (
                "validate_real_token_write_executor_inputs(final_gate, approval_token_payload, audit_payload)"
            ),
            "implemented_in_v0_1": False,
            "invoked_in_v0_1": False,
            "writes_files": False,
        },
        {
            "name": "prepare_real_token_write_payloads",
            "preview_only": True,
            "future_signature": (
                "prepare_real_token_write_payloads(approval_token_write_payload_preview, approval_audit_write_payload_preview)"
            ),
            "implemented_in_v0_1": False,
            "invoked_in_v0_1": False,
            "writes_files": False,
        },
        {
            "name": "commit_real_token_write_atomically",
            "preview_only": True,
            "future_signature": (
                "commit_real_token_write_atomically(token_payload, audit_payload, target_paths)"
            ),
            "implemented_in_v0_1": False,
            "invoked_in_v0_1": False,
            "writes_files": False,
        },
        {
            "name": "verify_real_token_write_audit_trail",
            "preview_only": True,
            "future_signature": (
                "verify_real_token_write_audit_trail(write_fingerprint, token_path, audit_path)"
            ),
            "implemented_in_v0_1": False,
            "invoked_in_v0_1": False,
            "writes_files": False,
        },
    ]


def _implementation_dry_run_idempotency_check_preview() -> dict[str, Any]:
    return {
        "preview_kind": "real_token_write_executor_idempotency_check_preview",
        "preview_only": True,
        "v0_1_effect": "no_idempotency_state_written",
        "fingerprint_inputs": [
            "source_implementation_plan_id",
            "source_contract_review_outcome_id",
            "source_contract_id",
            "approval_token_write_payload_preview",
            "approval_audit_write_payload_preview",
            "token_write_target_paths_preview",
        ],
        "future_checks": [
            "derive_write_fingerprint_before_future_write",
            "require_existing_token_and_audit_hashes_to_match_before_idempotent_success",
            "lock_on_mismatched_existing_token_or_audit_payload",
        ],
        "writes_token_files": False,
        "writes_approval_audit": False,
        "writes_operation_ledger": False,
        "written": False,
    }


def _implementation_dry_run_filesystem_safety_check_preview() -> dict[str, Any]:
    return {
        "preview_kind": "real_token_write_executor_filesystem_safety_check_preview",
        "preview_only": True,
        "v0_1_effect": "preview_only_no_filesystem_write_or_directory_creation",
        "future_checks": [
            "resolve_targets_under_hermes_home_memory_approval_scope",
            "reject_path_traversal_symlinks_and_unexpected_absolute_paths",
            "require_atomic_temp_file_replace_in_same_directory",
            "require_token_and_audit_pair_to_commit_together",
            "fsync_file_and_directory_after_future_write",
        ],
        "creates_executor_source_files": False,
        "writes_token_files": False,
        "writes_approval_audit": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "written": False,
    }


def _implementation_dry_run_audit_check_preview() -> dict[str, Any]:
    return {
        "preview_kind": "real_token_write_executor_audit_check_preview",
        "preview_only": True,
        "v0_1_effect": "no_approval_audit_file_write_and_no_operation_ledger_event",
        "future_checks": [
            "audit_payload_derives_from_approval_audit_write_payload_preview",
            "audit_record_references_source_implementation_plan_and_contract",
            "audit_success_requires_token_and_audit_payload_hash_match",
            "operation_ledger_events_remain_out_of_scope_for_token_executor",
        ],
        "created_operation_event": False,
        "writes_approval_audit": False,
        "writes_operation_ledger": False,
        "written": False,
    }


def _implementation_dry_run_rollback_check_preview() -> dict[str, Any]:
    return {
        "preview_kind": "real_token_write_executor_rollback_check_preview",
        "preview_only": True,
        "v0_1_effect": "no_rollback_action_because_no_write_is_performed",
        "future_checks": [
            "abort_before_first_write_when_preflight_fails",
            "remove_only_temp_files_created_by_same_future_attempt",
            "never_delete_preexisting_token_or_audit_files",
            "require_manual_recovery_packet_for_partial_commit",
        ],
        "writes_token_files": False,
        "writes_approval_audit": False,
        "writes_operation_ledger": False,
        "written": False,
    }


def _implementation_dry_run_test_harness_preview() -> list[dict[str, Any]]:
    return [
        {
            "id": "valid_implementation_plan_creates_code_review_plan_required_dry_run",
            "scope": "unit",
            "creates_files": False,
            "writes": False,
        },
        {
            "id": "locked_invalid_missing_and_integrity_failures_lock_dry_run",
            "scope": "unit",
            "creates_files": False,
            "writes": False,
        },
        {
            "id": "deterministic_previews_do_not_create_or_invoke_executor",
            "scope": "unit",
            "creates_files": False,
            "writes": False,
        },
        {
            "id": "benchmark_smoke_case_stays_read_only_executor_free_and_source_file_free",
            "scope": "benchmark",
            "creates_files": False,
            "writes": False,
        },
    ]


def _implementation_dry_run_readiness_checklist() -> list[str]:
    return [
        "source_implementation_plan_is_valid_and_implementation_plan_required",
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
        "executor_hard_lock_checks_are_present",
        "executor_audit_fields_are_present",
        "executor_rollback_rules_are_present",
        "executor_forbidden_side_effects_are_present",
        "executor_contract_checklist_is_present",
        "contract_review_checklist_is_present",
        "implementation_plan_interfaces_are_present_and_not_implemented",
        "implementation_plan_files_are_present_and_not_created",
        "implementation_plan_idempotency_strategy_is_present",
        "implementation_plan_filesystem_safety_model_is_present",
        "implementation_plan_audit_strategy_is_present",
        "implementation_plan_rollback_strategy_is_present",
        "implementation_plan_test_plan_is_present",
        "implementation_plan_forbidden_actions_are_present",
        "source_evidence_is_present",
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
    ]


def _dry_run_id(dry_run: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_VERSION,
        "dry_run_kind": dry_run.get("dry_run_kind"),
        "dry_run_status": dry_run.get("dry_run_status"),
        "routing": dry_run.get("routing"),
        "lock_reason": dry_run.get("lock_reason"),
        "source_implementation_plan_id": dry_run.get("source_implementation_plan_id"),
        "source_contract_review_outcome_id": dry_run.get("source_contract_review_outcome_id"),
        "source_contract_id": dry_run.get("source_contract_id"),
        "source_write_final_gate_id": dry_run.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": dry_run.get(
            "source_write_execution_dry_run_id"
        ),
        "source_write_execution_plan_id": dry_run.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": dry_run.get(
            "source_final_confirmation_review_outcome_id"
        ),
        "source_final_confirmation_request_id": dry_run.get(
            "source_final_confirmation_request_id"
        ),
        "source_token_write_lock_gate_id": dry_run.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": dry_run.get("source_token_issuance_dry_run_id"),
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
        "outcome": dry_run.get("outcome"),
        "rationale": dry_run.get("rationale"),
        "approval_token_record_preview": dry_run.get("approval_token_record_preview", {}),
        "approval_audit_record_preview": dry_run.get("approval_audit_record_preview", {}),
        "token_target_paths_preview": dry_run.get("token_target_paths_preview", {}),
        "proposal_record_preview": dry_run.get("proposal_record_preview", {}),
        "operation_ledger_preview": dry_run.get("operation_ledger_preview", {}),
        "target_paths_preview": dry_run.get("target_paths_preview", {}),
        "payload_preview": dry_run.get("payload_preview"),
        "source_pattern_ids": list(dry_run.get("source_pattern_ids", [])),
        "source_fact_ids": list(dry_run.get("source_fact_ids", [])),
        "token_write_execution_steps": list(
            dry_run.get("token_write_execution_steps", [])
        ),
        "token_write_execution_preflight_checks": list(
            dry_run.get("token_write_execution_preflight_checks", [])
        ),
        "approval_token_write_payload_preview": dry_run.get(
            "approval_token_write_payload_preview", {}
        ),
        "approval_audit_write_payload_preview": dry_run.get(
            "approval_audit_write_payload_preview", {}
        ),
        "token_write_target_paths_preview": dry_run.get(
            "token_write_target_paths_preview", {}
        ),
        "final_token_write_preflight_checklist": list(
            dry_run.get("final_token_write_preflight_checklist") or []
        ),
        "final_gate_checklist": list(dry_run.get("final_gate_checklist") or []),
        "executor_contract_inputs": dry_run.get("executor_contract_inputs", {}),
        "executor_hard_lock_checks": list(dry_run.get("executor_hard_lock_checks") or []),
        "executor_audit_fields": list(dry_run.get("executor_audit_fields") or []),
        "executor_rollback_rules": list(dry_run.get("executor_rollback_rules") or []),
        "executor_forbidden_side_effects": list(
            dry_run.get("executor_forbidden_side_effects") or []
        ),
        "executor_contract_checklist": list(
            dry_run.get("executor_contract_checklist") or []
        ),
        "contract_review_checklist": list(dry_run.get("contract_review_checklist") or []),
        "implementation_plan_interfaces": dry_run.get("implementation_plan_interfaces", []),
        "implementation_plan_files": dry_run.get("implementation_plan_files", []),
        "implementation_plan_idempotency_strategy": dry_run.get(
            "implementation_plan_idempotency_strategy", {}
        ),
        "implementation_plan_filesystem_safety_model": dry_run.get(
            "implementation_plan_filesystem_safety_model", {}
        ),
        "implementation_plan_audit_strategy": dry_run.get(
            "implementation_plan_audit_strategy", {}
        ),
        "implementation_plan_rollback_strategy": dry_run.get(
            "implementation_plan_rollback_strategy", {}
        ),
        "implementation_plan_test_plan": dry_run.get("implementation_plan_test_plan", []),
        "implementation_plan_forbidden_actions": list(
            dry_run.get("implementation_plan_forbidden_actions") or []
        ),
        "implementation_dry_run_module_boundary_preview": dry_run.get(
            "implementation_dry_run_module_boundary_preview", {}
        ),
        "implementation_dry_run_interface_preview": dry_run.get(
            "implementation_dry_run_interface_preview", []
        ),
        "implementation_dry_run_idempotency_check_preview": dry_run.get(
            "implementation_dry_run_idempotency_check_preview", {}
        ),
        "implementation_dry_run_filesystem_safety_check_preview": dry_run.get(
            "implementation_dry_run_filesystem_safety_check_preview", {}
        ),
        "implementation_dry_run_audit_check_preview": dry_run.get(
            "implementation_dry_run_audit_check_preview", {}
        ),
        "implementation_dry_run_rollback_check_preview": dry_run.get(
            "implementation_dry_run_rollback_check_preview", {}
        ),
        "implementation_dry_run_test_harness_preview": dry_run.get(
            "implementation_dry_run_test_harness_preview", []
        ),
        "implementation_dry_run_readiness_checklist": list(
            dry_run.get("implementation_dry_run_readiness_checklist") or []
        ),
        "implementation_plan_validation": dry_run.get("implementation_plan_validation", {}),
        "policy": dry_run.get("policy", {}),
    }
    return build_stable_digest(
        "memory-human-approval-token-real-write-executor-implementation-dry-run:v0.1",
        identity,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
