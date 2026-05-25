from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_write_final_gate import (
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_FINAL_GATE_ELIGIBLE,
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_FINAL_GATE_LOCKED,
    explain_human_approval_token_write_final_gate,
    recommend_human_approval_token_write_final_gate_action,
    validate_human_approval_token_write_final_gate,
)
from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_forbidden_true_keys_false_or_absent,
    validate_policy_flags,
)


MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_KIND = (
    "memory_human_approval_token_real_write_executor_contract_candidate"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_REQUIRED = (
    "real_token_write_executor_contract_required"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_ROUTING = (
    "real_token_write_executor_contract_review_required_before_implementation"
)
MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_real_write_executor_contract_candidates_only": True,
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

_DEFAULT_OPERATOR = "hermes_memory_human_approval_token_real_write_executor_contract_v0.1"
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
_LOCK_REASONS = {
    None,
    "token_write_final_gate_locked",
    "invalid_token_write_final_gate",
    "missing_approval_token_record_preview",
    "missing_approval_audit_record_preview",
    "missing_token_target_paths_preview",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_approval_token_write_payload_preview",
    "missing_approval_audit_write_payload_preview",
    "missing_token_write_target_paths_preview",
    "missing_final_gate_checklist",
    "missing_source_evidence",
    "preview_integrity_failed",
    "final_gate_integrity_failed",
    "token_write_execution_dry_run_locked",
    "missing_final_token_write_preflight_checklist",
    "final_preflight_integrity_failed",
    "final_confirmation_requested_changes",
    "final_confirmation_rejected",
    "final_confirmation_deferred",
    "invalid_final_confirmation_review_outcome",
}


def create_human_approval_token_real_write_executor_contract(
    write_final_gate: Mapping[str, Any],
    operator: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only real token write executor contract candidate."""
    source = deepcopy(dict(write_final_gate))
    write_final_gate_validation = validate_human_approval_token_write_final_gate(source)
    lock_reason = _contract_lock_reason(source, write_final_gate_validation)
    contract_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_LOCKED
    )

    contract = {
        "contract_id": None,
        "contract_kind": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_KIND,
        "contract_status": contract_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_ROUTING,
        "lock_reason": lock_reason,
        "source_write_final_gate_id": source.get("gate_id"),
        "source_write_execution_dry_run_id": source.get("source_write_execution_dry_run_id"),
        "source_write_execution_plan_id": source.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": source.get(
            "source_final_confirmation_review_outcome_id"
        ),
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
        "operator": operator if operator is not None else _DEFAULT_OPERATOR,
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
        "executor_contract_inputs": _executor_contract_inputs(source),
        "executor_hard_lock_checks": _executor_hard_lock_checks(),
        "executor_audit_fields": _executor_audit_fields(),
        "executor_rollback_rules": _executor_rollback_rules(),
        "executor_forbidden_side_effects": _executor_forbidden_side_effects(),
        "executor_contract_checklist": _executor_contract_checklist(),
        "write_final_gate_validation": deepcopy(write_final_gate_validation),
        "contract_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_write_final_gate_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_POLICY),
    }
    contract["contract_id"] = _contract_id(contract)
    contract["contract_validation"] = validate_human_approval_token_real_write_executor_contract(
        contract
    )
    contract["next_step_recommendation"] = (
        recommend_human_approval_token_real_write_executor_contract_action(contract)
    )
    return contract


def validate_human_approval_token_real_write_executor_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(contract.get("policy"))
    source_snapshot = _as_dict(contract.get("source_write_final_gate_snapshot"))
    gate_validation = _as_dict(contract.get("write_final_gate_validation"))
    expected_gate_validation = validate_human_approval_token_write_final_gate(source_snapshot)
    expected_lock_reason = _contract_lock_reason(source_snapshot, gate_validation)

    for key in _REQUIRED_CONTRACT_KEYS:
        if key not in contract:
            errors.append(f"missing_{key}")
    if (
        contract.get("contract_kind")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_KIND
    ):
        errors.append(
            "contract_kind_must_be_memory_human_approval_token_real_write_executor_contract_candidate"
        )
    if contract.get("contract_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_LOCKED,
    }:
        errors.append("contract_status_must_be_supported")
    if contract.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_ROUTING:
        errors.append("routing_must_require_contract_review_before_executor_implementation")
    if contract.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if contract.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_real_write_executor_contract_checks")
    if (
        expected_lock_reason is None
        and contract.get("contract_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_REQUIRED
    ):
        errors.append("eligible_token_write_final_gate_must_require_executor_contract")
    if (
        expected_lock_reason is not None
        and contract.get("contract_status")
        != MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_LOCKED
    ):
        errors.append("invalid_or_locked_token_write_final_gate_must_lock_contract")
    if not isinstance(contract.get("operator"), str) or not contract.get("operator"):
        errors.append("operator_must_be_non_empty_string")
    if gate_validation != expected_gate_validation:
        errors.append("write_final_gate_validation_must_match_source_write_final_gate_snapshot")
    if contract.get("source_write_final_gate_id") != source_snapshot.get("gate_id"):
        errors.append("source_write_final_gate_id_must_match_source_snapshot")
    for source_key in _SOURCE_KEYS:
        if contract.get(source_key) != source_snapshot.get(source_key):
            errors.append(f"{source_key}_must_match_source_snapshot")
    if not isinstance(contract.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(contract.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(contract.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(contract.get("source_pattern_ids"), list) and isinstance(
        contract.get("source_fact_ids"), list
    ):
        if (
            not contract.get("source_pattern_ids")
            and not contract.get("source_fact_ids")
            and contract.get("lock_reason") != "missing_source_evidence"
        ):
            errors.append("missing_source_evidence")
    for field in _PREVIEW_FIELDS + _WRITE_PREVIEW_FIELDS:
        if (
            not isinstance(contract.get(field), Mapping)
            and contract.get("lock_reason") != f"missing_{field}"
        ):
            errors.append(f"missing_{field}")
    if (
        not isinstance(contract.get("final_gate_checklist"), list)
        and contract.get("lock_reason") != "missing_final_gate_checklist"
    ):
        errors.append("missing_final_gate_checklist")
    if contract.get("lock_reason") != "preview_integrity_failed":
        errors.extend(_preview_integrity_errors(contract))
    if contract.get("lock_reason") != "final_gate_integrity_failed":
        errors.extend(_final_gate_integrity_errors(contract))
    if contract.get("executor_contract_inputs") != _executor_contract_inputs(source_snapshot):
        errors.append("executor_contract_inputs_must_match_v0_1_deterministic_contract")
    if contract.get("executor_hard_lock_checks") != _executor_hard_lock_checks():
        errors.append("executor_hard_lock_checks_must_match_v0_1_deterministic_contract")
    if contract.get("executor_audit_fields") != _executor_audit_fields():
        errors.append("executor_audit_fields_must_match_v0_1_deterministic_contract")
    if contract.get("executor_rollback_rules") != _executor_rollback_rules():
        errors.append("executor_rollback_rules_must_match_v0_1_deterministic_contract")
    if contract.get("executor_forbidden_side_effects") != _executor_forbidden_side_effects():
        errors.append("executor_forbidden_side_effects_must_match_v0_1_deterministic_contract")
    if contract.get("executor_contract_checklist") != _executor_contract_checklist():
        errors.append("executor_contract_checklist_must_match_v0_1_deterministic_contract")
    for field in _PREVIEW_FIELDS + _WRITE_PREVIEW_FIELDS + (
        "payload_preview",
        "final_token_write_preflight_checklist",
        "final_gate_checklist",
    ):
        if (
            field in contract
            and field in source_snapshot
            and contract.get(field) != source_snapshot.get(field)
        ):
            errors.append(f"{field}_must_match_source_write_final_gate_snapshot")
    if contract.get("source_pattern_ids") != list(source_snapshot.get("source_pattern_ids", []) or []):
        errors.append("source_pattern_ids_must_match_source_write_final_gate_snapshot")
    if contract.get("source_fact_ids") != list(source_snapshot.get("source_fact_ids", []) or []):
        errors.append("source_fact_ids_must_match_source_write_final_gate_snapshot")
    if contract.get("token_write_execution_steps") != list(
        source_snapshot.get("token_write_execution_steps", []) or []
    ):
        errors.append("token_write_execution_steps_must_match_source_write_final_gate_snapshot")
    if contract.get("token_write_execution_preflight_checks") != list(
        source_snapshot.get("token_write_execution_preflight_checks", []) or []
    ):
        errors.append(
            "token_write_execution_preflight_checks_must_match_source_write_final_gate_snapshot"
    )
    errors.extend(validate_forbidden_true_keys_false_or_absent(contract, _FORBIDDEN_TRUE_KEYS))
    errors.extend(
        validate_policy_flags(
            policy,
            MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_POLICY,
        )
    )

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_real_write_executor_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_real_write_executor_contract(contract)
    source_snapshot = _as_dict(contract.get("source_write_final_gate_snapshot"))
    return {
        "contract_id": contract.get("contract_id"),
        "contract_kind": contract.get("contract_kind"),
        "contract_status": contract.get("contract_status"),
        "routing": contract.get("routing"),
        "lock_reason": contract.get("lock_reason"),
        "source_write_final_gate_id": contract.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": contract.get("source_write_execution_dry_run_id"),
        "source_write_execution_plan_id": contract.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": contract.get(
            "source_final_confirmation_review_outcome_id"
        ),
        "block_id": contract.get("block_id"),
        "block_type": contract.get("block_type"),
        "project_scope": contract.get("project_scope"),
        "operator": contract.get("operator"),
        "outcome": contract.get("outcome"),
        "source_pattern_count": len(contract.get("source_pattern_ids", []) or []),
        "source_fact_count": len(contract.get("source_fact_ids", []) or []),
        "executor_contract_inputs": deepcopy(contract.get("executor_contract_inputs")),
        "executor_hard_lock_checks": deepcopy(contract.get("executor_hard_lock_checks")),
        "executor_audit_fields": deepcopy(contract.get("executor_audit_fields")),
        "executor_rollback_rules": deepcopy(contract.get("executor_rollback_rules")),
        "executor_forbidden_side_effects": deepcopy(
            contract.get("executor_forbidden_side_effects")
        ),
        "executor_contract_checklist": deepcopy(contract.get("executor_contract_checklist")),
        "validation": validation,
        "write_final_gate_explanation": (
            explain_human_approval_token_write_final_gate(source_snapshot)
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_POLICY),
    }


def recommend_human_approval_token_real_write_executor_contract_action(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_human_approval_token_real_write_executor_contract(contract)
    source_snapshot = _as_dict(contract.get("source_write_final_gate_snapshot"))
    if (
        validation["valid"]
        and contract.get("contract_status")
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_REQUIRED
    ):
        action = "route_to_real_token_write_executor_contract_review_without_implementation"
        reason = "Real token write executor contract candidate is ready for human review; this module does not implement or invoke the executor."
    elif validation["valid"]:
        action = "keep_real_token_write_executor_contract_locked"
        reason = f"Real token write executor contract candidate is locked by {contract.get('lock_reason')}."
    else:
        action = "repair_real_token_write_executor_contract_candidate"
        reason = "Real token write executor contract candidate failed validation and cannot proceed."
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_ROUTING,
        "validation": validation,
        "write_final_gate_recommendation": (
            recommend_human_approval_token_write_final_gate_action(source_snapshot)
            if source_snapshot
            else {}
        ),
        "creates_real_write_executor_contract_candidates_only": True,
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
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_POLICY),
    }


def summarize_human_approval_token_real_write_executor_contracts(
    contracts: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(contracts, "contract_status")
    lock_reason_summary = summarize_candidates(contracts, "lock_reason")
    required_count = 0
    locked_count = 0
    valid_count = 0
    invalid_count = 0
    for contract in contracts:
        if (
            contract.get("contract_status")
            == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_REQUIRED
        ):
            required_count += 1
        if (
            contract.get("contract_status")
            == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_LOCKED
        ):
            locked_count += 1
        validation = validate_human_approval_token_real_write_executor_contract(contract)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(contracts),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "real_token_write_executor_contract_required_count": required_count,
        "locked_count": locked_count,
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "by_lock_reason": lock_reason_summary["by_status"],
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_POLICY),
    }


_SOURCE_KEYS = (
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

_REQUIRED_CONTRACT_KEYS = (
    "contract_id",
    "contract_kind",
    "contract_status",
    "routing",
    "lock_reason",
    "source_write_final_gate_id",
) + _SOURCE_KEYS + (
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
    "write_final_gate_validation",
    "contract_validation",
    "next_step_recommendation",
    "source_write_final_gate_snapshot",
    "policy",
)


def _contract_lock_reason(
    gate: Mapping[str, Any],
    gate_validation: Mapping[str, Any],
) -> str | None:
    if gate.get("gate_status") == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_FINAL_GATE_LOCKED:
        return gate.get("lock_reason") or "token_write_final_gate_locked"
    for field in _PREVIEW_FIELDS:
        if not isinstance(gate.get(field), Mapping):
            return f"missing_{field}"
    for field in _WRITE_PREVIEW_FIELDS:
        if not isinstance(gate.get(field), Mapping):
            return f"missing_{field}"
    if not isinstance(gate.get("final_gate_checklist"), list):
        return "missing_final_gate_checklist"
    if not (gate.get("source_pattern_ids") or gate.get("source_fact_ids")):
        return "missing_source_evidence"
    if _preview_integrity_errors(gate):
        return "preview_integrity_failed"
    if _final_gate_integrity_errors(gate):
        return "final_gate_integrity_failed"
    if gate_validation.get("valid") is not True:
        return "invalid_token_write_final_gate"
    if gate.get("gate_status") != MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_FINAL_GATE_ELIGIBLE:
        return "invalid_token_write_final_gate"
    return None


def _executor_contract_inputs(gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "required_source_candidate": "eligible_for_real_token_write_executor_final_gate",
        "source_write_final_gate_id": gate.get("gate_id"),
        "source_write_execution_dry_run_id": gate.get("source_write_execution_dry_run_id"),
        "approval_token_record_preview_required": True,
        "approval_audit_record_preview_required": True,
        "token_target_paths_preview_required": True,
        "proposal_record_preview_required": True,
        "operation_ledger_preview_required": True,
        "target_paths_preview_required": True,
        "approval_token_write_payload_preview_required": True,
        "approval_audit_write_payload_preview_required": True,
        "token_write_target_paths_preview_required": True,
        "final_gate_checklist_required": True,
        "source_evidence_required": True,
        "executor_boundary": "contract_only_no_executor_implementation_or_invocation",
    }


def _executor_hard_lock_checks() -> list[str]:
    return [
        "lock_if_source_write_final_gate_is_locked",
        "lock_if_source_write_final_gate_validation_fails",
        "lock_if_source_write_final_gate_status_is_not_eligible_for_real_token_write_executor",
        "lock_if_approval_token_record_preview_missing_or_not_preview_only",
        "lock_if_approval_audit_record_preview_missing_or_not_preview_only",
        "lock_if_token_target_paths_preview_missing_or_not_preview_only",
        "lock_if_proposal_record_preview_missing_or_written",
        "lock_if_operation_ledger_preview_missing_or_written_or_event_created",
        "lock_if_target_paths_preview_missing_or_not_preview_only",
        "lock_if_approval_token_write_payload_preview_missing_or_not_preview_only",
        "lock_if_approval_audit_write_payload_preview_missing_or_not_preview_only",
        "lock_if_token_write_target_paths_preview_missing_or_not_preview_only",
        "lock_if_final_gate_checklist_missing_or_changed",
        "lock_if_source_evidence_missing",
        "lock_if_any_forbidden_write_or_executor_flag_is_true",
    ]


def _executor_audit_fields() -> list[str]:
    return [
        "contract_id",
        "contract_kind",
        "contract_status",
        "routing",
        "lock_reason",
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
        "operator",
        "outcome",
        "source_pattern_ids",
        "source_fact_ids",
        "write_final_gate_validation",
        "contract_validation",
        "policy",
    ]


def _executor_rollback_rules() -> list[str]:
    return [
        "no_rollback_action_in_v0_1_because_no_write_is_performed",
        "future_executor_must_abort_before_first_write_when_any_preflight_check_fails",
        "future_executor_must_use_atomic_token_and_audit_write_boundary_if_implemented_else_abort",
        "future_executor_must_record_reversible_target_paths_before_any_write_if_implemented",
        "future_executor_must_never_create_proposal_or_operation_ledger_events_inside_contract_module",
    ]


def _executor_forbidden_side_effects() -> list[str]:
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
        "write_token_files",
        "write_approval_audit_files",
        "persist_approvals",
        "invoke_real_token_write_executor",
        "implement_real_token_write_executor",
        "submit_to_governance",
        "convert_to_real_proposal",
    ]


def _executor_contract_checklist() -> list[str]:
    return [
        "contract_is_read_only_candidate_only",
        "source_final_gate_is_valid_and_executor_eligible",
        "all_preview_records_are_present_and_preview_only",
        "all_target_path_previews_are_present_and_preview_only",
        "proposal_and_operation_ledger_previews_are_not_written",
        "token_and_approval_audit_payload_previews_are_not_written",
        "source_evidence_is_present",
        "final_gate_checklist_is_present_and_unchanged",
        "hard_lock_checks_are_defined_before_executor_implementation",
        "audit_fields_are_defined_before_executor_implementation",
        "rollback_rules_are_defined_before_executor_implementation",
        "forbidden_side_effects_include_no_token_proposal_or_ledger_writes",
        "no_real_executor_is_invoked",
        "no_real_executor_is_implemented",
        "no_memory_or_graph_or_config_write",
    ]


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


def _final_gate_integrity_errors(container: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    checklist = container.get("final_gate_checklist")
    if isinstance(checklist, list):
        required_checks = {
            "no_real_approval_token_issued",
            "no_approval_persisted",
            "no_real_proposal_created",
            "no_operation_ledger_event_created",
            "no_token_file_written",
            "no_approval_audit_written",
            "real_token_write_executor_required_but_not_invoked",
        }
        missing = sorted(required_checks.difference(checklist))
        if missing:
            errors.append("final_gate_checklist_must_preserve_no_write_and_no_executor_checks")
    final_preflight = container.get("final_token_write_preflight_checklist")
    if isinstance(final_preflight, list) and "no_real_approval_token_issued" not in final_preflight:
        errors.append("final_token_write_preflight_checklist_must_preserve_no_token_write_checks")
    return errors


def _contract_id(contract: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CONTRACT_VERSION,
        "contract_kind": contract.get("contract_kind"),
        "contract_status": contract.get("contract_status"),
        "routing": contract.get("routing"),
        "lock_reason": contract.get("lock_reason"),
        "source_write_final_gate_id": contract.get("source_write_final_gate_id"),
        "source_write_execution_dry_run_id": contract.get("source_write_execution_dry_run_id"),
        "source_write_execution_plan_id": contract.get("source_write_execution_plan_id"),
        "source_final_confirmation_review_outcome_id": contract.get(
            "source_final_confirmation_review_outcome_id"
        ),
        "source_final_confirmation_request_id": contract.get(
            "source_final_confirmation_request_id"
        ),
        "source_token_write_lock_gate_id": contract.get("source_token_write_lock_gate_id"),
        "source_token_issuance_dry_run_id": contract.get("source_token_issuance_dry_run_id"),
        "source_token_issuance_plan_id": contract.get("source_token_issuance_plan_id"),
        "source_review_outcome_id": contract.get("source_review_outcome_id"),
        "source_request_id": contract.get("source_request_id"),
        "source_gate_id": contract.get("source_gate_id"),
        "source_dry_run_id": contract.get("source_dry_run_id"),
        "source_plan_id": contract.get("source_plan_id"),
        "source_outcome_id": contract.get("source_outcome_id"),
        "source_packet_id": contract.get("source_packet_id"),
        "source_submission_id": contract.get("source_submission_id"),
        "source_draft_id": contract.get("source_draft_id"),
        "source_decision_id": contract.get("source_decision_id"),
        "source_queue_item_id": contract.get("source_queue_item_id"),
        "block_id": contract.get("block_id"),
        "block_type": contract.get("block_type"),
        "project_scope": contract.get("project_scope"),
        "operator": contract.get("operator"),
        "outcome": contract.get("outcome"),
        "approval_token_record_preview": contract.get("approval_token_record_preview", {}),
        "approval_audit_record_preview": contract.get("approval_audit_record_preview", {}),
        "token_target_paths_preview": contract.get("token_target_paths_preview", {}),
        "proposal_record_preview": contract.get("proposal_record_preview", {}),
        "operation_ledger_preview": contract.get("operation_ledger_preview", {}),
        "target_paths_preview": contract.get("target_paths_preview", {}),
        "payload_preview": contract.get("payload_preview"),
        "source_pattern_ids": list(contract.get("source_pattern_ids", [])),
        "source_fact_ids": list(contract.get("source_fact_ids", [])),
        "token_write_execution_steps": list(contract.get("token_write_execution_steps", [])),
        "token_write_execution_preflight_checks": list(
            contract.get("token_write_execution_preflight_checks", [])
        ),
        "approval_token_write_payload_preview": contract.get(
            "approval_token_write_payload_preview", {}
        ),
        "approval_audit_write_payload_preview": contract.get(
            "approval_audit_write_payload_preview", {}
        ),
        "token_write_target_paths_preview": contract.get(
            "token_write_target_paths_preview", {}
        ),
        "final_token_write_preflight_checklist": list(
            contract.get("final_token_write_preflight_checklist") or []
        ),
        "final_gate_checklist": list(contract.get("final_gate_checklist") or []),
        "executor_contract_inputs": contract.get("executor_contract_inputs", {}),
        "executor_hard_lock_checks": list(contract.get("executor_hard_lock_checks", [])),
        "executor_audit_fields": list(contract.get("executor_audit_fields", [])),
        "executor_rollback_rules": list(contract.get("executor_rollback_rules", [])),
        "executor_forbidden_side_effects": list(
            contract.get("executor_forbidden_side_effects", [])
        ),
        "executor_contract_checklist": list(contract.get("executor_contract_checklist", [])),
        "write_final_gate_validation": contract.get("write_final_gate_validation", {}),
        "policy": contract.get("policy", {}),
    }
    return build_stable_digest(
        "memory-human-approval-token-real-write-executor-contract:v0.1",
        identity,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
