from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_real_proposal_dry_run import (
    MEMORY_REAL_PROPOSAL_DRY_RUN_STATUS,
    explain_real_proposal_dry_run,
    recommend_real_proposal_dry_run_action,
    validate_real_proposal_dry_run,
)


MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_VERSION = "0.1"
MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_KIND = "memory_real_proposal_write_lock_gate_candidate"
MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED = "locked"
MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE = "eligible_for_human_approval_token"
MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ROUTING = "human_approval_token_required_before_any_real_proposal_write"
MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_write_lock_candidates_only": True,
    "creates_real_proposals": False,
    "writes_proposal_files": False,
    "writes_operation_ledger": False,
    "applies_proposals": False,
    "persists_approvals": False,
    "submits_to_governance": False,
    "converts_to_real_proposal": False,
}

_DEFAULT_OPERATOR = "hermes_memory_real_proposal_write_lock_gate_v0.1"
_FORBIDDEN_TRUE_KEYS = (
    "written",
    "submitted",
    "applied",
    "persisted",
    "approved",
    "created_real_proposal",
    "created_operation_event",
    "writes_proposal_files",
    "writes_operation_ledger",
    "converted_to_real_proposal",
)
_PREVIEW_FIELDS = (
    "proposal_record_preview",
    "operation_ledger_preview",
    "target_paths_preview",
)
_LOCK_REASONS = {
    None,
    "invalid_real_proposal_dry_run",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_source_evidence",
    "preview_integrity_failed",
}


def create_real_proposal_write_lock_gate(
    dry_run: Mapping[str, Any],
    operator: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only write-lock gate candidate."""
    source = deepcopy(dict(dry_run))
    dry_run_validation = validate_real_proposal_dry_run(source)
    lock_reason = _write_lock_reason(source, dry_run_validation)
    gate_status = (
        MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE
        if lock_reason is None
        else MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED
    )

    gate = {
        "gate_id": None,
        "gate_kind": MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_KIND,
        "gate_status": gate_status,
        "routing": MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ROUTING,
        "lock_reason": lock_reason,
        "source_dry_run_id": source.get("dry_run_id"),
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
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "write_lock_checklist": _write_lock_checklist(),
        "dry_run_validation": deepcopy(dry_run_validation),
        "gate_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_dry_run_snapshot": deepcopy(source),
        "policy": dict(MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY),
    }
    gate["gate_id"] = _gate_id(gate)
    gate["gate_validation"] = validate_real_proposal_write_lock_gate(gate)
    gate["next_step_recommendation"] = recommend_real_proposal_write_lock_action(gate)
    return gate


def validate_real_proposal_write_lock_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(gate.get("policy"))
    source_snapshot = _as_dict(gate.get("source_dry_run_snapshot"))
    dry_run_validation = _as_dict(gate.get("dry_run_validation"))
    expected_lock_reason = _write_lock_reason(source_snapshot, dry_run_validation)
    preview_integrity_errors = _preview_integrity_errors(gate)

    for key in (
        "gate_id",
        "gate_kind",
        "gate_status",
        "routing",
        "lock_reason",
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
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "write_lock_checklist",
        "dry_run_validation",
        "gate_validation",
        "next_step_recommendation",
        "source_dry_run_snapshot",
        "policy",
    ):
        if key not in gate:
            errors.append(f"missing_{key}")
    if gate.get("gate_kind") != MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_KIND:
        errors.append("gate_kind_must_be_memory_real_proposal_write_lock_gate_candidate")
    if gate.get("gate_status") not in {
        MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED,
        MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE,
    }:
        errors.append("gate_status_must_be_supported")
    if gate.get("routing") != MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ROUTING:
        errors.append("routing_must_require_human_approval_token")
    if gate.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if gate.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_write_lock_checks")
    if expected_lock_reason is None and gate.get("gate_status") != MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE:
        errors.append("valid_dry_run_must_be_eligible_for_human_approval_token")
    if expected_lock_reason is not None and gate.get("gate_status") != MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED:
        errors.append("invalid_write_lock_check_must_be_locked")
    if not isinstance(gate.get("operator"), str) or not gate.get("operator"):
        errors.append("operator_must_be_non_empty_string")
    if dry_run_validation != validate_real_proposal_dry_run(source_snapshot):
        errors.append("dry_run_validation_must_match_source_dry_run_snapshot")
    if source_snapshot.get("dry_run_status") != MEMORY_REAL_PROPOSAL_DRY_RUN_STATUS:
        errors.append("source_dry_run_status_must_be_manual_final_preflight_required")
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
    errors.extend(preview_integrity_errors)
    for forbidden_key in _FORBIDDEN_TRUE_KEYS:
        if gate.get(forbidden_key) is True:
            errors.append(f"{forbidden_key}_must_be_false_or_absent")
    for key, expected in MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY.items():
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_real_proposal_write_lock_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_real_proposal_write_lock_gate(gate)
    source_snapshot = _as_dict(gate.get("source_dry_run_snapshot"))
    return {
        "gate_id": gate.get("gate_id"),
        "gate_kind": gate.get("gate_kind"),
        "gate_status": gate.get("gate_status"),
        "routing": gate.get("routing"),
        "lock_reason": gate.get("lock_reason"),
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
        "proposal_record_preview": deepcopy(gate.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(gate.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(gate.get("target_paths_preview")),
        "write_lock_checklist": deepcopy(gate.get("write_lock_checklist")),
        "validation": validation,
        "dry_run_explanation": explain_real_proposal_dry_run(source_snapshot) if source_snapshot else {},
        "written": False,
        "submitted": False,
        "applied": False,
        "persisted": False,
        "approved": False,
        "created_real_proposal": False,
        "created_operation_event": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "converted_to_real_proposal": False,
        "policy": dict(MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY),
    }


def recommend_real_proposal_write_lock_action(gate: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_real_proposal_write_lock_gate(gate)
    source_snapshot = _as_dict(gate.get("source_dry_run_snapshot"))
    if validation["valid"] and gate.get("gate_status") == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE:
        action = "request_separate_human_approval_token_before_real_proposal_write"
        reason = "Write-lock gate is eligible only for a separate human approval token flow; it does not create real proposal or operation-ledger records."
    else:
        action = "keep_real_proposal_write_locked"
        reason = _recommendation_reason(gate.get("lock_reason"), validation)
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ROUTING,
        "validation": validation,
        "dry_run_recommendation": recommend_real_proposal_dry_run_action(source_snapshot) if source_snapshot else {},
        "creates_write_lock_candidates_only": True,
        "creates_real_proposals": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "applies_proposals": False,
        "persists_approvals": False,
        "submits_to_governance": False,
        "converts_to_real_proposal": False,
        "policy": dict(MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY),
    }


def summarize_real_proposal_write_lock_gates(
    gates: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    by_block_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    locked_count = 0
    eligible_count = 0
    valid_count = 0
    invalid_count = 0
    for gate in gates:
        block_type = str(gate.get("block_type"))
        by_block_type[block_type] = by_block_type.get(block_type, 0) + 1
        status = str(gate.get("gate_status"))
        by_status[status] = by_status.get(status, 0) + 1
        if gate.get("gate_status") == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED:
            locked_count += 1
        if gate.get("gate_status") == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE:
            eligible_count += 1
        validation = validate_real_proposal_write_lock_gate(gate)
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
        "policy": dict(MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY),
    }


def _write_lock_reason(source: Mapping[str, Any], dry_run_validation: Mapping[str, Any]) -> str | None:
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
    if dry_run_validation.get("valid") is not True:
        return "invalid_real_proposal_dry_run"
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


def _write_lock_checklist() -> list[dict[str, str]]:
    return [
        {
            "id": "verify_dry_run_validation",
            "description": "Confirm the source dry run is valid and still requires manual final preflight.",
        },
        {
            "id": "verify_preview_artifacts_only",
            "description": "Confirm proposal, operation-ledger, and target-path previews are preview-only.",
        },
        {
            "id": "verify_no_written_or_created_flags",
            "description": "Confirm preview artifacts do not mark records as written or created.",
        },
        {
            "id": "verify_payload_and_source_evidence",
            "description": "Confirm payload preview and source pattern or fact evidence are present.",
        },
        {
            "id": "require_separate_human_approval_token",
            "description": "Require a separate human approval token flow before any real proposal write.",
        },
    ]


def _recommendation_reason(lock_reason: Any, validation: Mapping[str, Any]) -> str:
    if lock_reason == "invalid_real_proposal_dry_run":
        return "Source dry-run candidate is invalid."
    if lock_reason == "missing_proposal_record_preview":
        return "Write-lock gate is missing the proposal record preview."
    if lock_reason == "missing_operation_ledger_preview":
        return "Write-lock gate is missing the operation-ledger preview."
    if lock_reason == "missing_target_paths_preview":
        return "Write-lock gate is missing the target paths preview."
    if lock_reason == "missing_source_evidence":
        return "Write-lock gate is missing source pattern or fact evidence."
    if lock_reason == "preview_integrity_failed":
        return "Write-lock gate preview artifacts are not preview-only or include written/created flags."
    if validation.get("errors"):
        return "Write-lock gate violates the v0.1 validation contract."
    return "Real proposal write remains locked until a separate human approval token flow is completed."


def _gate_id(gate: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_VERSION,
        "gate_kind": gate.get("gate_kind"),
        "gate_status": gate.get("gate_status"),
        "routing": gate.get("routing"),
        "lock_reason": gate.get("lock_reason"),
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
        "proposal_record_preview": gate.get("proposal_record_preview", {}),
        "operation_ledger_preview": gate.get("operation_ledger_preview", {}),
        "target_paths_preview": gate.get("target_paths_preview", {}),
        "payload_preview": gate.get("payload_preview"),
        "source_pattern_ids": list(gate.get("source_pattern_ids", [])),
        "source_fact_ids": list(gate.get("source_fact_ids", [])),
        "write_lock_checklist": gate.get("write_lock_checklist", []),
        "dry_run_validation": gate.get("dry_run_validation", {}),
        "policy": gate.get("policy", {}),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"memory-real-proposal-write-lock-gate:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
