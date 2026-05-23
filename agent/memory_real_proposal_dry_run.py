from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_real_proposal_creation_plan import (
    MEMORY_REAL_PROPOSAL_CREATION_PLAN_STATUS,
    explain_real_proposal_creation_plan,
    recommend_real_proposal_creation_plan_action,
    validate_real_proposal_creation_plan,
)


MEMORY_REAL_PROPOSAL_DRY_RUN_VERSION = "0.1"
MEMORY_REAL_PROPOSAL_DRY_RUN_KIND = "memory_real_proposal_dry_run_candidate"
MEMORY_REAL_PROPOSAL_DRY_RUN_STATUS = "manual_final_preflight_required"
MEMORY_REAL_PROPOSAL_DRY_RUN_ROUTING = "manual_final_preflight_before_real_proposal_write"
MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_dry_run_candidates_only": True,
    "creates_real_proposals": False,
    "writes_proposal_files": False,
    "writes_operation_ledger": False,
    "applies_proposals": False,
    "persists_approvals": False,
    "submits_to_governance": False,
    "converts_to_real_proposal": False,
}

_DEFAULT_OPERATOR = "hermes_memory_real_proposal_dry_run_v0.1"
_FORBIDDEN_TRUE_KEYS = (
    "written",
    "submitted",
    "applied",
    "persisted",
    "approved",
    "created_real_proposal",
    "created_operation_event",
    "converted_to_real_proposal",
    "created_proposal_record",
    "submitted_to_governance",
    "persisted_approval",
)


def create_real_proposal_dry_run(
    plan: Mapping[str, Any],
    operator: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only dry-run preview for a real proposal."""
    source = deepcopy(dict(plan))
    plan_validation = validate_real_proposal_creation_plan(source)
    invalid_reason = _dry_run_invalid_reason(source, plan_validation)

    dry_run = {
        "dry_run_id": None,
        "dry_run_kind": MEMORY_REAL_PROPOSAL_DRY_RUN_KIND,
        "dry_run_status": MEMORY_REAL_PROPOSAL_DRY_RUN_STATUS,
        "routing": MEMORY_REAL_PROPOSAL_DRY_RUN_ROUTING,
        "source_plan_id": source.get("plan_id"),
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
        "proposal_record_preview": _proposal_record_preview(source),
        "operation_ledger_preview": _operation_ledger_preview(source),
        "target_paths_preview": _target_paths_preview(source),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "final_preflight_checklist": _final_preflight_checklist(),
        "plan_validation": deepcopy(plan_validation),
        "dry_run_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_plan_snapshot": deepcopy(source),
        "policy": dict(MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY),
    }
    if invalid_reason:
        dry_run["invalid_reason"] = invalid_reason
    dry_run["dry_run_id"] = _dry_run_id(dry_run)
    dry_run["dry_run_validation"] = validate_real_proposal_dry_run(dry_run)
    dry_run["next_step_recommendation"] = recommend_real_proposal_dry_run_action(dry_run)
    return dry_run


def validate_real_proposal_dry_run(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(dry_run.get("policy"))
    plan_validation = _as_dict(dry_run.get("plan_validation"))
    source_snapshot = _as_dict(dry_run.get("source_plan_snapshot"))
    proposal_preview = _as_dict(dry_run.get("proposal_record_preview"))
    ledger_preview = _as_dict(dry_run.get("operation_ledger_preview"))
    target_paths = _as_dict(dry_run.get("target_paths_preview"))

    for key in (
        "dry_run_id",
        "dry_run_kind",
        "dry_run_status",
        "routing",
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
        "final_preflight_checklist",
        "plan_validation",
        "dry_run_validation",
        "next_step_recommendation",
        "source_plan_snapshot",
        "policy",
    ):
        if key not in dry_run:
            errors.append(f"missing_{key}")
    if dry_run.get("dry_run_kind") != MEMORY_REAL_PROPOSAL_DRY_RUN_KIND:
        errors.append("dry_run_kind_must_be_memory_real_proposal_dry_run_candidate")
    if dry_run.get("dry_run_status") != MEMORY_REAL_PROPOSAL_DRY_RUN_STATUS:
        errors.append("dry_run_status_must_be_manual_final_preflight_required")
    if dry_run.get("routing") != MEMORY_REAL_PROPOSAL_DRY_RUN_ROUTING:
        errors.append("routing_must_require_manual_final_preflight")
    if not isinstance(dry_run.get("operator"), str) or not dry_run.get("operator"):
        errors.append("operator_must_be_non_empty_string")
    if plan_validation.get("valid") is not True:
        errors.append("invalid_real_proposal_creation_plan")
    if source_snapshot:
        snapshot_validation = validate_real_proposal_creation_plan(source_snapshot)
        if snapshot_validation != plan_validation:
            errors.append("plan_validation_must_match_source_plan_snapshot")
        if source_snapshot.get("plan_status") != MEMORY_REAL_PROPOSAL_CREATION_PLAN_STATUS:
            errors.append("source_plan_status_must_be_manual_creation_plan_required")
    if not isinstance(dry_run.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(dry_run.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(dry_run.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(dry_run.get("source_pattern_ids"), list) and isinstance(dry_run.get("source_fact_ids"), list):
        if not dry_run.get("source_pattern_ids") and not dry_run.get("source_fact_ids"):
            errors.append("missing_source_evidence")
    if not isinstance(source_snapshot.get("creation_steps"), list) or not source_snapshot.get("creation_steps"):
        errors.append("missing_plan_controls")
    if not isinstance(source_snapshot.get("preflight_checks"), list) or not source_snapshot.get("preflight_checks"):
        errors.append("missing_plan_controls")
    if not isinstance(dry_run.get("final_preflight_checklist"), list) or not dry_run.get("final_preflight_checklist"):
        errors.append("final_preflight_checklist_must_be_non_empty_list")
    if proposal_preview.get("preview_only") is not True:
        errors.append("proposal_record_preview_must_be_preview_only")
    if proposal_preview.get("written") is not False:
        errors.append("proposal_record_preview_must_not_be_written")
    if ledger_preview.get("preview_only") is not True:
        errors.append("operation_ledger_preview_must_be_preview_only")
    if ledger_preview.get("written") is not False:
        errors.append("operation_ledger_preview_must_not_be_written")
    if ledger_preview.get("created_operation_event") is not False:
        errors.append("operation_ledger_preview_must_not_create_operation_event")
    if target_paths.get("preview_only") is not True:
        errors.append("target_paths_preview_must_be_preview_only")
    for forbidden_key in _FORBIDDEN_TRUE_KEYS:
        if dry_run.get(forbidden_key) is True:
            errors.append(f"{forbidden_key}_must_be_false_or_absent")
    for key, expected in MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY.items():
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_real_proposal_dry_run(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_real_proposal_dry_run(dry_run)
    source_snapshot = _as_dict(dry_run.get("source_plan_snapshot"))
    return {
        "dry_run_id": dry_run.get("dry_run_id"),
        "dry_run_kind": dry_run.get("dry_run_kind"),
        "dry_run_status": dry_run.get("dry_run_status"),
        "routing": dry_run.get("routing"),
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
        "proposal_record_preview": deepcopy(dry_run.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(dry_run.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(dry_run.get("target_paths_preview")),
        "final_preflight_checklist": deepcopy(dry_run.get("final_preflight_checklist")),
        "validation": validation,
        "plan_explanation": explain_real_proposal_creation_plan(source_snapshot) if source_snapshot else {},
        "written": False,
        "submitted": False,
        "applied": False,
        "persisted": False,
        "approved": False,
        "created_real_proposal": False,
        "created_operation_event": False,
        "converted_to_real_proposal": False,
        "created_proposal_record": False,
        "submitted_to_governance": False,
        "persisted_approval": False,
        "policy": dict(MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY),
    }


def recommend_real_proposal_dry_run_action(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_real_proposal_dry_run(dry_run)
    source_snapshot = _as_dict(dry_run.get("source_plan_snapshot"))
    if validation["valid"]:
        action = "perform_manual_final_preflight_before_real_proposal_write"
        reason = "Dry-run candidate is valid for manual final preflight only; it does not create a proposal or operation-ledger event."
    else:
        action = "do_not_create_real_proposal"
        reason = _recommendation_reason(validation)
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_REAL_PROPOSAL_DRY_RUN_ROUTING,
        "validation": validation,
        "plan_recommendation": recommend_real_proposal_creation_plan_action(source_snapshot) if source_snapshot else {},
        "creates_dry_run_candidates_only": True,
        "creates_real_proposals": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "applies_proposals": False,
        "persists_approvals": False,
        "submits_to_governance": False,
        "converts_to_real_proposal": False,
        "policy": dict(MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY),
    }


def summarize_real_proposal_dry_runs(
    dry_runs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    by_block_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    valid_count = 0
    invalid_count = 0
    for dry_run in dry_runs:
        block_type = str(dry_run.get("block_type"))
        by_block_type[block_type] = by_block_type.get(block_type, 0) + 1
        status = str(dry_run.get("dry_run_status"))
        by_status[status] = by_status.get(status, 0) + 1
        validation = validate_real_proposal_dry_run(dry_run)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(dry_runs),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "by_block_type": dict(sorted(by_block_type.items())),
        "by_status": dict(sorted(by_status.items())),
        "policy": dict(MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY),
    }


def _dry_run_invalid_reason(source: Mapping[str, Any], plan_validation: Mapping[str, Any]) -> str | None:
    if not isinstance(source.get("payload_preview"), Mapping):
        return "missing_payload_preview"
    if not (source.get("source_pattern_ids") or source.get("source_fact_ids")):
        return "missing_source_evidence"
    if not isinstance(source.get("creation_steps"), list) or not source.get("creation_steps"):
        return "missing_plan_controls"
    if not isinstance(source.get("preflight_checks"), list) or not source.get("preflight_checks"):
        return "missing_plan_controls"
    if plan_validation.get("valid") is not True:
        return "invalid_real_proposal_creation_plan"
    if source.get("plan_status") != MEMORY_REAL_PROPOSAL_CREATION_PLAN_STATUS:
        return "invalid_real_proposal_creation_plan"
    return None


def _proposal_record_preview(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preview_only": True,
        "written": False,
        "proposal_id_preview": _preview_id("proposal", source),
        "proposal_status_preview": "pending_governed_write_if_manual_preflight_passes",
        "source_plan_id": source.get("plan_id"),
        "source_outcome_id": source.get("source_outcome_id"),
        "source_packet_id": source.get("source_packet_id"),
        "source_submission_id": source.get("source_submission_id"),
        "source_draft_id": source.get("source_draft_id"),
        "source_decision_id": source.get("source_decision_id"),
        "source_queue_item_id": source.get("source_queue_item_id"),
        "block_id": source.get("block_id"),
        "block_type": source.get("block_type"),
        "project_scope": source.get("project_scope"),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "created_real_proposal": False,
        "submitted_to_governance": False,
        "applied": False,
        "persisted": False,
        "approved": False,
    }


def _operation_ledger_preview(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preview_only": True,
        "written": False,
        "created_operation_event": False,
        "operation_event_id_preview": _preview_id("operation-ledger", source),
        "operation_type_preview": "memory_real_proposal_manual_write_preflight",
        "source_plan_id": source.get("plan_id"),
        "block_id": source.get("block_id"),
        "block_type": source.get("block_type"),
        "project_scope": source.get("project_scope"),
        "would_write_memory": False,
        "would_write_graph": False,
        "would_modify_config": False,
        "would_write_proposal_file": False,
        "would_write_operation_ledger": False,
    }


def _target_paths_preview(source: Mapping[str, Any]) -> dict[str, Any]:
    proposal_preview_id = _preview_id("proposal", source)
    ledger_preview_id = _preview_id("operation-ledger", source)
    return {
        "preview_only": True,
        "base": "${HERMES_HOME}",
        "proposal_file": "memory/proposals/memory_write_proposals.jsonl",
        "operation_ledger_file": "memory/audit/memory_operation_ledger.jsonl",
        "proposal_record_selector": proposal_preview_id,
        "operation_ledger_selector": ledger_preview_id,
        "written": False,
        "created_real_proposal": False,
        "created_operation_event": False,
    }


def _final_preflight_checklist() -> list[dict[str, str]]:
    return [
        {
            "id": "verify_plan_validation",
            "description": "Confirm the source plan is valid and still requires manual creation planning.",
        },
        {
            "id": "verify_payload_and_evidence",
            "description": "Confirm payload preview and source pattern or fact evidence are present.",
        },
        {
            "id": "verify_preview_records_only",
            "description": "Confirm proposal and operation-ledger records are previews only and were not written.",
        },
        {
            "id": "verify_no_memory_graph_config_write",
            "description": "Confirm the dry run did not write memory, graph state, or configuration.",
        },
        {
            "id": "manual_operator_final_preflight",
            "description": "Require a separate human-governed final preflight before any real proposal write.",
        },
    ]


def _recommendation_reason(validation: Mapping[str, Any]) -> str:
    errors = list(validation.get("errors", []) or [])
    if "missing_payload_preview" in errors:
        return "Dry-run candidate is missing payload preview required for final preflight."
    if "missing_source_evidence" in errors:
        return "Dry-run candidate is missing source pattern or fact evidence."
    if "missing_plan_controls" in errors:
        return "Dry-run candidate is missing source plan creation steps or preflight checks."
    if "invalid_real_proposal_creation_plan" in errors:
        return "Source real proposal creation plan is invalid."
    return "Dry-run candidate violates the v0.1 real proposal dry-run validation contract."


def _dry_run_id(dry_run: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_REAL_PROPOSAL_DRY_RUN_VERSION,
        "dry_run_kind": dry_run.get("dry_run_kind"),
        "dry_run_status": dry_run.get("dry_run_status"),
        "routing": dry_run.get("routing"),
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
        "proposal_record_preview": dry_run.get("proposal_record_preview", {}),
        "operation_ledger_preview": dry_run.get("operation_ledger_preview", {}),
        "target_paths_preview": dry_run.get("target_paths_preview", {}),
        "payload_preview": dry_run.get("payload_preview"),
        "source_pattern_ids": list(dry_run.get("source_pattern_ids", [])),
        "source_fact_ids": list(dry_run.get("source_fact_ids", [])),
        "final_preflight_checklist": dry_run.get("final_preflight_checklist", []),
        "plan_validation": dry_run.get("plan_validation", {}),
        "invalid_reason": dry_run.get("invalid_reason"),
        "policy": dry_run.get("policy", {}),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"memory-real-proposal-dry-run:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _preview_id(kind: str, source: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_REAL_PROPOSAL_DRY_RUN_VERSION,
        "kind": kind,
        "source_plan_id": source.get("plan_id"),
        "block_id": source.get("block_id"),
        "block_type": source.get("block_type"),
        "project_scope": source.get("project_scope"),
        "payload_preview": source.get("payload_preview"),
        "source_pattern_ids": list(source.get("source_pattern_ids", []) or []),
        "source_fact_ids": list(source.get("source_fact_ids", []) or []),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"preview:{kind}:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
