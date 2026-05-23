from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_approval_token_review_gate import (
    explain_human_approval_token_review_outcome,
    recommend_human_approval_token_review_action,
    validate_human_approval_token_review_outcome,
)


MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_VERSION = "0.1"
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_KIND = "memory_human_approval_token_issuance_plan_candidate"
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED = "manual_token_issuance_plan_required"
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED = "locked"
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_ROUTING = "manual_token_issuance_dry_run_required_before_any_token_write"
MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_token_issuance_plan_candidates_only": True,
    "issues_real_approval_tokens": False,
    "persists_approvals": False,
    "creates_real_proposals": False,
    "writes_proposal_files": False,
    "writes_operation_ledger": False,
    "applies_proposals": False,
    "submits_to_governance": False,
    "converts_to_real_proposal": False,
}

_DEFAULT_PLANNER = "hermes_memory_human_approval_token_issuance_plan_v0.1"
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
    "converted_to_real_proposal",
)
_LOCK_REASONS = {
    None,
    "invalid_token_review_outcome",
    "token_review_requested_changes",
    "token_review_rejected",
    "token_review_deferred",
    "missing_proposal_record_preview",
    "missing_operation_ledger_preview",
    "missing_target_paths_preview",
    "missing_source_evidence",
}


def create_human_approval_token_issuance_plan(
    review_outcome_candidate: Mapping[str, Any],
    planner: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only human approval token issuance plan candidate."""
    source = deepcopy(dict(review_outcome_candidate))
    review_outcome_validation = validate_human_approval_token_review_outcome(source)
    lock_reason = _plan_lock_reason(source, review_outcome_validation)
    plan_status = (
        MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED
        if lock_reason is None
        else MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    )

    plan = {
        "plan_id": None,
        "plan_kind": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_KIND,
        "plan_status": plan_status,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_ROUTING,
        "lock_reason": lock_reason,
        "source_review_outcome_id": source.get("review_outcome_id"),
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
        "planner": planner if planner is not None else _DEFAULT_PLANNER,
        "outcome": source.get("outcome"),
        "proposal_record_preview": deepcopy(source.get("proposal_record_preview")),
        "operation_ledger_preview": deepcopy(source.get("operation_ledger_preview")),
        "target_paths_preview": deepcopy(source.get("target_paths_preview")),
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "token_issuance_steps": _token_issuance_steps(),
        "token_preflight_checks": _token_preflight_checks(),
        "review_outcome_validation": deepcopy(review_outcome_validation),
        "plan_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_review_outcome_snapshot": deepcopy(source),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY),
    }
    plan["plan_id"] = _plan_id(plan)
    plan["plan_validation"] = validate_human_approval_token_issuance_plan(plan)
    plan["next_step_recommendation"] = recommend_human_approval_token_issuance_plan_action(plan)
    return plan


def validate_human_approval_token_issuance_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(plan.get("policy"))
    source_snapshot = _as_dict(plan.get("source_review_outcome_snapshot"))
    review_outcome_validation = _as_dict(plan.get("review_outcome_validation"))
    expected_review_outcome_validation = validate_human_approval_token_review_outcome(source_snapshot)
    expected_lock_reason = _plan_lock_reason(source_snapshot, review_outcome_validation)

    for key in (
        "plan_id",
        "plan_kind",
        "plan_status",
        "routing",
        "lock_reason",
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
        "planner",
        "outcome",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "token_issuance_steps",
        "token_preflight_checks",
        "review_outcome_validation",
        "plan_validation",
        "next_step_recommendation",
        "source_review_outcome_snapshot",
        "policy",
    ):
        if key not in plan:
            errors.append(f"missing_{key}")
    if plan.get("plan_kind") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_KIND:
        errors.append("plan_kind_must_be_memory_human_approval_token_issuance_plan_candidate")
    if plan.get("plan_status") not in {
        MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED,
        MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED,
    }:
        errors.append("plan_status_must_be_supported")
    if plan.get("routing") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_ROUTING:
        errors.append("routing_must_require_manual_token_issuance_dry_run")
    if plan.get("lock_reason") not in _LOCK_REASONS:
        errors.append("lock_reason_must_be_supported")
    if plan.get("lock_reason") != expected_lock_reason:
        errors.append("lock_reason_must_match_token_issuance_plan_checks")
    if expected_lock_reason is None and plan.get("plan_status") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED:
        errors.append("approved_token_review_outcome_must_require_manual_token_issuance_plan")
    if expected_lock_reason is not None and plan.get("plan_status") != MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED:
        errors.append("invalid_or_non_approved_token_review_outcome_must_lock_plan")
    if not isinstance(plan.get("planner"), str) or not plan.get("planner"):
        errors.append("planner_must_be_non_empty_string")
    if review_outcome_validation != expected_review_outcome_validation:
        errors.append("review_outcome_validation_must_match_source_review_outcome_snapshot")
    if plan.get("outcome") != source_snapshot.get("outcome"):
        errors.append("outcome_must_match_source_review_outcome_snapshot")
    if not isinstance(plan.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(plan.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(plan.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(plan.get("source_pattern_ids"), list) and isinstance(plan.get("source_fact_ids"), list):
        if not plan.get("source_pattern_ids") and not plan.get("source_fact_ids"):
            errors.append("missing_source_evidence")
    if not isinstance(plan.get("token_issuance_steps"), list) or plan.get("token_issuance_steps") != _token_issuance_steps():
        errors.append("token_issuance_steps_must_match_v0_1_deterministic_steps")
    if not isinstance(plan.get("token_preflight_checks"), list) or plan.get("token_preflight_checks") != _token_preflight_checks():
        errors.append("token_preflight_checks_must_match_v0_1_deterministic_checks")
    if not isinstance(plan.get("proposal_record_preview"), Mapping):
        errors.append("missing_proposal_record_preview")
    if not isinstance(plan.get("operation_ledger_preview"), Mapping):
        errors.append("missing_operation_ledger_preview")
    if not isinstance(plan.get("target_paths_preview"), Mapping):
        errors.append("missing_target_paths_preview")
    for field in ("proposal_record_preview", "operation_ledger_preview", "target_paths_preview", "payload_preview"):
        if field in plan and field in source_snapshot and plan.get(field) != source_snapshot.get(field):
            errors.append(f"{field}_must_match_source_review_outcome_snapshot")
    if plan.get("source_pattern_ids") != list(source_snapshot.get("source_pattern_ids", []) or []):
        errors.append("source_pattern_ids_must_match_source_review_outcome_snapshot")
    if plan.get("source_fact_ids") != list(source_snapshot.get("source_fact_ids", []) or []):
        errors.append("source_fact_ids_must_match_source_review_outcome_snapshot")
    for forbidden_key in _FORBIDDEN_TRUE_KEYS:
        if plan.get(forbidden_key) is True:
            errors.append(f"{forbidden_key}_must_be_false_or_absent")
    for key, expected in MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY.items():
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_human_approval_token_issuance_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_issuance_plan(plan)
    source_snapshot = _as_dict(plan.get("source_review_outcome_snapshot"))
    return {
        "plan_id": plan.get("plan_id"),
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
        "lock_reason": plan.get("lock_reason"),
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
        "planner": plan.get("planner"),
        "outcome": plan.get("outcome"),
        "source_pattern_count": len(plan.get("source_pattern_ids", []) or []),
        "source_fact_count": len(plan.get("source_fact_ids", []) or []),
        "token_issuance_steps": deepcopy(plan.get("token_issuance_steps")),
        "token_preflight_checks": deepcopy(plan.get("token_preflight_checks")),
        "validation": validation,
        "review_outcome_explanation": explain_human_approval_token_review_outcome(source_snapshot) if source_snapshot else {},
        "token_issued": False,
        "approved": False,
        "persisted": False,
        "submitted": False,
        "written": False,
        "created_real_proposal": False,
        "created_operation_event": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "converted_to_real_proposal": False,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY),
    }


def recommend_human_approval_token_issuance_plan_action(plan: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_human_approval_token_issuance_plan(plan)
    source_snapshot = _as_dict(plan.get("source_review_outcome_snapshot"))
    if validation["valid"] and plan.get("plan_status") == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED:
        action = "run_manual_token_issuance_dry_run_before_any_token_write"
        reason = "Token issuance plan candidate is ready only for a separate manual dry run; it does not issue, persist, submit, or write tokens."
    else:
        action = "keep_token_issuance_locked"
        reason = _recommendation_reason(plan.get("lock_reason"), validation)
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_ROUTING,
        "validation": validation,
        "review_outcome_recommendation": recommend_human_approval_token_review_action(source_snapshot) if source_snapshot else {},
        "creates_token_issuance_plan_candidates_only": True,
        "issues_real_approval_tokens": False,
        "persists_approvals": False,
        "creates_real_proposals": False,
        "writes_proposal_files": False,
        "writes_operation_ledger": False,
        "applies_proposals": False,
        "submits_to_governance": False,
        "converts_to_real_proposal": False,
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY),
    }


def summarize_human_approval_token_issuance_plans(
    plans: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    by_block_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_lock_reason: dict[str, int] = {}
    plan_required_count = 0
    locked_count = 0
    valid_count = 0
    invalid_count = 0
    for plan in plans:
        block_type = str(plan.get("block_type"))
        by_block_type[block_type] = by_block_type.get(block_type, 0) + 1
        status = str(plan.get("plan_status"))
        by_status[status] = by_status.get(status, 0) + 1
        reason = str(plan.get("lock_reason"))
        by_lock_reason[reason] = by_lock_reason.get(reason, 0) + 1
        if plan.get("plan_status") == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED:
            plan_required_count += 1
        if plan.get("plan_status") == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED:
            locked_count += 1
        validation = validate_human_approval_token_issuance_plan(plan)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(plans),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "plan_required_count": plan_required_count,
        "locked_count": locked_count,
        "by_block_type": dict(sorted(by_block_type.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_lock_reason": dict(sorted(by_lock_reason.items())),
        "policy": dict(MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY),
    }


def _plan_lock_reason(
    source: Mapping[str, Any],
    review_outcome_validation: Mapping[str, Any],
) -> str | None:
    outcome = source.get("outcome")
    if outcome in {"approve_token_issuance", "request_changes"}:
        if not isinstance(source.get("proposal_record_preview"), Mapping):
            return "missing_proposal_record_preview"
        if not isinstance(source.get("operation_ledger_preview"), Mapping):
            return "missing_operation_ledger_preview"
        if not isinstance(source.get("target_paths_preview"), Mapping):
            return "missing_target_paths_preview"
        if not (source.get("source_pattern_ids") or source.get("source_fact_ids")):
            return "missing_source_evidence"
    if review_outcome_validation.get("valid") is not True:
        return "invalid_token_review_outcome"
    if outcome == "request_changes":
        return "token_review_requested_changes"
    if outcome == "reject":
        return "token_review_rejected"
    if outcome == "defer":
        return "token_review_deferred"
    if outcome != "approve_token_issuance":
        return "invalid_token_review_outcome"
    return None


def _token_issuance_steps() -> list[str]:
    return [
        "verify_review_outcome_candidate_is_approve_token_issuance",
        "verify_preview_artifacts_and_source_evidence_are_present",
        "prepare_manual_token_issuance_dry_run_candidate",
        "require_separate_human_confirmation_before_any_token_write",
    ]


def _token_preflight_checks() -> list[str]:
    return [
        "no_real_approval_token_issued",
        "no_approval_persisted",
        "no_real_proposal_created",
        "no_operation_ledger_event_created",
        "no_proposal_file_written",
        "no_memory_or_graph_or_config_write",
    ]


def _recommendation_reason(lock_reason: Any, validation: Mapping[str, Any]) -> str:
    errors = list(validation.get("errors", []) or [])
    if lock_reason == "token_review_requested_changes":
        return "Token review outcome requested changes before any token issuance planning can proceed."
    if lock_reason == "token_review_rejected":
        return "Token review outcome rejected token issuance."
    if lock_reason == "token_review_deferred":
        return "Token review outcome deferred token issuance."
    if lock_reason in {
        "missing_proposal_record_preview",
        "missing_operation_ledger_preview",
        "missing_target_paths_preview",
        "missing_source_evidence",
    }:
        return f"Token issuance plan is locked because {lock_reason}."
    if "review_outcome_validation_must_match_source_review_outcome_snapshot" in errors:
        return "Token issuance plan validation does not match its source review outcome snapshot."
    return "Token issuance plan candidate violates the v0.1 validation contract."


def _plan_id(plan: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_VERSION,
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
        "lock_reason": plan.get("lock_reason"),
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
        "planner": plan.get("planner"),
        "outcome": plan.get("outcome"),
        "proposal_record_preview": plan.get("proposal_record_preview", {}),
        "operation_ledger_preview": plan.get("operation_ledger_preview", {}),
        "target_paths_preview": plan.get("target_paths_preview", {}),
        "payload_preview": plan.get("payload_preview"),
        "source_pattern_ids": list(plan.get("source_pattern_ids", [])),
        "source_fact_ids": list(plan.get("source_fact_ids", [])),
        "token_issuance_steps": list(plan.get("token_issuance_steps", [])),
        "token_preflight_checks": list(plan.get("token_preflight_checks", [])),
        "review_outcome_validation": plan.get("review_outcome_validation", {}),
        "policy": plan.get("policy", {}),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"memory-human-approval-token-issuance-plan:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
