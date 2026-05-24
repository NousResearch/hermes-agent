from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.memory_human_review_outcome_gate import (
    explain_human_review_outcome_candidate,
    recommend_human_review_outcome_action,
    validate_human_review_outcome_candidate,
)
from agent.memory_read_only_candidate_utils import (
    build_stable_digest,
    deep_copy_mapping,
    summarize_candidates,
    validate_policy_flags,
)


MEMORY_REAL_PROPOSAL_CREATION_PLAN_VERSION = "0.1"
MEMORY_REAL_PROPOSAL_CREATION_PLAN_KIND = "memory_real_proposal_creation_plan_candidate"
MEMORY_REAL_PROPOSAL_CREATION_PLAN_STATUS = "manual_creation_plan_required"
MEMORY_REAL_PROPOSAL_CREATION_PLAN_ROUTING = "manual_real_proposal_creation_required"
MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_plan_candidates_only": True,
    "creates_real_proposals": False,
    "applies_proposals": False,
    "persists_approvals": False,
    "submits_to_governance": False,
    "converts_to_real_proposal": False,
}

_DEFAULT_PLANNER = "hermes_memory_real_proposal_creation_plan_v0.1"
_FORBIDDEN_TRUE_KEYS = (
    "submitted",
    "applied",
    "persisted",
    "approved",
    "converted_to_real_proposal",
    "created_real_proposal",
    "created_operation_event",
    "created_proposal_record",
    "submitted_to_governance",
)


def create_real_proposal_creation_plan(
    outcome_candidate: Mapping[str, Any],
    planner: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic read-only real proposal creation plan candidate."""
    source = deepcopy(dict(outcome_candidate))
    outcome_validation = validate_human_review_outcome_candidate(source)
    invalid_reason = _plan_invalid_reason(source, outcome_validation)

    plan = {
        "plan_id": None,
        "plan_kind": MEMORY_REAL_PROPOSAL_CREATION_PLAN_KIND,
        "plan_status": MEMORY_REAL_PROPOSAL_CREATION_PLAN_STATUS,
        "routing": MEMORY_REAL_PROPOSAL_CREATION_PLAN_ROUTING,
        "source_outcome_id": source.get("outcome_id"),
        "source_packet_id": source.get("source_packet_id"),
        "source_submission_id": source.get("source_submission_id"),
        "source_draft_id": source.get("source_draft_id"),
        "source_decision_id": source.get("source_decision_id"),
        "source_queue_item_id": source.get("source_queue_item_id"),
        "block_id": source.get("block_id"),
        "block_type": source.get("block_type"),
        "project_scope": source.get("project_scope"),
        "planner": planner if planner is not None else _DEFAULT_PLANNER,
        "payload_preview": deepcopy(source.get("payload_preview")),
        "source_pattern_ids": deepcopy(list(source.get("source_pattern_ids", []) or [])),
        "source_fact_ids": deepcopy(list(source.get("source_fact_ids", []) or [])),
        "creation_steps": _creation_steps(),
        "preflight_checks": _preflight_checks(),
        "outcome_validation": deepcopy(outcome_validation),
        "plan_validation": {"valid": False, "errors": ["not_validated"]},
        "next_step_recommendation": {},
        "source_outcome_snapshot": deepcopy(source),
        "policy": dict(MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY),
    }
    if invalid_reason:
        plan["invalid_reason"] = invalid_reason
    plan["plan_id"] = _plan_id(plan)
    plan["plan_validation"] = validate_real_proposal_creation_plan(plan)
    plan["next_step_recommendation"] = recommend_real_proposal_creation_plan_action(plan)
    return plan


def validate_real_proposal_creation_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    policy = _as_dict(plan.get("policy"))
    outcome_validation = _as_dict(plan.get("outcome_validation"))
    source_snapshot = _as_dict(plan.get("source_outcome_snapshot"))

    for key in (
        "plan_id",
        "plan_kind",
        "plan_status",
        "routing",
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
        "payload_preview",
        "source_pattern_ids",
        "source_fact_ids",
        "creation_steps",
        "preflight_checks",
        "outcome_validation",
        "plan_validation",
        "next_step_recommendation",
        "source_outcome_snapshot",
        "policy",
    ):
        if key not in plan:
            errors.append(f"missing_{key}")
    if plan.get("plan_kind") != MEMORY_REAL_PROPOSAL_CREATION_PLAN_KIND:
        errors.append("plan_kind_must_be_memory_real_proposal_creation_plan_candidate")
    if plan.get("plan_status") != MEMORY_REAL_PROPOSAL_CREATION_PLAN_STATUS:
        errors.append("plan_status_must_be_manual_creation_plan_required")
    if plan.get("routing") != MEMORY_REAL_PROPOSAL_CREATION_PLAN_ROUTING:
        errors.append("routing_must_require_manual_real_proposal_creation")
    if not isinstance(plan.get("planner"), str) or not plan.get("planner"):
        errors.append("planner_must_be_non_empty_string")
    if outcome_validation.get("valid") is not True:
        errors.append("invalid_human_review_outcome_candidate")
    if source_snapshot:
        snapshot_validation = validate_human_review_outcome_candidate(source_snapshot)
        if snapshot_validation != outcome_validation:
            errors.append("outcome_validation_must_match_source_outcome_snapshot")
    if source_snapshot.get("outcome") == "request_changes":
        errors.append("human_review_requested_changes")
    elif source_snapshot.get("outcome") == "reject":
        errors.append("human_review_rejected")
    elif source_snapshot.get("outcome") == "defer":
        errors.append("human_review_deferred")
    elif source_snapshot.get("outcome") != "approve_real_proposal_creation":
        errors.append("outcome_must_approve_real_proposal_creation")
    if not isinstance(plan.get("payload_preview"), Mapping):
        errors.append("missing_payload_preview")
    if not isinstance(plan.get("source_pattern_ids"), list):
        errors.append("source_pattern_ids_must_be_list")
    if not isinstance(plan.get("source_fact_ids"), list):
        errors.append("source_fact_ids_must_be_list")
    if isinstance(plan.get("source_pattern_ids"), list) and isinstance(plan.get("source_fact_ids"), list):
        if not plan.get("source_pattern_ids") and not plan.get("source_fact_ids"):
            errors.append("missing_source_evidence")
    if not isinstance(plan.get("creation_steps"), list) or not plan.get("creation_steps"):
        errors.append("creation_steps_must_be_non_empty_list")
    if not isinstance(plan.get("preflight_checks"), list) or not plan.get("preflight_checks"):
        errors.append("preflight_checks_must_be_non_empty_list")
    for forbidden_key in _FORBIDDEN_TRUE_KEYS:
        if plan.get(forbidden_key) is True:
            errors.append(f"{forbidden_key}_must_be_false_or_absent")
    errors.extend(validate_policy_flags(policy, MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY))

    return {"valid": not errors, "errors": _dedupe(errors)}


def explain_real_proposal_creation_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_real_proposal_creation_plan(plan)
    source_snapshot = _as_dict(plan.get("source_outcome_snapshot"))
    return {
        "plan_id": plan.get("plan_id"),
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
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
        "source_pattern_count": len(plan.get("source_pattern_ids", []) or []),
        "source_fact_count": len(plan.get("source_fact_ids", []) or []),
        "creation_steps": deepcopy(plan.get("creation_steps")),
        "preflight_checks": deepcopy(plan.get("preflight_checks")),
        "validation": validation,
        "outcome_explanation": explain_human_review_outcome_candidate(source_snapshot) if source_snapshot else {},
        "submitted": False,
        "applied": False,
        "persisted": False,
        "approved": False,
        "converted_to_real_proposal": False,
        "created_real_proposal": False,
        "created_operation_event": False,
        "created_proposal_record": False,
        "submitted_to_governance": False,
        "policy": dict(MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY),
    }


def recommend_real_proposal_creation_plan_action(plan: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_real_proposal_creation_plan(plan)
    source_snapshot = _as_dict(plan.get("source_outcome_snapshot"))
    if validation["valid"]:
        action = "perform_manual_real_proposal_creation_preflight"
        reason = "Plan candidate is valid for manual real proposal creation planning only; it does not create or submit a real proposal."
    else:
        action = "do_not_create_real_proposal"
        reason = _recommendation_reason(validation)
    return {
        "action": action,
        "reason": reason,
        "routing": MEMORY_REAL_PROPOSAL_CREATION_PLAN_ROUTING,
        "validation": validation,
        "outcome_recommendation": recommend_human_review_outcome_action(source_snapshot) if source_snapshot else {},
        "creates_plan_candidates_only": True,
        "creates_real_proposals": False,
        "applies_proposals": False,
        "persists_approvals": False,
        "submits_to_governance": False,
        "converts_to_real_proposal": False,
        "policy": dict(MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY),
    }


def summarize_real_proposal_creation_plans(
    plans: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    candidate_summary = summarize_candidates(plans, "plan_status")
    valid_count = 0
    invalid_count = 0
    for plan in plans:
        validation = validate_real_proposal_creation_plan(plan)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
    return {
        "total": len(plans),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "by_block_type": candidate_summary["by_block_type"],
        "by_status": candidate_summary["by_status"],
        "policy": dict(MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY),
    }


def _plan_invalid_reason(source: Mapping[str, Any], outcome_validation: Mapping[str, Any]) -> str | None:
    if outcome_validation.get("valid") is not True:
        return "invalid_human_review_outcome_candidate"
    outcome = source.get("outcome")
    if outcome == "request_changes":
        return "human_review_requested_changes"
    if outcome == "reject":
        return "human_review_rejected"
    if outcome == "defer":
        return "human_review_deferred"
    if outcome != "approve_real_proposal_creation":
        return "invalid_human_review_outcome_candidate"
    if not isinstance(source.get("payload_preview"), Mapping):
        return "missing_payload_preview"
    if not (source.get("source_pattern_ids") or source.get("source_fact_ids")):
        return "missing_source_evidence"
    return None


def _creation_steps() -> list[dict[str, str]]:
    return [
        {
            "id": "verify_human_review_approval",
            "description": "Verify the source outcome candidate explicitly approves real proposal creation.",
        },
        {
            "id": "verify_payload_preview",
            "description": "Verify the payload preview matches the reviewed memory block payload.",
        },
        {
            "id": "verify_source_evidence",
            "description": "Verify source pattern or fact evidence supports the proposal payload.",
        },
        {
            "id": "manual_real_proposal_creation",
            "description": "Create a real proposal only through the separate governed manual creation path.",
        },
    ]


def _preflight_checks() -> list[dict[str, str]]:
    return [
        {
            "id": "no_memory_write",
            "description": "Confirm this planning step did not write durable memory.",
        },
        {
            "id": "no_graph_write",
            "description": "Confirm this planning step did not write the real Memory Graph.",
        },
        {
            "id": "no_config_change",
            "description": "Confirm this planning step did not modify OpenClaw or Hermes configuration.",
        },
        {
            "id": "no_proposal_or_ledger_record",
            "description": "Confirm this planning step did not create proposal records or operation-ledger events.",
        },
    ]


def _recommendation_reason(validation: Mapping[str, Any]) -> str:
    errors = list(validation.get("errors", []) or [])
    if "human_review_requested_changes" in errors:
        return "Human review requested changes before any real proposal creation."
    if "human_review_rejected" in errors:
        return "Human review rejected real proposal creation."
    if "human_review_deferred" in errors:
        return "Human review deferred real proposal creation."
    if "missing_payload_preview" in errors:
        return "Plan candidate is missing payload preview required before manual real proposal creation."
    if "missing_source_evidence" in errors:
        return "Plan candidate is missing source pattern or fact evidence."
    if "invalid_human_review_outcome_candidate" in errors:
        return "Source human review outcome candidate is invalid."
    return "Plan candidate violates the v0.1 real proposal creation plan validation contract."


def _plan_id(plan: Mapping[str, Any]) -> str:
    identity = {
        "version": MEMORY_REAL_PROPOSAL_CREATION_PLAN_VERSION,
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "routing": plan.get("routing"),
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
        "payload_preview": plan.get("payload_preview"),
        "source_pattern_ids": list(plan.get("source_pattern_ids", [])),
        "source_fact_ids": list(plan.get("source_fact_ids", [])),
        "creation_steps": plan.get("creation_steps", []),
        "preflight_checks": plan.get("preflight_checks", []),
        "outcome_validation": plan.get("outcome_validation", {}),
        "invalid_reason": plan.get("invalid_reason"),
        "policy": plan.get("policy", {}),
    }
    return build_stable_digest("memory-real-proposal-creation-plan:v0.1", identity)


def _as_dict(value: Any) -> dict[str, Any]:
    return deep_copy_mapping(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
