#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

try:
    from scripts.hermes_pm.backlog_candidate_approval_scope import sha256_payload
    from scripts.hermes_pm.backlog_selection_packet import (
        BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
    )
    from scripts.hermes_pm.forge_write_plan import (
        FORGE_WRITE_PLAN_SCHEMA_VERSION,
        build_forge_write_plan,
        format_forge_write_plan_text,
        redact_secret_values,
    )
    from scripts.hermes_pm.kanban_proposal_packet import (
        KANBAN_PACKET_SCHEMA_VERSION,
    )
    from scripts.hermes_pm.selected_candidate_execution_payload import (
        STABLE_EXECUTION_PAYLOAD_HASH_FIELD,
        approval_scope_from_execution_payload,
        validate_selected_candidate_execution_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from backlog_candidate_approval_scope import (  # type: ignore[no-redef]
        sha256_payload,
    )
    from backlog_selection_packet import (  # type: ignore[no-redef]
        BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
    )
    from forge_write_plan import (  # type: ignore[no-redef]
        FORGE_WRITE_PLAN_SCHEMA_VERSION,
        build_forge_write_plan,
        format_forge_write_plan_text,
        redact_secret_values,
    )
    from kanban_proposal_packet import (  # type: ignore[no-redef]
        KANBAN_PACKET_SCHEMA_VERSION,
    )
    from selected_candidate_execution_payload import (  # type: ignore[no-redef]
        STABLE_EXECUTION_PAYLOAD_HASH_FIELD,
        approval_scope_from_execution_payload,
        validate_selected_candidate_execution_payload,
    )


def _selected_candidate_from_scope(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": scope.get("selected_candidate_id"),
        "title": scope.get("selected_candidate_title")
        or scope.get("proposed_issue_title"),
        "proposed_issue_body_summary": (
            "Selected PM-10 backlog candidate approval scope."
        ),
        "proposed_issue_body": scope.get("proposed_issue_body"),
        "rationale": (
            "PM-10 selected-candidate approval scope for future issue-only "
            "backlog expansion."
        ),
        "source": "backlog_candidate_approval_scope",
        "suggested_priority": "P1",
        "suggested_authority_class": "propose",
        "task_class": "propose",
        "selection_status": "selected_for_review",
        "approval_required": True,
        "blocked": False,
        "blockers": [],
        "ci_or_workflow_related": False,
        "non_executable": True,
        "proposal_only": True,
        "duplicates_issue_1": False,
        "future_create_issue_requires_approval": True,
    }


def build_selected_candidate_forge_plan(
    *,
    approval_scope: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    if approval_scope.get("blocked"):
        raise ValueError("Approval scope is blocked; no forge plan can be generated.")
    constraints = (
        approval_scope.get("exact_future_constraints")
        if isinstance(approval_scope.get("exact_future_constraints"), dict)
        else {}
    )
    candidate_id = str(approval_scope.get("selected_candidate_id") or "")
    if not candidate_id:
        raise ValueError("Approval scope must include selected_candidate_id.")
    selected_candidate = _selected_candidate_from_scope(approval_scope)
    selection_packet = {
        "schema_version": BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
        "selection_packet_id": approval_scope.get("source_selection_packet_id"),
        "project_id": approval_scope.get("project_id") or "crypto_bot",
        "selected_candidates": [selected_candidate],
        "review_pending_candidates": [],
        "deferred_candidates": [],
        "rejected_candidates": [],
        "blocked_candidates": [],
        "recommended_future_write_scope": [
            {
                "candidate_id": candidate_id,
                "proposed_title": approval_scope.get("proposed_issue_title"),
                "proposed_body_summary": (
                    "Selected PM-10 backlog candidate approval scope."
                ),
                "authority_class": "forge_write",
                "future_operation_type": "create_issue",
                "requires_exact_approval_token": True,
                "requires_exact_plan_hash": True,
                "requires_operator_confirmation": True,
                "approval_required": True,
                "executable_by_default": False,
            }
        ],
        "approval_required_for_write": True,
        "calls_gitea_write_api": False,
        "mutation_executed": False,
    }
    issue_ref = (
        approval_scope.get("issue_1_reference")
        if isinstance(approval_scope.get("issue_1_reference"), dict)
        else {}
    )
    kanban_packet = {
        "schema_version": KANBAN_PACKET_SCHEMA_VERSION,
        "project_id": approval_scope.get("project_id") or "crypto_bot",
        "existing_issue_refs": [issue_ref] if issue_ref else [],
        "proposed_cards": [],
        "suggested_labels": [],
        "proposed_columns": [],
        "suggested_issue_updates": [],
        "suggested_pr_attention": [],
    }
    plan = build_forge_write_plan(
        kanban_packet=kanban_packet,
        backlog_selection_packet=selection_packet,
        project_id=str(approval_scope.get("project_id") or "crypto_bot"),
        gitea_base_url=str(
            constraints.get("gitea_base_url") or "http://127.0.0.1:3005"
        ),
        owner=str(constraints.get("owner") or "preston"),
        repo=str(constraints.get("repo") or "crypto_bot"),
        created_at=created_at,
    )
    operations = [
        item
        for item in plan.get("operations") or []
        if isinstance(item, dict)
    ]
    if len(operations) != 1 or operations[0].get("operation_type") != "create_issue":
        raise ValueError(
            "Selected candidate forge plan must contain exactly one create_issue "
            "operation."
        )
    operation = operations[0]
    operation["source_approval_scope_id"] = approval_scope.get("approval_scope_id")
    operation["approval_required"] = True
    operation["future_exact_token_required"] = True
    operation["no_labels"] = True
    operation["no_projects"] = True
    operation["no_comments"] = True
    operation["requires_exact_stable_execution_payload_sha256"] = False
    plan["source_approval_scope_id"] = approval_scope.get("approval_scope_id")
    plan["selected_candidate_approval_scope_sha256"] = sha256_payload(
        redact_secret_values(approval_scope)
    )
    plan["schema_version"] = FORGE_WRITE_PLAN_SCHEMA_VERSION
    plan["approval_required"] = True
    plan["future_exact_token_required"] = True
    plan["future_write_checkpoint"] = "PM-11"
    plan["execution_payload_freeze_state"] = "not_frozen_for_execution"
    plan["requires_stable_execution_payload_sha256"] = False
    plan["approval_hash_kind"] = "forge_write_plan_sha256"
    plan["warnings"] = sorted(
        dict.fromkeys(
            [
                *[
                    str(item)
                    for item in plan.get("warnings") or []
                    if str(item).strip()
                ],
                (
                    "Selected-candidate forge plan is not frozen for execution; "
                    "future PM-11 approval must use a frozen execution payload."
                ),
            ]
        )
    )
    plan["selected_candidate_scope_constraints"] = constraints
    plan["operations"] = operations
    plan["endpoint_preview"] = [
        {
            "operation_id": operation["operation_id"],
            "method": operation["http_method_that_would_be_used"],
            "endpoint": operation["expected_gitea_endpoint"],
            "called": False,
        }
    ]
    plan["payload_preview_redacted"] = [
        {
            "operation_id": operation["operation_id"],
            "payload": operation["proposed_payload_redacted"],
        }
    ]
    return redact_secret_values(plan)


def build_selected_candidate_forge_plan_from_execution_payload(
    *,
    execution_payload: dict[str, Any],
    created_at: str | None = None,
    frozen_payload_path: str | None = None,
) -> dict[str, Any]:
    payload = validate_selected_candidate_execution_payload(execution_payload)
    stable_hash = payload[STABLE_EXECUTION_PAYLOAD_HASH_FIELD]
    approval_scope = approval_scope_from_execution_payload(payload)
    selection_packet = {
        "schema_version": BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
        "selection_packet_id": f"frozen-execution-payload-{stable_hash[:16]}",
        "project_id": payload["project_id"],
        "selected_candidates": [
            {
                "candidate_id": payload["candidate_id"],
                "title": payload["issue_title"],
                "candidate_title": payload["candidate_title"],
                "proposed_issue_body_summary": (
                    "Frozen selected-candidate PM-11 issue payload."
                ),
                "proposed_issue_body": payload["issue_body"],
                "rationale": (
                    "PM-10B selected-candidate execution payload freeze for "
                    "future issue-only backlog governance."
                ),
                "source": "selected_candidate_execution_payload",
                "suggested_priority": "P1",
                "suggested_authority_class": "propose",
                "task_class": "propose",
                "selection_status": "selected_for_review",
                "approval_required": True,
                "blocked": False,
                "blockers": [],
                "ci_or_workflow_related": False,
                "non_executable": True,
                "proposal_only": True,
                "duplicates_issue_1": False,
                "future_create_issue_requires_approval": True,
            }
        ],
        "review_pending_candidates": [],
        "deferred_candidates": [],
        "rejected_candidates": [],
        "blocked_candidates": [],
        "recommended_future_write_scope": [
            {
                "candidate_id": payload["candidate_id"],
                "proposed_title": payload["issue_title"],
                "proposed_body_summary": (
                    "Frozen selected-candidate PM-11 issue payload."
                ),
                "authority_class": "forge_write",
                "future_operation_type": "create_issue",
                "requires_exact_approval_token": True,
                "requires_exact_stable_execution_payload_sha256": True,
                "requires_exact_plan_hash": False,
                "requires_operator_confirmation": True,
                "approval_required": True,
                "executable_by_default": False,
            }
        ],
        "approval_required_for_write": True,
        "calls_gitea_write_api": False,
        "mutation_executed": False,
    }
    kanban_packet = {
        "schema_version": KANBAN_PACKET_SCHEMA_VERSION,
        "project_id": payload["project_id"],
        "existing_issue_refs": [],
        "proposed_cards": [],
        "suggested_labels": [],
        "proposed_columns": [],
        "suggested_issue_updates": [],
        "suggested_pr_attention": [],
    }
    plan = build_forge_write_plan(
        kanban_packet=kanban_packet,
        backlog_selection_packet=selection_packet,
        project_id=payload["project_id"],
        gitea_base_url=payload["gitea_base_url"],
        owner=payload["owner"],
        repo=payload["repo"],
        created_at=created_at,
    )
    operations = [
        item
        for item in plan.get("operations") or []
        if isinstance(item, dict)
    ]
    if len(operations) != 1 or operations[0].get("operation_type") != "create_issue":
        raise ValueError(
            "Frozen selected-candidate payload must generate exactly one "
            "create_issue operation."
        )
    operation = operations[0]
    if operation.get("operation_id") != payload["operation_id"]:
        raise ValueError(
            "Frozen selected-candidate payload operation_id must match the "
            "generated single-operation forge plan."
        )
    proposed_payload = operation.get("proposed_payload_redacted")
    if not isinstance(proposed_payload, dict):
        raise ValueError("Frozen selected-candidate operation payload is missing.")
    if proposed_payload.get("title") != payload["issue_title"]:
        raise ValueError("Frozen selected-candidate issue title drifted.")
    if proposed_payload.get("body") != payload["issue_body"]:
        raise ValueError("Frozen selected-candidate issue body drifted.")

    operation["source_candidate_id"] = payload["candidate_id"]
    operation["source_candidate_title"] = payload["candidate_title"]
    operation["source_execution_payload_schema_version"] = payload["schema_version"]
    operation["stable_execution_payload_sha256"] = stable_hash
    operation["requires_exact_stable_execution_payload_sha256"] = True
    operation["requires_exact_plan_hash"] = False
    operation["approval_hash_kind"] = "stable_execution_payload_sha256"
    operation["source_approval_scope_id"] = approval_scope.get("approval_scope_id")
    operation["approval_required"] = True
    operation["future_exact_token_required"] = True
    operation["no_labels"] = True
    operation["no_projects"] = True
    operation["no_comments"] = True
    operation["executed"] = False

    plan["schema_version"] = FORGE_WRITE_PLAN_SCHEMA_VERSION
    plan["source_approval_scope_id"] = approval_scope.get("approval_scope_id")
    plan["selected_candidate_approval_scope_sha256"] = sha256_payload(
        redact_secret_values(approval_scope)
    )
    plan["stable_execution_payload_sha256"] = stable_hash
    plan["selected_candidate_execution_payload"] = payload
    plan["frozen_payload_path"] = frozen_payload_path
    plan["execution_payload_freeze_state"] = "frozen_for_execution"
    plan["requires_stable_execution_payload_sha256"] = True
    plan["approval_hash_kind"] = "stable_execution_payload_sha256"
    plan["approval_required"] = True
    plan["future_exact_token_required"] = True
    plan["future_write_checkpoint"] = "PM-11"
    plan["selected_candidate_scope_constraints"] = (
        approval_scope.get("exact_future_constraints") or {}
    )
    plan["whole_plan_sha256_is_reference_only_for_selected_candidate"] = True
    plan["not_frozen_for_execution"] = False
    plan["calls_gitea_write_api"] = False
    plan["mutation_executed"] = False
    plan["operations"] = operations
    plan["endpoint_preview"] = [
        {
            "operation_id": operation["operation_id"],
            "method": operation["http_method_that_would_be_used"],
            "endpoint": operation["expected_gitea_endpoint"],
            "called": False,
        }
    ]
    plan["payload_preview_redacted"] = [
        {
            "operation_id": operation["operation_id"],
            "payload": operation["proposed_payload_redacted"],
        }
    ]
    plan["approval_requirements"] = [
        {
            "operation_id": operation["operation_id"],
            "operation_type": operation["operation_type"],
            "approval_required": True,
            "blocked": False,
            "required_evidence": [
                "explicit Operator approval naming this operation_id",
                "matching stable_execution_payload_sha256 from the frozen payload",
                "reviewed endpoint preview and exact issue title/body payload",
                "confirmation that no forbidden PM/runtime/financial/secret "
                "surface is involved",
            ],
        }
    ]
    plan["evidence_requirements"] = [
        "Current PM status report generated in read-only mode.",
        "Frozen selected-candidate execution payload and its SHA-256.",
        "Operator approval scoped to stable_execution_payload_sha256 and "
        "operation_id.",
        "Redacted endpoint and exact payload previews reviewed by the Operator.",
        "Proof that no Gitea write API was called during planning.",
        "Proof that no workflow, runner, deploy, runtime, financial, secret, "
        "or branch-writer action occurred.",
        "Rollback or manual correction notes for the future issue mutation.",
    ]
    plan["warnings"] = sorted(
        dict.fromkeys(
            [
                *[
                    str(item)
                    for item in plan.get("warnings") or []
                    if str(item).strip()
                ],
                (
                    "Whole-plan SHA-256 is reference-only for selected-candidate "
                    "PM-11 execution; approval must reference "
                    "stable_execution_payload_sha256."
                ),
            ]
        )
    )
    return redact_secret_values(plan)


def format_selected_candidate_forge_plan_text(plan: dict[str, Any]) -> str:
    return format_forge_write_plan_text(plan)
