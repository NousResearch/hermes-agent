#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.forge_write_plan import (
        FORGE_WRITE_PLAN_SCHEMA_VERSION,
        redact_secret_values,
        sha256_payload,
    )
    from scripts.hermes_operator.operator_policy import redact_text
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    repo_root_for_import = Path(__file__).resolve().parents[2]
    if str(repo_root_for_import) not in sys.path:
        sys.path.insert(0, str(repo_root_for_import))
    from scripts.hermes_pm.forge_write_plan import (  # type: ignore[no-redef]
        FORGE_WRITE_PLAN_SCHEMA_VERSION,
        redact_secret_values,
        sha256_payload,
    )
    from scripts.hermes_operator.operator_policy import (  # type: ignore[no-redef]
        redact_text,
    )


FORGE_APPROVAL_PACKET_SCHEMA_VERSION = "hermes.pm.forge_approval_packet.v1"
FORGE_APPROVAL_TOKEN_SCHEMA_VERSION = "hermes.pm.forge_approval_token.v1"

EXPLICIT_NON_ACTIONS = {
    "no_gitea_write_performed": True,
    "no_issue_created": True,
    "no_label_created": True,
    "no_project_mutated": True,
    "no_workflow_run": True,
    "no_runner_started": True,
    "no_branch_writer_invoked": True,
    "no_secret_access": True,
    "no_runtime_action": True,
    "no_financial_action": True,
}

SUPPORTED_FUTURE_OPERATION_TYPES = {
    "create_issue",
    "create_label",
    "create_project_column",
    "create_project_card",
    "comment_on_issue",
    "update_issue",
    "request_pr_review",
}

ENDPOINT_UNKNOWN_UNTIL_CAPABILITY_MAP_TYPES = {
    "create_project_column",
    "create_project_card",
    "comment_on_issue",
    "update_issue",
    "request_pr_review",
}


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _operation_ids(operations: list[dict[str, Any]]) -> list[str]:
    return [
        str(operation.get("operation_id"))
        for operation in operations
        if operation.get("operation_id")
    ]


def _approval_operation_ids(operations: list[dict[str, Any]]) -> list[str]:
    return [
        str(operation.get("operation_id"))
        for operation in operations
        if operation.get("requires_operator_approval") and not operation.get("blocked")
    ]


def _operation_types_for_ids(
    operations: list[dict[str, Any]],
    operation_ids: list[str],
) -> list[str]:
    selected = set(operation_ids)
    return sorted(
        {
            str(operation.get("operation_type"))
            for operation in operations
            if operation.get("operation_id") in selected
            and operation.get("operation_type")
        }
    )


def _blocked_operations(plan: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = plan.get("blocked_operations")
    if isinstance(blocked, list):
        return [item for item in blocked if isinstance(item, dict)]
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return []
    return [
        {
            "operation_id": operation.get("operation_id"),
            "operation_type": operation.get("operation_type"),
            "title": operation.get("title"),
            "blockers": operation.get("blockers") or [],
        }
        for operation in operations
        if isinstance(operation, dict) and operation.get("blocked")
    ]


def _unsupported_operation_ids(operations: list[dict[str, Any]]) -> list[str]:
    return [
        str(operation.get("operation_id"))
        for operation in operations
        if operation.get("operation_id")
        and str(operation.get("operation_type") or "")
        not in SUPPORTED_FUTURE_OPERATION_TYPES
    ]


def _endpoint_unknown_operation_ids(operations: list[dict[str, Any]]) -> list[str]:
    return [
        str(operation.get("operation_id"))
        for operation in operations
        if operation.get("operation_id")
        and str(operation.get("operation_type") or "")
        in ENDPOINT_UNKNOWN_UNTIL_CAPABILITY_MAP_TYPES
    ]


def _recommended_first_write_subset(
    operations: list[dict[str, Any]],
    *,
    existing_issue_refs: list[dict[str, Any]],
    selected_backlog_plan: bool = False,
) -> dict[str, Any]:
    if selected_backlog_plan:
        eligible = [
            operation
            for operation in operations
            if operation.get("operation_type") == "create_issue"
            and operation.get("operation_id")
            and not operation.get("blocked")
            and operation.get("source_candidate_id")
            and operation.get("low_risk_backlog_candidate")
        ]
        first = eligible[:1]
        return {
            "max_operations": 1 if first else 0,
            "operation_ids": [
                str(operation.get("operation_id")) for operation in first
            ],
            "candidate_ids": [
                str(operation.get("source_candidate_id")) for operation in first
            ],
            "operation_types": ["create_issue"] if first else [],
            "excluded_operation_types": [
                "create_label",
                "create_project_column",
                "create_project_card",
                "comment_on_issue",
                "update_issue",
                "request_pr_review",
            ],
            "requires_exact_plan_sha256": True,
            "requires_exact_operation_id": True,
            "requires_exact_candidate_id": True,
            "requires_explicit_future_checkpoint_approval": True,
            "reason": (
                "PM-9 selected backlog candidates may only advance as exact "
                "future create_issue approval scopes. Labels, projects, "
                "comments, PRs, workflows, runners, deploys, runtime actions, "
                "trading, and secrets remain excluded."
            ),
        }
    if existing_issue_refs:
        return {
            "max_operations": 0,
            "operation_ids": [],
            "operation_types": [],
            "excluded_operation_types": [
                "create_issue",
                "create_label",
                "create_project_column",
                "create_project_card",
                "comment_on_issue",
                "update_issue",
                "request_pr_review",
            ],
            "requires_exact_plan_sha256": True,
            "requires_exact_operation_id": True,
            "requires_explicit_future_checkpoint_approval": True,
            "reason": (
                "The initial PM seed issue already exists; PM-7 recommends "
                "Operator review instead of approving another first-write "
                "create_issue subset."
            ),
        }
    issue_id = next(
        (
            str(operation.get("operation_id"))
            for operation in operations
            if operation.get("operation_type") == "create_issue"
            and operation.get("operation_id")
            and not operation.get("blocked")
        ),
        None,
    )
    operation_ids = [issue_id] if issue_id else []
    return {
        "max_operations": 1,
        "operation_ids": operation_ids,
        "operation_types": ["create_issue"] if operation_ids else [],
        "excluded_operation_types": [
            "create_label",
            "create_project_column",
            "create_project_card",
            "comment_on_issue",
            "update_issue",
            "request_pr_review",
        ],
        "requires_exact_plan_sha256": True,
        "requires_exact_operation_id": True,
        "requires_explicit_future_checkpoint_approval": True,
        "reason": (
            "The first future forge write should rehearse exactly one "
            "create_issue operation with no labels, projects, cards, comments, "
            "workflows, runners, deploys, runtime actions, trading, or secrets."
        ),
    }


def _recommended_decision(
    *,
    approval_operation_ids: list[str],
    blocked_operations: list[dict[str, Any]],
    existing_issue_refs: list[dict[str, Any]],
    selected_backlog_plan: bool = False,
    selected_low_risk_count: int = 0,
) -> str:
    if blocked_operations:
        return "request_revision"
    if selected_backlog_plan:
        return (
            "approve_selected"
            if selected_low_risk_count == 1
            else "request_operator_review"
        )
    if existing_issue_refs:
        return "request_operator_review"
    if approval_operation_ids:
        return "approve_selected"
    return "approve_none"


def _suggested_token_scope(
    *,
    plan: dict[str, Any],
    plan_sha: str,
    approval_ids: list[str],
    blocked_ids: list[str],
    expires_at: str,
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    stable_payload_sha = plan.get("stable_execution_payload_sha256")
    token = {
        "schema_version": FORGE_APPROVAL_TOKEN_SCHEMA_VERSION,
        "approval_id": "<operator-supplied>",
        "operator": "<operator-name>",
        "approved_at": "<approval-time-utc>",
        "expires_at": expires_at,
        "project_id": plan.get("project_id") or "crypto_bot",
        "gitea_base_url": plan.get("gitea_base_url"),
        "owner": plan.get("owner"),
        "repo": plan.get("repo"),
        "approved_operation_ids": approval_ids,
        "allowed_operation_types": _operation_types_for_ids(
            operations,
            approval_ids,
        ),
        "max_operations": len(approval_ids),
        "reason": "<operator reason>",
        "constraints": {
            "exact_plan_sha256_required": True,
            "exact_operation_ids_required": True,
            "no_wildcards": True,
            "does_not_execute": True,
            "separate_future_checkpoint_required": True,
            "blocked_operation_ids_not_approvable": blocked_ids,
        },
        "forbidden_operation_types": [
            "deploy",
            "financial",
            "runtime_admin",
            "secret",
            "start_runner",
            "trigger_workflow",
        ],
        "forbidden_authority_classes": [
            "deploy",
            "financial",
            "runtime_admin",
            "secret",
        ],
        "single_use_intent": True,
    }
    if isinstance(stable_payload_sha, str) and stable_payload_sha:
        token["stable_execution_payload_sha256"] = stable_payload_sha
        token["whole_plan_sha256_for_reference"] = plan_sha
        token["approval_reference_type"] = "stable_execution_payload_sha256"
        token["constraints"]["exact_plan_sha256_required"] = False
        token["constraints"]["exact_stable_execution_payload_sha256_required"] = True
    else:
        token["forge_write_plan_sha256"] = plan_sha
        token["approval_reference_type"] = "forge_write_plan_sha256"
    return token


def build_forge_approval_packet(
    *,
    forge_write_plan: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    redacted_plan = redact_secret_values(forge_write_plan)
    created = (
        dt.datetime.fromisoformat(created_at)
        if created_at
        else _utc_now()
    )
    if created.tzinfo is None:
        created = created.replace(tzinfo=dt.timezone.utc)
    created = created.astimezone(dt.timezone.utc)
    expires = created + dt.timedelta(hours=24)
    operations = [
        item
        for item in redacted_plan.get("operations") or []
        if isinstance(item, dict)
    ]
    approval_ids = _approval_operation_ids(operations)
    blocked = _blocked_operations(redacted_plan)
    plan_sha = sha256_payload(redacted_plan)
    stable_payload_sha = redacted_plan.get("stable_execution_payload_sha256")
    if not isinstance(stable_payload_sha, str):
        stable_payload_sha = None
    packet_id = f"forge-approval-{plan_sha[:16]}"
    blocked_ids = _operation_ids(blocked)
    unsupported_ids = _unsupported_operation_ids(operations)
    endpoint_unknown_ids = _endpoint_unknown_operation_ids(operations)
    existing_issue_refs = [
        item
        for item in redacted_plan.get("existing_issue_refs") or []
        if isinstance(item, dict)
    ]
    selected_backlog_plan = bool(redacted_plan.get("backlog_selection_supplied"))
    selected_candidate_ids = [
        str(item)
        for item in redacted_plan.get("selected_candidate_ids") or []
        if str(item)
    ]
    selected_low_risk_count = len(
        [
            operation
            for operation in operations
            if operation.get("operation_type") == "create_issue"
            and operation.get("source_candidate_id")
            and operation.get("low_risk_backlog_candidate")
            and not operation.get("blocked")
        ]
    )
    first_write_subset = _recommended_first_write_subset(
        operations,
        existing_issue_refs=existing_issue_refs,
        selected_backlog_plan=selected_backlog_plan,
    )
    recommended_decision = _recommended_decision(
        approval_operation_ids=approval_ids,
        blocked_operations=blocked,
        existing_issue_refs=existing_issue_refs,
        selected_backlog_plan=selected_backlog_plan,
        selected_low_risk_count=selected_low_risk_count,
    )
    selectable_ids = (
        []
        if recommended_decision == "request_operator_review"
        or selected_backlog_plan
        else approval_ids
    )
    if selected_backlog_plan:
        next_checkpoint = {
            "checkpoint": (
                "PM-10: exact approved backlog issue creation scope review"
            ),
            "allow_only_selected_create_issue_operations": True,
            "recommended_operation_ids": first_write_subset["operation_ids"],
            "recommended_candidate_ids": first_write_subset.get("candidate_ids", []),
            "no_labels": True,
            "no_project_cards": True,
            "no_comments": True,
            "exact_candidate_id_required": True,
            "exact_plan_hash_required": True,
            "exact_operation_id_required": True,
            "explicit_approval_token_required": True,
            "approval_packet_does_not_execute": True,
        }
    elif first_write_subset["operation_ids"]:
        next_checkpoint = {
            "checkpoint": "PM-6: one approved Gitea issue creation rehearsal",
            "allow_only_one_create_issue_operation": True,
            "recommended_operation_ids": first_write_subset["operation_ids"],
            "no_labels": True,
            "no_project_cards": True,
            "no_comments": True,
            "exact_plan_hash_required": True,
            "exact_operation_id_required": True,
            "explicit_approval_token_required": True,
            "capability_map_required": True,
            "post_write_attestation_required": True,
        }
    else:
        next_checkpoint = {
            "checkpoint": "PM-7: no-mutation issue lifecycle review",
            "allow_only_one_create_issue_operation": False,
            "recommended_operation_ids": [],
            "no_labels": True,
            "no_project_cards": True,
            "no_comments": True,
            "exact_plan_hash_required": True,
            "exact_operation_id_required": True,
            "explicit_approval_token_required": True,
            "capability_map_required": True,
            "post_write_attestation_required": True,
            "reason": (
                "Issue #1 is existing durable PM state; review and expand "
                "proposal-only backlog instead of recreating it."
            ),
        }
    packet = {
        "schema_version": FORGE_APPROVAL_PACKET_SCHEMA_VERSION,
        "approval_packet_id": packet_id,
        "created_at": created.isoformat(),
        "project_id": redacted_plan.get("project_id") or "crypto_bot",
        "forge_write_plan_id": redacted_plan.get("plan_id"),
        "forge_write_plan_schema_version": redacted_plan.get("schema_version")
        or FORGE_WRITE_PLAN_SCHEMA_VERSION,
        "plan_sha256": plan_sha,
        "whole_plan_sha256_for_reference": plan_sha,
        "stable_execution_payload_sha256": stable_payload_sha,
        "future_approval_token_should_reference": (
            "stable_execution_payload_sha256"
            if stable_payload_sha
            else "forge_write_plan_sha256"
        ),
        "operation_count": len(operations),
        "selected_candidate_ids": selected_candidate_ids,
        "selected_backlog_candidate_count": len(selected_candidate_ids),
        "selected_backlog_plan": selected_backlog_plan,
        "operations_requiring_approval": approval_ids,
        "operations_requiring_approval_count": len(approval_ids),
        "blocked_operations": blocked,
        "blocked_operation_count": len(blocked),
        "selectable_operation_ids": selectable_ids,
        "blocked_operation_ids": blocked_ids,
        "unsupported_operation_ids": unsupported_ids,
        "endpoint_unknown_operation_ids": endpoint_unknown_ids,
        "existing_issue_refs": existing_issue_refs,
        "deduplication_summary": redacted_plan.get("deduplication_summary") or {},
        "recommended_first_write_subset": first_write_subset,
        "recommended_first_backlog_write_subset": first_write_subset
        if selected_backlog_plan
        else {},
        "next_write_checkpoint_recommendation": next_checkpoint,
        "recommended_expiration": expires.isoformat(),
        "recommended_approval_mode": recommended_decision,
        "recommended_decision": recommended_decision,
        "suggested_token_scope": _suggested_token_scope(
            plan=redacted_plan,
            plan_sha=plan_sha,
            approval_ids=selectable_ids,
            blocked_ids=blocked_ids,
            expires_at=expires.isoformat(),
            operations=operations,
        ),
        "approval_scope": {
            "plan_sha256": plan_sha,
            "stable_execution_payload_sha256": stable_payload_sha,
            "forge_write_plan_id": redacted_plan.get("plan_id"),
            "operator_may_approve_operation_ids": selectable_ids,
            "selected_candidate_ids": selected_candidate_ids,
            "recommended_first_write_operation_ids": (
                first_write_subset["operation_ids"]
            ),
            "blocked_operation_ids": blocked_ids,
            "approval_does_not_execute": True,
            "requires_future_mutation_tool": True,
            "allowed_authority_class": "forge_write",
            "forbidden_authority_classes": [
                "deploy",
                "financial",
                "runtime_admin",
                "secret",
            ],
        },
        "expiration_suggestion": {
            "expires_at": expires.isoformat(),
            "duration_hours": 24,
            "reason": (
                "Forge-write approval should be short-lived and tied to the "
                "exact reviewed plan hash."
            ),
        },
        "review_instructions": [
            "Compare the plan SHA-256 with the dry-run forge-write plan.",
            "Review each endpoint preview and redacted payload preview.",
            "Approve selected operation_id values only if the scope is exact.",
            "Reject or request revision for blocked operations.",
            "Do not treat this packet as executable approval by itself.",
        ],
        "approval_token_signing": {
            "performed": False,
            "reason": (
                "This packet suggests exact token scope only; it is not an "
                "active approval token and does not execute anything."
            ),
        },
        "explicit_non_actions": dict(EXPLICIT_NON_ACTIONS),
    }
    return redact_secret_values(packet)


def format_forge_approval_packet_text(packet: dict[str, Any]) -> str:
    lines = [
        "Hermes PM forge approval packet",
        f"Project: {packet.get('project_id') or '<unknown>'}",
        f"Approval packet: {packet.get('approval_packet_id') or '<unknown>'}",
        f"Plan: {packet.get('forge_write_plan_id') or '<unknown>'}",
        f"Plan sha256: {packet.get('plan_sha256') or '<missing>'}",
        f"Operations: {packet.get('operation_count', 0)}",
        (
            "Approval required: "
            f"{packet.get('operations_requiring_approval_count', 0)}"
        ),
        f"Blocked: {packet.get('blocked_operation_count', 0)}",
        f"Decision: {packet.get('recommended_decision')}",
        "Gitea writes performed: no",
        "Workflow/runner/runtime/financial/secret actions: no",
    ]
    return redact_text("\n".join(lines))
