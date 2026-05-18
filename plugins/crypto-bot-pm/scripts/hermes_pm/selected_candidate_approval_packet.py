#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from typing import Any

try:
    from scripts.hermes_pm.forge_approval_packet import (
        FORGE_APPROVAL_TOKEN_SCHEMA_VERSION,
    )
    from scripts.hermes_pm.forge_write_plan import redact_secret_values
    from scripts.hermes_pm.backlog_candidate_approval_scope import (
        sha256_payload,
    )
    from scripts.hermes_pm.selected_candidate_execution_payload import (
        STABLE_EXECUTION_PAYLOAD_HASH_FIELD,
        validate_selected_candidate_execution_payload,
    )
    from scripts.hermes_operator.operator_policy import redact_text
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from forge_approval_packet import (  # type: ignore[no-redef]
        FORGE_APPROVAL_TOKEN_SCHEMA_VERSION,
    )
    from forge_write_plan import (  # type: ignore[no-redef]
        redact_secret_values,
    )
    from backlog_candidate_approval_scope import (  # type: ignore[no-redef]
        sha256_payload,
    )
    from selected_candidate_execution_payload import (  # type: ignore[no-redef]
        STABLE_EXECUTION_PAYLOAD_HASH_FIELD,
        validate_selected_candidate_execution_payload,
    )
    from scripts.hermes_operator.operator_policy import (  # type: ignore[no-redef]
        redact_text,
    )


SELECTED_CANDIDATE_APPROVAL_PACKET_SCHEMA_VERSION = (
    "hermes.pm.selected_candidate_approval_packet.v1"
)

EXPLICIT_NON_ACTIONS = {
    "no_gitea_write_performed": True,
    "no_issue_created": True,
    "no_label_created": True,
    "no_project_mutated": True,
    "no_comment_created": True,
    "no_workflow_run": True,
    "no_runner_started": True,
    "no_branch_writer_invoked": True,
    "no_secret_access": True,
    "no_runtime_action": True,
    "no_financial_action": True,
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _single_create_issue_operation(plan: dict[str, Any]) -> dict[str, Any]:
    operations = [
        item
        for item in plan.get("operations") or []
        if isinstance(item, dict)
    ]
    if len(operations) != 1:
        raise ValueError("Selected-candidate approval packet requires one operation.")
    operation = operations[0]
    if operation.get("operation_type") != "create_issue":
        raise ValueError("Selected-candidate operation must be create_issue.")
    return operation


def _suggested_future_token_fields(
    *,
    approval_scope: dict[str, Any],
    forge_write_plan: dict[str, Any],
    plan_sha: str,
    stable_execution_payload_sha256: str | None,
    operation_id: str,
    created_at: str,
) -> dict[str, Any]:
    constraints = (
        approval_scope.get("exact_future_constraints")
        if isinstance(approval_scope.get("exact_future_constraints"), dict)
        else {}
    )
    expires = (
        dt.datetime.fromisoformat(created_at)
        + dt.timedelta(hours=24)
    ).isoformat()
    token = {
        "schema_version": FORGE_APPROVAL_TOKEN_SCHEMA_VERSION,
        "approval_id": "<operator-supplied-pm-11>",
        "operator": "<operator-name>",
        "approved_at": "<approval-time-utc>",
        "expires_at": expires,
        "project_id": approval_scope.get("project_id") or "crypto_bot",
        "selected_candidate_id": approval_scope.get("selected_candidate_id"),
        "gitea_base_url": constraints.get("gitea_base_url")
        or forge_write_plan.get("gitea_base_url"),
        "owner": constraints.get("owner") or forge_write_plan.get("owner"),
        "repo": constraints.get("repo") or forge_write_plan.get("repo"),
        "approved_operation_ids": [operation_id],
        "allowed_operation_types": ["create_issue"],
        "max_operations": 1,
        "reason": "<operator reason>",
        "constraints": {
            "exact_candidate_id_required": True,
            "exact_plan_sha256_required": True,
            "exact_operation_ids_required": True,
            "exact_repo_required": True,
            "no_wildcards": True,
            "no_labels": True,
            "no_projects": True,
            "no_comments": True,
            "no_prs": True,
            "no_workflows": True,
            "no_runners": True,
            "no_runtime_actions": True,
            "no_financial_actions": True,
            "no_secret_access": True,
            "single_use_intent": True,
        },
        "not_active_approval": True,
    }
    if stable_execution_payload_sha256:
        token["stable_execution_payload_sha256"] = stable_execution_payload_sha256
        token["whole_plan_sha256_for_reference"] = plan_sha
        token["approval_reference_type"] = "stable_execution_payload_sha256"
        token["constraints"]["exact_stable_execution_payload_sha256_required"] = True
        token["constraints"]["exact_plan_sha256_required"] = False
    else:
        token["forge_write_plan_sha256"] = plan_sha
        token["approval_reference_type"] = "forge_write_plan_sha256"
    return token


def build_selected_candidate_approval_packet(
    *,
    approval_scope: dict[str, Any],
    forge_write_plan: dict[str, Any],
    frozen_payload_path: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    redacted_scope = redact_secret_values(approval_scope)
    redacted_plan = redact_secret_values(forge_write_plan)
    operation = _single_create_issue_operation(redacted_plan)
    candidate_id = str(redacted_scope.get("selected_candidate_id") or "")
    if operation.get("source_candidate_id") != candidate_id:
        raise ValueError("Forge operation candidate ID does not match approval scope.")
    created = created_at or _utc_now()
    plan_sha = sha256_payload(redacted_plan)
    execution_payload = redacted_plan.get("selected_candidate_execution_payload")
    stable_execution_payload_sha256 = None
    if isinstance(execution_payload, dict):
        frozen = validate_selected_candidate_execution_payload(execution_payload)
        stable_execution_payload_sha256 = frozen[
            STABLE_EXECUTION_PAYLOAD_HASH_FIELD
        ]
    elif isinstance(
        redacted_plan.get(STABLE_EXECUTION_PAYLOAD_HASH_FIELD),
        str,
    ):
        stable_execution_payload_sha256 = str(
            redacted_plan.get(STABLE_EXECUTION_PAYLOAD_HASH_FIELD)
        )
    operation_id = str(operation.get("operation_id"))
    packet_seed = stable_execution_payload_sha256 or plan_sha
    packet_id = f"selected-candidate-approval-{packet_seed[:16]}"
    not_frozen = stable_execution_payload_sha256 is None
    packet = {
        "schema_version": SELECTED_CANDIDATE_APPROVAL_PACKET_SCHEMA_VERSION,
        "approval_packet_id": packet_id,
        "created_at": created,
        "selected_candidate_id": candidate_id,
        "candidate_title": redacted_scope.get("selected_candidate_title")
        or redacted_scope.get("proposed_issue_title"),
        "forge_write_plan_id": redacted_plan.get("plan_id"),
        "forge_write_plan_sha256": plan_sha,
        "whole_plan_sha256_for_reference": plan_sha,
        "stable_execution_payload_sha256": stable_execution_payload_sha256,
        "frozen_payload_path": frozen_payload_path
        or redacted_plan.get("frozen_payload_path"),
        "future_approval_token_should_reference": (
            "stable_execution_payload_sha256"
            if stable_execution_payload_sha256
            else "forge_write_plan_sha256"
        ),
        "plan_frozen_for_execution": bool(stable_execution_payload_sha256),
        "warning": (
            "Plan is not frozen for selected-candidate execution; PM-11 must "
            "not use a volatile whole-plan hash for this write."
            if not_frozen
            else None
        ),
        "operation_id": operation_id,
        "operation_type": "create_issue",
        "approval_scope": redacted_scope,
        "operation_payload_preview_redacted": operation.get(
            "proposed_payload_redacted"
        ),
        "suggested_future_approval_token_fields": _suggested_future_token_fields(
            approval_scope=redacted_scope,
            forge_write_plan=redacted_plan,
            plan_sha=plan_sha,
            stable_execution_payload_sha256=stable_execution_payload_sha256,
            operation_id=operation_id,
            created_at=created,
        ),
        "recommended_decision": "request_operator_review",
        "future_write_checkpoint": "PM-11",
        "approval_token_signing": {
            "performed": False,
            "reason": (
                "PM-10 creates a review packet only. PM-11 must receive exact "
                "Operator approval before any write."
            ),
        },
        "review_instructions": [
            "Review the approval scope and proposed issue body.",
            (
                "Compare stable_execution_payload_sha256 with the frozen "
                "execution payload before PM-11."
                if stable_execution_payload_sha256
                else "Do not execute selected-candidate PM-11 from this packet "
                "until a frozen execution payload is available."
            ),
            "Confirm the exact operation_id before preparing any PM-11 token.",
            "Do not approve labels, projects, comments, workflows, runners, "
            "runtime actions, financial actions, or secret access.",
        ],
        "calls_gitea_write_api": False,
        "mutation_executed": False,
        "explicit_non_actions": dict(EXPLICIT_NON_ACTIONS),
    }
    return packet


def format_selected_candidate_approval_packet_text(
    packet: dict[str, Any],
) -> str:
    stable_hash = packet.get("stable_execution_payload_sha256")
    lines = [
        "Hermes PM selected-candidate approval packet",
        (
            f"Candidate: {packet.get('selected_candidate_id') or '<missing>'} "
            f"{packet.get('candidate_title') or ''}"
        ),
        f"Packet: {packet.get('approval_packet_id') or '<unknown>'}",
        (
            "Stable payload sha256: "
            f"{stable_hash or '<not frozen>'}"
        ),
        (
            "Plan sha256 reference: "
            f"{packet.get('whole_plan_sha256_for_reference') or '<missing>'}"
        ),
        f"Operation: {packet.get('operation_id') or '<missing>'}",
        f"Decision: {packet.get('recommended_decision')}",
        f"Future checkpoint: {packet.get('future_write_checkpoint')}",
        "Gitea writes performed: no",
        "Issue created: no",
        "Approval token active: no",
    ]
    return redact_text("\n".join(lines))
