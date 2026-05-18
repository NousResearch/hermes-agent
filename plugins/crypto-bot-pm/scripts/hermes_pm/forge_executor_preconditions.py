#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from typing import Any

try:
    from scripts.hermes_pm.forge_write_plan import (
        redact_secret_values,
        sha256_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from forge_write_plan import (  # type: ignore[no-redef]
        redact_secret_values,
        sha256_payload,
    )


FORGE_EXECUTOR_PRECONDITIONS_SCHEMA_VERSION = (
    "hermes.pm.forge_executor_preconditions.v1"
)

NON_ACTION_BOOLEANS = {
    "calls_gitea_write_api": False,
    "mutation_executed": False,
    "creates_issues": False,
    "creates_labels": False,
    "mutates_projects": False,
    "comments": False,
    "starts_workflows": False,
    "starts_runners": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
}

FORBIDDEN_OPERATION_TYPES = [
    "create_label",
    "create_project_column",
    "create_project_card",
    "comment_on_issue",
    "update_issue",
    "request_pr_review",
    "create_status",
    "create_check",
    "create_release",
    "merge_pr",
    "package_publish",
    "webhook_mutation",
    "trigger_workflow",
    "start_runner",
    "deploy",
    "runtime_admin",
    "financial",
    "secret",
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _operations(plan: dict[str, Any]) -> list[dict[str, Any]]:
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return []
    return [item for item in operations if isinstance(item, dict)]


def _operation_ids(operations: list[dict[str, Any]]) -> list[str]:
    return [
        str(operation.get("operation_id"))
        for operation in operations
        if operation.get("operation_id")
    ]


def _first_issue_operation(operations: list[dict[str, Any]]) -> dict[str, Any] | None:
    for operation in operations:
        if (
            operation.get("operation_type") == "create_issue"
            and not operation.get("blocked")
            and operation.get("operation_id")
        ):
            return operation
    return None


def _operation_type_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for operation in operations:
        operation_type = str(operation.get("operation_type") or "unknown")
        counts[operation_type] = counts.get(operation_type, 0) + 1
    return counts


def _capability_summary(capability_map: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(capability_map, dict):
        return {
            "present": False,
            "schema_version": None,
            "endpoint_ready_operation_types": [],
            "blocked_or_unknown_operation_types": [],
            "project_endpoint_status": {},
        }
    return {
        "present": True,
        "schema_version": capability_map.get("schema_version"),
        "endpoint_ready_operation_types": (
            capability_map.get("endpoint_ready_operation_types") or []
        ),
        "blocked_or_unknown_operation_types": (
            capability_map.get("blocked_or_unknown_operation_types") or []
        ),
        "project_endpoint_status": capability_map.get("project_endpoint_status")
        or {},
    }


def build_forge_executor_preconditions(
    *,
    forge_write_plan: dict[str, Any] | None = None,
    approval_token: dict[str, Any] | None = None,
    capability_map: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    plan = redact_secret_values(forge_write_plan or {})
    token = redact_secret_values(approval_token or {})
    redacted_capability_map = (
        redact_secret_values(capability_map)
        if isinstance(capability_map, dict)
        else None
    )
    operations = _operations(plan)
    first_issue = _first_issue_operation(operations)
    first_issue_ids = (
        [str(first_issue["operation_id"])]
        if isinstance(first_issue, dict) and first_issue.get("operation_id")
        else []
    )
    plan_sha = sha256_payload(plan) if plan else None
    recommended_subset = {
        "max_operations": 1,
        "operation_ids": first_issue_ids,
        "operation_types": ["create_issue"] if first_issue_ids else [],
        "requires_exact_operation_id": True,
        "requires_exact_plan_hash": True,
        "excludes_operation_types": [
            "create_label",
            "create_project_column",
            "create_project_card",
            "comment_on_issue",
            "update_issue",
            "request_pr_review",
        ],
        "reason": (
            "The first real forge write should prove one issue creation only, "
            "without labels, project cards, comments, workflows, runners, "
            "deploys, runtime actions, trading, or secrets."
        ),
    }
    preconditions = {
        "schema_version": FORGE_EXECUTOR_PRECONDITIONS_SCHEMA_VERSION,
        "created_at": created_at or _utc_now(),
        "read_only": True,
        "this_is_not_an_executor": True,
        "project_id": plan.get("project_id") or "crypto_bot",
        "plan_id": plan.get("plan_id"),
        "plan_sha256": plan_sha,
        "operation_count": len(operations),
        "operation_type_counts": _operation_type_counts(operations),
        "required_valid_plan_hash": {
            "required": True,
            "plan_sha256": plan_sha,
            "exact_match_required": True,
            "wildcards_allowed": False,
            "satisfied": bool(plan_sha),
        },
        "required_approval_token": {
            "required": True,
            "present": bool(token),
            "token_value_exposed": False,
            "must_be_non_expired": True,
            "must_match_plan_sha256": True,
            "must_name_exact_operation_ids": True,
            "must_forbid_wildcards": True,
            "must_be_single_use_intent": True,
        },
        "required_operation_ids": {
            "required": True,
            "all_plan_operation_ids": _operation_ids(operations),
            "recommended_first_write_operation_ids": first_issue_ids,
            "exact_operation_ids_required": True,
            "max_recommended_first_write_operations": 1,
        },
        "required_capability_map": {
            "required": True,
            "present": isinstance(redacted_capability_map, dict),
            "required_schema_version": "hermes.pm.gitea_forge_capability_map.v1",
            "schema_version": (
                redacted_capability_map.get("schema_version")
                if isinstance(redacted_capability_map, dict)
                else None
            ),
            "issue_read_endpoint_required": True,
            "project_endpoints_do_not_block_issue_only_rehearsal": True,
            "permission_proof_still_required_by_future_executor": True,
        },
        "required_auth_scope": {
            "required": True,
            "must_be_scoped_to_base_url_owner_repo": True,
            "must_allow_only_selected_operation_ids": True,
            "must_not_expose_token_value": True,
            "must_not_use_wildcards": True,
            "write_permission_unproven_until_future_checkpoint": True,
        },
        "required_artifact_store_or_audit_log": {
            "required": True,
            "must_record_pre_write_plan_hash": True,
            "must_record_operator_confirmation": True,
            "must_record_post_write_attestation": True,
            (
                "must_record_no_workflow_runner_deploy_runtime_trading_"
                "secret_activity"
            ): True,
        },
        "allowed_future_operation_types": ["create_issue"],
        "forbidden_operation_types": FORBIDDEN_OPERATION_TYPES,
        "rollback_or_reversal_notes": [
            "Issue creation cannot be atomically rolled back by this model.",
            "The future executor must record the created issue URL/index.",
            (
                "Manual reversal, if needed, is close the issue with an "
                "Operator-approved note."
            ),
            (
                "No labels, project cards, or comments should be created in "
                "the first write."
            ),
        ],
        "operator_confirmation_required": True,
        "future_executor_must_stop_if": [
            "plan hash differs from the approved plan hash",
            "approval token is missing, expired, wildcarded, or scope-mismatched",
            "operation id differs from the one approved create_issue operation",
            "more than one operation is requested",
            "operation type is not create_issue",
            "capability map is missing or issue read endpoint is not ready",
            "Gitea auth scope is absent or broader than the exact repo operation",
            (
                "any label, project card, comment, PR, status, release, "
                "package, webhook, workflow, runner, deploy, runtime, "
                "trading, financial, secret, or branch-writer action is "
                "requested"
            ),
            "post-write attestation path or audit log is unavailable",
        ],
        "capability_summary": _capability_summary(redacted_capability_map),
        "recommended_first_write_subset": recommended_subset,
        "next_write_checkpoint_recommendation": {
            "checkpoint": "PM-6: one approved Gitea issue creation rehearsal",
            "allow_only_one_create_issue_operation": True,
            "no_labels": True,
            "no_project_cards": True,
            "no_comments": True,
            "exact_plan_hash_required": True,
            "exact_operation_id_required": True,
            "explicit_token_required": True,
            "endpoint_capability_map_required": True,
            "post_write_attestation_required": True,
            "no_workflow_runner_deploy_runtime_trading_secret_activity": True,
        },
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }
    return preconditions


def format_forge_executor_preconditions_text(preconditions: dict[str, Any]) -> str:
    subset = preconditions.get("recommended_first_write_subset")
    if not isinstance(subset, dict):
        subset = {}
    operation_ids = ", ".join(subset.get("operation_ids") or ["<none>"])
    lines = [
        "Hermes PM forge executor preconditions",
        f"Project: {preconditions.get('project_id') or '<unknown>'}",
        f"Plan sha256: {preconditions.get('plan_sha256') or '<required>'}",
        f"Operations: {preconditions.get('operation_count', 0)}",
        f"Recommended first write ids: {operation_ids}",
        "Allowed future operation types: create_issue",
        "Operator confirmation required: yes",
        "This is an executor: no",
        "Gitea writes performed: no",
    ]
    return "\n".join(lines)
