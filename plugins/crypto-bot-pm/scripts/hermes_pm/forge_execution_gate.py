#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from typing import Any

try:
    from scripts.hermes_pm.forge_approval_token import (
        FORBIDDEN_AUTHORITY_CLASSES,
        FORBIDDEN_OPERATION_TYPES,
        FORBIDDEN_TASK_CLASSES,
        validate_forge_approval_token,
    )
    from scripts.hermes_pm.forge_write_plan import (
        redact_secret_values,
        sha256_payload,
    )
    from scripts.hermes_operator.operator_policy import redact_text
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from forge_approval_token import (  # type: ignore[no-redef]
        FORBIDDEN_AUTHORITY_CLASSES,
        FORBIDDEN_OPERATION_TYPES,
        FORBIDDEN_TASK_CLASSES,
        validate_forge_approval_token,
    )
    from forge_write_plan import (  # type: ignore[no-redef]
        redact_secret_values,
        sha256_payload,
    )
    from operator_policy import redact_text  # type: ignore[no-redef]


FORGE_EXECUTION_GATE_SCHEMA_VERSION = "hermes.pm.forge_execution_gate.v1"
GITEA_FORGE_CAPABILITY_MAP_SCHEMA_VERSION = (
    "hermes.pm.gitea_forge_capability_map.v1"
)

SUPPORTED_CAPABILITY_KEYS = {
    "create_issue": "issues_write_preview_supported",
    "create_label": "labels_write_preview_supported",
    "update_issue": "issues_write_preview_supported",
    "comment_on_issue": "comments_write_preview_supported",
    "create_project_column": "projects_write_preview_supported",
    "create_project_card": "projects_write_preview_supported",
}

PROJECT_OPERATION_TYPES = {"create_project_column", "create_project_card"}
READY_CAPABILITY_CLASSIFICATIONS = {
    "endpoint_ready",
    "read_endpoint_ready_write_unproven",
}

NON_ACTION_BOOLEANS = {
    "action_executable_by_this_tool": False,
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
    "branch_writer_invoked": False,
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _operations_by_id(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return {}
    return {
        str(operation.get("operation_id")): operation
        for operation in operations
        if isinstance(operation, dict) and operation.get("operation_id")
    }


def _task_class(operation: dict[str, Any]) -> str:
    classification = operation.get("task_classification")
    if not isinstance(classification, dict):
        return ""
    return str(classification.get("task_class") or "").strip().lower()


def _operation_type(operation: dict[str, Any]) -> str:
    return str(operation.get("operation_type") or "").strip().lower()


def _authority(operation: dict[str, Any]) -> str:
    return str(operation.get("authority_class") or "").strip().lower()


def _policy_decision(operation: dict[str, Any]) -> str:
    policy = operation.get("policy_result")
    if not isinstance(policy, dict):
        return ""
    return str(policy.get("decision") or "")


def _endpoint_is_well_formed(operation: dict[str, Any], plan: dict[str, Any]) -> bool:
    method = str(operation.get("http_method_that_would_be_used") or "").upper()
    endpoint = str(operation.get("expected_gitea_endpoint") or "")
    owner = str(plan.get("owner") or "")
    repo = str(plan.get("repo") or "")
    prefix = f"/api/v1/repos/{owner}/{repo}/"
    return method == "POST" and endpoint.startswith(prefix)


def _operation_capability_from_map(
    capabilities: dict[str, Any],
    operation_type: str,
) -> dict[str, Any]:
    operation_capabilities = capabilities.get("operation_capabilities")
    if not isinstance(operation_capabilities, dict):
        return {
            "state": "unknown",
            "classification": "endpoint_unknown",
            "reason": "Capability map does not include operation_capabilities.",
            "endpoint": None,
            "status_code": None,
            "evidence_strength": "not_sampled",
            "blockers": [],
            "warnings": [],
        }
    detail = operation_capabilities.get(operation_type)
    if not isinstance(detail, dict):
        return {
            "state": "unknown",
            "classification": "endpoint_unknown",
            "reason": f"No capability map entry for {operation_type}.",
            "endpoint": None,
            "status_code": None,
            "evidence_strength": "not_sampled",
            "blockers": [],
            "warnings": [],
        }
    classification = str(detail.get("classification") or "endpoint_unknown")
    blockers = [
        str(item)
        for item in detail.get("blockers") or []
        if str(item).strip()
    ]
    warnings = [
        str(item)
        for item in detail.get("warnings") or []
        if str(item).strip()
    ]
    endpoint = detail.get("endpoint")
    status_code = detail.get("status_code")
    evidence_strength = str(detail.get("evidence_strength") or "not_sampled")
    if classification in READY_CAPABILITY_CLASSIFICATIONS and not blockers:
        reason = (
            f"{operation_type} has read endpoint evidence "
            f"({classification}) at {endpoint or '<unknown endpoint>'}."
        )
        state = "supported"
    else:
        reason = (
            "; ".join(blockers)
            if blockers
            else f"{operation_type} capability is {classification}."
        )
        state = classification
    return {
        "state": state,
        "classification": classification,
        "reason": reason,
        "endpoint": endpoint,
        "method": detail.get("method"),
        "status_code": status_code,
        "evidence_strength": evidence_strength,
        "blockers": blockers,
        "warnings": warnings,
    }


def _operation_capability_detail(
    capabilities: dict[str, Any] | None,
    operation_type: str,
) -> dict[str, Any]:
    key = SUPPORTED_CAPABILITY_KEYS.get(operation_type)
    if key is None:
        return {
            "state": "unsupported",
            "classification": "unsupported",
            "reason": "Endpoint type is not supported by this gate.",
            "endpoint": None,
            "status_code": None,
            "evidence_strength": "unavailable",
            "blockers": ["Endpoint type is not supported by this gate."],
            "warnings": [],
        }
    if capabilities is None:
        return {
            "state": "unknown",
            "classification": "endpoint_unknown",
            "reason": "No Gitea forge capability report was supplied.",
            "endpoint": None,
            "status_code": None,
            "evidence_strength": "not_sampled",
            "blockers": ["No Gitea forge capability report was supplied."],
            "warnings": [],
        }
    if (
        capabilities.get("schema_version")
        == GITEA_FORGE_CAPABILITY_MAP_SCHEMA_VERSION
        or isinstance(capabilities.get("operation_capabilities"), dict)
    ):
        return _operation_capability_from_map(capabilities, operation_type)
    value = capabilities.get(key)
    if value is True:
        return {
            "state": "supported",
            "classification": "read_endpoint_ready_write_unproven",
            "reason": f"{key} is true.",
            "endpoint": None,
            "status_code": None,
            "evidence_strength": "direct_read",
            "blockers": [],
            "warnings": ["Write permission is not proven by read evidence."],
        }
    if value is False:
        return {
            "state": "unsupported",
            "classification": "endpoint_not_found",
            "reason": f"{key} is false.",
            "endpoint": None,
            "status_code": None,
            "evidence_strength": "unavailable",
            "blockers": [f"{key} is false."],
            "warnings": [],
        }
    return {
        "state": "unknown",
        "classification": "endpoint_unknown",
        "reason": f"{key} is unknown.",
        "endpoint": None,
        "status_code": None,
        "evidence_strength": "not_sampled",
        "blockers": [f"{key} is unknown."],
        "warnings": [],
    }


def _operation_blockers(
    *,
    operation: dict[str, Any],
    plan: dict[str, Any],
    capabilities: dict[str, Any] | None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    blockers: list[str] = []
    endpoint_blockers: list[dict[str, Any]] = []
    operation_id = str(operation.get("operation_id") or "")
    operation_type = _operation_type(operation)
    task_class = _task_class(operation)
    authority = _authority(operation)

    if operation.get("blocked"):
        blockers.append("Operation is blocked in the forge-write plan.")
    if task_class in FORBIDDEN_TASK_CLASSES:
        blockers.append(f"Task class {task_class!r} is forbidden.")
    if authority in FORBIDDEN_AUTHORITY_CLASSES:
        blockers.append(f"Authority class {authority!r} is forbidden.")
    if operation_type in FORBIDDEN_OPERATION_TYPES:
        blockers.append(f"Operation type {operation_type!r} is forbidden.")
    if _policy_decision(operation) == "DENY":
        blockers.append("Operator policy denied this operation preview.")
    if not _endpoint_is_well_formed(operation, plan):
        blockers.append("Endpoint preview is missing or not well-formed.")

    capability_detail = _operation_capability_detail(
        capabilities,
        operation_type,
    )
    capability_state = str(capability_detail.get("state") or "unknown")
    capability_reason = str(capability_detail.get("reason") or "")
    if capability_state != "supported":
        endpoint_blockers.append(
            {
                "operation_id": operation_id,
                "operation_type": operation_type,
                "capability_state": capability_state,
                "classification": capability_detail.get("classification"),
                "endpoint": capability_detail.get("endpoint"),
                "status_code": capability_detail.get("status_code"),
                "evidence_strength": capability_detail.get("evidence_strength"),
                "reason": capability_reason,
            }
        )
        blockers.append(capability_reason)

    return list(dict.fromkeys(blockers)), endpoint_blockers, capability_detail


def _operation_summary(
    operation: dict[str, Any],
    *,
    future_executable: bool,
    reasons: list[str] | None = None,
    endpoint_capability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    endpoint_reason = (
        str(endpoint_capability.get("reason"))
        if isinstance(endpoint_capability, dict)
        and endpoint_capability.get("reason")
        else ""
    )
    return {
        "operation_id": operation.get("operation_id"),
        "operation_type": operation.get("operation_type"),
        "title": operation.get("title"),
        "authority_class": operation.get("authority_class"),
        "future_executable": bool(future_executable),
        "executed": False,
        "requires_separate_future_checkpoint_approval": True,
        "endpoint_capability_reason": endpoint_reason,
        "endpoint_capability": endpoint_capability or {},
        "reasons": reasons or [],
    }


def _evidence_requirements(
    *,
    has_capabilities: bool,
    endpoint_blockers: list[dict[str, Any]],
    requires_stable_payload_hash: bool = False,
) -> list[str]:
    requirements = [
        (
            "Exact stable execution payload SHA-256 in the approval token."
            if requires_stable_payload_hash
            else "Exact forge-write plan SHA-256 in the approval token."
        ),
        "Explicit approved_operation_ids in the approval token.",
        "Operator review of each redacted payload preview and endpoint preview.",
        "Proof that selected operations are not blocked by PM classification "
        "or operator policy.",
        "Rollback or manual correction note for each future forge mutation.",
        "Separate future checkpoint approval naming the exact plan hash and "
        "selected operation IDs.",
        "Proof that no workflow, runner, deploy, runtime, financial, secret, "
        "or branch-writer action is included.",
    ]
    if not has_capabilities:
        requirements.append(
            "Read-only Gitea forge capability report generated with GET/HEAD only."
        )
    if endpoint_blockers:
        requirements.append(
            "Endpoint capability blockers must be resolved or explicitly "
            "excluded before any future write."
        )
    if any(
        item.get("operation_type") in PROJECT_OPERATION_TYPES
        for item in endpoint_blockers
    ):
        requirements.append(
            "Project/card endpoint shape must be confirmed before project or "
            "card writes are eligible."
        )
    return requirements


def _next_action(
    *,
    validation: dict[str, Any],
    ready: bool,
    endpoint_blockers: list[dict[str, Any]],
    approved_ids: list[str],
) -> str:
    if not validation.get("valid"):
        return (
            "Provide a non-expired exact-scope approval token that matches the "
            "plan SHA-256 and selected operation IDs."
        )
    if endpoint_blockers:
        return (
            "Resolve endpoint capability blockers or approve a revised future "
            "checkpoint excluding unsupported operations."
        )
    if ready:
        return (
            "Request a separate future forge-write checkpoint approval naming "
            f"plan {validation.get('plan_sha256')} and operation IDs "
            f"{', '.join(approved_ids)}."
        )
    return "Revise the plan or approval scope before any future forge write."


def build_forge_execution_gate(
    *,
    forge_write_plan: dict[str, Any],
    approval_token: dict[str, Any],
    gitea_capabilities: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    plan = redact_secret_values(
        forge_write_plan if isinstance(forge_write_plan, dict) else {}
    )
    token = redact_secret_values(
        approval_token if isinstance(approval_token, dict) else {}
    )
    capabilities = (
        redact_secret_values(gitea_capabilities)
        if isinstance(gitea_capabilities, dict)
        else None
    )
    validation = validate_forge_approval_token(
        forge_write_plan=plan,
        approval_token=token,
    )
    operations = _operations_by_id(plan)
    selected_ids = [
        operation_id
        for operation_id in validation.get("approved_operation_ids") or []
        if isinstance(operation_id, str)
    ]
    token_valid = bool(validation.get("valid"))

    approved_operations: list[dict[str, Any]] = []
    rejected_operations: list[dict[str, Any]] = []
    blocked_operations: list[dict[str, Any]] = []
    endpoint_blockers: list[dict[str, Any]] = []

    for operation_id, operation in operations.items():
        if operation.get("blocked"):
            blocked_operations.append(
                _operation_summary(
                    operation,
                    future_executable=False,
                    reasons=operation.get("blockers") or ["Operation is blocked."],
                )
            )

    for operation_id in selected_ids:
        operation = operations.get(operation_id)
        if operation is None:
            rejected_operations.append(
                {
                    "operation_id": operation_id,
                    "future_executable": False,
                    "executed": False,
                    "reasons": ["Operation ID is not present in the plan."],
                }
            )
            continue
        if not token_valid:
            rejected_operations.append(
                _operation_summary(
                    operation,
                    future_executable=False,
                    reasons=validation.get("reasons")
                    or ["Approval token is not valid."],
                )
            )
            continue
        blockers, blockers_for_endpoint, capability_detail = _operation_blockers(
            operation=operation,
            plan=plan,
            capabilities=capabilities,
        )
        endpoint_blockers.extend(blockers_for_endpoint)
        if blockers:
            rejected_operations.append(
                _operation_summary(
                    operation,
                    future_executable=False,
                    reasons=blockers,
                    endpoint_capability=capability_detail,
                )
            )
            if operation_id not in {
                str(item.get("operation_id")) for item in blocked_operations
            }:
                blocked_operations.append(
                    _operation_summary(
                        operation,
                        future_executable=False,
                        reasons=blockers,
                        endpoint_capability=capability_detail,
                    )
                )
            continue
        approved_operations.append(
            _operation_summary(
                operation,
                future_executable=True,
                endpoint_capability=capability_detail,
            )
        )

    for rejected_id in validation.get("rejected_operation_ids") or []:
        if rejected_id not in selected_ids:
            rejected_operations.append(
                {
                    "operation_id": rejected_id,
                    "future_executable": False,
                    "executed": False,
                    "reasons": ["Approval validation rejected this operation ID."],
                }
            )

    selected_blocked_ids = {
        str(item.get("operation_id"))
        for item in rejected_operations + blocked_operations
        if item.get("operation_id") in selected_ids
    }
    ready = (
        bool(validation.get("valid"))
        and bool(selected_ids)
        and len(approved_operations) == len(selected_ids)
        and not selected_blocked_ids
        and not rejected_operations
    )
    plan_sha = sha256_payload(plan)
    seed = {
        "plan_sha256": plan_sha,
        "approval_id": validation.get("approval_id"),
        "approved_ids": selected_ids,
        "created_at": created_at or "",
    }
    gate_id = f"forge-gate-{sha256_payload(seed)[:16]}"
    endpoint_blockers = [
        item
        for index, item in enumerate(endpoint_blockers)
        if item not in endpoint_blockers[:index]
    ]
    result = {
        "schema_version": FORGE_EXECUTION_GATE_SCHEMA_VERSION,
        "gate_id": gate_id,
        "created_at": created_at or _utc_now(),
        "project_id": plan.get("project_id"),
        "plan_sha256": plan_sha,
        "stable_execution_payload_sha256": validation.get(
            "stable_execution_payload_sha256"
        ),
        "approval_reference_type": validation.get("approval_reference_type"),
        "approval_id": validation.get("approval_id"),
        "approval_validation": validation,
        "ready_for_future_execution": ready,
        "dry_run": True,
        "mutation_executed": False,
        "calls_gitea_write_api": False,
        "approved_operations": approved_operations,
        "rejected_operations": rejected_operations,
        "blocked_operations": blocked_operations,
        "endpoint_capability_blockers": endpoint_blockers,
        "evidence_requirements": _evidence_requirements(
            has_capabilities=capabilities is not None,
            endpoint_blockers=endpoint_blockers,
            requires_stable_payload_hash=bool(
                validation.get("stable_execution_payload_sha256_required")
            ),
        ),
        "next_required_operator_action": _next_action(
            validation=validation,
            ready=ready,
            endpoint_blockers=endpoint_blockers,
            approved_ids=selected_ids,
        ),
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }
    return redact_secret_values(result)


def format_forge_execution_gate_text(gate: dict[str, Any]) -> str:
    lines = [
        "Hermes PM forge execution gate",
        f"Gate: {gate.get('gate_id') or '<unknown>'}",
        f"Project: {gate.get('project_id') or '<unknown>'}",
        f"Approval: {gate.get('approval_id') or '<missing>'}",
        (
            "Ready for future execution: "
            + ("yes" if gate.get("ready_for_future_execution") else "no")
        ),
        "Dry run: yes",
        "Gitea writes performed: no",
        f"Approved operations: {len(gate.get('approved_operations') or [])}",
        f"Rejected operations: {len(gate.get('rejected_operations') or [])}",
        f"Blocked operations: {len(gate.get('blocked_operations') or [])}",
        (
            "Endpoint blockers: "
            f"{len(gate.get('endpoint_capability_blockers') or [])}"
        ),
        "Future checkpoint approval required: yes",
    ]
    blockers = gate.get("endpoint_capability_blockers") or []
    for blocker in blockers[:3]:
        if isinstance(blocker, dict):
            lines.append(
                "- "
                f"{blocker.get('operation_id')}: "
                f"{blocker.get('capability_state')} "
                f"{blocker.get('operation_type')}"
            )
    return redact_text("\n".join(lines))
