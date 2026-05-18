#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

try:
    from scripts.hermes_pm.forge_write_plan import (
        FORGE_WRITE_PLAN_SCHEMA_VERSION,
        redact_secret_values,
        sha256_payload,
    )
    from scripts.hermes_operator.operator_policy import redact_text
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from forge_write_plan import (  # type: ignore[no-redef]
        FORGE_WRITE_PLAN_SCHEMA_VERSION,
        redact_secret_values,
        sha256_payload,
    )
    from operator_policy import redact_text  # type: ignore[no-redef]


FORGE_APPROVAL_TOKEN_SCHEMA_VERSION = "hermes.pm.forge_approval_token.v1"
FORGE_APPROVAL_VALIDATION_SCHEMA_VERSION = (
    "hermes.pm.forge_approval_validation.v1"
)

FORBIDDEN_AUTHORITY_CLASSES = {
    "deploy",
    "financial",
    "runtime_admin",
    "secret",
}
FORBIDDEN_TASK_CLASSES = {
    "deploy",
    "financial",
    "runtime_admin",
    "secret",
}
FORBIDDEN_OPERATION_TYPES = {
    "cancel_order",
    "create_check",
    "create_release",
    "create_status",
    "delete",
    "deploy",
    "financial",
    "merge_pr",
    "package_publish",
    "runtime_admin",
    "secret",
    "start_runner",
    "trigger_workflow",
    "webhook_mutation",
}
WILDCARD_SCOPE_VALUES = {
    "",
    "*",
    "**",
    "all",
    "any",
    "project",
    "projects",
    "repo",
    "repository",
}
GLOB_CHARS = set("*?[")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(token|secret|password|api[_-]?key)(\s*=\s*)[^\s,;]+"),
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
)

NON_ACTION_FLAGS = {
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
}


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_datetime(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _has_wildcard(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in WILDCARD_SCOPE_VALUES or any(
        char in value for char in GLOB_CHARS
    )


def _valid_sha256(value: str) -> bool:
    return bool(SHA256_RE.fullmatch(value))


def _normalize_base_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _operation_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in operations:
        if isinstance(item, dict) and item.get("operation_id"):
            result[str(item["operation_id"])] = item
    return result


def _operation_task_class(operation: dict[str, Any]) -> str:
    classification = operation.get("task_classification")
    if not isinstance(classification, dict):
        return ""
    return str(classification.get("task_class") or "").strip().lower()


def _operation_authority(operation: dict[str, Any]) -> str:
    return str(operation.get("authority_class") or "").strip().lower()


def _operation_type(operation: dict[str, Any]) -> str:
    return str(operation.get("operation_type") or "").strip().lower()


def _token_contains_secret_value(token: dict[str, Any]) -> bool:
    serialized = json.dumps(
        token,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return any(pattern.search(serialized) for pattern in SECRET_VALUE_PATTERNS)


def _scope_detail(
    *,
    token: dict[str, Any],
    plan: dict[str, Any],
    reasons: list[str],
) -> tuple[dict[str, Any], bool]:
    token_project = str(token.get("project_id") or "").strip()
    token_base_url = _normalize_base_url(token.get("gitea_base_url"))
    token_owner = str(token.get("owner") or "").strip()
    token_repo = str(token.get("repo") or "").strip()
    plan_project = str(plan.get("project_id") or "").strip()
    plan_base_url = _normalize_base_url(plan.get("gitea_base_url"))
    plan_owner = str(plan.get("owner") or "").strip()
    plan_repo = str(plan.get("repo") or "").strip()

    token_too_broad = False
    for label, value in (
        ("project_id", token_project),
        ("gitea_base_url", token_base_url),
        ("owner", token_owner),
        ("repo", token_repo),
    ):
        if not value:
            token_too_broad = True
            reasons.append(f"Approval token {label} is required.")
        elif _has_wildcard(value):
            token_too_broad = True
            reasons.append(f"Approval token {label} must not be a wildcard.")

    detail = {
        "project_id_matched": token_project == plan_project and bool(token_project),
        "gitea_base_url_matched": (
            token_base_url == plan_base_url and bool(token_base_url)
        ),
        "owner_matched": token_owner == plan_owner and bool(token_owner),
        "repo_matched": token_repo == plan_repo and bool(token_repo),
        "project_id": token_project,
        "gitea_base_url": redact_text(token_base_url),
        "owner": token_owner,
        "repo": token_repo,
    }
    if not detail["project_id_matched"]:
        reasons.append("Approval token project_id must match the plan exactly.")
    if not detail["gitea_base_url_matched"]:
        reasons.append("Approval token gitea_base_url must match the plan exactly.")
    if not detail["owner_matched"]:
        reasons.append("Approval token owner must match the plan exactly.")
    if not detail["repo_matched"]:
        reasons.append("Approval token repo must match the plan exactly.")
    return detail, token_too_broad


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _reject_broad_list(
    *,
    values: list[str],
    label: str,
    reasons: list[str],
) -> bool:
    broad = False
    for value in values:
        if _has_wildcard(value):
            broad = True
            reasons.append(f"Approval token {label} must not contain wildcards.")
    return broad


def _blocked_plan_operation_ids(plan: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for operation in _operation_map(plan).values():
        if operation.get("blocked") and operation.get("operation_id"):
            ids.append(str(operation["operation_id"]))
    return sorted(ids)


def _validation_result(
    *,
    token: dict[str, Any],
    plan: dict[str, Any],
    valid: bool,
    plan_sha256: str,
    plan_hash_required: bool,
    plan_sha256_matched: bool,
    stable_execution_payload_sha256: str | None,
    approval_stable_execution_payload_sha256: str,
    stable_execution_payload_sha256_matched: bool,
    stable_execution_payload_sha256_required: bool,
    operation_ids_matched: bool,
    scope_matched: bool,
    token_expired: bool,
    token_too_broad: bool,
    approved_operation_ids: list[str],
    rejected_operation_ids: list[str],
    blocked_operation_ids: list[str],
    reasons: list[str],
    warnings: list[str],
    scope_matches: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": FORGE_APPROVAL_VALIDATION_SCHEMA_VERSION,
        "valid": bool(valid),
        "approval_id": str(token.get("approval_id") or ""),
        "plan_sha256": plan_sha256,
        "plan_hash_required": bool(plan_hash_required),
        "approval_forge_write_plan_sha256": str(
            token.get("forge_write_plan_sha256") or ""
        ).strip(),
        "plan_sha256_matched": bool(plan_sha256_matched),
        "stable_execution_payload_sha256": stable_execution_payload_sha256,
        "approval_stable_execution_payload_sha256": (
            approval_stable_execution_payload_sha256
        ),
        "stable_execution_payload_sha256_matched": bool(
            stable_execution_payload_sha256_matched
        ),
        "stable_execution_payload_sha256_required": bool(
            stable_execution_payload_sha256_required
        ),
        "approval_reference_type": (
            "stable_execution_payload_sha256"
            if stable_execution_payload_sha256_required
            else "forge_write_plan_sha256"
        ),
        "operation_ids_matched": bool(operation_ids_matched),
        "scope_matched": bool(scope_matched),
        "scope_matches": scope_matches,
        "token_expired": bool(token_expired),
        "token_too_broad": bool(token_too_broad),
        "approved_operation_ids": approved_operation_ids,
        "rejected_operation_ids": sorted(_dedupe(rejected_operation_ids)),
        "blocked_operation_ids": sorted(_dedupe(blocked_operation_ids)),
        "unapproved_operation_ids": [
            operation_id
            for operation_id in sorted(_operation_map(plan))
            if operation_id not in approved_operation_ids
        ],
        "reasons": _dedupe([redact_text(reason) for reason in reasons]),
        "warnings": _dedupe([redact_text(warning) for warning in warnings]),
        **NON_ACTION_FLAGS,
    }
    return redact_secret_values(payload)


def validate_forge_approval_token(
    *,
    forge_write_plan: dict[str, Any],
    approval_token: dict[str, Any],
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    plan = (
        forge_write_plan if isinstance(forge_write_plan, dict) else {}
    )
    token = approval_token if isinstance(approval_token, dict) else {}
    redacted_plan = redact_secret_values(plan)
    plan_sha256 = sha256_payload(redacted_plan)
    stable_execution_payload_sha256 = redacted_plan.get(
        "stable_execution_payload_sha256"
    )
    if not isinstance(stable_execution_payload_sha256, str):
        stable_execution_payload_sha256 = None
    stable_execution_payload_sha256_required = bool(
        stable_execution_payload_sha256
        and (
            redacted_plan.get("requires_stable_execution_payload_sha256") is True
            or redacted_plan.get("approval_hash_kind")
            == "stable_execution_payload_sha256"
        )
    )
    plan_hash_required = not stable_execution_payload_sha256_required
    operations = _operation_map(redacted_plan)

    reasons: list[str] = []
    warnings: list[str] = []
    rejected_operation_ids: list[str] = []
    blocked_operation_ids = _blocked_plan_operation_ids(redacted_plan)
    token_too_broad = False

    if redacted_plan.get("schema_version") != FORGE_WRITE_PLAN_SCHEMA_VERSION:
        reasons.append("Forge write plan schema_version is missing or unsupported.")
    if token.get("schema_version") != FORGE_APPROVAL_TOKEN_SCHEMA_VERSION:
        reasons.append("Approval token schema_version is missing or unsupported.")
    if not str(token.get("approval_id") or "").strip():
        reasons.append("Approval token approval_id is required.")
    if not str(token.get("operator") or "").strip():
        reasons.append("Approval token operator is required.")
    if not str(token.get("reason") or "").strip():
        reasons.append("Approval token reason is required.")
    if _token_contains_secret_value(token):
        reasons.append("Approval token appears to contain a secret-like value.")

    constraints = token.get("constraints")
    if isinstance(constraints, dict) and (
        constraints.get("example_only") or constraints.get("not_active_approval")
    ):
        reasons.append("Example-only approval tokens are not active approval.")

    current_time = now or _utc_now()
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=dt.timezone.utc)
    current_time = current_time.astimezone(dt.timezone.utc)
    expires_at = _parse_datetime(token.get("expires_at"))
    token_expired = expires_at is None or expires_at <= current_time
    if expires_at is None:
        reasons.append("Approval token expires_at is missing or invalid.")
    elif token_expired:
        reasons.append("Approval token is expired.")

    approval_plan_sha = str(token.get("forge_write_plan_sha256") or "").strip()
    if plan_hash_required and not approval_plan_sha:
        reasons.append("Approval token forge_write_plan_sha256 is required.")
    elif approval_plan_sha and not _valid_sha256(approval_plan_sha):
        reasons.append(
            "Approval token forge_write_plan_sha256 must be a lowercase SHA-256."
        )
    plan_sha256_matched = approval_plan_sha == plan_sha256 and bool(approval_plan_sha)
    if plan_hash_required and not plan_sha256_matched:
        reasons.append("Approval token plan hash must match the plan exactly.")
    elif (
        stable_execution_payload_sha256_required
        and approval_plan_sha
        and not plan_sha256_matched
    ):
        warnings.append(
            "Approval token whole-plan SHA-256 did not match; selected-candidate "
            "execution uses stable_execution_payload_sha256 instead."
        )

    approval_stable_sha = str(
        token.get("stable_execution_payload_sha256") or ""
    ).strip()
    if stable_execution_payload_sha256_required and not approval_stable_sha:
        reasons.append(
            "Approval token stable_execution_payload_sha256 is required for "
            "selected-candidate execution payload plans."
        )
    elif approval_stable_sha and not _valid_sha256(approval_stable_sha):
        reasons.append(
            "Approval token stable_execution_payload_sha256 must be a lowercase "
            "SHA-256."
        )
    stable_execution_payload_sha256_matched = (
        bool(stable_execution_payload_sha256)
        and approval_stable_sha == stable_execution_payload_sha256
    )
    if (
        stable_execution_payload_sha256_required
        and not stable_execution_payload_sha256_matched
    ):
        reasons.append(
            "Approval token stable execution payload hash must match exactly."
        )

    scope_matches, scope_too_broad = _scope_detail(
        token=token,
        plan=redacted_plan,
        reasons=reasons,
    )
    token_too_broad = token_too_broad or scope_too_broad
    scope_matched = all(
        bool(scope_matches.get(key))
        for key in (
            "project_id_matched",
            "gitea_base_url_matched",
            "owner_matched",
            "repo_matched",
        )
    )

    approved_ids = _as_string_list(token.get("approved_operation_ids"))
    approved_operation_ids = _dedupe(approved_ids)
    if not approved_ids:
        reasons.append("Approval token approved_operation_ids must be explicit.")
    if len(approved_ids) != len(approved_operation_ids):
        reasons.append("Approval token approved_operation_ids must not repeat IDs.")
    if _reject_broad_list(
        values=approved_operation_ids,
        label="approved_operation_ids",
        reasons=reasons,
    ):
        token_too_broad = True
        rejected_operation_ids.extend(approved_operation_ids)

    allowed_operation_types = _as_string_list(token.get("allowed_operation_types"))
    if not allowed_operation_types:
        token_too_broad = True
        reasons.append("Approval token allowed_operation_types must be explicit.")
    if _reject_broad_list(
        values=allowed_operation_types,
        label="allowed_operation_types",
        reasons=reasons,
    ):
        token_too_broad = True
    allowed_type_set = {item.lower() for item in allowed_operation_types}

    forbidden_operation_types = {
        item.lower()
        for item in _as_string_list(token.get("forbidden_operation_types"))
    }
    if not forbidden_operation_types:
        warnings.append(
            "Approval token did not list forbidden_operation_types; "
            "checkpoint defaults are still enforced."
        )
    if _reject_broad_list(
        values=sorted(forbidden_operation_types),
        label="forbidden_operation_types",
        reasons=reasons,
    ):
        token_too_broad = True
    effective_forbidden_operation_types = (
        forbidden_operation_types | FORBIDDEN_OPERATION_TYPES
    )

    forbidden_authority_classes = {
        item.lower()
        for item in _as_string_list(token.get("forbidden_authority_classes"))
    }
    missing_default_forbidden = sorted(
        FORBIDDEN_AUTHORITY_CLASSES - forbidden_authority_classes
    )
    if missing_default_forbidden:
        reasons.append(
            "Approval token forbidden_authority_classes must include "
            + ", ".join(missing_default_forbidden)
            + "."
        )
    if _reject_broad_list(
        values=sorted(forbidden_authority_classes),
        label="forbidden_authority_classes",
        reasons=reasons,
    ):
        token_too_broad = True
    effective_forbidden_authorities = (
        forbidden_authority_classes | FORBIDDEN_AUTHORITY_CLASSES
    )

    max_operations = token.get("max_operations")
    if not isinstance(max_operations, int) or isinstance(max_operations, bool):
        token_too_broad = True
        reasons.append("Approval token max_operations must be an integer.")
    elif max_operations != len(approved_operation_ids):
        token_too_broad = True
        reasons.append(
            "Approval token max_operations must equal the selected operation count."
        )

    if token.get("single_use_intent") is not True:
        reasons.append("Approval token single_use_intent must be true.")

    if not operations:
        reasons.append("Forge write plan contains no operations to approve.")

    operation_ids_matched = bool(approved_operation_ids)
    for operation_id in approved_operation_ids:
        operation = operations.get(operation_id)
        if operation is None:
            operation_ids_matched = False
            rejected_operation_ids.append(operation_id)
            reasons.append(
                f"Approval token operation_id {operation_id!r} is not in the plan."
            )
            continue
        operation_type = _operation_type(operation)
        operation_authority = _operation_authority(operation)
        task_class = _operation_task_class(operation)
        if operation_type not in allowed_type_set:
            operation_ids_matched = False
            rejected_operation_ids.append(operation_id)
            reasons.append(
                f"Operation {operation_id} type {operation_type!r} is not allowed."
            )
        if operation_type in effective_forbidden_operation_types:
            operation_ids_matched = False
            rejected_operation_ids.append(operation_id)
            blocked_operation_ids.append(operation_id)
            reasons.append(
                f"Operation {operation_id} type {operation_type!r} is forbidden."
            )
        if operation_authority in effective_forbidden_authorities:
            operation_ids_matched = False
            rejected_operation_ids.append(operation_id)
            blocked_operation_ids.append(operation_id)
            reasons.append(
                f"Operation {operation_id} authority {operation_authority!r} "
                "is forbidden."
            )
        if task_class in FORBIDDEN_TASK_CLASSES:
            operation_ids_matched = False
            rejected_operation_ids.append(operation_id)
            blocked_operation_ids.append(operation_id)
            reasons.append(
                f"Operation {operation_id} task class {task_class!r} is forbidden."
            )
        if operation.get("blocked"):
            operation_ids_matched = False
            rejected_operation_ids.append(operation_id)
            blocked_operation_ids.append(operation_id)
            reasons.append(f"Operation {operation_id} is blocked in the plan.")

    valid = (
        not reasons
        and (
            stable_execution_payload_sha256_matched
            if stable_execution_payload_sha256_required
            else plan_sha256_matched
        )
        and operation_ids_matched
        and scope_matched
        and not token_expired
        and not token_too_broad
    )
    if valid:
        warnings.append(
            "Validation is non-mutating; future execution still requires a "
            "separate explicit checkpoint approval."
        )

    return _validation_result(
        token=token,
        plan=redacted_plan,
        valid=valid,
        plan_sha256=plan_sha256,
        plan_hash_required=plan_hash_required,
        plan_sha256_matched=plan_sha256_matched,
        stable_execution_payload_sha256=stable_execution_payload_sha256,
        approval_stable_execution_payload_sha256=approval_stable_sha,
        stable_execution_payload_sha256_matched=(
            stable_execution_payload_sha256_matched
        ),
        stable_execution_payload_sha256_required=(
            stable_execution_payload_sha256_required
        ),
        operation_ids_matched=operation_ids_matched,
        scope_matched=scope_matched,
        token_expired=token_expired,
        token_too_broad=token_too_broad,
        approved_operation_ids=approved_operation_ids,
        rejected_operation_ids=rejected_operation_ids,
        blocked_operation_ids=blocked_operation_ids,
        reasons=reasons,
        warnings=warnings,
        scope_matches=scope_matches,
    )


def format_forge_approval_validation_text(validation: dict[str, Any]) -> str:
    lines = [
        "Hermes PM forge approval validation",
        f"Approval: {validation.get('approval_id') or '<missing>'}",
        f"Valid: {'yes' if validation.get('valid') else 'no'}",
        (
            "Plan hash matched: "
            + ("yes" if validation.get("plan_sha256_matched") else "no")
        ),
        (
            "Stable payload hash matched: "
            + (
                "yes"
                if validation.get("stable_execution_payload_sha256_matched")
                else "no"
            )
        ),
        (
            "Operation IDs matched: "
            + ("yes" if validation.get("operation_ids_matched") else "no")
        ),
        f"Scope matched: {'yes' if validation.get('scope_matched') else 'no'}",
        f"Expired: {'yes' if validation.get('token_expired') else 'no'}",
        f"Too broad: {'yes' if validation.get('token_too_broad') else 'no'}",
        f"Approved IDs: {len(validation.get('approved_operation_ids') or [])}",
        f"Rejected IDs: {len(validation.get('rejected_operation_ids') or [])}",
        f"Blocked IDs: {len(validation.get('blocked_operation_ids') or [])}",
        "Executable by this tool: no",
        "Gitea writes performed: no",
    ]
    reasons = validation.get("reasons") or []
    if reasons:
        lines.append(f"Reasons: {len(reasons)}")
        for reason in reasons[:3]:
            lines.append(f"- {redact_text(str(reason))}")
    return redact_text("\n".join(lines))
