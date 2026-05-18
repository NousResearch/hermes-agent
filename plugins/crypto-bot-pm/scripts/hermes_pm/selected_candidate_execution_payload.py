#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )


SELECTED_CANDIDATE_EXECUTION_PAYLOAD_SCHEMA_VERSION = (
    "hermes.pm.selected_candidate_execution_payload.v1"
)
STABLE_EXECUTION_PAYLOAD_HASH_FIELD = "stable_execution_payload_sha256"

FORBIDDEN_PAYLOAD_FIELDS = (
    "labels",
    "assignees",
    "milestone",
    "due_date",
    "project",
    "comments",
    "attachments",
)

VOLATILE_PAYLOAD_FIELDS = {
    "approval_packet_id",
    "created_at",
    "generated_at",
    "live_issue_counts",
    "plan_id",
    "snapshot_timestamp",
    "source_packet_id",
    "source_packet_ids",
    "status",
    "transient_status",
    "warnings",
}

REQUIRED_CONSTRAINTS = {
    "max_operations": 1,
    "endpoint": "/api/v1/repos/preston/crypto_bot/issues",
    "no_labels": True,
    "no_projects": True,
    "no_comments": True,
    "no_prs": True,
    "no_workflows": True,
    "no_runners": True,
    "no_runtime_actions": True,
    "no_financial_actions": True,
    "no_secret_access": True,
    "no_branch_writer": True,
}

PAYLOAD_FIELDS = {
    "schema_version",
    "project_id",
    "candidate_id",
    "candidate_title",
    "gitea_base_url",
    "owner",
    "repo",
    "operation_type",
    "operation_id",
    "issue_title",
    "issue_body",
    "forbidden_payload_fields",
    "constraints",
}
ALLOWED_PAYLOAD_FIELDS = PAYLOAD_FIELDS | {STABLE_EXECUTION_PAYLOAD_HASH_FIELD}


class SelectedCandidateExecutionPayloadError(ValueError):
    """Raised when a selected-candidate execution payload is not exact."""


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SelectedCandidateExecutionPayloadError(f"{key} is required.")
    return redact_text(value.strip())


def _normalized_core_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload JSON must be an object."
        )
    keys = set(payload)
    volatile = sorted(keys & VOLATILE_PAYLOAD_FIELDS)
    if volatile:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload includes volatile field(s): "
            + ", ".join(volatile)
            + "."
        )
    unknown = sorted(keys - ALLOWED_PAYLOAD_FIELDS)
    if unknown:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload includes unsupported field(s): "
            + ", ".join(unknown)
            + "."
        )
    missing = sorted(PAYLOAD_FIELDS - keys)
    if missing:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload is missing required field(s): "
            + ", ".join(missing)
            + "."
        )
    if (
        payload.get("schema_version")
        != SELECTED_CANDIDATE_EXECUTION_PAYLOAD_SCHEMA_VERSION
    ):
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload schema_version is unsupported."
        )
    if payload.get("operation_type") != "create_issue":
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload operation_type must be create_issue."
        )

    constraints = payload.get("constraints")
    if not isinstance(constraints, dict):
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload constraints must be an object."
        )
    missing_constraints = sorted(set(REQUIRED_CONSTRAINTS) - set(constraints))
    if missing_constraints:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload constraints are missing required field(s): "
            + ", ".join(missing_constraints)
            + "."
        )
    wrong_constraints = [
        key
        for key, expected in REQUIRED_CONSTRAINTS.items()
        if constraints.get(key) != expected
    ]
    if wrong_constraints:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload constraints do not match required value(s): "
            + ", ".join(sorted(wrong_constraints))
            + "."
        )
    if sorted(constraints) != sorted(REQUIRED_CONSTRAINTS):
        extras = sorted(set(constraints) - set(REQUIRED_CONSTRAINTS))
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload constraints include unsupported field(s): "
            + ", ".join(extras)
            + "."
        )

    forbidden = payload.get("forbidden_payload_fields")
    if forbidden != list(FORBIDDEN_PAYLOAD_FIELDS):
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload forbidden_payload_fields must match the exact "
            "selected-candidate create_issue denylist."
        )

    owner = _require_text(payload, "owner")
    repo = _require_text(payload, "repo")
    endpoint = f"/api/v1/repos/{owner}/{repo}/issues"
    if constraints.get("endpoint") != endpoint:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload endpoint must match owner/repo exactly."
        )
    if owner != DEFAULT_OWNER or repo != DEFAULT_REPO:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload owner/repo must be preston/crypto_bot."
        )

    normalized = {
        "schema_version": SELECTED_CANDIDATE_EXECUTION_PAYLOAD_SCHEMA_VERSION,
        "project_id": _require_text(payload, "project_id"),
        "candidate_id": _require_text(payload, "candidate_id"),
        "candidate_title": _require_text(payload, "candidate_title"),
        "gitea_base_url": _require_text(payload, "gitea_base_url").rstrip("/"),
        "owner": owner,
        "repo": repo,
        "operation_type": "create_issue",
        "operation_id": _require_text(payload, "operation_id"),
        "issue_title": _require_text(payload, "issue_title"),
        "issue_body": _require_text(payload, "issue_body"),
        "forbidden_payload_fields": list(FORBIDDEN_PAYLOAD_FIELDS),
        "constraints": dict(REQUIRED_CONSTRAINTS),
    }
    if normalized["gitea_base_url"] != DEFAULT_GITEA_BASE_URL:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload gitea_base_url must be the approved local Gitea URL."
        )
    return normalized


def stable_execution_payload_sha256(payload: dict[str, Any]) -> str:
    core = _normalized_core_payload(payload)
    return _sha256(core)


def validate_selected_candidate_execution_payload(
    payload: dict[str, Any],
    *,
    require_hash_match: bool = True,
) -> dict[str, Any]:
    core = _normalized_core_payload(payload)
    stable_hash = _sha256(core)
    supplied_hash = payload.get(STABLE_EXECUTION_PAYLOAD_HASH_FIELD)
    if require_hash_match and supplied_hash != stable_hash:
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload stable_execution_payload_sha256 does not match "
            "the canonical payload."
        )
    if supplied_hash is not None and not isinstance(supplied_hash, str):
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload stable_execution_payload_sha256 must be a string."
        )
    return {
        **core,
        STABLE_EXECUTION_PAYLOAD_HASH_FIELD: stable_hash,
    }


def freeze_selected_candidate_execution_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    return validate_selected_candidate_execution_payload(
        {
            key: value
            for key, value in payload.items()
            if key != STABLE_EXECUTION_PAYLOAD_HASH_FIELD
        },
        require_hash_match=False,
    )


def load_selected_candidate_execution_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SelectedCandidateExecutionPayloadError(
            "Execution payload file must contain a JSON object."
        )
    return validate_selected_candidate_execution_payload(payload)


def issue_payload_from_execution_payload(payload: dict[str, Any]) -> dict[str, str]:
    frozen = validate_selected_candidate_execution_payload(payload)
    return {
        "title": frozen["issue_title"],
        "body": frozen["issue_body"],
    }


def approval_scope_from_execution_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frozen = validate_selected_candidate_execution_payload(payload)
    return {
        "schema_version": "hermes.pm.backlog_candidate_approval_scope.v1",
        "approval_scope_id": (
            "approval-scope-selected-payload-"
            f"{frozen[STABLE_EXECUTION_PAYLOAD_HASH_FIELD][:16]}"
        ),
        "project_id": frozen["project_id"],
        "selected_candidate_id": frozen["candidate_id"],
        "selected_candidate_title": frozen["candidate_title"],
        "source_selection_packet_id": None,
        "issue_1_reference": {},
        "proposed_issue_title": frozen["issue_title"],
        "proposed_issue_body": frozen["issue_body"],
        "future_operation_type": "create_issue",
        "authority_class": "forge_write",
        "exact_future_constraints": {
            "owner": frozen["owner"],
            "repo": frozen["repo"],
            "gitea_base_url": frozen["gitea_base_url"],
            "operation_type": frozen["operation_type"],
            "max_operations": frozen["constraints"]["max_operations"],
            "no_labels": frozen["constraints"]["no_labels"],
            "no_projects": frozen["constraints"]["no_projects"],
            "no_comments": frozen["constraints"]["no_comments"],
            "no_prs": frozen["constraints"]["no_prs"],
            "no_workflows": frozen["constraints"]["no_workflows"],
            "no_runners": frozen["constraints"]["no_runners"],
            "no_runtime_actions": frozen["constraints"]["no_runtime_actions"],
            "no_financial_actions": frozen["constraints"]["no_financial_actions"],
            "no_secret_access": frozen["constraints"]["no_secret_access"],
            "no_branch_writer": frozen["constraints"]["no_branch_writer"],
        },
        "blocked": False,
        "blockers": [],
        "warnings": [
            "Approval scope was reconstructed from a frozen selected-candidate "
            "execution payload.",
        ],
        "approval_required_for_write": True,
        "calls_gitea_write_api": False,
        "mutation_executed": False,
        "candidate_task_class": "propose",
        "future_create_issue_executable_now": False,
        "requires_exact_candidate_id": True,
        "requires_exact_operation_id": True,
        "requires_exact_stable_execution_payload_sha256": True,
        "requires_exact_plan_hash": False,
        "requires_exact_approval_token": True,
    }
