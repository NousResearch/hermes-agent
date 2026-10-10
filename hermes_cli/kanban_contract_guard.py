"""Fail-closed validation for deterministic fleet task contracts.

Only tasks carrying the fleet reconciler identity are governed here. Ordinary
Kanban cards retain their existing dispatch behaviour.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import Any, Optional


FLEET_CREATED_BY = "fleet-linear-reconciler"
FLEET_IDEMPOTENCY_PREFIX = "linear:"
CONTRACT_BEGIN = "<!-- FLEET_CONTRACT_SNAPSHOT_V1 BEGIN -->"
CONTRACT_END = "<!-- FLEET_CONTRACT_SNAPSHOT_V1 END -->"
CONTRACT_SCHEMA_VERSION = 1
VALID_EXECUTION_MODES = {"plan_only", "implement"}

CONTRACT_HASH_FIELDS = (
    "schema_version",
    "issue_id",
    "linear_id",
    "title",
    "description",
    "project_slug",
    "repository",
    "labels",
    "relations",
    "agent_ready",
    "blocked",
    "redactions",
    "execution_mode",
)

CONTRACT_FIELDS = set(CONTRACT_HASH_FIELDS) | {
    "status",
    "url",
    "source_updated_at",
    "contract_sha256",
    "fetched_at",
}


def _value(row: Mapping[str, Any], key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _is_sorted_unique_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and bool(item) for item in value)
        and value == sorted(set(value))
    )


def _contract_hash(contract: Mapping[str, Any]) -> str:
    payload = {key: contract[key] for key in CONTRACT_HASH_FIELDS}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _extract_contract(body: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not isinstance(body, str):
        return None, "body_missing"
    if body.count(CONTRACT_BEGIN) != 1 or body.count(CONTRACT_END) != 1:
        return None, "snapshot_marker_invalid"
    try:
        before, remainder = body.split(CONTRACT_BEGIN, 1)
        payload, after = remainder.split(CONTRACT_END, 1)
    except ValueError:
        # Individually unique but reversed markers leave no END marker after
        # BEGIN.  Malformed user-controlled text must be a rejection, never an
        # exception that aborts the rest of the board's dispatch tick.
        return None, "snapshot_marker_invalid"
    if CONTRACT_END in before or CONTRACT_BEGIN in after:
        return None, "snapshot_marker_invalid"
    payload = payload.strip()
    if not payload.startswith("```json") or not payload.endswith("```"):
        return None, "snapshot_fence_invalid"
    payload = payload[len("```json"):-3].strip()
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        return None, "snapshot_json_invalid"
    if not isinstance(parsed, dict):
        return None, "snapshot_json_invalid"
    return parsed, None


def _validate_relations(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"blocks", "blockedBy", "relatedTo", "duplicateOf"}:
        return False
    for key in ("blocks", "blockedBy", "relatedTo"):
        items = value[key]
        if not isinstance(items, list):
            return False
        expected = []
        for item in items:
            if not isinstance(item, dict) or set(item) != {"id", "title"}:
                return False
            if not all(isinstance(item[field], str) and item[field].strip() for field in ("id", "title")):
                return False
            expected.append(item)
        if items != sorted(expected, key=lambda item: (item["id"], item["title"])):
            return False
    duplicate = value["duplicateOf"]
    return duplicate is None or (
        isinstance(duplicate, dict)
        and set(duplicate) == {"id", "title"}
        and all(isinstance(duplicate[field], str) and duplicate[field].strip() for field in ("id", "title"))
    )


def _linked_project_slug(project_id: str) -> Optional[str]:
    """Resolve immutable first-class project identity for a task link.

    Branch names and workspace paths are derived presentation data.  The
    linked Project record is the authoritative repository identity.
    """
    try:
        from hermes_cli import projects_db

        with projects_db.connect_closing() as conn:
            project = projects_db.get_project(conn, project_id)
    except Exception:
        return None
    return project.slug if project is not None else None


def _fleet_contract_guard_reason(row: Mapping[str, Any]) -> Optional[str]:
    """Return a stable rejection code, or ``None`` when dispatch may proceed.

    A row is fleet-governed when either provenance marker is present. Requiring
    both markers prevents a hand-written card from impersonating a reconciled
    task by copying only the body or only an idempotency key.
    """
    created_by = _value(row, "created_by")
    idempotency_key = _value(row, "idempotency_key")
    fleet_creator = created_by == FLEET_CREATED_BY
    fleet_key = isinstance(idempotency_key, str) and idempotency_key.startswith(FLEET_IDEMPOTENCY_PREFIX)
    if not fleet_creator and not fleet_key:
        return None
    if not fleet_creator or not fleet_key:
        return "provenance_mismatch"

    contract, error = _extract_contract(_value(row, "body"))
    if error:
        return error
    assert contract is not None
    if set(contract) != CONTRACT_FIELDS:
        return "snapshot_fields_invalid"
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        return "snapshot_schema_unsupported"

    for key in (
        "issue_id", "linear_id", "title", "project_slug", "repository",
        "status", "source_updated_at", "fetched_at",
    ):
        value = contract.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"snapshot_{key}_invalid"
    for key in ("description", "url"):
        if not isinstance(contract.get(key), str):
            return f"snapshot_{key}_invalid"
    for key in ("agent_ready", "blocked"):
        if not isinstance(contract.get(key), bool):
            return f"snapshot_{key}_invalid"
    execution_mode = contract.get("execution_mode")
    if not isinstance(execution_mode, str) or execution_mode not in VALID_EXECUTION_MODES:
        return "snapshot_execution_mode_invalid"
    for key in ("labels", "redactions"):
        if not _is_sorted_unique_strings(contract.get(key)):
            return f"snapshot_{key}_invalid"
    if not _validate_relations(contract.get("relations")):
        return "snapshot_relations_invalid"
    if not contract["agent_ready"]:
        return "snapshot_not_agent_ready"
    if contract["blocked"]:
        return "snapshot_blocked"
    if contract["redactions"]:
        return "snapshot_redacted"

    digest = contract.get("contract_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return "snapshot_hash_invalid"
    try:
        actual = _contract_hash(contract)
    except (KeyError, TypeError, ValueError):
        return "snapshot_hash_invalid"
    if digest != actual:
        return "snapshot_hash_mismatch"

    linear_id = idempotency_key[len(FLEET_IDEMPOTENCY_PREFIX):]
    if not linear_id or contract["linear_id"] != linear_id:
        return "linear_identity_mismatch"
    expected_title = f"{contract['issue_id']}: {contract['title']}"
    if _value(row, "title") != expected_title:
        return "task_title_mismatch"

    project_id = _value(row, "project_id")
    if not isinstance(project_id, str) or not project_id.strip():
        return "task_project_missing"
    project_slug = _linked_project_slug(project_id)
    if project_slug is None:
        return "task_project_unknown"
    if project_slug != contract["repository"]:
        return "repository_project_mismatch"
    if _value(row, "workspace_kind") != "worktree":
        return "task_workspace_kind_invalid"
    branch_name = _value(row, "branch_name")
    if not isinstance(branch_name, str) or not branch_name.startswith(f"{contract['repository']}/"):
        return "repository_project_mismatch"
    workspace_path = _value(row, "workspace_path")
    if not isinstance(workspace_path, str) or not workspace_path.strip():
        return "task_workspace_path_invalid"
    expected_suffix = os.path.normpath(os.path.join(".worktrees", str(_value(row, "id"))))
    normalized_workspace = os.path.normpath(workspace_path)
    if not normalized_workspace.endswith(os.sep + expected_suffix):
        return "task_workspace_path_invalid"
    return None


def fleet_contract_guard_reason(row: Mapping[str, Any]) -> Optional[str]:
    """Return a stable rejection code, never an exception, for any task row.

    Fleet task bodies are external input.  A malformed JSON shape, marker
    order, or unexpected SQLite value must quarantine only that card; it must
    not prevent unrelated ready/review work from progressing.
    """
    try:
        return _fleet_contract_guard_reason(row)
    except Exception:
        return "snapshot_invalid"
