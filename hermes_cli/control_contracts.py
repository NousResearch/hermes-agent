from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ALLOWED_TASK_TYPES = {"inspect_repo", "run_tests", "parse_sections", "review_diff", "generic"}
ALLOWED_TASK_PERMISSIONS = {"read", "write", "test", "git"}
STRICT_CONSTRAINTS = {"no_live_db_mutation"}


class ContractError(ValueError):
    pass


def _as_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError("dispatch payload must be an object")
    return payload


def _norm_abs(path: Any, *, field: str) -> str:
    if not isinstance(path, str) or not path:
        raise ContractError(f"{field} must be a non-empty string")
    p = Path(path)
    if not p.is_absolute():
        raise ContractError(f"{field} must be absolute")
    return os.path.normpath(str(p))


def _within(child: str, parent: str) -> bool:
    try:
        Path(child).relative_to(Path(parent))
        return True
    except ValueError:
        return child == parent


def validate_statute_dispatch_v1(payload: dict[str, Any], *, parent: dict[str, Any] | None = None, require_parent: bool = False) -> dict[str, Any]:
    payload = dict(_as_dict(payload))
    if payload.get("schema") != "statute_dispatch_v1":
        raise ContractError("schema must be statute_dispatch_v1")
    if payload.get("silo") != "statute":
        raise ContractError("silo must be statute")

    repo_root = _norm_abs(payload.get("repo_root"), field="repo_root")
    allowed_paths_raw = payload.get("allowed_paths")
    if not isinstance(allowed_paths_raw, list) or not allowed_paths_raw:
        raise ContractError("allowed_paths must be a non-empty list")
    allowed_paths = [_norm_abs(p, field="allowed_paths[]") for p in allowed_paths_raw]
    for path in allowed_paths:
        if not _within(path, repo_root):
            raise ContractError(f"allowed path escapes repo_root: {path}")

    task_type = payload.get("task_type", "generic")
    if task_type not in ALLOWED_TASK_TYPES:
        raise ContractError(f"invalid task_type: {task_type}")
    permissions_raw = payload.get("task_permissions", [])
    if not isinstance(permissions_raw, list):
        raise ContractError("task_permissions must be a list")
    permissions = set(permissions_raw)
    invalid = permissions - ALLOWED_TASK_PERMISSIONS
    if invalid:
        raise ContractError(f"invalid task_permissions: {sorted(invalid)}")

    constraints = payload.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise ContractError("constraints must be an object")
    for key in STRICT_CONSTRAINTS:
        if constraints.get(key) is not True:
            raise ContractError(f"constraint {key} must be true")
    if constraints.get("no_push") is not True and constraints.get("push_at_successful_wave_closeout") is not True:
        raise ContractError("constraint no_push must be true unless push_at_successful_wave_closeout is true")

    parent_dispatch_id = payload.get("parent_dispatch_id")
    if require_parent and not parent_dispatch_id:
        raise ContractError("child dispatch requires parent_dispatch_id")

    payload["repo_root"] = repo_root
    payload["allowed_paths"] = allowed_paths
    payload["task_type"] = task_type
    payload["task_permissions"] = sorted(permissions)
    payload["constraints"] = dict(constraints)

    if parent is not None:
        parent_v = validate_statute_dispatch_v1(parent)
        if repo_root != parent_v["repo_root"]:
            raise ContractError("child repo_root must equal parent repo_root")
        parent_allowed = set(parent_v["allowed_paths"])
        for path in allowed_paths:
            if not any(_within(path, base) for base in parent_allowed):
                raise ContractError("child allowed_paths must be within parent allowed_paths")
        if not permissions.issubset(set(parent_v["task_permissions"])):
            raise ContractError("child task_permissions cannot widen parent permissions")
        parent_constraints = parent_v.get("constraints") or {}
        for key, value in parent_constraints.items():
            if value is True and payload["constraints"].get(key) is not True:
                raise ContractError(f"child constraint {key} cannot relax parent")
        for key in STRICT_CONSTRAINTS:
            if parent_constraints.get(key) is True and payload["constraints"].get(key) is not True:
                raise ContractError(f"child constraint {key} cannot relax parent")
    return payload


def make_child_payload(parent_payload: dict[str, Any], *, parent_dispatch_id: str, instructions: str | None = None) -> dict[str, Any]:
    parent = validate_statute_dispatch_v1(parent_payload)
    child = dict(parent)
    child["parent_dispatch_id"] = parent_dispatch_id
    if instructions is not None:
        child["instructions"] = instructions
    return validate_statute_dispatch_v1(child, parent=parent, require_parent=True)
