"""Task-creation validation and project-link resolution for Kanban."""
from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any, Optional


def require_dispatchable_assignee(assignee: Optional[str]) -> None:
    """Fail closed for typed/non-local work while preserving legacy local rows."""
    if not assignee:
        raise ValueError("typed or non-local tasks require an assignee")
    try:
        from hermes_cli.profiles import profile_exists
    except Exception as exc:
        raise ValueError("cannot verify dispatchable assignee") from exc
    if not profile_exists(assignee):
        raise ValueError(f"assignee is not a dispatchable profile: {assignee}")


def validate_task_attributes(
    task_type: str, delivery_type: str, completion_contract: Optional[str],
    assignee: Optional[str], valid_task_types: set[str], valid_delivery_types: set[str],
) -> tuple[str, str]:
    """Normalize and validate task/delivery fields at create and edit boundaries."""
    normalized_task = str(task_type or "general").strip().lower()
    normalized_delivery = str(delivery_type or "local").strip()
    if normalized_task not in valid_task_types:
        raise ValueError(f"task_type must be one of {sorted(valid_task_types)}")
    if normalized_delivery not in valid_delivery_types:
        raise ValueError(f"delivery_type must be one of {sorted(valid_delivery_types)}")
    if normalized_delivery == "PR" and completion_contract == "local-only":
        raise ValueError("delivery_type=PR requires a non-local completion_contract")
    if (normalized_task != "general" or normalized_delivery != "local"):
        require_dispatchable_assignee(assignee)
    return normalized_task, normalized_delivery


def resolve_project_link(
    conn: sqlite3.Connection, project_id: Optional[str], project_source_task_id: Optional[str],
    workspace_kind: str, workspace_path: Optional[str],
) -> tuple[Optional[str], Any, Optional[str], str]:
    """Resolve a project link and its repository/workspace kind for task creation."""
    project_id = (str(project_id).strip() or None) if project_id is not None else None
    if not project_id:
        return None, None, None, workspace_kind
    from hermes_cli import projects_db as _pdb

    project_repo: Optional[str] = None
    try:
        with _pdb.connect_closing() as _pconn:
            project_obj = _pdb.get_project(_pconn, project_id)
    except (OSError, RuntimeError, ValueError, sqlite3.Error):
        project_obj = None
    if project_obj is None and project_source_task_id:
        project_obj, project_repo = _project_from_source_task(
            conn, _pdb, project_id, str(project_source_task_id),
        )
        if project_obj is not None and workspace_kind == "scratch":
            workspace_kind = "worktree"
    if project_obj is None:
        return None, None, None, workspace_kind
    if workspace_kind == "scratch" and project_obj.primary_path:
        workspace_kind = "worktree"
    if workspace_kind == "worktree" and workspace_path is None and project_obj.primary_path:
        project_repo = str(project_obj.primary_path)
    return project_obj.id, project_obj, project_repo, workspace_kind


def _project_from_source_task(
    conn: sqlite3.Connection, _pdb: Any, project_id: str, source_task_id: str,
) -> tuple[Any, Optional[str]]:
    """Recover a project from a canonical project-linked worktree task."""
    from hermes_cli.kanban_db import get_task

    source_task = get_task(conn, source_task_id)
    if not (
        source_task is not None
        and source_task.project_id == project_id
        and source_task.workspace_kind == "worktree"
        and source_task.workspace_path
    ):
        return None, None
    source_path = Path(source_task.workspace_path)
    if not (
        source_path.is_absolute()
        and source_path.name == source_task.id
        and source_path.parent.name == ".worktrees"
    ):
        return None, None
    project_slug = None
    if source_task.branch_name:
        prefix, separator, leaf = source_task.branch_name.partition("/")
        if separator and (leaf == source_task.id or leaf.startswith(f"{source_task.id}-")):
            with contextlib.suppress(ValueError):
                project_slug = _pdb.normalize_slug(prefix)
    if project_slug is None:
        with contextlib.suppress(ValueError):
            project_slug = _pdb.normalize_slug(project_id)
    if not project_slug:
        return None, None
    project_repo = str(source_path.parent.parent)
    project_obj = _pdb.Project(
        id=project_id, slug=project_slug, name=project_slug, created_at=0, primary_path=project_repo,
    )
    return project_obj, project_repo
