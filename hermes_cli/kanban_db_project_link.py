"""Project identity recovery from canonical tasks on a shared Kanban board."""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional


def _project_from_source_task(
    conn: sqlite3.Connection, _pdb: Any, project_id: str, source_task_id: str,
    *, child_workspace_path: Optional[str] = None,
    child_workspace_kind: Optional[str] = None,
) -> tuple[Any, Optional[str]]:
    """Recover a Project (and its repo) from a canonical project-linked task on
    this board. Worker profiles have their own projects.db while the Kanban DB
    is shared, so this carries the repo + branch convention forward without
    opening the creator's store and without reusing the source task's literal
    worktree path. ``(None, None)`` when the source task is not a qualified
    project task.

    A ``dir`` source task is qualified only when the child is itself an
    explicit ``dir`` task in the same governed directory: both paths must be
    absolute and canonically equal. The explicit dir workspace is then kept
    (``project_repo`` is None, so no worktree path or branch is derived)."""
    from hermes_cli.kanban_db import get_task

    source_task = get_task(conn, source_task_id)
    if not (source_task is not None and source_task.project_id == project_id):
        return None, None
    if source_task.workspace_kind == "dir":
        if child_workspace_kind != "dir":
            return None, None
        if not source_task.workspace_path or not child_workspace_path:
            return None, None
        dir_path = Path(str(source_task.workspace_path))
        child_path = Path(str(child_workspace_path))
        if not (
            dir_path.is_absolute()
            and child_path.is_absolute()
            and os.path.normcase(os.path.realpath(str(dir_path)))
            == os.path.normcase(os.path.realpath(str(child_path)))
        ):
            return None, None
        project_slug = None
        with contextlib.suppress(ValueError):
            project_slug = _pdb.normalize_slug(project_id)
        if not project_slug:
            return None, None
        project_obj = _pdb.Project(
            id=project_id, slug=project_slug, name=project_slug,
            created_at=0, primary_path=str(dir_path),
        )
        return project_obj, None
    if not (source_task.workspace_kind == "worktree" and source_task.workspace_path):
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
