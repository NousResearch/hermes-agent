"""Project links on task creation: resolve a requested project id to the project, its repo and
the workspace kind, recovering it from this board's canonical worktree rows when the filer's own
per-profile projects.db does not hold it, and report what became of the request.

Split out of ``hermes_cli.kanban_db``; facade helpers are late-imported inside functions.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Optional


def _resolve_project_link(
    conn: sqlite3.Connection, project_id: Optional[str], project_source_task_id: Optional[str],
    workspace_kind: str, workspace_path: Optional[str],
) -> tuple[Optional[str], Any, Optional[str], str, Optional[dict]]:
    """``(project_id, project_obj, project_repo, workspace_kind, link_report)``.

    A project-linked task is anchored to the project's primary repo as a
    worktree with a deterministic branch (slug + task id). Projects live in the
    creator's per-profile projects.db, but the stored repo path is absolute so
    the cross-profile dispatcher needs no projects.db access. ``project_repo``
    is set when the worktree path must still be derived from the new task id.

    ``link_report`` is ``None`` for the ordinary case (the requested id resolved
    in this profile's own store) and otherwise says what became of the request:
    recovered from a board row that carries the project
    (``recovered_from_source_task`` / ``recovered_from_board_task``) or
    ``dropped``. The caller records it, because a dropped link is otherwise
    indistinguishable from a caller that never asked for a project — and a
    worker profile whose projects.db is empty loses every link it requests.
    """
    project_id = (str(project_id).strip() or None) if project_id is not None else None
    if not project_id:
        return None, None, None, workspace_kind, None
    from hermes_cli import projects_db as _pdb

    project_repo: Optional[str] = None
    try:
        with _pdb.connect_closing() as _pconn:
            project_obj = _pdb.get_project(_pconn, project_id)
    except Exception:
        project_obj = None
    link_report: Optional[dict] = None
    if project_obj is None:
        # projects.db is per-profile while the board is shared, so a request
        # that is valid board-wide is unresolvable from here (worker profiles
        # can hold no projects at all). The board rows are the shared truth:
        # recover the project from a canonical worktree card that carries it,
        # the filer's own card first.
        source_id = str(project_source_task_id) if project_source_task_id else None
        project_obj, project_repo, recovered_from = _project_from_board(
            conn, _pdb, project_id, source_id,
        )
        if project_obj is not None:
            link_report = {
                "requested": project_id,
                "resolution": (
                    "recovered_from_source_task" if recovered_from == source_id
                    else "recovered_from_board_task"
                ),
                "source_task": recovered_from,
            }
            if workspace_kind == "scratch":
                workspace_kind = "worktree"
    if project_obj is None:
        # Unresolvable id/slug: drop the link (never a dangling reference,
        # never a crash) and create an ordinary scratch task — silently no
        # longer: the report lands on the card's ``created`` event.
        link_report = {
            "requested": project_id,
            "resolution": "dropped",
            "source_task": str(project_source_task_id) if project_source_task_id else None,
            "reason": (
                f"project {project_id!r} is not in this profile's projects.db and no card "
                "on this board carries it as a <repo>/.worktrees/<task-id> worktree, so "
                "this task was created without a project link"
            ),
        }
        return None, None, None, workspace_kind, link_report
    # Canonicalise (a slug may have been passed) and anchor the worktree
    # under the project's primary repo.
    if workspace_kind == "scratch" and project_obj.primary_path:
        workspace_kind = "worktree"
    if workspace_kind == "worktree" and workspace_path is None and project_obj.primary_path:
        # Concrete path is deferred to the insert loop: a fresh
        # ``<repo>/.worktrees/<task-id>`` keyed on the new task id.
        project_repo = str(project_obj.primary_path)
    return project_obj.id, project_obj, project_repo, workspace_kind, link_report


def _project_from_board(
    conn: sqlite3.Connection, _pdb: Any, project_id: str, source_task_id: Optional[str],
) -> tuple[Any, Optional[str], Optional[str]]:
    """``(project_obj, repo, task_id)`` for ``project_id``, read off THIS board.

    ``projects_db`` is per-profile and the board is shared: a filer whose store
    never heard of the project (a worker profile with an empty store, a profile
    tracking that repo under another id) is the ordinary case, not an error.
    A canonical project worktree row carries everything the link needs — the
    project id it was filed under and the repo the worktree sits in — so the
    filer's own card wins when it carries the project, and otherwise the newest
    canonical worktree row does: the repo is a property of the project, not of
    the card that happens to record it. The board scan is bounded (the newest 25
    rows for this project that look like project worktrees); past it the caller
    reports the drop rather than walking a whole history. ``(None, None, None)``
    when no row qualifies.
    """
    candidates: list[str] = [source_task_id] if source_task_id else []
    rows = conn.execute(
        "SELECT id FROM tasks WHERE project_id = ? AND workspace_kind = 'worktree' "
        "AND REPLACE(workspace_path, '\\', '/') LIKE '%/.worktrees/%' "
        "ORDER BY created_at DESC, id DESC LIMIT 25",
        (project_id,),
    ).fetchall()
    candidates += [row["id"] for row in rows if row["id"] not in candidates]
    for task_id in candidates:
        project_obj, project_repo = _project_from_worktree_task(
            conn, _pdb, project_id, str(task_id),
        )
        if project_obj is not None:
            return project_obj, project_repo, str(task_id)
    return None, None, None


# The board is shared across hosts, so a row written on Windows may be read on
# POSIX (and the reverse): the shape check cannot go through the host's pathlib.
_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_WORKTREE_PATH_RE = re.compile(
    r"^(?P<repo>.*?)[/\\]+\.worktrees[/\\]+(?P<leaf>[^/\\]+)[/\\]*$"
)


def _worktree_repo_root(workspace_path: str, task_id: str) -> Optional[str]:
    """The repo of an absolute ``<repo>/.worktrees/<task_id>`` path, in the
    path's own separator style, on any host. ``None`` for any other shape."""
    if not _ABSOLUTE_PATH_RE.match(workspace_path):
        return None
    match = _WORKTREE_PATH_RE.match(workspace_path)
    if match is None or match.group("leaf") != task_id:
        return None
    repo = match.group("repo")
    if repo == "" or re.fullmatch(r"[A-Za-z]:", repo):
        # Filesystem root (``/`` or ``C:\``): keep the separator that makes it one.
        repo = workspace_path[: match.end("repo") + 1]
    return repo if _ABSOLUTE_PATH_RE.match(repo) else None


def _project_from_worktree_task(
    conn: sqlite3.Connection, _pdb: Any, project_id: str, task_id: str,
) -> tuple[Any, Optional[str]]:
    """Recover a Project (and its repo) from a canonical project-linked
    worktree task on this board. Worker profiles have their own projects.db
    while the Kanban DB is shared, so this carries the repo + branch
    convention forward without opening the creator's store and without
    reusing the task's literal worktree path. ``(None, None)`` when the task
    is not a ``<repo>/.worktrees/<id>`` project worktree."""
    from hermes_cli.kanban_db import get_task

    source_task = get_task(conn, task_id)
    if not (
        source_task is not None
        and source_task.project_id == project_id
        and source_task.workspace_kind == "worktree"
        and source_task.workspace_path
    ):
        return None, None
    project_repo = _worktree_repo_root(source_task.workspace_path, source_task.id)
    if project_repo is None:
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
    project_obj = _pdb.Project(
        id=project_id, slug=project_slug, name=project_slug, created_at=0, primary_path=project_repo,
    )
    return project_obj, project_repo
