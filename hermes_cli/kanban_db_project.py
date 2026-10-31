"""Project linkage for ``kanban_db.create_task``: which project a task belongs to,
and whether that project can anchor the task's workspace as a worktree.

Lives beside ``kanban_db`` (the facade) as a topical sibling. The rule that
matters here is a creation-time one: a project's ``primary_path`` can only
anchor ``<path>/.worktrees/<id>`` when it is inside a git work tree, and that
has to be decided when the card is created — the spawn-time check runs after the
card is already on the board, where a bad anchor is a dispatch crash instead of
a rejected create.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)


def worktree_anchor(primary_path: Optional[str]) -> Optional[Path]:
    """The git repo that can anchor ``<primary_path>/.worktrees/<id>``, or ``None``.

    Mirrors the spawn-time resolver (``kanban_db_workspace._repo_root_for_worktree_target``
    climbs to the nearest existing ancestor), so a card is upgraded to a worktree
    only when dispatch will actually be able to materialize one there.
    """
    if not primary_path:
        return None
    from hermes_cli import kanban_db_workspace as _kbw

    return _kbw._repo_root_for_worktree_target(Path(str(primary_path)))


def worktree_unavailable(project_obj: Any) -> Optional[str]:
    """Why *project_obj* cannot anchor a worktree; ``None`` when it can."""
    if worktree_anchor(project_obj.primary_path) is not None:
        return None
    slug = getattr(project_obj, "slug", None) or project_obj.id
    return (
        f"project {slug!r} primary_path {str(project_obj.primary_path)!r} "
        "is not inside a git repo"
    )


def worktree_path_unavailable(workspace_path: Optional[str]) -> Optional[str]:
    """Why an explicit worktree ``workspace_path`` cannot host a worktree; ``None`` when it can.

    Held to the rule dispatch applies (``kanban_db_workspace._resolve_worktree_workspace``):
    the path must be absolute and must be a git repo root, an existing linked
    worktree, or inside a repo. A path *inside* a repo is fine even when it does
    not exist yet — dispatch materializes it there.
    """
    if not workspace_path:
        return None
    path = Path(str(workspace_path)).expanduser()
    if not path.is_absolute():
        return (
            f"workspace_path {str(workspace_path)!r} is not absolute, so it cannot "
            "host a worktree (worktree paths must be absolute)"
        )
    if worktree_anchor(str(path)) is not None:
        return None
    return (
        f"workspace_path {str(workspace_path)!r} is not inside a git repo and does "
        "not point at a git repo root, so it cannot host a worktree"
    )


def board_workspace_path(
    board: Optional[str], workspace_kind: str, workspace_path: Optional[str],
    project_repo: Optional[str],
) -> Optional[str]:
    """``workspace_path`` a new card starts with, board ``default_workdir`` included.

    Only persistent kinds inherit the board default — a scratch task inheriting
    it would point cleanup at the user's source tree — and only when nothing
    more specific was supplied. A ``worktree`` default must be able to host a
    worktree at dispatch, or the card is accepted here and dies at spawn.
    """
    if workspace_path is not None or project_repo is not None:
        return workspace_path
    if workspace_kind not in {"dir", "worktree"}:
        return workspace_path
    from hermes_cli import kanban_db as _kb

    board_slug = board if board else _kb.get_current_board()
    board_default = (
        _kb.read_board_metadata(board_slug).get("default_workdir") or ""
    ).strip()
    if not board_default:
        return workspace_path
    if workspace_kind == "worktree":
        refusal = worktree_path_unavailable(board_default)
        if refusal:
            raise ValueError(
                f"board {board_slug!r} default_workdir cannot host a worktree: "
                f"{refusal}. Point the board's default workdir at a folder inside "
                "a git repo, or create the task with an explicit workspace_path "
                "inside one."
            )
    return board_default


def resolve_project_link(
    conn: sqlite3.Connection, project_id: Optional[str], project_source_task_id: Optional[str],
    workspace_kind: str, workspace_path: Optional[str],
) -> tuple[Optional[str], Any, Optional[str], str, Optional[str]]:
    """``(project_id, project_obj, project_repo, workspace_kind, skipped_reason)`` for ``create_task``.

    A project-linked task is anchored to the project's primary repo as a
    worktree with a deterministic branch (slug + task id). Projects live in the
    creator's per-profile projects.db, but the stored repo path is absolute so
    the cross-profile dispatcher needs no projects.db access. ``project_repo``
    is set when the worktree path must still be derived from the new task id.

    The upgrade is a *default* (board- or surface-supplied), so it may never
    strand a card: when ``primary_path`` cannot host a worktree the task stays on
    a ``scratch`` workspace and ``skipped_reason`` says why, rather than creating
    a card whose worker dies at spawn. An explicit ``workspace_kind="worktree"``
    on such a project raises here instead — same gap, closed at the same place —
    and so does an explicit ``workspace_path`` that dispatch could not anchor a
    worktree on. That path check runs *before* the project lookup: a
    project-less card, an unresolvable id/slug and a project-backed card all
    reach dispatch with the same literal path, so the same rule must refuse it
    whichever way the lookup goes.
    """
    if workspace_kind == "worktree":
        # An explicit worktree path overrides every default at dispatch, so it
        # is validated before any project resolution: whichever way the lookup
        # below goes (no project, unresolvable id/slug, project that cannot
        # anchor), the card reaches dispatch with this literal path.
        refusal = worktree_path_unavailable(workspace_path)
        if refusal:
            raise ValueError(
                f"{refusal}. Create the task with workspace_kind='scratch', point "
                "workspace_path at a folder inside a git repo, or omit it and let a "
                "project anchor the worktree."
            )
    project_id = (str(project_id).strip() or None) if project_id is not None else None
    if not project_id:
        return None, None, None, workspace_kind, None
    from hermes_cli import projects_db as _pdb

    project_repo: Optional[str] = None
    skipped_reason: Optional[str] = None
    try:
        with _pdb.connect_closing() as _pconn:
            project_obj = _pdb.get_project(_pconn, project_id)
    except Exception:  # health: allow BLE001 -- a broken projects.db must never block card creation
        _log.debug("kanban: project %r lookup failed", project_id, exc_info=True)
        project_obj = None
    if project_obj is None and project_source_task_id:
        project_obj, project_repo = project_from_source_task(
            conn, _pdb, project_id, str(project_source_task_id),
        )
        if project_obj is not None and workspace_kind == "scratch":
            # The recovered repo is ``<repo>`` by construction, but the board is
            # shared across profiles and hosts: the source task's checkout may be
            # absent here (another machine, or a moved/deleted repo), and then
            # this anchor is no better than an unavailable ``primary_path``.
            if worktree_unavailable(project_obj):
                project_repo = None
            else:
                workspace_kind = "worktree"
    if project_obj is None:
        # Unresolvable id/slug: drop the link (never a dangling reference,
        # never a crash) and create an ordinary scratch task.
        return None, None, None, workspace_kind, None
    # Canonicalise (a slug may have been passed) and anchor the worktree under
    # the project's primary repo — only when that path can host one.
    unavailable = worktree_unavailable(project_obj) if project_obj.primary_path else None
    if workspace_kind == "scratch" and project_obj.primary_path:
        if unavailable:
            skipped_reason = f"{unavailable}; created as a scratch workspace"
        else:
            workspace_kind = "worktree"
    if unavailable and workspace_kind == "worktree" and workspace_path is None:
        raise ValueError(
            f"{unavailable}, so it cannot host a worktree. Create the task with "
            "workspace_kind='scratch', point the project at a folder inside a git repo, "
            "or pass an explicit workspace_path inside a git repo."
        )
    if workspace_kind == "worktree" and workspace_path is None and project_obj.primary_path:
        # Concrete path is deferred to the insert loop: a fresh
        # ``<repo>/.worktrees/<task-id>`` keyed on the new task id.
        project_repo = str(project_obj.primary_path)
    return project_obj.id, project_obj, project_repo, workspace_kind, skipped_reason


def record_worktree_skip(conn: sqlite3.Connection, task_id: str, reason: Optional[str]) -> None:
    """Record why a project could not anchor the task as a worktree.

    No-op when the upgrade happened — the call site stays branch-free so
    ``create_task`` (already over its complexity cap) does not grow one.
    """
    if not reason:
        return
    from hermes_cli import kanban_db as _kb

    _log.warning("kanban: %s", reason)
    _kb._append_event(
        conn, task_id, "project_worktree_skipped",
        {"reason": "primary_path_not_a_git_repo", "message": reason},
    )


def project_from_source_task(
    conn: sqlite3.Connection, _pdb: Any, project_id: str, source_task_id: str,
) -> tuple[Any, Optional[str]]:
    """Recover a Project (and its repo) from a canonical project-linked
    worktree task on this board. Worker profiles have their own projects.db
    while the Kanban DB is shared, so this carries the repo + branch
    convention forward without opening the creator's store and without
    reusing the source task's literal worktree path. ``(None, None)`` when
    the source task is not a ``<repo>/.worktrees/<id>`` project worktree."""
    from hermes_cli import kanban_db as _kb

    source_task = _kb.get_task(conn, source_task_id)
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
