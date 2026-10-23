"""Guarded project binding for existing Kanban tasks."""
from __future__ import annotations

import json
import os
import subprocess
import unicodedata
from typing import Any

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


# A spawn fingerprint without worker_pid is historical metadata; complete_task clears the PID but leaves it behind.
_ACTIVE_FIELDS = (
    "claim_lock", "claim_expires", "worker_pid", "current_run_id",
)
_WORKSPACE_FIELDS = ("workspace_kind", "workspace_path", "branch_name")


def _find_project(reference: str) -> pdb.Project:
    if not reference or reference != reference.strip():
        raise ValueError("--project must be a non-empty project ID or slug")
    try:
        with pdb.connect_closing() as conn:
            projects = pdb.list_projects(conn, include_archived=True)
    except Exception as exc:
        raise RuntimeError(f"cannot read the active profile's project registry: {exc}") from exc

    try:
        slug = pdb.normalize_slug(reference)
    except ValueError as exc:
        slug = None
        slug_error = exc
    else:
        slug_error = None
    candidates = [p for p in projects if p.id == reference or (slug and p.slug == slug)]
    if len(candidates) > 1:
        labels = ", ".join(f"{p.slug} ({p.id})" for p in candidates)
        raise ValueError(f"project reference {reference!r} is ambiguous: {labels}")
    if not candidates:
        if slug_error is not None:
            raise ValueError(f"invalid project reference {reference!r}: {slug_error}") from slug_error
        raise ValueError(f"project {reference!r} was not found in the active profile's project registry")
    project = candidates[0]
    if project.archived:
        raise ValueError(f"destination project {project.slug!r} ({project.id}) is archived")
    return project


def _project_root(project: pdb.Project) -> str:
    root = project.primary_path or next(
        (folder.path for folder in project.folders if folder.is_primary),
        project.folders[0].path if project.folders else None,
    )
    if not root:
        raise ValueError(f"destination project {project.slug!r} has no primary repository path")
    root = os.path.expanduser(root)
    if not os.path.isabs(root):
        raise ValueError(f"destination project {project.slug!r} has a non-absolute repository path")
    return os.path.realpath(root)


def _canonical(path: str) -> str:
    value = os.path.normcase(os.path.realpath(os.path.normpath(os.path.expanduser(path))))
    return unicodedata.normalize("NFC", value)


def _same_path(left: str, right: str) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return _canonical(left) == _canonical(right)


def _git(path: str, *args: str) -> str:
    """Query repository metadata using the supplied worktree, not ambient Git selectors."""
    from tools.environments.local import served_profile_child_env

    env = served_profile_child_env()
    for name in (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CEILING_DIRECTORIES",
        "GIT_PREFIX",
    ):
        env.pop(name, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"timed out verifying workspace {path!r}") from exc
    except OSError as exc:
        raise ValueError(f"cannot verify Git workspace {path!r}: {exc}") from exc
    if result.returncode:
        raise ValueError(f"cannot verify Git workspace {path!r}: {result.stderr.strip() or 'Git check failed'}")
    value = result.stdout.rstrip("\r\n")
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"cannot verify Git workspace {path!r}: expected one unambiguous result")
    return value


def _common_dir(path: str) -> str:
    value = _git(path, "--git-common-dir")
    if not os.path.isabs(value):
        value = os.path.join(path, value)
    return _canonical(value)


def _check_workspace(project: pdb.Project, task: Any) -> None:
    """Require the task's stored workspace to already match the project workspace."""
    workspace_kind = task["workspace_kind"]
    task_id = task["id"]
    if not task_id or task_id in {".", ".."} or any(c in task_id for c in ("/", "\\", ":")) or any(ord(c) < 32 for c in task_id):
        raise ValueError("workspace is incompatible: unsafe task ID path component")

    root = _project_root(project)
    path = task["workspace_path"]
    if not path or not os.path.isabs(os.path.expanduser(path)):
        raise ValueError("workspace is incompatible: task workspace path must be absolute")

    if workspace_kind == "dir":
        if not os.path.isdir(root) or not os.path.isdir(path):
            raise ValueError("workspace is incompatible: project primary directory and task directory must already exist")
        if _canonical(path) != _canonical(root):
            raise ValueError(
                f"workspace is incompatible: existing directory path must resolve to project primary directory {root!r}"
            )
        return

    if workspace_kind != "worktree":
        raise ValueError("workspace is incompatible: task must use a project directory or existing worktree")

    expected = os.path.join(root, ".worktrees", task_id)
    if not _same_path(path, expected):
        raise ValueError(f"workspace is incompatible: existing path must remain at {expected!r}")

    branch = task["branch_name"] or ""
    prefix = f"{project.slug}/{task_id}"
    if branch != prefix and not branch.startswith(prefix + "-"):
        raise ValueError(f"workspace is incompatible: branch must belong to project task prefix {prefix!r}")
    if not os.path.isdir(root) or not os.path.isdir(path):
        raise ValueError("workspace is incompatible: project repository and task worktree must already exist")

    project_top = _git(root, "--show-toplevel")
    task_top = _git(path, "--show-toplevel")
    if not _same_path(project_top, root):
        raise ValueError("workspace is incompatible: project primary path is not a Git repository root")
    if not _same_path(task_top, path):
        raise ValueError("workspace is incompatible: stored path is not a Git worktree root")
    if _common_dir(root) != _common_dir(path):
        raise ValueError("workspace is incompatible: task worktree belongs to another Git repository")
    actual_branch = _git(path, "--abbrev-ref", "HEAD")
    if actual_branch != branch:
        raise ValueError(
            f"workspace is incompatible: Git branch {actual_branch!r} does not match stored branch {branch!r}"
        )


def _board_project_scope(board: str) -> str | None:
    """Read board scope without accepting ``read_board_metadata``'s fallback defaults."""
    path = kb.board_metadata_path(board)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        if board == kb.DEFAULT_BOARD:
            # The default board predates board.json and is legitimately unscoped
            # when no metadata file exists. A broken symlink or unreadable
            # parent is not equivalent to an absent file.
            try:
                path.lstat()
            except FileNotFoundError:
                return None
            except OSError as stat_exc:
                raise RuntimeError(f"cannot verify board {board!r} metadata: {stat_exc}") from stat_exc
        raise RuntimeError(f"cannot verify board {board!r} metadata: file is missing") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot verify board {board!r} metadata: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"cannot verify board {board!r} metadata: expected a JSON object")
    if raw.get("slug") != board:
        raise RuntimeError(f"cannot verify board {board!r} metadata: slug integrity check failed")
    if "project_id" not in raw:
        raise RuntimeError(f"cannot verify board {board!r} metadata: required project_id scope is missing")
    project_id = raw["project_id"]
    if project_id is None:
        return None
    if not isinstance(project_id, str) or not project_id.strip():
        raise RuntimeError(f"cannot verify board {board!r} metadata: project_id scope is invalid")
    return project_id.strip()


def _check_scope(project: pdb.Project, board: str) -> None:
    if project.board_slug:
        project_board = kb._normalize_board_slug(project.board_slug)
        if project_board != board:
            raise ValueError(f"project {project.slug!r} belongs to board {project_board!r}, not {board!r}")
    board_project = _board_project_scope(board)
    if board_project and board_project != project.id:
        raise ValueError(f"board {board!r} is bound to project {board_project!r}, not {project.id!r}")


def _active(row: Any) -> bool:
    return row["status"] == "running" or any(row[field] is not None for field in _ACTIVE_FIELDS)


def _success(task_id: str, project: pdb.Project, board: str, changed: bool) -> dict[str, Any]:
    return {
        "ok": True,
        "result": "bound" if changed else "already_bound",
        "changed": changed,
        "task_id": task_id,
        "project_id": project.id,
        "project_slug": project.slug,
        "board": board,
    }


def bind_task_project(
    conn, task_id: str, project_ref: str, *, board: str | None = None, actor: str = "user",
) -> dict[str, Any]:
    """Bind a projectless task to an explicit project, preserving every other field."""
    if not task_id or task_id != task_id.strip():
        raise ValueError("task ID is required and must not contain surrounding whitespace")
    if not isinstance(project_ref, str) or not project_ref:
        raise ValueError("an explicit destination project is required")

    board_slug = kb._normalize_board_slug(board or kb.get_current_board()) or kb.DEFAULT_BOARD
    project = _find_project(project_ref)
    _check_scope(project, board_slug)
    select = (
        "SELECT id, status, project_id, workspace_kind, workspace_path, branch_name, "
        "claim_lock, claim_expires, worker_pid, worker_started_at, current_run_id "
        "FROM tasks WHERE id = ?"
    )
    snapshot = conn.execute(select, (task_id,)).fetchone()
    if snapshot is None:
        raise ValueError(f"task {task_id!r} was not found on board {board_slug!r}")
    if _active(snapshot):
        raise ValueError(f"cannot bind project to running or claimed task {task_id!r}")
    if snapshot["project_id"] is None:
        _check_workspace(project, snapshot)

    with kb.write_txn(conn):
        row = conn.execute(select, (task_id,)).fetchone()
        if row is None:
            raise ValueError(f"task {task_id!r} was not found on board {board_slug!r}")
        if _active(row):
            raise ValueError(f"cannot bind project to running or claimed task {task_id!r}")
        if row["project_id"] is not None:
            if row["project_id"] == project.id:
                return _success(task_id, project, board_slug, False)
            raise ValueError(
                f"task {task_id!r} is already bound to project {row['project_id']!r}; "
                "only projectless tasks can be bound"
            )
        if snapshot["project_id"] is not None:
            raise ValueError(f"task {task_id!r} project binding changed during validation; inspect and retry")
        if any(row[field] != snapshot[field] for field in _WORKSPACE_FIELDS):
            raise ValueError(f"task {task_id!r} workspace changed during validation; inspect and retry")

        changed = conn.execute(
            "UPDATE tasks SET project_id = ? WHERE id = ? AND project_id IS NULL "
            "AND status != 'running' AND claim_lock IS NULL AND claim_expires IS NULL "
            "AND worker_pid IS NULL AND current_run_id IS NULL",
            (project.id, task_id),
        )
        if changed.rowcount != 1:
            raise ValueError(f"task {task_id!r} changed during project binding; inspect and retry")
        kb._append_event(
            conn,
            task_id,
            "project_bound",
            {
                "from_project_id": None,
                "to_project_id": project.id,
                "project_slug": project.slug,
                "actor": actor or "user",
            },
        )

    kb.notify_task_updated(conn, task_id, ["project_id"], board=board_slug)
    return _success(task_id, project, board_slug, True)
