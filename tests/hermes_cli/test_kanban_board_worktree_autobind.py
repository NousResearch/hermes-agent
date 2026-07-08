"""Board-linked worktree autobind: a task created on a board whose
``default_workdir`` points at a git repo (set by ``hermes project bind-board``
/ ``project create --board``) is materialized as a real linked worktree on that
repo instead of an ephemeral scratch dir — even with no explicit ``project_id``
and even when a *different* profile creates the card (the binding lives in the
shared board metadata, not the per-profile projects.db).
"""

from __future__ import annotations

import os
import subprocess

import pytest

from hermes_cli import kanban_db as kb


def _git_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "config", "commit.gpgsign", "false"],
    ):
        subprocess.run(cmd, cwd=path, env=env, check=True)
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "-A"], cwd=path, env=env, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, env=env, check=True)
    return path


@pytest.fixture
def conn():
    c = kb.connect()
    try:
        yield c
    finally:
        c.close()


def test_board_bound_repo_upgrades_scratch_to_worktree(conn, tmp_path):
    repo = _git_repo(tmp_path / "repo")
    kb.write_board_metadata("default", default_workdir=str(repo))

    tid = kb.create_task(conn, title="Build screen", board="default")
    task = kb.get_task(conn, tid)

    # No explicit project link, yet the task is upgraded to a worktree anchored
    # under the bound repo, keyed on the task id.
    assert task.project_id is None
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == os.path.join(str(repo), ".worktrees", tid)
    # Board links have no deterministic project branch; the worker's wt/<id>
    # fallback owns the branch (materialized at dispatch).
    assert task.branch_name is None


def test_board_bound_repo_materializes_on_resolve(conn, tmp_path):
    repo = _git_repo(tmp_path / "repo")
    kb.write_board_metadata("default", default_workdir=str(repo))

    tid = kb.create_task(conn, title="x", board="default")
    task = kb.get_task(conn, tid)
    workspace, branch = kb._resolve_worktree_workspace(task, board="default")

    assert workspace == (repo / ".worktrees" / tid)
    assert branch == f"wt/{tid}"
    listing = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert str(repo / ".worktrees" / tid) in listing


def test_goal_mode_root_stays_scratch_on_git_backed_board(conn, tmp_path):
    # goal_mode coordination roots produce no code and must not enter the
    # worktree completion gate — they stay scratch even on a git-backed board.
    repo = _git_repo(tmp_path / "repo")
    kb.write_board_metadata("default", default_workdir=str(repo))

    tid = kb.create_task(conn, title="coordinate", board="default", goal_mode=True)
    task = kb.get_task(conn, tid)
    assert task.workspace_kind == "scratch"


def test_explicit_workspace_path_is_respected(conn, tmp_path):
    # An explicit workspace_path must not be overridden by board resolution.
    repo = _git_repo(tmp_path / "repo")
    kb.write_board_metadata("default", default_workdir=str(repo))

    explicit = str(tmp_path / "elsewhere")
    tid = kb.create_task(
        conn, title="x", board="default",
        workspace_kind="dir", workspace_path=explicit,
    )
    task = kb.get_task(conn, tid)
    assert task.workspace_kind == "dir"
    assert task.workspace_path == explicit


def test_non_git_board_default_degrades_to_scratch(conn, tmp_path):
    # default_workdir set to a path that is not inside a git repo (e.g. an
    # unmounted external drive or a plain dir) must degrade to scratch, not
    # crash task creation.
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    kb.write_board_metadata("default", default_workdir=str(plain))

    tid = kb.create_task(conn, title="x", board="default")
    task = kb.get_task(conn, tid)
    assert task.workspace_kind == "scratch"


def test_no_board_default_leaves_scratch(conn):
    # A board with no default_workdir is unchanged: plain scratch task.
    tid = kb.create_task(conn, title="plain", board="default")
    task = kb.get_task(conn, tid)
    assert task.workspace_kind == "scratch"
    assert task.project_id is None
