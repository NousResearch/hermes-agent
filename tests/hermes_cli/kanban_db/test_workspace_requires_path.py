"""A persistent-workspace task is refused at create when no path can be built.

``dir``/``worktree`` used to accept a missing ``workspace_path`` on a board with
no ``default_workdir``; the card was created, failed to spawn twice, and landed
in ``gave_up`` — which reads downstream as two abnormal exits instead of as a
misconfiguration. ``resolve_workspace_path`` now refuses it at create time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import sqlite_support  # registers the sqlite3 shim other DB tests need
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    run = lambda *args: subprocess.run(args, check=True, capture_output=True, text=True)
    run("git", "init", "-b", "main", str(repo))
    run("git", "-C", str(repo), "config", "user.email", "kanban@example.com")
    run("git", "-C", str(repo), "config", "user.name", "Kanban Test")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    run("git", "-C", str(repo), "add", "README.md")
    run("git", "-C", str(repo), "commit", "-m", "init")


def test_worktree_task_without_path_is_rejected_at_create(kanban_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="needs a workspace"):
            kb.create_task(conn, title="spawn die", workspace_kind="worktree")
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_dir_task_without_path_is_rejected_at_create(kanban_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="workspace_kind='dir' needs a workspace"):
            kb.create_task(conn, title="spawn die", workspace_kind="dir")
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_scratch_task_needs_no_workspace_path(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no repo", workspace_kind="scratch")
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "scratch"
    assert not task.workspace_path


def test_worktree_task_with_explicit_path_is_still_created(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="with repo", workspace_kind="worktree",
                             workspace_path=str(repo))
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(repo)


def test_worktree_task_inherits_the_board_default_workdir(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.write_board_metadata("default", default_workdir=str(repo))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="inherits default", workspace_kind="worktree")
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_path == str(repo)
