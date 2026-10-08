from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

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
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kanban@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kanban Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


def test_create_task_omitted_kind_derives_worktree_from_board_default(kanban_home, tmp_path):
    """Docs-promised inheritance (#69787): a task created with no workspace kind
    on a board whose default_workdir is a git repo lands as ``worktree`` anchored
    on that workdir, not scratch."""
    repo = tmp_path / "proj"
    _init_git_repo(repo)
    kb.write_board_metadata(None, default_workdir=str(repo))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="from-board", assignee="worker")
        t = kb.get_task(conn, tid)
    assert t is not None
    assert t.workspace_kind == "worktree"
    assert t.workspace_path == str(repo)


def test_create_task_omitted_kind_derives_dir_for_plain_dir_default(kanban_home, tmp_path):
    """Board default_workdir that is an existing plain directory → ``dir`` kind
    inheriting that path (#69787)."""
    plain = tmp_path / "notes"
    plain.mkdir()
    kb.write_board_metadata(None, default_workdir=str(plain))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="from-board-dir", assignee="worker")
        t = kb.get_task(conn, tid)
    assert t is not None
    assert t.workspace_kind == "dir"
    assert t.workspace_path == str(plain)


def test_create_task_omitted_kind_stays_scratch_without_board_default(kanban_home, tmp_path):
    """No default_workdir (or a non-directory one) → the disposable scratch
    default is unchanged; scratch never gets a real path (#28818/#30917)."""
    kb.write_board_metadata(None, default_workdir=str(tmp_path / "missing"))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")
        t = kb.get_task(conn, tid)
    assert t is not None
    assert t.workspace_kind == "scratch"
    assert t.workspace_path is None


def test_create_task_explicit_scratch_wins_over_board_default(kanban_home, tmp_path):
    """An explicit ``workspace_kind="scratch"`` is a request for no inheritance:
    even on a git-repo board it must stay a path-less scratch (#30917)."""
    repo = tmp_path / "proj"
    _init_git_repo(repo)
    kb.write_board_metadata(None, default_workdir=str(repo))
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="force-scratch", assignee="worker", workspace_kind="scratch",
        )
        t = kb.get_task(conn, tid)
    assert t is not None
    assert t.workspace_kind == "scratch"
    assert t.workspace_path is None
