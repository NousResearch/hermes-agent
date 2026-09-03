"""Tests: `kanban reclaim` refuses a dirty worktree unless --force (issue #101788).

Reclaiming a task whose worktree holds uncommitted work lets the next
takeover rebase/reset it away. The CLI guard lists what is at risk and
refuses; --force overrides. Inspection failures are fail-clean (proceed).
The dispatcher's automatic reclaim calls reclaim_task directly and keeps
its behavior.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb

needs_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary required"
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


def _git(*args, cwd):
    subprocess.run(
        ["git", *args], cwd=cwd, check=True,
        capture_output=True, timeout=30,
    )


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", cwd=path)
    _git("config", "user.email", "test@example.com", cwd=path)
    _git("config", "user.name", "test", cwd=path)
    (path / "base.txt").write_text("base\n", encoding="utf-8")
    _git("add", "base.txt", cwd=path)
    _git("commit", "-m", "base", cwd=path)
    return path


def _running_task_with_worktree(conn, tree: Path) -> str:
    tid = kb.create_task(conn, title="takeover", assignee="w")
    conn.execute(
        "UPDATE tasks SET workspace_kind='worktree', workspace_path=? WHERE id=?",
        (str(tree), tid),
    )
    conn.commit()
    kb.claim_task(conn, tid, claimer="test-host:worker")
    return tid


def _reclaim_ns(task_id, *, force=False):
    return argparse.Namespace(task_id=task_id, reason=None, force=force)


@needs_git
def test_reclaim_clean_worktree_succeeds(kanban_home, tmp_path, capsys):
    tree = _make_repo(tmp_path / "tree-clean")
    with kb.connect() as conn:
        tid = _running_task_with_worktree(conn, tree)
    rc = kb_cli._cmd_reclaim(_reclaim_ns(tid))
    assert rc == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status != "running"


@needs_git
def test_reclaim_dirty_worktree_refuses_and_keeps_claim(
    kanban_home, tmp_path, capsys
):
    tree = _make_repo(tmp_path / "tree-dirty")
    (tree / "wip.txt").write_text("uncommitted fix\n", encoding="utf-8")
    (tree / "base.txt").write_text("base\nmodified\n", encoding="utf-8")
    with kb.connect() as conn:
        tid = _running_task_with_worktree(conn, tree)
    rc = kb_cli._cmd_reclaim(_reclaim_ns(tid))
    assert rc == 1
    err = capsys.readouterr().err
    assert "wip.txt" in err
    assert "--force" in err
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.claim_lock is not None


@needs_git
def test_reclaim_dirty_worktree_force_overrides(kanban_home, tmp_path, capsys):
    tree = _make_repo(tmp_path / "tree-force")
    (tree / "wip.txt").write_text("uncommitted fix\n", encoding="utf-8")
    with kb.connect() as conn:
        tid = _running_task_with_worktree(conn, tree)
    rc = kb_cli._cmd_reclaim(_reclaim_ns(tid, force=True))
    assert rc == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status != "running"


@needs_git
def test_reclaim_scratch_task_skips_guard(kanban_home, capsys):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="scratch", assignee="w")
        kb.claim_task(conn, tid, claimer="test-host:worker")
    rc = kb_cli._cmd_reclaim(_reclaim_ns(tid))
    assert rc == 0


@needs_git
def test_reclaim_missing_worktree_path_fails_clean(kanban_home, tmp_path, capsys):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ghost", assignee="w")
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', workspace_path=? WHERE id=?",
            (str(tmp_path / "does-not-exist"), tid),
        )
        conn.commit()
        kb.claim_task(conn, tid, claimer="test-host:worker")
    rc = kb_cli._cmd_reclaim(_reclaim_ns(tid))
    assert rc == 0
