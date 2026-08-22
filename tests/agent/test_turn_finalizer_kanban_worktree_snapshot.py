"""Kanban worktree snapshot on iteration-budget exhaustion (#73234).

When a kanban worker dies on budget exhaustion, whatever it left
staged/uncommitted in ``$HERMES_KANBAN_WORKSPACE`` must be snapshotted
into ONE durable task comment (worktree path, changed-file list, diff
stat, handoff hint) BEFORE the failure record releases the claim.
Clean worktrees and non-git workspaces stay zero-noise, and any
snapshot failure must never break the exit path.

Models the worker-path patterns of
``tests/hermes_cli/test_kanban_goal_mode.py`` (temp HERMES_HOME +
real kanban DB) and ``test_turn_finalizer_iteration_limit_exit.py``
(stub agent driving ``finalize_turn`` — no live model).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent.turn_finalizer import (
    _KANBAN_WORKTREE_SNAPSHOT_PREFIX,
    finalize_turn,
)
from tests.agent.test_turn_finalizer_iteration_limit_exit import _LimitAgent


@pytest.fixture
def kanban_task_id(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + real kanban DB containing ONE fresh card.

    Returns the card id — tests point ``HERMES_KANBAN_TASK`` at it.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    kb.init_db()
    with kb.connect() as conn:
        return kb.create_task(conn, title="snapshot me")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture
def git_repo(tmp_path):
    """A minimal git repo usable as a worker workspace."""
    repo = tmp_path / "ws"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "worker@example.com")
    _git(repo, "config", "user.name", "worker")
    (repo / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _exhaust_budget(monkeypatch):
    """Drive a budget-exhausted worker turn through the real finalizer."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    return finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
    )


def _snapshot_bodies(conn, task_id):
    from hermes_cli import kanban_db as kb
    bodies = [
        c.body for c in kb.list_comments(conn, task_id)
        if c.body.startswith(_KANBAN_WORKTREE_SNAPSHOT_PREFIX)
    ]
    return bodies


def _parse_snapshot(body: str) -> dict:
    first_line = body.splitlines()[0]
    return json.loads(first_line[len(_KANBAN_WORKTREE_SNAPSHOT_PREFIX):])


def test_dirty_worktree_snapshots_one_comment(kanban_task_id, git_repo, monkeypatch):
    """Budget exhaustion on a dirty worktree appends exactly ONE durable
    comment carrying the machine-readable recovery summary."""
    from hermes_cli import kanban_db as kb

    # One tracked modification + one staged new file + one untracked file.
    (git_repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    (git_repo / "staged.py").write_text("y = 1\n", encoding="utf-8")
    _git(git_repo, "add", "staged.py")
    (git_repo / "notes.md").write_text("wip\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(git_repo))

    result = _exhaust_budget(monkeypatch)

    assert result["completed"] is False
    with kb.connect() as conn:
        bodies = _snapshot_bodies(conn, kanban_task_id)
    assert len(bodies) == 1, "exactly ONE snapshot comment per exhaustion"
    body = bodies[0]

    payload = _parse_snapshot(body)
    assert payload["type"] == "worktree_snapshot"
    assert payload["reason"] == "iteration_budget_exhausted"
    assert payload["workspace"] == str(git_repo)
    assert payload["changed_files"] == 3
    assert payload["staged_files"] == 1
    assert payload["untracked_files"] == 1
    assert payload["files_truncated"] is False
    assert sorted(payload["files"]) == ["notes.md", "staged.py", "tracked.py"]
    assert payload["budget_used"] == 60
    assert payload["budget_max"] == 60
    assert isinstance(payload["branch"], str)

    # Human handoff notes ride along with the machine-readable header.
    assert "iteration budget (60/60)" in body
    assert "Handoff:" in body
    # Tracked diff stat covers the modified file.
    assert "```text" in body
    assert "tracked.py" in body


def test_clean_worktree_stays_silent(kanban_task_id, git_repo, monkeypatch):
    """A clean worktree produces NO snapshot comment — zero noise."""
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(git_repo))

    result = _exhaust_budget(monkeypatch)

    assert isinstance(result, dict)
    with kb.connect() as conn:
        assert _snapshot_bodies(conn, kanban_task_id) == []


def test_non_git_workspace_fails_safe(kanban_task_id, tmp_path, monkeypatch):
    """A scratch (non-git) workspace must not raise and must not block the
    terminal failure record — the dispatcher contract still runs."""
    from unittest.mock import MagicMock

    from hermes_cli import kanban_db as kb

    scratch = tmp_path / "scratch-ws"
    scratch.mkdir()

    monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(scratch))
    record = MagicMock(name="record_task_failure")
    monkeypatch.setattr(kb, "_record_task_failure", record)

    result = _exhaust_budget(monkeypatch)

    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    record.assert_called_once()
    args, kwargs = record.call_args
    assert args[1] == kanban_task_id
    assert kwargs["outcome"] == "timed_out"
    with kb.connect() as conn:
        assert _snapshot_bodies(conn, kanban_task_id) == []


def test_snapshot_failure_never_breaks_exit_path(
    kanban_task_id, git_repo, monkeypatch
):
    """If appending the snapshot comment blows up, the exit path still
    completes normally and the failure record still lands."""
    from unittest.mock import MagicMock

    from hermes_cli import kanban_db as kb

    (git_repo / "tracked.py").write_text("x = 3\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(git_repo))
    record = MagicMock(name="record_task_failure")
    monkeypatch.setattr(kb, "_record_task_failure", record)

    def _boom(*_a, **_kw):
        raise RuntimeError("db locked")

    monkeypatch.setattr(kb, "add_comment", _boom)

    result = _exhaust_budget(monkeypatch)

    assert isinstance(result, dict)
    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    record.assert_called_once()


def test_file_list_capped_with_truncation_flag(
    kanban_task_id, git_repo, monkeypatch
):
    """More changed files than the cap → list capped, count exact, flag set."""
    from hermes_cli import kanban_db as kb
    from agent.turn_finalizer import _KANBAN_SNAPSHOT_MAX_FILES

    for i in range(_KANBAN_SNAPSHOT_MAX_FILES + 5):
        (git_repo / f"f{i:03}.txt").write_text(f"{i}\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(git_repo))

    _exhaust_budget(monkeypatch)

    with kb.connect() as conn:
        bodies = _snapshot_bodies(conn, kanban_task_id)
    assert len(bodies) == 1
    payload = _parse_snapshot(bodies[0])
    assert payload["changed_files"] == _KANBAN_SNAPSHOT_MAX_FILES + 5
    assert len(payload["files"]) == _KANBAN_SNAPSHOT_MAX_FILES
    assert payload["files_truncated"] is True


def test_no_workspace_env_no_snapshot(kanban_task_id, monkeypatch):
    """No HERMES_KANBAN_WORKSPACE → nothing to inspect, no comment."""
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task_id)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)

    result = _exhaust_budget(monkeypatch)

    assert isinstance(result, dict)
    with kb.connect() as conn:
        assert _snapshot_bodies(conn, kanban_task_id) == []
