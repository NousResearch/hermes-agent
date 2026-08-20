"""Tests for kanban goal_mode — per-card Ralph-style goal loop.

Covers three layers:

1. DB: goal_mode / goal_max_turns persist through create_task + from_row,
   and a legacy DB (without the columns) migrates cleanly.
2. Spawn: _default_spawn sets the HERMES_KANBAN_GOAL_MODE env vars only
   when the card opts in.
3. Loop: goals.run_kanban_goal_loop continuation / completion / budget
   behaviour, driven entirely through injected callbacks (no live model).
"""

from __future__ import annotations

import subprocess
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import goals
import cli


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# DB layer
# ---------------------------------------------------------------------------





def test_legacy_db_migrates_goal_columns(tmp_path, monkeypatch):
    """A tasks table created without goal columns must gain them on init."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal legacy schema: tasks table missing goal_mode / goal_max_turns.
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind) "
        "VALUES ('legacy1', 'old', 'ready', 0, 1, 'scratch')"
    )
    legacy.commit()
    legacy.close()

    # init_db runs the additive migration.
    kb.init_db()
    with kb.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "goal_mode" in cols
        assert "goal_max_turns" in cols
        task = kb.get_task(conn, "legacy1")
    # Existing row keeps the safe default.
    assert task.goal_mode is False
    assert task.goal_max_turns is None


# ---------------------------------------------------------------------------
# Spawn env
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Goal loop logic (callback-injected, no live model)
# ---------------------------------------------------------------------------

def _patch_judge(monkeypatch, verdicts):
    """Make judge_goal return a scripted sequence of verdicts."""
    seq = list(verdicts)

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        v = seq.pop(0) if seq else "done"
        # 5-tuple contract: verdict, reason, parse failure, wait, transport failure.
        return v, f"scripted:{v}", False, None, False

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)


def test_loop_stops_when_worker_already_completed(monkeypatch):
    # Worker called kanban_complete on its first turn — no judging needed.
    _patch_judge(monkeypatch, ["continue"])  # should never be consulted
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "x",
        task_status_fn=lambda: "done",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="done already",
    )
    assert res["outcome"] == "completed_by_worker"
    assert turns == []  # no extra turns


def test_loop_blocks_at_budget_without_progress_signal(monkeypatch):
    """Fail-closed: no progress_check_fn (or a falsy result) blocks exactly
    like before this feature existed — no silent extension."""
    _patch_judge(monkeypatch, ["continue"])
    blocked = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: pytest.fail("should not run another turn"),
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.append(r),
        max_turns=1,
        first_response="working on it",
        progress_check_fn=lambda: False,
    )
    assert res["outcome"] == "blocked_budget"
    assert res["turns_used"] == 1
    assert len(blocked) == 1


# ---------------------------------------------------------------------------
# cli.py's real progress_check_fn wiring (heartbeat is NOT a progress
# signal — see #81990 review; git HEAD movement is what actually gets
# checked for workspace-backed tasks).
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path) -> None:
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test"],
    ):
        subprocess.run(args, cwd=path, check=True, capture_output=True)


def _git_commit(path: Path, filename: str, content: str) -> None:
    (path / filename).write_text(content)
    subprocess.run(["git", "add", filename], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", f"add {filename}"], cwd=path, check=True, capture_output=True,
    )


def test_workspace_git_head_detects_new_commit(tmp_path):
    """The real signal cli.py wires in: HEAD moving means a commit landed —
    unlike a heartbeat, this can't be true for a worker that is merely
    alive and thrashing without producing anything."""
    _init_git_repo(tmp_path)
    _git_commit(tmp_path, "a.txt", "1")
    initial_head = cli._kanban_workspace_git_head(str(tmp_path))
    assert initial_head

    # No new commit yet — HEAD hasn't moved, so there is no progress signal.
    assert cli._kanban_workspace_git_head(str(tmp_path)) == initial_head

    _git_commit(tmp_path, "b.txt", "2")
    new_head = cli._kanban_workspace_git_head(str(tmp_path))
    assert new_head and new_head != initial_head


def test_workspace_git_head_none_for_non_git_dir(tmp_path):
    (tmp_path / "not_a_repo").mkdir()
    assert cli._kanban_workspace_git_head(str(tmp_path / "not_a_repo")) is None


def test_workspace_git_head_none_for_missing_dir(tmp_path):
    assert cli._kanban_workspace_git_head(str(tmp_path / "does_not_exist")) is None


def test_goal_max_turns_ceiling_leaves_room_for_extension_above_configured_budget():
    """A card whose goal_max_turns already meets or exceeds the default
    ceiling must still get at least one extension's worth of headroom,
    or the extension mechanism is a no-op for it (#81990 review: a card
    budgeted at exactly the 200-turn default ceiling was blocked before
    the progress signals ever ran)."""
    from hermes_cli.goals import DEFAULT_GOAL_EXTENSION_TURNS, DEFAULT_GOAL_MAX_TURNS_CEILING

    assert cli._kanban_goal_max_turns_ceiling(DEFAULT_GOAL_MAX_TURNS_CEILING) == (
        DEFAULT_GOAL_MAX_TURNS_CEILING + DEFAULT_GOAL_EXTENSION_TURNS
    )
    assert cli._kanban_goal_max_turns_ceiling(DEFAULT_GOAL_MAX_TURNS_CEILING + 50) == (
        DEFAULT_GOAL_MAX_TURNS_CEILING + 50 + DEFAULT_GOAL_EXTENSION_TURNS
    )
    # Small configured budgets keep the default ceiling unchanged.
    assert cli._kanban_goal_max_turns_ceiling(20) == DEFAULT_GOAL_MAX_TURNS_CEILING


def test_loop_extends_budget_when_progress_detected(monkeypatch):
    """A worker that hits its turn budget but is showing observable
    progress gets a bounded extension instead of an immediate block, but
    never past ``max_turns_ceiling``."""
    _patch_judge(monkeypatch, ["continue"] * 5)
    turns = []
    blocked = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "still working",
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.append(r),
        max_turns=1,
        first_response="working on it",
        progress_check_fn=lambda: True,
        extension_turns=1,
        max_turns_ceiling=2,
    )
    assert res["outcome"] == "blocked_budget"
    assert res["turns_used"] == 2  # one extended turn was granted, then the ceiling hit
    assert len(turns) == 1
    assert len(blocked) == 1






# ---------------------------------------------------------------------------
# CLI judge gate tests (hermes kanban complete bypass fix)
# ---------------------------------------------------------------------------

class TestCLIJudgeGate:
    """hermes kanban complete must apply the same goal_mode judge gate as the
    kanban_complete tool (Issue #38367 sibling gap).

    Uses mocks for kb.get_task and kb.complete_task to avoid depending on the
    full kanban_db schema; the gate logic is the unit under test.
    """

    def _run(self, monkeypatch, *, goal_mode=True, judge_available=True,
             verdict="done", reason="", complete_ok=True, summary="done"):
        import argparse
        import types
        from unittest.mock import MagicMock
        from hermes_cli.kanban import _cmd_complete

        fake_task = types.SimpleNamespace(
            goal_mode=goal_mode,
            title="Finish report",
            body="acceptance: criteria",
        )
        fake_conn = MagicMock()
        complete_calls: list = []

        def fake_connect_closing():
            from contextlib import contextmanager
            @contextmanager
            def _cm():
                yield fake_conn
            return _cm()

        def fake_complete_task(conn, tid, **kw):
            complete_calls.append(tid)
            return complete_ok

        monkeypatch.setattr("hermes_cli.kanban.kb.get_task", lambda conn, tid: fake_task)
        monkeypatch.setattr("hermes_cli.kanban.kb.complete_task", fake_complete_task)
        monkeypatch.setattr("hermes_cli.kanban.kb.connect_closing", fake_connect_closing)
        monkeypatch.setattr("hermes_cli.kanban._worker_run_id_for", lambda _: None)

        _aux_client = (object(), "judge-model") if judge_available else (None, None)
        monkeypatch.setattr(
            "agent.auxiliary_client.get_text_auxiliary_client",
            lambda name: _aux_client,
        )
        # Match the real judge_goal contract:
        # (verdict, reason, parse_failed, wait_directive, transport_failed)
        monkeypatch.setattr(
            "hermes_cli.goals.judge_goal",
            lambda **kw: (verdict, reason, False, None, False),
        )

        args = argparse.Namespace(task_ids=["t1"], summary=summary, result=None, metadata=None)
        return _cmd_complete(args), complete_calls

    def test_judge_rejects_premature_completion(self, monkeypatch):
        rc, complete_calls = self._run(
            monkeypatch, verdict="continue", reason="criteria not met"
        )
        assert rc != 0, "judge rejection must produce non-zero exit code"
        assert complete_calls == [], (
            "complete_task must NOT be invoked when the judge rejects"
        )


    def test_non_goal_mode_task_skips_gate(self, monkeypatch):
        """Plain (non-goal_mode) tasks are never sent to the judge."""
        rc, complete_calls = self._run(monkeypatch, goal_mode=False)
        assert rc == 0
        assert complete_calls == ["t1"]
