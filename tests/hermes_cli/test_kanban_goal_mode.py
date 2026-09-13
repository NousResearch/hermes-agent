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

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import goals



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
    with kbc.connect() as conn:
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
        monkeypatch.setattr("hermes_cli.kanban.kbc.connect_closing", fake_connect_closing)
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

    def test_judge_blocked_verdict_rejects_completion(self, monkeypatch, capsys):
        """#100954: an unachievable goal must not complete silently.

        The judge's ``blocked`` verdict is a refusal, not a completion —
        ``complete_task`` must never run and stderr must steer the user
        toward re-scoping / recording the block.
        """
        rc, complete_calls = self._run(
            monkeypatch,
            verdict="blocked",
            reason="the target repository does not exist",
        )
        err = capsys.readouterr().err
        assert rc != 0, "blocked verdict must reject the completion"
        assert complete_calls == [], "an unachievable goal must never reach complete_task"
        assert "unachievable" in err.lower()
        assert "kanban block" in err.lower()


# ---------------------------------------------------------------------------
# Verifier / goal-loop budget default (t_bb181168)
# ---------------------------------------------------------------------------


class TestVerifierGoalBudgetDefault:
    """Spawn-time default ``goal_max_turns=40`` for verifier-style cards.

    Verifier-style = goal_mode AND ``goal_max_turns`` NULL AND
    (assignee==\"reviewer\" OR title starts with Verify / Re-verify / Review).
    Default applies via a one-shot UPDATE gated on ``goal_max_turns IS
    NULL`` so explicit overrides and idempotency across ticks are
    preserved.
    """

    def _create_task(self, conn, *, title, assignee, goal_mode=True, goal_max_turns=None):
        return kb.create_task(
            conn,
            title=title,
            body="x",
            assignee=assignee,
            goal_mode=goal_mode,
            goal_max_turns=goal_max_turns,
        )

    def test_reviewer_assignee_gets_default(self, kanban_home):
        from hermes_cli.kanban_db_dispatch import (
            _apply_default_verifier_goal_max_turns,
            _DEFAULT_VERIFIER_GOAL_MAX_TURNS,
        )

        with kbc.connect() as conn:
            tid = self._create_task(conn, title="Ship the patch", assignee="reviewer")
            task = kb.get_task(conn, tid)
            assert task.goal_max_turns is None
            effective = _apply_default_verifier_goal_max_turns(conn, task)
            assert effective == _DEFAULT_VERIFIER_GOAL_MAX_TURNS == 40
            assert kb.get_task(conn, tid).goal_max_turns == 40

    def test_verify_title_prefix_gets_default(self, kanban_home):
        from hermes_cli.kanban_db_dispatch import _apply_default_verifier_goal_max_turns

        with kbc.connect() as conn:
            tid = self._create_task(conn, title="Verify auth flow", assignee="engineer")
            task = kb.get_task(conn, tid)
            assert _apply_default_verifier_goal_max_turns(conn, task) == 40
            assert kb.get_task(conn, tid).goal_max_turns == 40

    def test_re_verify_title_prefix_gets_default(self, kanban_home):
        from hermes_cli.kanban_db_dispatch import _apply_default_verifier_goal_max_turns

        with kbc.connect() as conn:
            tid = self._create_task(
                conn, title="Re-verify regression suite", assignee="engineer"
            )
            task = kb.get_task(conn, tid)
            assert _apply_default_verifier_goal_max_turns(conn, task) == 40
            assert kb.get_task(conn, tid).goal_max_turns == 40

    def test_review_title_prefix_gets_default(self, kanban_home):
        from hermes_cli.kanban_db_dispatch import _apply_default_verifier_goal_max_turns

        with kbc.connect() as conn:
            tid = self._create_task(
                conn, title="Review the PR diff", assignee="engineer"
            )
            task = kb.get_task(conn, tid)
            assert _apply_default_verifier_goal_max_turns(conn, task) == 40
            assert kb.get_task(conn, tid).goal_max_turns == 40

    def test_explicit_override_is_not_stomped(self, kanban_home):
        from hermes_cli.kanban_db_dispatch import _apply_default_verifier_goal_max_turns

        with kbc.connect() as conn:
            tid = self._create_task(
                conn,
                title="Verify auth flow",
                assignee="reviewer",
                goal_max_turns=12,
            )
            task = kb.get_task(conn, tid)
            assert _apply_default_verifier_goal_max_turns(conn, task) == 12
            assert kb.get_task(conn, tid).goal_max_turns == 12

    def test_non_goal_mode_skipped(self, kanban_home):
        from hermes_cli.kanban_db_dispatch import _apply_default_verifier_goal_max_turns

        with kbc.connect() as conn:
            tid = self._create_task(
                conn,
                title="Verify something",
                assignee="reviewer",
                goal_mode=False,
            )
            task = kb.get_task(conn, tid)
            assert _apply_default_verifier_goal_max_turns(conn, task) is None
            assert kb.get_task(conn, tid).goal_max_turns is None

    def test_non_matching_goal_mode_left_null(self, kanban_home):
        from hermes_cli.kanban_db_dispatch import _apply_default_verifier_goal_max_turns

        with kbc.connect() as conn:
            tid = self._create_task(
                conn,
                title="Investigate flaky test",
                assignee="engineer",
            )
            task = kb.get_task(conn, tid)
            assert _apply_default_verifier_goal_max_turns(conn, task) is None
            assert kb.get_task(conn, tid).goal_max_turns is None

    def test_idempotent_on_second_call(self, kanban_home):
        from hermes_cli.kanban_db_dispatch import _apply_default_verifier_goal_max_turns

        with kbc.connect() as conn:
            tid = self._create_task(conn, title="Verify login", assignee="reviewer")
            task = kb.get_task(conn, tid)
            assert _apply_default_verifier_goal_max_turns(conn, task) == 40
            task2 = kb.get_task(conn, tid)
            assert _apply_default_verifier_goal_max_turns(conn, task2) == 40
            assert kb.get_task(conn, tid).goal_max_turns == 40

    def test_spawned_task_carries_default_through_dispatch_once(
        self, kanban_home, all_assignees_spawnable, monkeypatch, caplog
    ):
        """Claim + spawn path stamps 40 and logs ``goal_budget=40``."""
        import logging

        from hermes_cli import kanban_db_dispatch as kbd

        monkeypatch.setattr(
            kbd, "_system_memory_sample",
            lambda: {"mem_available_kib": 8 * 1024 * 1024, "mem_total_kib": 16 * 1024 * 1024},
        )

        seen: list[tuple[str, object]] = []

        def _stub_spawn(task, workspace, board=None):
            seen.append((task.id, task.goal_max_turns))
            return 4242

        caplog.set_level(logging.INFO)
        with kbc.connect() as conn:
            tid = kb.create_task(
                conn,
                title="Verify smoke pack",
                assignee="reviewer",
                goal_mode=True,
            )
            res = kbd.dispatch_once(conn, spawn_fn=_stub_spawn)
        assert tid in [row[0] for row in res.spawned]
        assert seen == [(tid, 40)]
        with kbc.connect() as conn:
            assert kb.get_task(conn, tid).goal_max_turns == 40
        assert any(
            "goal_budget=40" in rec.getMessage() and tid in rec.getMessage()
            for rec in caplog.records
        )

    def test_explicit_override_survives_dispatch_once(
        self, kanban_home, all_assignees_spawnable, monkeypatch
    ):
        from hermes_cli import kanban_db_dispatch as kbd

        monkeypatch.setattr(
            kbd, "_system_memory_sample",
            lambda: {"mem_available_kib": 8 * 1024 * 1024, "mem_total_kib": 16 * 1024 * 1024},
        )
        seen: list[tuple[str, object]] = []

        def _stub_spawn(task, workspace, board=None):
            seen.append((task.id, task.goal_max_turns))
            return 4242

        with kbc.connect() as conn:
            tid = kb.create_task(
                conn,
                title="Verify smoke pack",
                assignee="reviewer",
                goal_mode=True,
                goal_max_turns=80,
            )
            res = kbd.dispatch_once(conn, spawn_fn=_stub_spawn)
        assert tid in [row[0] for row in res.spawned]
        assert seen == [(tid, 80)]
        with kbc.connect() as conn:
            assert kb.get_task(conn, tid).goal_max_turns == 80
