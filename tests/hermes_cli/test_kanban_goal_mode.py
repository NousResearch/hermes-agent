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


def test_loop_stops_when_worker_already_blocked(monkeypatch):
    _patch_judge(monkeypatch, ["done"])  # should never be consulted
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "x",
        task_status_fn=lambda: "blocked",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="blocked for review",
    )
    assert res["outcome"] == "blocked_by_worker"
    assert turns == []


def test_loop_skips_synthetic_block_if_worker_blocks_on_finalize_nudge(monkeypatch):
    """Worker calls kanban_block during the finalize-nudge turn.

    The loop must treat that as terminal and must not call block_fn.
    """
    _patch_judge(monkeypatch, ["done", "done"])
    state = {"status": "running"}
    turns = []

    def _run_turn(prompt):
        turns.append(prompt)
        state["status"] = "blocked"
        return "called kanban_block for review"

    blocked = []
    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=_run_turn,
        task_status_fn=lambda: state["status"],
        block_fn=blocked.append,
        first_response="looks complete",
        max_turns=5,
    )
    assert res["outcome"] == "blocked_by_worker"
    assert len(turns) == 1
    assert blocked == []


def test_loop_still_blocks_when_worker_never_calls_terminal(monkeypatch):
    """Holdout: worker exits without kanban_complete/block → synthetic block."""
    _patch_judge(monkeypatch, ["done", "done"])
    turns = []
    blocked = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "forgot to call the tool",
        task_status_fn=lambda: "running",
        block_fn=blocked.append,
        first_response="looks complete",
        max_turns=5,
    )
    assert res["outcome"] == "blocked_budget"
    assert res["reason"] == "judged done, never finalized"
    assert len(turns) == 1
    assert blocked and "never called kanban_complete after a finalize nudge" in blocked[0]


def _goal_loop_block_fn(task_id, expected_run_id):
    """Mirror cli.py's goal-loop _block wrapper (idempotent finalizer)."""

    def _block(reason: str) -> None:
        with kb.connect() as conn:
            if kb.goal_run_already_terminal(conn, task_id, expected_run_id):
                return
            kb.block_task(
                conn,
                task_id,
                reason=reason,
                expected_run_id=expected_run_id,
            )

    return _block


def test_finalize_does_not_overwrite_worker_dependency_block(kanban_home, monkeypatch):
    """Production bug: kanban_block(kind=dependency) then promote-to-ready.

    Replay cards t_8812f1d4 / t_b121dde5: the worker recorded a
    REVIEW-REQUIRED block, the dispatcher promoted the card back to
    ready, and the goal-mode finalizer synthesized a second run that
    overwrote the original summary/kind. Finalization must no-op.
    """
    review_reason = "REVIEW-REQUIRED: merge the repair PR before redeploy"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Daily production deployment",
            body="Ship the daily deploy or block for human review.",
            assignee="replay-agent",
            goal_mode=True,
        )
        claimed = kb.claim_task(conn, tid, claimer="replay-agent:test")
        assert claimed is not None
        run_id = claimed.current_run_id
        assert kb.block_task(
            conn,
            tid,
            reason=review_reason,
            kind="dependency",
            expected_run_id=run_id,
        )
        kb.recompute_ready(conn)
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status in ("ready", "todo")
        assert kb.goal_run_status(conn, tid, run_id) == "blocked"
        assert kb.goal_run_already_terminal(conn, tid, run_id)
        # Legacy wrapper with no run id must still see the ended terminal run.
        assert kb.goal_run_already_terminal(conn, tid, None)

    _patch_judge(monkeypatch, ["done", "done"])
    blocked_reasons: list[str] = []
    inner_block = _goal_loop_block_fn(tid, run_id)

    def _block(reason: str) -> None:
        blocked_reasons.append(reason)
        inner_block(reason)

    def _status():
        with kb.connect() as conn:
            return kb.goal_run_status(conn, tid, run_id)

    res = goals.run_kanban_goal_loop(
        task_id=tid,
        goal_text="Daily production deployment\n\nShip the daily deploy or block for human review.",
        run_turn=lambda p: pytest.fail("must not run another turn after a terminal block"),
        task_status_fn=_status,
        block_fn=_block,
        first_response="blocked pending human review of the repair PR",
        max_turns=5,
    )
    assert res["outcome"] == "blocked_by_worker"
    assert blocked_reasons == []

    with kb.connect() as conn:
        runs = list(
            conn.execute(
                "SELECT id, outcome, summary FROM task_runs WHERE task_id = ? ORDER BY id",
                (tid,),
            )
        )
        assert len(runs) == 1
        assert runs[0]["outcome"] == "blocked"
        assert runs[0]["summary"] == review_reason
        latest = kb.latest_run(conn, tid)
        assert latest is not None
        assert latest.summary == review_reason
        assert "never called kanban_complete" not in (latest.summary or "")


def test_legacy_finalize_wrapper_does_not_synthesize_over_ended_block(kanban_home):
    """Missing HERMES_KANBAN_RUN_ID must still refuse a synthetic replacement."""
    review_reason = "REVIEW-REQUIRED: human eyes please"
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs review", assignee="worker")
        claimed = kb.claim_task(conn, tid, claimer="worker:test")
        assert claimed is not None
        assert kb.block_task(
            conn,
            tid,
            reason=review_reason,
            kind="needs_input",
            expected_run_id=claimed.current_run_id,
        )
        kb.recompute_ready(conn)

    # Old cli.py wrapper: expected_run_id=None, no already-terminal guard.
    # That path synthesized a zero-duration run and overwrote the summary.
    # The new guard must no-op even without a run id.
    with kb.connect() as conn:
        assert kb.goal_run_already_terminal(conn, tid, None)
        before = kb.latest_run(conn, tid)
        assert before is not None
        before_id = before.id
        before_task = kb.get_task(conn, tid)
        assert before_task is not None
        before_kind = before_task.block_kind

    _goal_loop_block_fn(tid, None)(
        "Goal-mode worker's output looked complete but it never "
        "called kanban_complete after a finalize nudge (looks done)."
    )

    with kb.connect() as conn:
        runs = list(
            conn.execute(
                "SELECT id, summary FROM task_runs WHERE task_id = ? ORDER BY id",
                (tid,),
            )
        )
        assert len(runs) == 1
        assert runs[0]["id"] == before_id
        assert runs[0]["summary"] == review_reason
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.block_kind == before_kind


def test_finalize_still_blocks_open_run_without_terminal_call(kanban_home, monkeypatch):
    """Holdout: no worker terminal action → synthetic block still lands."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Finish the report",
            body="acceptance: write the report",
            assignee="worker",
            goal_mode=True,
        )
        claimed = kb.claim_task(conn, tid, claimer="worker:test")
        assert claimed is not None
        run_id = claimed.current_run_id
        assert not kb.goal_run_already_terminal(conn, tid, run_id)

    _patch_judge(monkeypatch, ["done", "done"])
    blocked = []
    inner_block = _goal_loop_block_fn(tid, run_id)

    def _block(reason: str) -> None:
        blocked.append(reason)
        inner_block(reason)

    def _status():
        with kb.connect() as conn:
            return kb.goal_run_status(conn, tid, run_id)

    res = goals.run_kanban_goal_loop(
        task_id=tid,
        goal_text="Finish the report\n\nacceptance: write the report",
        run_turn=lambda p: "I think I'm done but I forgot the tool",
        task_status_fn=_status,
        block_fn=_block,
        first_response="the report looks finished",
        max_turns=5,
    )
    assert res["outcome"] == "blocked_budget"
    assert blocked and "never called kanban_complete" in blocked[0]
    with kb.connect() as conn:
        latest = kb.latest_run(conn, tid)
        assert latest is not None
        assert latest.outcome == "blocked"
        assert "never called kanban_complete" in (latest.summary or "")






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
