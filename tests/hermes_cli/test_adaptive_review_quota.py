"""Regression coverage for phase-aware review, model pin handoff and quota budget."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from tools import kanban_tools as kt


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def test_review_unpins_author_and_restores_on_requested_changes(board):
    tid = kb.create_task(
        board, title="feature", assignee="coder",
        model_override="gpt-6-luna", provider_override="openai-codex",
    )
    impl = kb.claim_task(board, tid)
    assert impl is not None
    assert kb.request_review(
        board, tid, reviewer="reviewer", summary="Tests passed",
        expected_run_id=impl.current_run_id,
    )
    card = kb.get_task(board, tid)
    assert card.status == "review"
    assert card.assignee == "reviewer"
    assert card.model_override is None
    assert card.provider_override is None
    event = board.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='review_requested' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    provenance = json.loads(event["payload"])
    assert provenance["implementation_model_override"] == "gpt-6-luna"

    review = kb.claim_review_task(board, tid)
    assert review is not None
    result, owner = kb.request_changes(
        board, tid, reason="Please add a test",
        expected_run_id=review.current_run_id,
    )
    assert result and owner == "coder"
    card = kb.get_task(board, tid)
    assert card.assignee == "coder"
    assert card.model_override == "gpt-6-luna"
    assert card.provider_override == "openai-codex"


def test_goal_gate_uses_implementation_phase_only_for_review(monkeypatch):
    seen = []
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)
    def fake_goal(*, goal, last_response):
        seen.append(goal)
        return "done", "ready", False, None, False
    monkeypatch.setattr(kt, "judge_goal", fake_goal)
    task = SimpleNamespace(
        goal_mode=True, title="Ship after independent approval",
        body="Require independent reviewer PASS before final completion.",
    )
    kt._goal_gate("kanban_request_review", task, "t_test", "Tests passed")
    assert "PHASE: IMPLEMENTATION READINESS" in seen[-1]
    assert "START review" in seen[-1]
    kt._goal_gate("kanban_complete", task, "t_test", "Reviewer approved")
    assert "PHASE: IMPLEMENTATION READINESS" not in seen[-1]


def test_quota_streak_is_bounded_and_changes_reset_it(board, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_ATTEMPTS", "3")
    tid = kb.create_task(board, title="quota", assignee="coder")
    now = int(time.time())
    def add_run(profile="coder", outcome="rate_limited", ended_at=None):
        ended = now - 100 if ended_at is None else ended_at
        board.execute(
            "INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,outcome,metadata)"
            " VALUES(?,?,?,?,?,?,?)",
            (tid, profile, outcome, ended-1, ended, outcome,
             json.dumps({"retry_status":"ready"})),
        )
    add_run()
    add_run()
    assert not dispatch._quota_budget_exhausted(board, tid, "coder", "ready")
    add_run()
    assert dispatch._quota_budget_exhausted(board, tid, "coder", "ready")
    assert not dispatch._quota_budget_exhausted(board, tid, "another-profile", "ready")
    # A manual model switch after those attempts clears the old quota streak.
    kb._append_event(board, tid, "model_override_set", {"model":"new"})
    assert not dispatch._quota_budget_exhausted(board, tid, "coder", "ready")


def test_invalid_quota_cap_cannot_unbound_retries(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_ATTEMPTS", "0")
    assert dispatch._rate_limit_attempt_budget() == 1
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_ATTEMPTS", "bogus")
    assert dispatch._rate_limit_attempt_budget() == 3


def test_real_reclaim_parks_after_three_rate_limited_worker_exits(
    board, monkeypatch,
):
    """Real task/run/event state transitions, with only the OS exit mocked."""
    from hermes_cli import kanban_owner_recovery as recovery
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_ATTEMPTS", "3")
    monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)
    monkeypatch.setattr(dispatch, "_worker_alive", lambda *_args, **_kw: False)
    monkeypatch.setattr(recovery, "managed_run_exit_code", lambda *_args: 75)
    tid = kb.create_task(board, title="quota reaper", assignee="coder")
    for attempt in range(1, 4):
        claimed = kb.claim_task(board, tid)
        assert claimed is not None
        board.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
            (919191, int(time.time()) - 120, tid),
        )
        sweep = dispatch._reclaim_dead_workers(board)
        assert tid in sweep.rate_limited
        status = kb.get_task(board, tid).status
        if attempt < 3:
            assert status == "ready"
        else:
            assert status == "blocked"
            runs = board.execute(
                "SELECT outcome FROM task_runs WHERE task_id = ? ORDER BY id",
                (tid,),
            ).fetchall()
            assert [r["outcome"] for r in runs] == ["rate_limited"] * 3
            events = board.execute(
                "SELECT kind FROM task_events WHERE task_id = ? "
                "AND kind = 'quota_retry_budget_exhausted'", (tid,),
            ).fetchall()
            assert len(events) == 1
