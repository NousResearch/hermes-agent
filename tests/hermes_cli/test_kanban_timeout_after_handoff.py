"""A worker that exhausts its iteration budget AFTER handing off must not
record ``timed_out`` (which drives a false "timed out; dispatcher will retry"
notification) nor fail a successor's run."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _budget_timeout(conn, task_id, *, expected_run_id=None):
    return kbd._record_task_failure(
        conn, task_id, "Iteration budget exhausted (80/80)",
        outcome="timed_out", release_claim=True, end_run=True,
        event_payload_extra={"budget_used": 80, "budget_max": 80},
        expected_run_id=expected_run_id,
    )


def _kinds(conn, task_id):
    return [e.kind for e in kb.list_events(conn, task_id)]


def _failures(conn, task_id):
    return conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()[0]


@pytest.mark.parametrize("pass_run_id", [True, False])
def test_budget_exhaustion_after_complete_is_a_noop(conn, pass_run_id):
    task_id = kb.create_task(conn, title="finish then run out", assignee="builder")
    claimed = kb.claim_task(conn, task_id, claimer="builder:1")
    run_id = claimed.current_run_id
    assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)

    assert _budget_timeout(conn, task_id, expected_run_id=run_id if pass_run_id else None) is False

    task = kb.get_task(conn, task_id)
    assert task.status == "done"
    assert "timed_out" not in _kinds(conn, task_id)
    assert _failures(conn, task_id) == 0
    run = [r for r in kb.list_runs(conn, task_id) if r.id == run_id][0]
    assert run.outcome == "completed"


def test_stale_worker_cannot_fail_successor_run(conn):
    task_id = kb.create_task(conn, title="reclaimed then respawned", assignee="builder")
    first = kb.claim_task(conn, task_id, claimer="builder:1")
    assert kb.reclaim_task(conn, task_id, reason="operator retry", signal_fn=lambda *_a: None)
    second = kb.claim_task(conn, task_id, claimer="builder:2")
    assert second.current_run_id != first.current_run_id

    assert _budget_timeout(conn, task_id, expected_run_id=first.current_run_id) is False

    task = kb.get_task(conn, task_id)
    assert task.status == "running"
    assert task.current_run_id == second.current_run_id
    assert "timed_out" not in _kinds(conn, task_id)


def test_budget_exhaustion_on_live_own_run_still_records_timeout(conn):
    task_id = kb.create_task(conn, title="really ran out", assignee="builder")
    claimed = kb.claim_task(conn, task_id, claimer="builder:1")

    assert _budget_timeout(conn, task_id, expected_run_id=claimed.current_run_id) is False

    task = kb.get_task(conn, task_id)
    assert task.status == "ready"
    assert task.current_run_id is None
    assert _failures(conn, task_id) == 1
    event = [e for e in kb.list_events(conn, task_id) if e.kind == "timed_out"][-1]
    assert event.run_id == claimed.current_run_id
    assert event.payload["retry_status"] == "ready"
