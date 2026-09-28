"""Durable, event-bound review handoff reads across review attempts and recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _handoff(conn, title, summary, metadata):
    task_id = kb.create_task(conn, title=title, assignee="builder")
    claim = kb.claim_task(conn, task_id)
    assert claim is not None
    assert kb.request_review(
        conn, task_id, summary=summary, metadata=metadata,
        reviewer="reviewer", expected_run_id=claim.current_run_id,
    )
    return task_id, claim.current_run_id


def test_review_handoff_uses_durable_event_and_run_with_bounded_history(conn):
    typed = {"results": [True, 3, None, {"verdict": "PASS"}]}
    task_id, first_run = _handoff(conn, "implementation", "First handoff\nfull detail", typed)
    other_id, other_run = _handoff(conn, "unrelated", "Other", {"ok": False})
    first = kb.latest_review_handoff(conn, task_id)
    assert first is not None
    writes = conn.total_changes
    assert kb.latest_review_handoff(conn, task_id) == first
    assert conn.total_changes == writes
    assert first["task_id"] == task_id
    assert first["run_id"] == first_run
    assert first["summary"] == "First handoff\nfull detail"
    assert first["metadata"] == typed
    assert first["implementer"] == "builder"
    assert first["reviewer"] == "reviewer"
    assert first["event_id"] == kb.list_events(conn, task_id)[-1].id
    assert kb.latest_review_handoff(conn, task_id, before_run_id=first_run) is None
    other = kb.latest_review_handoff(conn, other_id)
    assert other is not None and other["run_id"] == other_run

    review = kb.claim_review_task(conn, task_id)
    assert review is not None
    assert kb.request_changes(conn, task_id, reason="fix", expected_run_id=review.current_run_id)[0]
    retry = kb.claim_task(conn, task_id)
    assert retry is not None
    assert kb.request_review(
        conn, task_id, summary="Corrected", metadata={"iteration": 2},
        expected_run_id=retry.current_run_id,
    )
    newest = kb.latest_review_handoff(conn, task_id)
    assert newest is not None and newest["run_id"] == retry.current_run_id
    assert newest["metadata"] == {"iteration": 2}
    assert kb.latest_review_handoff(conn, task_id, before_run_id=retry.current_run_id) == first
    assert kb.latest_review_handoff(conn, task_id, before_run_id=first_run) is None
    assert kb.latest_review_handoff(conn, "missing") is None

    empty_id = kb.create_task(conn, title="empty review request", assignee="builder")
    assert kb.request_review(conn, empty_id, reviewer="reviewer")
    empty = kb.latest_review_handoff(conn, empty_id)
    assert empty is not None
    assert empty["run_id"] is None and empty["summary"] is None and empty["metadata"] == {}
    assert kb.latest_review_handoff(conn, empty_id, before_run_id=retry.current_run_id) is None


def test_recovered_handoff_is_bound_and_corruption_does_not_fall_back(conn):
    task_id, run_id = _handoff(conn, "recoverable", "First", {"value": 1})
    event_id = kb.list_events(conn, task_id)[-1].id
    receipt = {"task_id": task_id, "run_id": run_id, "profile": "builder"}
    repaired = {
        "summary": "Recovered", "implementer": "builder", "reviewer": "reviewer",
        "recovery": receipt, "handoff_summary": "Recovered\nfull detail",
        "handoff_metadata": {"scores": [2, False]},
    }
    # A blocked-to-review recovery rewrites the closed run and appends one
    # review_requested event in the SAME run; model that persisted shape.
    with kb.write_txn(conn):
        conn.execute("UPDATE task_events SET kind = 'review_request_superseded' WHERE id = ?", (event_id,))
        conn.execute("UPDATE task_runs SET summary = ?, metadata = ? WHERE id = ?", (
            repaired["handoff_summary"], json.dumps(repaired["handoff_metadata"]), run_id,
        ))
        kb._append_event(conn, task_id, "review_requested", repaired, run_id=run_id)
    latest = kb.latest_review_handoff(conn, task_id)
    assert latest is not None
    assert latest["run_id"] == run_id and latest["summary"] == repaired["handoff_summary"]
    assert latest["metadata"] == repaired["handoff_metadata"]
    assert latest["event_id"] != event_id
    assert kb.latest_review_handoff(conn, task_id) == latest
    assert kb.latest_review_handoff(conn, task_id, before_run_id=run_id) is None

    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "review_requested", {**repaired, "recovery": {**receipt, "run_id": True}}, run_id=run_id)
    assert kb.latest_review_handoff(conn, task_id) is None  # no older fallback
    with kb.write_txn(conn):
        conn.execute("UPDATE task_events SET kind = 'review_request_invalid' WHERE id > ? "
                     "AND task_id = ? AND kind = 'review_requested'", (latest["event_id"], task_id))
        conn.execute("UPDATE task_events SET payload = ? WHERE id = ?", ('{"summary": "bad"}', latest["event_id"]))
    assert kb.latest_review_handoff(conn, task_id) is None
    with kb.write_txn(conn):
        conn.execute("UPDATE task_events SET payload = ? WHERE id = ?", (json.dumps(repaired), latest["event_id"]))
        conn.execute("UPDATE tasks SET result = ? WHERE id = ?", ("not a handoff", task_id))
        conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?", ('{"scores": [999]}', run_id))
    assert kb.latest_review_handoff(conn, task_id) is None  # run/event conflict
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?", (
            json.dumps(repaired["handoff_metadata"]), run_id,
        ))
        conn.execute("UPDATE task_events SET run_id = ? WHERE id = ?", (run_id + 100000, latest["event_id"]))
    assert kb.latest_review_handoff(conn, task_id) is None
    with kb.write_txn(conn):
        conn.execute("UPDATE task_events SET run_id = ? WHERE id = ?", (run_id, latest["event_id"]))
        conn.execute("UPDATE task_runs SET task_id = 'other-task' WHERE id = ?", (run_id,))
    assert kb.latest_review_handoff(conn, task_id) is None
