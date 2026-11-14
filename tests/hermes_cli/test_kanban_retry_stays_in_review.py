"""A crash after review must never restart implementation work.

``_retry_status_for_run`` decides the status a dead run is retried into. The
``claimed`` event of the dying run is the primary witness (``source_status=
review``), but that record is not always readable — the run id can already be
cleared, the event can be pruned, or a claim can predate the field. In those
cases the card's own lane history decides: once ``review_requested`` is the
newest lane decision, the retry is a review run.

Only an explicit implementer handoff moves the card back to the implement
lane: a reviewer's ``changes_requested``, or an operator's ``review_reopened``
(reopening review so the implementer re-runs on new comments).
"""

from __future__ import annotations

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


def _review_card(conn, title: str = "Review me"):
    """Card claimed for implementation, then handed to the review lane."""
    task_id = kb.create_task(conn, title=title, assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:test")
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        summary="ready for independent review",
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    return task_id


def _drop_claims(conn, task_id: str) -> None:
    """Simulate a claim record we can no longer read (pruned/legacy row)."""
    conn.execute(
        "DELETE FROM task_events WHERE task_id = ? AND kind = 'claimed'", (task_id,)
    )
    conn.commit()


def test_retry_with_cleared_run_id_stays_in_review(conn):
    task_id = _review_card(conn)
    conn.execute(
        "UPDATE tasks SET current_run_id = NULL WHERE id = ?", (task_id,)
    )
    conn.commit()

    assert kb.get_task(conn, task_id).current_run_id is None
    assert kb._retry_status_for_run(conn, task_id) == "review"


def test_retry_without_readable_claim_event_stays_in_review(conn):
    task_id = _review_card(conn)
    _drop_claims(conn, task_id)

    # No run id at all, and a dangling id whose claim record is gone: both must
    # resolve from the card's lane history, not fall back to implementation.
    assert kb._retry_status_for_run(conn, task_id) == "review"
    assert kb._retry_status_for_run(conn, task_id, run_id=999999) == "review"


def test_retry_after_changes_requested_is_implementation(conn):
    task_id = _review_card(conn)
    review = kb.claim_review_task(conn, task_id)
    assert review is not None
    ok, implementer = kb.request_changes(
        conn, task_id, reason="fix the parser", expected_run_id=review.current_run_id
    )
    assert ok, implementer

    implementation = kb.claim_task(conn, task_id, claimer="builder:2")
    assert implementation is not None
    # Legacy/pruned claim witness: history must send it back to implementation.
    _drop_claims(conn, task_id)
    assert kb._retry_status_for_run(conn, task_id) == "ready"


def test_retry_after_review_reopened_is_implementation(conn):
    task_id = _review_card(conn)
    assert kb.reopen_review_task(conn, task_id)
    _drop_claims(conn, task_id)

    assert kb.get_task(conn, task_id).status in {"ready", "todo"}
    assert kb._retry_status_for_run(conn, task_id) == "ready"


def test_retry_of_card_that_never_reviewed_is_implementation(conn):
    task_id = kb.create_task(conn, title="Never reviewed", assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:test")
    assert implementation is not None

    _drop_claims(conn, task_id)
    assert kb._retry_status_for_run(conn, task_id) == "ready"


def test_claim_witness_still_wins_for_review_runs(conn):
    task_id = _review_card(conn)
    review = kb.claim_review_task(conn, task_id)
    assert review is not None

    # The claimed event of the reviewer run is authoritative on its own, even
    # before any lane history is consulted.
    assert kb._retry_status_for_run(conn, task_id, run_id=review.current_run_id) == "review"
