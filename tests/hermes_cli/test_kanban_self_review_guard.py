"""Tests for the kanban self-review guard (#97910).

When ``reviewer`` is omitted on a first review, the assignee stays as the
implementer, silently routing the card back to the same profile that just
built it.  The guard refuses this unless ``kanban.allow_self_review`` is
enabled, and records a ``self_review`` flag on the event when opted in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _event(events, kind):
    for e in reversed(events):
        if e.kind == kind:
            return e
    return None


def test_first_review_without_reviewer_is_rejected(conn) -> None:
    """Omitting ``reviewer`` on a first review returns False with a diagnostic."""
    task_id = kb.create_task(conn, title="Implement feature X", assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    ok, reason = kb.request_review(
        conn,
        task_id,
        summary="ready for review",
        expected_run_id=implementation.current_run_id,
        with_reason=True,
    )
    assert ok is False
    assert "implementer" in (reason or "")
    assert "allow_self_review" in (reason or "")


def test_first_review_with_explicit_different_reviewer_succeeds(conn) -> None:
    """Passing ``reviewer=`` with a different profile works normally."""
    task_id = kb.create_task(conn, title="Implement feature Y", assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        summary="ready for review",
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "review"
    assert task.assignee == "reviewer"


def test_explicit_self_review_is_rejected(conn) -> None:
    """Passing ``reviewer=`` equal to the implementer is also rejected."""
    task_id = kb.create_task(conn, title="Implement feature Z", assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    ok, reason = kb.request_review(
        conn,
        task_id,
        summary="ready for review",
        reviewer="builder",
        expected_run_id=implementation.current_run_id,
        with_reason=True,
    )
    assert ok is False
    assert "implementer" in (reason or "")


def test_rereview_without_reviewer_uses_provenance(conn) -> None:
    """Re-review without ``reviewer=`` recovers prior reviewer from events — not blocked."""
    task_id = kb.create_task(conn, title="Re-review flow", assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        summary="v1",
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
    assert review is not None
    assert kb.request_changes(
        conn,
        task_id,
        reason="needs fixes",
        expected_run_id=review.current_run_id,
    ) == (True, "builder")

    # Re-review without reviewer= — should recover "reviewer" from prior event
    retry = kb.claim_task(conn, task_id, claimer="builder:2")
    assert retry is not None
    assert kb.request_review(
        conn,
        task_id,
        summary="v2",
        expected_run_id=retry.current_run_id,
    )
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "review"
    assert task.assignee == "reviewer"