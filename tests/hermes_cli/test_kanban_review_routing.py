"""Tests for ``review_required`` block routing.

The review pipeline (dispatcher review column → ``claim_review_task`` →
reviewer agent) was unreachable dead code: nothing in the codebase ever
transitioned a task TO ``review`` status. ``block_task`` now routes a
``review_required`` block to ``review`` (where the dispatcher picks it up)
instead of ``blocked`` (the human queue).

Two trigger paths:
* Explicit: ``kind="review_required"``
* Auto-detected: ``reason`` starts with ``"review-required"`` and ``kind``
  is ``None`` (the convention taught to workers via KANBAN_GUIDANCE).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t", assignee="glm-executor"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


# ---------------------------------------------------------------------------
# Explicit kind
# ---------------------------------------------------------------------------

def test_explicit_kind_routes_to_review(kanban_home: Path) -> None:
    """``kind="review_required"`` transitions to ``review`` status."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        result = kb.block_task(
            conn, tid, reason="review-required: PR open",
            kind="review_required",
        )
        assert result is True
        task = kb.get_task(conn, tid)
        assert task.status == "review"


def test_explicit_kind_assignee_normalized(kanban_home: Path) -> None:
    """Executor assignee is rewritten to reviewer on review routing."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="glm-executor")
        kb.block_task(
            conn, tid, reason="review-required: PR open",
            kind="review_required",
        )
        task = kb.get_task(conn, tid)
        assert task.assignee == "glm-reviewer"
        assert task.original_executor == "glm-executor"


# ---------------------------------------------------------------------------
# Auto-detection from reason prefix
# ---------------------------------------------------------------------------

def test_auto_detected_from_reason_prefix(kanban_home: Path) -> None:
    """``reason="review-required: …"`` with ``kind=None`` auto-routes."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="kimi-coder")
        result = kb.block_task(
            conn, tid, reason="review-required: ready for review",
        )
        assert result is True
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == "kimi-reviewer"
        assert task.original_executor == "kimi-coder"


def test_auto_detected_case_insensitive(kanban_home: Path) -> None:
    """The reason prefix match is case-insensitive."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="glm-executor")
        kb.block_task(conn, tid, reason="Review-Required: PR ready")
        task = kb.get_task(conn, tid)
        assert task.status == "review"


# ---------------------------------------------------------------------------
# Non-review blocks still go to blocked (regression)
# ---------------------------------------------------------------------------

def test_non_review_block_still_blocked(kanban_home: Path) -> None:
    """A capability block must NOT route to review."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="missing creds", kind="capability")
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"


def test_unrelated_reason_not_routed(kanban_home: Path) -> None:
    """A reason that merely contains 'review' but doesn't start with the
    prefix must not trigger review routing."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="peer review needed manually")
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"


# ---------------------------------------------------------------------------
# original_executor preservation
# ---------------------------------------------------------------------------

def test_original_executor_preserved_across_reentry(kanban_home: Path) -> None:
    """COALESCE preserves original_executor if already set (re-review)."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="glm-executor")
        # First review-required block
        kb.block_task(
            conn, tid, reason="review-required: PR #1",
            kind="review_required",
        )
        task1 = kb.get_task(conn, tid)
        assert task1.original_executor == "glm-executor"
        assert task1.assignee == "glm-reviewer"

        # Simulate reviewer returning it: manually set back to running
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='running', assignee='glm-executor' "
                "WHERE id=?",
                (tid,),
            )
        # Second review-required block — original_executor should NOT be
        # overwritten by whatever assignee was current at re-block time.
        kb.block_task(
            conn, tid, reason="review-required: PR #2",
            kind="review_required",
        )
        task2 = kb.get_task(conn, tid)
        assert task2.original_executor == "glm-executor"


# ---------------------------------------------------------------------------
# Already-reviewer assignee
# ---------------------------------------------------------------------------

def test_already_reviewer_assignee_unchanged(kanban_home: Path) -> None:
    """If the assignee is already ``*-reviewer``, it stays as-is."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="glm-reviewer")
        kb.block_task(
            conn, tid, reason="review-required: re-review",
            kind="review_required",
        )
        task = kb.get_task(conn, tid)
        assert task.assignee == "glm-reviewer"


# ---------------------------------------------------------------------------
# claim_review_task compatibility
# ---------------------------------------------------------------------------

def test_claim_review_task_works_after_routing(kanban_home: Path) -> None:
    """After review routing, ``claim_review_task`` can claim the task."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="glm-executor")
        kb.block_task(
            conn, tid, reason="review-required: PR open",
            kind="review_required",
        )
        # The task is now in 'review' — claim_review_task should be able to
        # transition it to 'running'.
        claimed = kb.claim_review_task(conn, tid, claimer="reviewer-1")
        assert claimed is not None
        assert claimed.status == "running"


# ---------------------------------------------------------------------------
# Event emitted
# ---------------------------------------------------------------------------

def test_review_required_event_emitted(kanban_home: Path) -> None:
    """A ``review_required`` event is emitted with the right metadata."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="glm-executor")
        kb.block_task(
            conn, tid, reason="review-required: PR open",
            kind="review_required",
        )
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "review_required"]
        assert events, "expected a review_required event"
        payload = events[-1].payload or {}
        assert payload.get("original_executor") == "glm-executor"
        assert payload.get("review_assignee") == "glm-reviewer"
        assert payload.get("kind") == "review_required"


# ---------------------------------------------------------------------------
# block_recurrences reset
# ---------------------------------------------------------------------------

def test_block_recurrences_reset_on_review(kanban_home: Path) -> None:
    """Review routing resets block_recurrences to 0 (fresh start)."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="glm-executor")
        # Block once with capability to set recurrences
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        # Now block with review_required
        kb.block_task(
            conn, tid, reason="review-required: PR open",
            kind="review_required",
        )
        task = kb.get_task(conn, tid)
        assert task.block_recurrences == 0
