"""Regression tests for the circuit-breaker orphan-source fix
(post-mortem 260822-critic-lane-orphan-blindspot, action #3).

The bug: ``_record_task_failure`` (the dispatcher's circuit breaker)
flipped ``tasks.status`` to ``blocked`` on retry exhaustion while
emitting ONLY a ``gave_up`` event — no ``blocked`` event, NULL
``block_kind``/``block_reason``. Every event-keyed detector (critic
lane, kanban-block-watcher) was structurally blind to these "orphan"
tickets; 16 of them dead-ended on the default board for 10-48h.

The contract under test (events = audit trail, status = state):

* EVERY transition into ``status='blocked'`` must leave a ``blocked``
  event row in ``task_events`` — including the circuit breaker.
* The breaker's ``blocked`` event carries ``kind: circuit_breaker`` in
  its payload so ``_has_sticky_block`` still classifies it as an
  auto-recoverable block (preserving #35072 / #28712 semantics).
* Worker/operator ``block_task`` events remain sticky (no marker).
* ``tasks.block_kind`` stays NULL on the breaker path — userland
  escalation classifiers key retry-exhaustion tickets off
  ``block_kind in {None, transient}`` and must not see a new value.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn: sqlite3.Connection, task_id: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()
    out = []
    for r in rows:
        try:
            out.append(json.loads(r["payload"]) if r["payload"] else {})
        except (json.JSONDecodeError, TypeError):
            out.append({})
    return out


# ---------------------------------------------------------------------------
# AC: every blocked-status transition emits a blocked event — breaker path
# ---------------------------------------------------------------------------


def test_breaker_trip_emits_blocked_event(kanban_home: Path) -> None:
    """Retry exhaustion (the exact #28712-loop shape that produced live
    orphans t_dd7d6af9 / t_f801ed90) must leave BOTH a ``gave_up`` and a
    ``blocked`` event. Before the fix the ``blocked`` row was absent —
    the orphan class."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="breaker reproducer")
        kb.claim_task(conn, tid)
        # Crash path: release_claim=False, end_run=False — the caller
        # (detect_crashed_workers) has already restored the source phase.
        tripped = kbd._record_task_failure(
            conn, tid, "worker exited cleanly (rc=0) without terminal call",
            outcome="crashed", failure_limit=1,
            release_claim=False, end_run=False,
        )
        assert tripped is True
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert len(_events(conn, tid, "gave_up")) == 1
        blocked = _events(conn, tid, "blocked")
        assert len(blocked) == 1, (
            "circuit-breaker block must be visible in the event audit trail"
        )
        assert blocked[0].get("kind") == kb.CIRCUIT_BREAKER_BLOCK_KIND
        assert blocked[0].get("trigger_outcome") == "crashed"
        assert blocked[0].get("failures") == 1


def test_breaker_spawn_path_emits_blocked_event(kanban_home: Path) -> None:
    """The spawn-failure variant (release_claim=True, end_run=True) must
    emit the same audit row."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="spawn failure reproducer")
        kb.claim_task(conn, tid)
        tripped = kbd._record_task_failure(
            conn, tid, "spawn: image pull backoff",
            outcome="spawn_failed", failure_limit=1,
            release_claim=True, end_run=True,
        )
        assert tripped is True
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        blocked = _events(conn, tid, "blocked")
        assert len(blocked) == 1
        assert blocked[0].get("kind") == kb.CIRCUIT_BREAKER_BLOCK_KIND


def test_breaker_block_leaves_block_kind_null(kanban_home: Path) -> None:
    """``tasks.block_kind`` must stay NULL on the breaker path: userland
    escalation tooling (classify_rca_tier) routes retry-exhaustion
    tickets through the ``block_kind in {None, transient}`` lane and the
    scheduler's structural/transient vocab does not know this value."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="kind stays null")
        kb.claim_task(conn, tid)
        kbd._record_task_failure(
            conn, tid, "boom", outcome="crashed", failure_limit=1,
            release_claim=False, end_run=False,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind is None


def test_below_threshold_no_blocked_event(kanban_home: Path) -> None:
    """A single failure below the limit must NOT produce a blocked
    event — the audit row appears exactly when the status flips."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="not yet exhausted")
        kb.claim_task(conn, tid)
        tripped = kbd._record_task_failure(
            conn, tid, "first crash", outcome="crashed", failure_limit=3,
            release_claim=False, end_run=False,
        )
        assert tripped is False
        assert kb.get_task(conn, tid).status != "blocked"  # type: ignore[union-attr]
        assert _events(conn, tid, "blocked") == []


# ---------------------------------------------------------------------------
# AC: breaker blocks remain auto-recoverable (not sticky)
# ---------------------------------------------------------------------------


def test_breaker_block_is_not_sticky(kanban_home: Path) -> None:
    """A circuit-breaker block (marker-carrying event) must remain
    auto-recoverable via recompute_ready once the failure counter drops
    below the effective limit — the #35072/#28712 semantics. Sticky is
    reserved for deliberate worker/operator blocks."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="breaker auto-recovery")
        kb.claim_task(conn, tid)
        kbd._record_task_failure(
            conn, tid, "exhausted", outcome="crashed", failure_limit=1,
            release_claim=False, end_run=False,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert kb._has_sticky_block(conn, tid) is False

        # Simulate the recovery condition: the dispatcher's unblock lane
        # resets consecutive_failures (see unblock_task), after which
        # recompute_ready may promote the task.
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0 WHERE id = ?", (tid,)
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, tid).status == "ready"


def test_breaker_marker_does_not_leak_stickiness_to_later_worker_block(
    kanban_home: Path,
) -> None:
    """Sequence: breaker block (non-sticky) → task recovers → worker
    blocks for review (sticky). The LATER unmarked event must win — the
    task stays blocked across recompute_ready."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="marker then real block")
        kb.claim_task(conn, tid)
        kbd._record_task_failure(
            conn, tid, "exhausted", outcome="crashed", failure_limit=1,
            release_claim=False, end_run=False,
        )
        # Recover: reset counter, promote, reclaim.
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0 WHERE id = ?", (tid,)
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 1
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid, reason="review-required: human eyes",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert kb._has_sticky_block(conn, tid) is True
        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.status == "blocked"


def test_worker_block_still_sticky_after_fix(kanban_home: Path) -> None:
    """Plain guard: the fix must not weaken the original #28712
    contract — an unmarked worker block stays sticky."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain sticky block")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid, reason="review-required: verify",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb._has_sticky_block(conn, tid) is True
        assert kb.recompute_ready(conn) == 0
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"


def test_unblock_after_breaker_clears_event_pair(kanban_home: Path) -> None:
    """unblock_task on a breaker-blocked task emits ``unblocked`` — the
    marker/event pair stays coherent for event-keyed consumers."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unblock after breaker")
        kb.claim_task(conn, tid)
        kbd._record_task_failure(
            conn, tid, "exhausted", outcome="crashed", failure_limit=1,
            release_claim=False, end_run=False,
        )
        assert kb.unblock_task(conn, tid) is True
        unblocked = kb.get_task(conn, tid)
        assert unblocked is not None
        assert unblocked.status in ("ready", "review")
        assert len(_events(conn, tid, "unblocked")) == 1
        # The newest {blocked, unblocked} row is the unblock → not sticky.
        assert kb._has_sticky_block(conn, tid) is False


# ---------------------------------------------------------------------------
# Port guard: the breaker's blocked event is the NEWEST lifecycle row, so
# it must carry the resume signal itself (post-dispatch-split main).
# ---------------------------------------------------------------------------


def test_breaker_blocked_event_carries_review_retry_status(kanban_home: Path) -> None:
    """``_resume_status_from_events`` reads the NEWEST lifecycle event to
    pick a resume phase. The breaker's ``blocked`` row lands AFTER
    ``gave_up``, so it must carry ``retry_status`` itself — otherwise a
    breaker trip during review would resume to ``ready`` instead of
    ``review`` after unblock."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="breaker during review")
        implementation = kb.claim_task(conn, tid)
        assert implementation is not None
        assert kb.request_review(
            conn, tid, summary="needs eyes",
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        kbd._record_task_failure(
            conn, tid, "reviewer cannot start", outcome="spawn_failed",
            failure_limit=1, release_claim=True, end_run=True,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        blocked = _events(conn, tid, "blocked")
        assert len(blocked) == 1
        assert blocked[0].get("kind") == kb.CIRCUIT_BREAKER_BLOCK_KIND
        assert blocked[0].get("retry_status") == "review"
        assert kb.unblock_task(conn, tid) is True
        resumed = kb.get_task(conn, tid)
        assert resumed is not None
        assert resumed.status == "review"
