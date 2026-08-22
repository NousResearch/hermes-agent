"""Tests: heartbeat disposition contract + stranded-review auto-recovery.

Two gaps this covers, both observed live while benchmarking an external
agent-org manager (Paperclip) against the Hermes board:

1. ``heartbeat_worker`` accepted only a free-text note, so a worker parked on
   a blocker it cannot clear (needs a human decision, needs credentials) was
   indistinguishable from a worker making progress. It kept its concurrency
   slot for the full ``_STALE_HEARTBEAT_GAP_SECONDS`` window and no operator
   signal existed. Workers can now declare a structured disposition, and a
   sustained ``blocked`` disposition routes the task to the blocked lane
   instead of burning the slot.

2. A ``review`` task that cannot be dispatched — unassigned, pointed at a
   profile that does not exist, or sitting while ``kanban.review_dispatch`` is
   off — was swept by nothing. ``detect_stale_running`` only scans
   ``status='running'``. Such a card sat in the review column indefinitely
   while its parent goal stayed open. It is now recovered into ``blocked``
   with kind ``needs_input`` so it surfaces to a human.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _running_task(conn, *, title="t", assignee="w"):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    return tid


def _backdate_heartbeat(conn, tid, seconds_ago):
    """Age the task's heartbeat + run start so sweepers consider it."""
    old = int(time.time()) - seconds_ago
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, started_at = ? WHERE id = ?",
            (old, old, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
            (old, tid),
        )


# --------------------------------------------------------------------------
# 1. Heartbeat disposition contract
# --------------------------------------------------------------------------


def test_heartbeat_records_structured_disposition(conn):
    """A worker can declare WHY it is still alive, not just that it is."""
    tid = _running_task(conn)

    assert kb.heartbeat_worker(conn, tid, disposition="progressing") is True

    row = conn.execute(
        "SELECT last_disposition FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["last_disposition"] == "progressing"

    events = [e for e in kb.list_events(conn, tid) if e.kind == "heartbeat"]
    assert events, "heartbeat event should be recorded"
    assert events[-1].payload["disposition"] == "progressing"


def test_heartbeat_rejects_unknown_disposition(conn):
    """The contract is a closed set — free-text status strings are refused."""
    tid = _running_task(conn)

    with pytest.raises(ValueError):
        kb.heartbeat_worker(conn, tid, disposition="vibing")


def test_blocked_disposition_does_not_trip_immediately(conn):
    """One blocked heartbeat is not enough — transient waits are normal."""
    tid = _running_task(conn)
    kb.heartbeat_worker(conn, tid, disposition="blocked", note="waiting on auth")

    assert kb.detect_blocked_dispositions(conn, blocked_timeout_seconds=600) == []
    assert kb.get_task(conn, tid).status == "running"


def test_sustained_blocked_disposition_releases_the_slot(conn):
    """A worker parked on a blocker past the window is routed to blocked.

    This is the concurrency-slot leak: without it the task holds a worker slot
    for the full stale-heartbeat window while making no progress.
    """
    tid = _running_task(conn)
    kb.heartbeat_worker(conn, tid, disposition="blocked", note="needs owner key")
    _backdate_heartbeat(conn, tid, 900)

    recovered = kb.detect_blocked_dispositions(conn, blocked_timeout_seconds=600)

    assert recovered == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"


def test_progressing_disposition_is_never_swept(conn):
    """A long-running worker that reports progress keeps its slot."""
    tid = _running_task(conn)
    kb.heartbeat_worker(conn, tid, disposition="progressing")
    _backdate_heartbeat(conn, tid, 9000)

    assert kb.detect_blocked_dispositions(conn, blocked_timeout_seconds=600) == []
    assert kb.get_task(conn, tid).status == "running"


def test_blocked_disposition_sweep_is_disabled_at_zero(conn):
    """Timeout of 0 disables the check, matching detect_stale_running."""
    tid = _running_task(conn)
    kb.heartbeat_worker(conn, tid, disposition="blocked")
    _backdate_heartbeat(conn, tid, 9000)

    assert kb.detect_blocked_dispositions(conn, blocked_timeout_seconds=0) == []
    assert kb.get_task(conn, tid).status == "running"


def test_recovering_disposition_clears_a_prior_blocked_report(conn):
    """A worker that unsticks itself must not be swept for its old report."""
    tid = _running_task(conn)
    kb.heartbeat_worker(conn, tid, disposition="blocked")
    _backdate_heartbeat(conn, tid, 900)
    kb.heartbeat_worker(conn, tid, disposition="progressing")

    assert kb.detect_blocked_dispositions(conn, blocked_timeout_seconds=600) == []
    assert kb.get_task(conn, tid).status == "running"


# --------------------------------------------------------------------------
# 2. Stranded-review auto-recovery
# --------------------------------------------------------------------------


def _review_task(conn, *, assignee, age_seconds):
    """Put a task into review via the real transition, then age the event.

    Review entry has no dedicated timestamp column, so the age of the
    ``review_requested`` event is the source of truth for how long a card has
    been sitting in the review column.
    """
    tid = kb.create_task(conn, title="shipped work", assignee="builder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]
    assert kb.request_review(
        conn, tid, summary="PR opened", expected_run_id=run_id
    ) is True
    assert kb.get_task(conn, tid).status == "review"
    old = int(time.time()) - age_seconds
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET assignee = ? WHERE id = ?", (assignee, tid)
        )
        conn.execute(
            "UPDATE task_events SET created_at = ? "
            "WHERE task_id = ? AND kind = 'review_requested'",
            (old, tid),
        )
    return tid


def test_unassigned_review_card_is_recovered(conn):
    """A review card with no assignee is dispatched by nothing — recover it."""
    tid = _review_task(conn, assignee=None, age_seconds=7200)

    recovered = kb.detect_stranded_review(conn, stranded_timeout_seconds=3600)

    assert recovered == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"


def test_review_card_on_missing_profile_is_recovered(conn):
    """An assignee that resolves to no profile can never spawn a reviewer."""
    tid = _review_task(conn, assignee="ghost-profile-does-not-exist", age_seconds=7200)

    assert kb.detect_stranded_review(conn, stranded_timeout_seconds=3600) == [tid]
    assert kb.get_task(conn, tid).status == "blocked"


def test_dispatch_keeps_stranded_review_blocked_after_recompute(conn):
    """Full dispatch must not recover review, then auto-promote it ready."""
    tid = _review_task(conn, assignee="ghost-profile-does-not-exist", age_seconds=7200)

    result = kb.dispatch_once(
        conn,
        dry_run=True,
        stranded_review_timeout_seconds=3600,
    )

    assert result.stranded_reviews == [tid]
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"
    assert result.spawned == []
    assert tid not in result.skipped_nonspawnable


def test_fresh_review_card_is_left_alone(conn):
    """Recovery must not race a reviewer that simply has not spawned yet."""
    tid = _review_task(conn, assignee=None, age_seconds=60)

    assert kb.detect_stranded_review(conn, stranded_timeout_seconds=3600) == []
    assert kb.get_task(conn, tid).status == "review"


def test_claimed_review_card_is_left_alone(conn):
    """A claimed review card has a live reviewer — never touch it."""
    tid = _review_task(conn, assignee=None, age_seconds=7200)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_lock = ? WHERE id = ?",
            (kb._claimer_id(), tid),
        )

    assert kb.detect_stranded_review(conn, stranded_timeout_seconds=3600) == []
    assert kb.get_task(conn, tid).status == "review"


def test_stranded_review_sweep_is_disabled_at_zero(conn):
    tid = _review_task(conn, assignee=None, age_seconds=99999)

    assert kb.detect_stranded_review(conn, stranded_timeout_seconds=0) == []
    assert kb.get_task(conn, tid).status == "review"


def test_recovered_review_card_records_why(conn):
    """The blocked reason must name the strand cause for the human reading it."""
    tid = _review_task(conn, assignee=None, age_seconds=7200)

    kb.detect_stranded_review(conn, stranded_timeout_seconds=3600)

    events = [
        e for e in kb.list_events(conn, tid)
        if e.kind == "stranded_recovered"
    ]
    assert events, "recovery should be auditable on the task timeline"
    assert events[-1].payload["reason"] == "unassigned"


def test_review_dispatch_disabled_strands_every_review_card(conn, monkeypatch):
    """With review dispatch off, even a well-formed card is never spawned."""
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: False)
    tid = _review_task(conn, assignee="worker-that-exists", age_seconds=7200)

    assert kb.detect_stranded_review(conn, stranded_timeout_seconds=3600) == [tid]
    events = [
        e for e in kb.list_events(conn, tid)
        if e.kind == "stranded_recovered"
    ]
    assert events[-1].payload["reason"] == "review_dispatch_disabled"
