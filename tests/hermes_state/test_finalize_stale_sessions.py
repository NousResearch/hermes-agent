"""Regression coverage for monotone stale-session finalization."""

import sqlite3
import time

import pytest

from hermes_state import SessionDB


def _make_stale(db: SessionDB, session_id: str, *, started_at: float) -> None:
    db.create_session(session_id=session_id, source="cli")
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (started_at, session_id),
        )
    )


@pytest.fixture()
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    database.close()


def test_finalize_stale_dry_run_observes_without_mutating(db):
    now = time.time()
    _make_stale(db, "old", started_at=now - 10_000)

    result = db.list_stale_open_candidates(idle_for_seconds=3600, now=now)

    assert result["observed_ids"] == ["old"]
    assert result["mutated_ids"] == []
    assert result["transaction_committed"] is False
    assert db.get_session("old")["ended_at"] is None


def test_finalize_stale_uses_last_message_activity_and_is_idempotent(db):
    now = time.time()
    _make_stale(db, "active", started_at=now - 10_000)
    db.append_message("active", role="user", content="still here", timestamp=now - 30)
    _make_stale(db, "old", started_at=now - 10_000)

    result = db.finalize_stale_sessions(
        idle_for_seconds=3600,
        reason="stale_open_cleanup",
        now=now,
    )

    assert result["mutated_ids"] == ["old"]
    assert result["skipped_after_revalidation"] == []
    assert result["transaction_committed"] is True
    assert db.get_session("active")["ended_at"] is None
    assert db.get_session("old")["end_reason"] == "stale_open_cleanup"
    assert db.finalize_stale_sessions(
        idle_for_seconds=3600, reason="stale_open_cleanup", now=now
    )["mutated_ids"] == []


def test_finalize_stale_preserves_existing_finalization_and_orders_limited_batch(db):
    now = time.time()
    _make_stale(db, "third", started_at=now - 3_000)
    _make_stale(db, "first", started_at=now - 5_000)
    _make_stale(db, "second", started_at=now - 4_000)
    db.end_session("second", end_reason="already_closed")

    result = db.finalize_stale_sessions(
        idle_for_seconds=3600, reason="stale_open_cleanup", limit=1, now=now
    )

    assert result["mutated_ids"] == ["first"]
    assert db.get_session("second")["end_reason"] == "already_closed"
    assert db.get_session("third")["ended_at"] is None


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_finalize_stale_rejects_out_of_bounds_limit(db, limit):
    with pytest.raises(ValueError, match="limit"):
        db.finalize_stale_sessions(
            idle_for_seconds=3600, reason="stale_open_cleanup", limit=limit
        )


def test_recent_writer_commit_before_apply_is_preserved(tmp_path):
    db_path = tmp_path / "state.db"
    cleaner = SessionDB(db_path)
    writer = SessionDB(db_path)
    now = time.time()
    _make_stale(cleaner, "raced", started_at=now - 10_000)

    # A second real WAL connection commits activity after a dry-run preview but
    # before the cleanup acquires its write transaction.
    assert cleaner.list_stale_open_candidates(
        idle_for_seconds=3600, now=now
    )["observed_ids"] == ["raced"]
    writer.append_message("raced", role="user", content="new", timestamp=now - 5)

    result = cleaner.finalize_stale_sessions(
        idle_for_seconds=3600, reason="stale_open_cleanup", now=now
    )

    assert result["mutated_ids"] == []
    assert cleaner.get_session("raced")["ended_at"] is None
    writer.close()
    cleaner.close()


def test_finalize_stale_rolls_back_entire_batch_on_update_failure(db, monkeypatch):
    now = time.time()
    _make_stale(db, "first", started_at=now - 5_000)
    _make_stale(db, "second", started_at=now - 4_000)
    original = db._finalize_stale_session
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected update failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_finalize_stale_session", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        db.finalize_stale_sessions(
            idle_for_seconds=3600, reason="stale_open_cleanup", now=now
        )

    assert db.get_session("first")["ended_at"] is None
    assert db.get_session("second")["ended_at"] is None


def test_finalize_stale_lock_contention_leaves_rows_open(db, monkeypatch):
    now = time.time()
    _make_stale(db, "old", started_at=now - 10_000)
    blocker = sqlite3.connect(str(db.db_path), isolation_level=None, timeout=0.01)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        monkeypatch.setattr(db, "_WRITE_MAX_RETRIES", 1)
        with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
            db.finalize_stale_sessions(
                idle_for_seconds=3600, reason="stale_open_cleanup", now=now
            )
    finally:
        blocker.rollback()
        blocker.close()

    assert db.get_session("old")["ended_at"] is None


def test_prune_limit_is_applied_inside_transaction_and_receipt_is_honest(db):
    now = time.time()
    for session_id, age in (("first", 5_000), ("second", 4_000)):
        _make_stale(db, session_id, started_at=now - age)
        db.end_session(session_id, end_reason="stale_open_cleanup")

    preview = db.list_prune_candidates(
        older_than_days=None, started_before=now, limit=1
    )
    receipt = db.prune_sessions_with_receipt(
        older_than_days=None,
        started_before=now,
        limit=1,
    )

    assert [row["id"] for row in preview] == ["first"]
    assert receipt == {
        "db_pruned_count": 1,
        "db_pruned_ids": ["first"],
        "filesystem_cleanup_status": "best_effort_unverified",
    }
    assert db.get_session("first") is None
    assert db.get_session("second") is not None
