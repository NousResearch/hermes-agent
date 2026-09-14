"""Retention and idempotency tests for messaging Retry receipts."""

import sqlite3

import pytest

from gateway import hosted_room_messaging_retries as retries


def _plan(db, command_id, *, now, task_id="task-1"):
    return retries.retry_receipt_plan(
        db,
        command_id=command_id,
        room_id="room-1",
        actor={"kind": "user", "id": "owner"},
        task_ids=[task_id],
        now=now,
    )


def test_completed_receipts_make_room_without_pruning_pending(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(retries, "MAX_RETRY_RECEIPTS", 2)
    _plan(db, "completed", now=1)
    retries.complete_retry_receipt(
        db,
        command_id="completed",
        result="done",
        now=2,
    )
    _plan(db, "pending", now=3)

    assert _plan(db, "new", now=4, task_id="task-2") == (["task-2"], None)

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT command_id, state FROM hosted_room_messaging_retries "
            "ORDER BY command_id"
        ).fetchall()
    assert rows == [("new", "pending"), ("pending", "pending")]


def test_pending_receipts_fail_closed_at_the_cap(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(retries, "MAX_RETRY_RECEIPTS", 2)
    _plan(db, "pending-1", now=1)
    _plan(db, "pending-2", now=2)

    with pytest.raises(retries.MessagingRetryReceiptError, match="Finish pending"):
        _plan(db, "pending-3", now=3)


def test_abandoned_pending_receipt_becomes_non_retargetable_tombstone(
    tmp_path, monkeypatch
):
    db = tmp_path / "state.db"
    monkeypatch.setattr(retries, "MAX_RETRY_RECEIPTS", 2)
    monkeypatch.setattr(retries, "PENDING_RETRY_RECEIPT_TTL_SECONDS", 10)
    monkeypatch.setattr(retries, "RETRY_RECEIPT_RETENTION_SECONDS", 20)
    _plan(db, "same-delivery", now=1, task_id="old-task")

    assert _plan(db, "same-delivery", now=20, task_id="new-task") == (
        ["old-task"],
        retries.EXPIRED_RETRY_RESULT,
    )
    assert _plan(db, "pending", now=20) == (["task-1"], None)
    assert _plan(db, "new", now=41) == (["task-1"], None)

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT command_id, state FROM hosted_room_messaging_retries "
            "ORDER BY command_id"
        ).fetchall()
    assert rows == [("new", "pending"), ("pending", "expired")]


def test_expired_completed_receipts_are_pruned(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(retries, "RETRY_RECEIPT_RETENTION_SECONDS", 10)
    _plan(db, "old", now=1)
    retries.complete_retry_receipt(db, command_id="old", result="done", now=2)

    _plan(db, "new", now=20)

    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM hosted_room_messaging_retries "
                "WHERE command_id='old'"
            ).fetchone()[0]
            == 0
        )
