"""Real SQLite contract tests for same-board delivery receipts."""

from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_surface import (
    DeliveryReceiptIdentityError,
    DeliveryReceiptLeaseLost,
    DeliveryReceiptNotRetryable,
    DeliveryReceiptRevisionError,
    DeliveryReceiptUnknown,
    authorize_delivery_replacement,
    claim_delivery_receipt,
    ensure_delivery_receipt,
    get_delivery_receipt,
    record_delivery_outcome,
)


def _receipt_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    db_path = home / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(db_path=db_path)
    return db_path


def _task_and_revision(db_path: Path) -> tuple[str, int, str]:
    with kbc.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="receipt task")
        row = conn.execute(
            "SELECT MAX(task_events.id) AS revision, tasks.status FROM task_events JOIN tasks ON tasks.id = task_events.task_id "
            "WHERE task_events.task_id = ?",
            (task_id,),
        ).fetchone()
        task = kb.get_task(conn, task_id)
        assert task is not None
        return task_id, int(row["revision"]), str(task.status)


def _claim_process(db_path: str, receipt_id: int, owner: str, queue) -> None:
    """Spawn-safe race worker; each process opens its own authoritative board connection."""
    from hermes_cli import kanban_db_connect as _kbc
    from hermes_cli.kanban_db_surface import claim_delivery_receipt as _claim

    conn = _kbc.connect(Path(db_path))
    try:
        lease = _claim(conn, receipt_id, owner_id=owner, now=100, lease_seconds=30)
        queue.put(("claimed", owner, lease.owner_epoch, lease.attempt_id))
    except Exception as exc:  # the loser is expected to observe a CAS/lease failure
        queue.put((type(exc).__name__, owner, "", ""))
    finally:
        conn.close()


def test_schema_is_added_to_fresh_and_legacy_boards_idempotently(tmp_path, monkeypatch):
    db_path = _receipt_home(tmp_path, monkeypatch)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("DROP TABLE kanban_action_records; DROP TABLE kanban_delivery_receipts;")
        conn.commit()
    kb._INITIALIZED_PATHS.clear()

    with kbc.connect(db_path) as conn:
        names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"kanban_delivery_receipts", "kanban_action_records"} <= names
        task_id = kb.create_task(conn, title="legacy survives")

    # The second migration must not duplicate or rewrite feature rows.
    with kbc.connect(db_path) as conn:
        receipt = ensure_delivery_receipt(
            conn, task_id=task_id, platform="telegram", chat_id="-100",
            thread_id="topic-1", notifier_profile="alpha", desired_revision=1,
        )
        before = conn.execute(
            "SELECT COUNT(*) AS n FROM kanban_delivery_receipts"
        ).fetchone()["n"]
    kb.init_db(db_path=db_path)
    with kbc.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM kanban_delivery_receipts"
        ).fetchone()["n"] == before
        assert get_delivery_receipt(conn, receipt.id).task_id == task_id


def test_intent_is_durable_before_send_and_lane_metadata_is_exact(tmp_path, monkeypatch):
    db_path = _receipt_home(tmp_path, monkeypatch)
    task_id, revision, _ = _task_and_revision(db_path)
    with kbc.connect(db_path) as conn:
        receipt = ensure_delivery_receipt(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="-100123",
            thread_id="42",
            notifier_profile="profile-a",
            routing_metadata={"topic": 42, "chat_type": "forum"},
            renderer_version="receipt-v1",
            renderer_hash="sha256:renderer",
            desired_revision=revision,
        )
    # Reopen a different connection before any remote transport call.
    with kbc.connect(db_path) as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        stored = get_delivery_receipt(conn, receipt.id)
        assert stored is not None
        assert stored.task_id == task.id
        assert stored.state == "pending"
        assert stored.desired_revision == revision
        assert stored.delivered_revision is None
        assert stored.notifier_profile == "profile-a"
        assert stored.routing_metadata == {"chat_type": "forum", "topic": 42}
        assert stored.platform == "telegram"
        assert stored.chat_id == "-100123"
        assert stored.thread_id == "42"


def test_receipt_revision_cas_lease_expiry_and_stale_owner_are_fail_closed(tmp_path, monkeypatch):
    db_path = _receipt_home(tmp_path, monkeypatch)
    task_id, revision, _ = _task_and_revision(db_path)
    with kbc.connect(db_path) as conn:
        receipt = ensure_delivery_receipt(
            conn, task_id=task_id, platform="telegram", chat_id="chat",
            thread_id="topic", desired_revision=revision,
        )
        lease = claim_delivery_receipt(
            conn, receipt.id, owner_id="owner-a", now=100, lease_seconds=5,
        )
        with pytest.raises(DeliveryReceiptRevisionError):
            ensure_delivery_receipt(
                conn, task_id=task_id, platform="telegram", chat_id="chat",
                thread_id="topic", desired_revision=revision - 1,
            )
        with pytest.raises(DeliveryReceiptLeaseLost):
            record_delivery_outcome(
                conn, receipt.id, owner_id="owner-a", owner_epoch=lease.owner_epoch,
                attempt_id=lease.attempt_id, desired_revision=revision,
                state="sent", message_id="m1", destination_profile="profile-a",
                delivered_revision=revision, now=106,
            )
        with pytest.raises(DeliveryReceiptUnknown):
            claim_delivery_receipt(
                conn, receipt.id, owner_id="owner-b", now=106, lease_seconds=5,
            )
        stored = get_delivery_receipt(conn, receipt.id)
        assert stored is not None
        assert stored.state == "unknown"


def test_receipt_stale_owner_generation_cannot_record_after_retry(tmp_path, monkeypatch):
    db_path = _receipt_home(tmp_path, monkeypatch)
    task_id, revision, _ = _task_and_revision(db_path)
    with kbc.connect(db_path) as conn:
        receipt = ensure_delivery_receipt(
            conn, task_id=task_id, platform="telegram", chat_id="chat",
            desired_revision=revision,
        )
        first = claim_delivery_receipt(conn, receipt.id, owner_id="a", now=100)
        record_delivery_outcome(
            conn, receipt.id, owner_id="a", owner_epoch=first.owner_epoch,
            attempt_id=first.attempt_id, desired_revision=revision,
            state="failed", retry_disposition="retryable", error="timeout", now=101,
        )
        second = claim_delivery_receipt(conn, receipt.id, owner_id="b", now=102)
        with pytest.raises(DeliveryReceiptLeaseLost):
            record_delivery_outcome(
                conn, receipt.id, owner_id="a", owner_epoch=first.owner_epoch,
                attempt_id=first.attempt_id, desired_revision=revision,
                state="sent", message_id="stale", destination_profile="profile-a",
                delivered_revision=revision, now=103,
            )
        sent = record_delivery_outcome(
            conn, receipt.id, owner_id="b", owner_epoch=second.owner_epoch,
            attempt_id=second.attempt_id, desired_revision=revision,
            state="sent", message_id="m2", destination_profile="profile-a",
            delivered_revision=revision, now=103,
        )
        assert sent.state == "sent"
        assert sent.destination_message_id == "m2"


def test_unknown_create_is_not_auto_replayed_and_known_delete_has_one_replacement(tmp_path, monkeypatch):
    db_path = _receipt_home(tmp_path, monkeypatch)
    task_id, revision, _ = _task_and_revision(db_path)
    with kbc.connect(db_path) as conn:
        receipt = ensure_delivery_receipt(
            conn, task_id=task_id, platform="telegram", chat_id="chat",
            desired_revision=revision, replacement_budget=1,
        )
        first = claim_delivery_receipt(conn, receipt.id, owner_id="sender", now=100)
        record_delivery_outcome(
            conn, receipt.id, owner_id="sender", owner_epoch=first.owner_epoch,
            attempt_id=first.attempt_id, desired_revision=revision,
            state="sent", message_id="known-1", destination_profile="profile-a",
            delivered_revision=revision, now=101,
        )
        next_event = kb.add_comment(conn, task_id, "tester", "edit")
        new_revision = conn.execute(
            "SELECT MAX(id) AS n FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()["n"]
        ensure_delivery_receipt(
            conn, task_id=task_id, platform="telegram", chat_id="chat",
            desired_revision=int(new_revision),
        )
        edit = claim_delivery_receipt(conn, receipt.id, owner_id="editor", now=102)
        deleted = record_delivery_outcome(
            conn, receipt.id, owner_id="editor", owner_epoch=edit.owner_epoch,
            attempt_id=edit.attempt_id, desired_revision=int(new_revision),
            state="deleted", message_id="known-1", now=103,
        )
        assert deleted.state == "deleted"
        replacement = authorize_delivery_replacement(
            conn, receipt.id, desired_revision=int(new_revision), now=104,
        )
        assert replacement.state == "pending"
        replacement_lease = claim_delivery_receipt(conn, receipt.id, owner_id="replacement", now=105)
        sent = record_delivery_outcome(
            conn, receipt.id, owner_id="replacement",
            owner_epoch=replacement_lease.owner_epoch,
            attempt_id=replacement_lease.attempt_id,
            desired_revision=int(new_revision), state="sent", message_id="known-2",
            destination_profile="profile-a", delivered_revision=int(new_revision), now=106,
        )
        assert sent.destination_message_id == "known-2"
        with pytest.raises(DeliveryReceiptNotRetryable):
            authorize_delivery_replacement(
                conn, receipt.id, desired_revision=int(new_revision), now=107,
            )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert next_event > 0


def test_receipt_apis_join_outer_transaction_without_premature_commit(tmp_path, monkeypatch):
    db_path = _receipt_home(tmp_path, monkeypatch)
    task_id, revision, _ = _task_and_revision(db_path)
    with kbc.connect(db_path) as conn:
        with pytest.raises(RuntimeError):
            with kb.write_txn(conn):
                receipt = ensure_delivery_receipt(
                    conn, task_id=task_id, platform="telegram", chat_id="chat",
                    desired_revision=revision,
                )
                assert get_delivery_receipt(conn, receipt.id) is not None
                raise RuntimeError("caller rollback")
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM kanban_delivery_receipts"
        ).fetchone()["n"] == 0


def test_shared_board_receipt_survives_profile_a_to_b_to_a(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    board = root / "kanban.db"
    profile_a = root / "profiles" / "a"
    profile_b = root / "profiles" / "b"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board))

    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(db_path=board)
    with kbc.connect(board) as conn:
        task_id = kb.create_task(conn, title="profile round trip")
        revision = conn.execute(
            "SELECT MAX(id) AS n FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()["n"]
        receipt = ensure_delivery_receipt(
            conn, task_id=task_id, platform="telegram", chat_id="chat",
            thread_id="topic", notifier_profile="a", desired_revision=int(revision),
        )

    monkeypatch.setenv("HERMES_HOME", str(profile_b))
    with kbc.connect(board) as conn:
        from_b = get_delivery_receipt(conn, receipt.id)
        assert from_b is not None
        assert from_b.task_id == task_id
    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    with kbc.connect(board) as conn:
        round_trip = get_delivery_receipt(conn, receipt.id)
        assert round_trip is not None
        assert round_trip.notifier_profile == "a"


def test_independent_processes_only_one_receipt_owner_wins(tmp_path, monkeypatch):
    db_path = _receipt_home(tmp_path, monkeypatch)
    task_id, revision, _ = _task_and_revision(db_path)
    with kbc.connect(db_path) as conn:
        receipt = ensure_delivery_receipt(
            conn, task_id=task_id, platform="telegram", chat_id="chat",
            desired_revision=revision,
        )

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    workers = [
        ctx.Process(target=_claim_process, args=(str(db_path), receipt.id, owner, queue))
        for owner in ("proc-a", "proc-b")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
    assert all(worker.exitcode == 0 for worker in workers)
    results = [queue.get(timeout=5) for _ in workers]
    assert sum(result[0] == "claimed" for result in results) == 1
    assert sum(result[0] != "claimed" for result in results) == 1
