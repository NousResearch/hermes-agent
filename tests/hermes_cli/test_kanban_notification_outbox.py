"""Durability contracts for the Kanban notification outbox."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def outbox_db(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban-outbox.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="durable notification", assignee="worker")
        subscription_id = kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        yield conn, task_id, subscription_id
    finally:
        conn.close()


def _action(event_id: int, message: str = "done") -> dict:
    return {
        "event_id": event_id,
        "action": "message",
        "payload": {"kind": "completed", "message": message},
    }


def _outbox_row(conn, subscription_id: str, event_id: int = 1):
    return conn.execute(
        "SELECT * FROM kanban_notification_outbox "
        "WHERE subscription_id = ? AND event_id = ? AND action = 'message'",
        (subscription_id, event_id),
    ).fetchone()


def test_stage_batch_persists_before_cursor_and_deduplicates_rewind(outbox_db):
    conn, task_id, subscription_id = outbox_db

    assert kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=2,
        actions=[_action(1, "first"), _action(2, "second")],
    ) == 2

    row = _outbox_row(conn, subscription_id)
    sub = kb.list_notify_subs(conn, task_id)[0]
    assert row["state"] == "pending"
    assert int(sub["last_event_id"]) == 2

    kb.advance_notify_cursor(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="chat-1",
        new_cursor=0,
    )
    assert kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=2,
        actions=[_action(1, "first"), _action(2, "second")],
    ) == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM kanban_notification_outbox"
    ).fetchone()[0] == 2
    assert int(kb.list_notify_subs(conn, task_id)[0]["last_event_id"]) == 2


def test_expired_lease_is_reclaimed_and_stale_owner_cannot_ack(outbox_db):
    conn, _task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=1,
        actions=[_action(1)],
    )

    first = kb.lease_notification_outbox(conn, now=100, lease_seconds=10)
    assert len(first) == 1
    assert first[0]["attempts"] == 0
    assert kb.lease_notification_outbox(conn, now=109, lease_seconds=10) == []

    second = kb.lease_notification_outbox(conn, now=110, lease_seconds=10)
    assert len(second) == 1
    assert second[0]["attempts"] == 0
    assert second[0]["lease_token"] != first[0]["lease_token"]
    assert kb.ack_notification_delivery(
        conn,
        subscription_id=subscription_id,
        event_id=1,
        action="message",
        lease_token=first[0]["lease_token"],
    ) is False
    assert kb.ack_notification_delivery(
        conn,
        subscription_id=subscription_id,
        event_id=1,
        action="message",
        lease_token=second[0]["lease_token"],
    ) is True
    assert _outbox_row(conn, subscription_id)["state"] == "acknowledged"


def test_renewed_lease_prevents_a_second_connection_from_reclaiming(outbox_db):
    conn, _task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=1,
        actions=[_action(1)],
    )
    first = kb.lease_notification_outbox(conn, now=100, lease_seconds=10)[0]

    assert kb.renew_notification_lease(
        conn,
        subscription_id=subscription_id,
        event_id=first["event_id"],
        action=first["action"],
        lease_token=first["lease_token"],
        now=109,
        lease_seconds=10,
    ) is True

    contender = kb.connect()
    try:
        assert kb.lease_notification_outbox(
            contender, now=110, lease_seconds=10,
        ) == []
    finally:
        contender.close()


def test_ordered_lease_keeps_a_pending_tail_behind_an_active_head(outbox_db):
    conn, _task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=2,
        actions=[_action(1, "head"), _action(2, "tail")],
    )
    head = kb.lease_notification_outbox(
        conn,
        now=100,
        lease_seconds=10,
        limit=1,
        enforce_subscription_order=True,
    )[0]

    contender = kb.connect()
    try:
        assert kb.lease_notification_outbox(
            contender,
            now=105,
            lease_seconds=10,
            limit=1,
            enforce_subscription_order=True,
        ) == []
        assert kb.ack_notification_delivery(
            conn,
            subscription_id=subscription_id,
            event_id=head["event_id"],
            action=head["action"],
            lease_token=head["lease_token"],
        ) is True
        tail = kb.lease_notification_outbox(
            contender,
            now=106,
            lease_seconds=10,
            limit=1,
            enforce_subscription_order=True,
        )
        assert [item["event_id"] for item in tail] == [2]
    finally:
        contender.close()


def test_staged_snapshot_omits_unused_task_and_event_fields(outbox_db):
    conn, _task_id, _subscription_id = outbox_db
    task_id = kb.create_task(
        conn,
        title="minimal snapshot",
        assignee="worker",
        session_id="unused-session-marker",
    )
    subscription_id = kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="chat-minimal",
    )
    kb._append_event(
        conn,
        task_id,
        "status",
        {
            "status": "visible",
            "debug_context": "unused-event-marker",
        },
    )

    kb.stage_unseen_notifications_for_sub(
        conn,
        subscription_id=subscription_id,
        kinds=["status"],
        action_kinds=["status"],
    )

    row = conn.execute(
        "SELECT payload FROM kanban_notification_outbox WHERE subscription_id = ?",
        (subscription_id,),
    ).fetchone()
    snapshot = json.loads(row["payload"])
    assert snapshot["task"] == {
        "title": "minimal snapshot",
        "assignee": "worker",
        "result": None,
    }
    assert snapshot["event_payload"] == {"status": "visible"}
    assert "unused-session-marker" not in row["payload"]
    assert "unused-event-marker" not in row["payload"]


def test_crash_expiry_does_not_charge_unsent_batch_actions(outbox_db):
    conn, _task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=3,
        actions=[_action(1), _action(2), _action(3)],
    )

    # Each connection leases the whole batch and then "crashes" before any
    # platform send, ACK, failure record, or graceful release. Reopening a
    # real SQLite connection after lease expiry must not consume retry budget
    # for those unsent actions.
    conn.close()
    for claim_number in range(kb.NOTIFICATION_MAX_ATTEMPTS + 2):
        restarted = kb.connect()
        try:
            leased = kb.lease_notification_outbox(
                restarted,
                now=10_000 + claim_number * 10,
                lease_seconds=1,
            )
            assert len(leased) == 3
            assert {item["attempts"] for item in leased} == {0}
            rows = restarted.execute(
                "SELECT state, attempts FROM kanban_notification_outbox "
                "WHERE subscription_id = ? ORDER BY event_id",
                (subscription_id,),
            ).fetchall()
            assert [(row["state"], row["attempts"]) for row in rows] == [
                ("leased", 0),
                ("leased", 0),
                ("leased", 0),
            ]
        finally:
            restarted.close()


def test_retry_threshold_dead_letters_and_recovery_are_durable(outbox_db):
    conn, _task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=1,
        actions=[_action(1)],
    )

    for attempt in range(1, kb.NOTIFICATION_MAX_ATTEMPTS + 1):
        leased = kb.lease_notification_outbox(conn, now=attempt * 100)
        assert len(leased) == 1
        state = kb.fail_notification_delivery(
            conn,
            subscription_id=subscription_id,
            event_id=1,
            action="message",
            lease_token=leased[0]["lease_token"],
            error="temporary outage",
        )
        expected = (
            "dead_letter"
            if attempt == kb.NOTIFICATION_MAX_ATTEMPTS
            else "pending"
        )
        assert state == expected

    assert kb.notification_outbox_counts(conn) == {
        "pending": 0,
        "leased": 0,
        "acknowledged": 0,
        "dead_letter": 1,
    }
    conn.close()

    restarted = kb.connect()
    try:
        row = _outbox_row(restarted, subscription_id)
        assert row["state"] == "dead_letter"
        assert row["attempts"] == kb.NOTIFICATION_MAX_ATTEMPTS
        assert kb.requeue_dead_letter_notifications(
            restarted, subscription_id=subscription_id
        ) == 1
        recovered = kb.lease_notification_outbox(restarted, now=1_000)
        assert len(recovered) == 1
        assert recovered[0]["attempts"] == 0
    finally:
        restarted.close()


def test_release_before_send_does_not_spend_attempt(outbox_db):
    conn, _task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=1,
        actions=[_action(1)],
    )
    leased = kb.lease_notification_outbox(conn, now=100)[0]

    assert kb.release_notification_lease(
        conn,
        subscription_id=subscription_id,
        event_id=1,
        action="message",
        lease_token=leased["lease_token"],
    ) is True
    row = _outbox_row(conn, subscription_id)
    assert row["state"] == "pending"
    assert row["attempts"] == 0


def test_legacy_subscription_migration_adds_identity_and_outbox(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy-kanban.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE kanban_notify_subs ("
        "task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL, "
        "thread_id TEXT NOT NULL DEFAULT '', user_id TEXT, "
        "created_at INTEGER NOT NULL, last_event_id INTEGER DEFAULT 0, "
        "PRIMARY KEY (task_id, platform, chat_id, thread_id))"
    )
    conn.execute(
        "INSERT INTO kanban_notify_subs "
        "(task_id, platform, chat_id, thread_id, created_at, last_event_id) "
        "VALUES ('t-old', 'telegram', 'chat-old', '', 1, 0)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    migrated = kb.connect()
    try:
        sub = kb.list_notify_subs(migrated)[0]
        assert sub["subscription_id"]
        assert migrated.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='kanban_notification_outbox'"
        ).fetchone()
        assert kb.notification_outbox_counts(migrated) == {
            "pending": 0,
            "leased": 0,
            "acknowledged": 0,
            "dead_letter": 0,
        }

        # A still-running pre-outbox process writes only the original route
        # columns. The migration trigger must give that row an addressable
        # identity in the same SQLite statement.
        legacy_writer = sqlite3.connect(db_path)
        try:
            legacy_writer.execute(
                "INSERT INTO kanban_notify_subs "
                "(task_id, platform, chat_id, thread_id, created_at, last_event_id) "
                "VALUES ('t-new', 'telegram', 'chat-new', '', 2, 0)"
            )
            legacy_writer.commit()
        finally:
            legacy_writer.close()
        inserted = migrated.execute(
            "SELECT subscription_id FROM kanban_notify_subs WHERE task_id = 't-new'"
        ).fetchone()
        assert inserted is not None
        assert inserted["subscription_id"]
    finally:
        migrated.close()


def test_identity_migration_keeps_an_already_assigned_subscription_id(tmp_path, monkeypatch):
    db_path = tmp_path / "assigned-identity.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE kanban_notify_subs ("
        "subscription_id TEXT, task_id TEXT NOT NULL, platform TEXT NOT NULL, "
        "chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', user_id TEXT, "
        "created_at INTEGER NOT NULL, last_event_id INTEGER DEFAULT 0, "
        "PRIMARY KEY (task_id, platform, chat_id, thread_id))"
    )
    conn.execute(
        "INSERT INTO kanban_notify_subs "
        "(subscription_id, task_id, platform, chat_id, thread_id, created_at, last_event_id) "
        "VALUES ('writer-assigned-id', 't-assigned', 'telegram', 'chat-assigned', '', 1, 0)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    migrated = kb.connect()
    try:
        row = migrated.execute(
            "SELECT subscription_id FROM kanban_notify_subs WHERE task_id = 't-assigned'"
        ).fetchone()
        assert row["subscription_id"] == "writer-assigned-id"
    finally:
        migrated.close()


def test_rebuilds_drifted_subscription_rows_before_creating_identity_index(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "drifted-subscription-identities.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE kanban_notify_subs ("
        "subscription_id TEXT, task_id TEXT NOT NULL, platform TEXT NOT NULL, "
        "chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', user_id TEXT, "
        "notifier_profile TEXT, created_at INTEGER NOT NULL, last_event_id TEXT, "
        "PRIMARY KEY (task_id, platform, chat_id, thread_id))"
    )
    conn.executemany(
        "INSERT INTO kanban_notify_subs "
        "(subscription_id, task_id, platform, chat_id, thread_id, created_at, last_event_id) "
        "VALUES (?, ?, 'telegram', ?, '', 1, ?)",
        [
            ("", "t-empty-one", "chat-one", "4"),
            ("", "t-empty-two", "chat-two", "not-a-number"),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    migrated = kb.connect()
    try:
        rows = migrated.execute(
            "SELECT subscription_id, last_event_id FROM kanban_notify_subs "
            "ORDER BY task_id"
        ).fetchall()
        assert len(rows) == 2
        assert all(row["subscription_id"] for row in rows)
        assert len({row["subscription_id"] for row in rows}) == 2
        assert [row["last_event_id"] for row in rows] == [4, 0]
        indexes = {
            row["name"]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'kanban_notify_subs'"
            )
        }
        assert {"idx_notify_subscription_id", "idx_notify_task"} <= indexes
        legacy_table = migrated.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'kanban_notify_subs_legacy'"
        ).fetchone()
        assert legacy_table is None
    finally:
        migrated.close()


def test_identity_migration_waits_for_writer_and_preserves_assigned_identity(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "identity-writer-race.db"
    setup = sqlite3.connect(db_path)
    setup.execute(
        "CREATE TABLE kanban_notify_subs ("
        "subscription_id TEXT, task_id TEXT NOT NULL, platform TEXT NOT NULL, "
        "chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', user_id TEXT, "
        "notifier_profile TEXT, created_at INTEGER NOT NULL, "
        "last_event_id INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (task_id, platform, chat_id, thread_id))"
    )
    setup.execute(
        "INSERT INTO kanban_notify_subs "
        "(task_id, platform, chat_id, thread_id, created_at, last_event_id) "
        "VALUES ('t-race', 'telegram', 'chat-race', '', 1, 0)"
    )
    setup.commit()
    setup.close()

    writer = sqlite3.connect(db_path, isolation_level=None)
    migrator = sqlite3.connect(
        db_path,
        isolation_level=None,
        check_same_thread=False,
    )
    migrator.row_factory = sqlite3.Row
    entered_write_transaction = threading.Event()
    migration_done = threading.Event()
    errors = []
    original_write_txn = kb.write_txn

    @contextlib.contextmanager
    def observed_write_txn(connection):
        entered_write_transaction.set()
        with original_write_txn(connection):
            yield connection

    monkeypatch.setattr(kb, "write_txn", observed_write_txn)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "UPDATE kanban_notify_subs SET subscription_id = 'writer-won-id' "
        "WHERE task_id = 't-race'"
    )

    def migrate():
        try:
            kb._ensure_notify_subscription_identity(migrator)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            migration_done.set()

    thread = threading.Thread(target=migrate)
    thread.start()
    assert entered_write_transaction.wait(timeout=5)
    writer.execute("COMMIT")
    assert migration_done.wait(timeout=5)
    thread.join(timeout=5)
    assert errors == []

    try:
        row = writer.execute(
            "SELECT subscription_id FROM kanban_notify_subs WHERE task_id = 't-race'"
        ).fetchone()
        assert row[0] == "writer-won-id"
    finally:
        writer.close()
        migrator.close()


def test_delete_task_removes_pending_and_dead_letter_outbox_rows(outbox_db):
    conn, task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=2,
        actions=[_action(2, "dead letter")],
    )
    for attempt in range(kb.NOTIFICATION_MAX_ATTEMPTS):
        leased = kb.lease_notification_outbox(conn, now=1_000 + attempt)[0]
        assert kb.fail_notification_delivery(
            conn,
            subscription_id=subscription_id,
            event_id=leased["event_id"],
            action=leased["action"],
            lease_token=leased["lease_token"],
            error="temporary outage",
        ) in {"pending", "dead_letter"}
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=3,
        actions=[_action(3, "pending")],
    )
    assert kb.notification_outbox_counts(conn)["pending"] == 1
    assert kb.notification_outbox_counts(conn)["dead_letter"] == 1

    assert kb.delete_task(conn, task_id) is True
    assert conn.execute(
        "SELECT COUNT(*) FROM kanban_notification_outbox"
    ).fetchone()[0] == 0


def test_delete_archived_task_removes_pending_and_dead_letter_outbox_rows(outbox_db):
    conn, task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=2,
        actions=[_action(2, "dead letter")],
    )
    for attempt in range(kb.NOTIFICATION_MAX_ATTEMPTS):
        leased = kb.lease_notification_outbox(conn, now=2_000 + attempt)[0]
        assert kb.fail_notification_delivery(
            conn,
            subscription_id=subscription_id,
            event_id=leased["event_id"],
            action=leased["action"],
            lease_token=leased["lease_token"],
            error="temporary outage",
        ) in {"pending", "dead_letter"}
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=3,
        actions=[_action(3, "pending")],
    )
    assert kb.archive_task(conn, task_id) is True
    assert kb.delete_archived_task(conn, task_id) is True
    assert conn.execute(
        "SELECT COUNT(*) FROM kanban_notification_outbox"
    ).fetchone()[0] == 0


def test_terminal_cleanup_cannot_remove_a_recreated_subscription(outbox_db):
    conn, task_id, old_subscription_id = outbox_db
    assert kb.complete_task(conn, task_id, summary="done")
    assert kb.remove_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="chat-1",
    ) is True
    replacement_subscription_id = kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="chat-1",
    )

    assert kb.remove_notify_sub_if_settled(
        conn,
        subscription_id=old_subscription_id,
        task_id=task_id,
        platform="telegram",
        chat_id="chat-1",
    ) is False
    assert kb.list_notify_subs(conn, task_id)[0]["subscription_id"] == replacement_subscription_id


def test_terminal_cleanup_requires_acknowledgement_for_pending_leased_and_dead_rows(
    outbox_db,
):
    conn, task_id, subscription_id = outbox_db
    assert kb.complete_task(conn, task_id, summary="done")
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=1,
        actions=[_action(1)],
    )

    def cleanup() -> bool:
        return kb.remove_notify_sub_if_settled(
            conn,
            subscription_id=subscription_id,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )

    assert cleanup() is False
    leased = kb.lease_notification_outbox(conn, now=4_000)[0]
    assert cleanup() is False
    assert kb.fail_notification_delivery(
        conn,
        subscription_id=subscription_id,
        event_id=leased["event_id"],
        action=leased["action"],
        lease_token=leased["lease_token"],
        error="temporary outage",
    ) == "pending"
    for attempt in range(1, kb.NOTIFICATION_MAX_ATTEMPTS):
        leased = kb.lease_notification_outbox(conn, now=4_000 + attempt)[0]
        state = kb.fail_notification_delivery(
            conn,
            subscription_id=subscription_id,
            event_id=leased["event_id"],
            action=leased["action"],
            lease_token=leased["lease_token"],
            error="temporary outage",
        )
    assert state == "dead_letter"
    assert cleanup() is False

    assert kb.requeue_dead_letter_notifications(
        conn,
        subscription_id=subscription_id,
    ) == 1
    recovered = kb.lease_notification_outbox(conn, now=5_000)[0]
    assert kb.ack_notification_delivery(
        conn,
        subscription_id=subscription_id,
        event_id=recovered["event_id"],
        action=recovered["action"],
        lease_token=recovered["lease_token"],
    ) is True
    assert cleanup() is True


def test_delivery_error_is_redacted_before_outbox_storage(outbox_db):
    conn, _task_id, subscription_id = outbox_db
    kb.stage_notification_batch(
        conn,
        subscription_id=subscription_id,
        new_cursor=1,
        actions=[_action(1)],
    )
    leased = kb.lease_notification_outbox(conn, now=3_000)[0]
    secret = "notification-secret-1234567890"
    assert kb.fail_notification_delivery(
        conn,
        subscription_id=subscription_id,
        event_id=leased["event_id"],
        action=leased["action"],
        lease_token=leased["lease_token"],
        error=(
            "websocket rejected wss://example.invalid/socket?access_token="
            f"{secret} Authorization: Bearer {secret}"
        ),
    ) == "pending"

    stored = _outbox_row(conn, subscription_id)["last_error"]
    assert secret not in stored
    assert "access_token=" in stored
