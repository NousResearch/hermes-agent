"""Focused durable-inbound contract tests for Telegram."""

import asyncio
import os
import sqlite3
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import plugins.platforms.telegram.inbound_store as inbound_store_module
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram.inbound_store import (
    MAX_ATTEMPTS,
    CaptureDecision,
    DurableTelegramUpdateQueue,
    TelegramInboundStore,
    TelegramQueueLifecycleRegistry,
    bot_account_key,
    canonical_bot_account_id,
)


def payload(update_id, *, chat_id="10", message_id="20", text="same"):
    return {
        "update_id": update_id,
        "message": {
            "message_id": int(message_id),
            "date": 1,
            "chat": {"id": int(chat_id), "type": "private"},
            "from": {"id": 30, "is_bot": False, "first_name": "U"},
            "text": text,
        },
    }


def decision(data, *, actionable=True, profile="default", update_kind="message"):
    message = data["message"]
    return CaptureDecision(
        actionable=actionable,
        profile=profile,
        account_id="telegram",
        update_kind=update_kind,
        chat_id=str(message["chat"]["id"]),
        message_id=str(message["message_id"]),
        session_key=f"telegram:dm:{message['chat']['id']}",
        priority=10,
        payload=data,
        receipt_required=False,
    )


def test_existing_v1_schema_accepts_new_persist(tmp_path):
    db_path = tmp_path / "telegram_inbound.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE telegram_inbound_event (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                bot_account_id INTEGER NOT NULL,
                update_id INTEGER NOT NULL,
                profile TEXT NOT NULL,
                account_id TEXT NOT NULL,
                update_kind TEXT NOT NULL,
                chat_id TEXT,
                message_id TEXT,
                callback_query_id TEXT,
                session_key TEXT NOT NULL,
                priority INTEGER NOT NULL,
                payload_json BLOB,
                payload_sha256 TEXT NOT NULL,
                work_state TEXT NOT NULL,
                receipt_state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_epoch INTEGER NOT NULL DEFAULT 0,
                lease_expires_at REAL,
                next_attempt_at REAL,
                received_at REAL NOT NULL,
                persisted_at REAL NOT NULL,
                queued_at REAL,
                leased_at REAL,
                context_committed_at REAL,
                consumed_at REAL,
                terminal_at REAL,
                receipt_attempted_at REAL,
                receipt_confirmed_at REAL,
                duplicate_count INTEGER NOT NULL DEFAULT 0,
                last_duplicate_at REAL,
                replay_count INTEGER NOT NULL DEFAULT 0,
                last_replay_at REAL,
                identity_conflict_count INTEGER NOT NULL DEFAULT 0,
                last_error_class TEXT,
                terminal_reason TEXT,
                dispatch_state TEXT NOT NULL DEFAULT 'pending',
                UNIQUE(bot_account_id, update_id)
            );
            CREATE TABLE telegram_inbound_alias (
                bot_account_id INTEGER NOT NULL,
                alias_kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                alias_value TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(bot_account_id, alias_kind, scope, alias_value)
            );
            """
        )

    store = TelegramInboundStore(db_path)
    result = store.persist(
        bot_account_id=111,
        update_id=6,
        decision=decision(payload(6)),
        now=1.0,
    )

    assert result.row is not None
    assert result.row.work_state == "queued"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT receipt_state FROM telegram_inbound_event WHERE event_id=?",
            (result.event_id,),
        ).fetchone() == ("not_required",)


def test_existing_v1_schema_migrates_identifiers_to_lossless_text(tmp_path):
    db_path = tmp_path / "telegram_inbound.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE telegram_inbound_event (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                bot_account_id INTEGER NOT NULL,
                update_id INTEGER NOT NULL,
                profile TEXT NOT NULL,
                account_id TEXT NOT NULL,
                update_kind TEXT NOT NULL,
                chat_id TEXT,
                message_id TEXT,
                callback_query_id TEXT,
                session_key TEXT NOT NULL,
                priority INTEGER NOT NULL,
                payload_json BLOB,
                payload_sha256 TEXT NOT NULL,
                work_state TEXT NOT NULL,
                receipt_state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_epoch INTEGER NOT NULL DEFAULT 0,
                lease_expires_at REAL,
                next_attempt_at REAL,
                received_at REAL NOT NULL,
                persisted_at REAL NOT NULL,
                queued_at REAL,
                leased_at REAL,
                context_committed_at REAL,
                consumed_at REAL,
                terminal_at REAL,
                receipt_attempted_at REAL,
                receipt_confirmed_at REAL,
                duplicate_count INTEGER NOT NULL DEFAULT 0,
                last_duplicate_at REAL,
                replay_count INTEGER NOT NULL DEFAULT 0,
                last_replay_at REAL,
                identity_conflict_count INTEGER NOT NULL DEFAULT 0,
                last_error_class TEXT,
                terminal_reason TEXT,
                dispatch_state TEXT NOT NULL DEFAULT 'pending',
                UNIQUE(bot_account_id, update_id)
            );
            CREATE TABLE telegram_inbound_alias (
                bot_account_id INTEGER NOT NULL,
                alias_kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                alias_value TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(bot_account_id, alias_kind, scope, alias_value),
                FOREIGN KEY(event_id) REFERENCES telegram_inbound_event(event_id)
            );
            CREATE INDEX custom_alias_scope ON telegram_inbound_alias(scope);
            INSERT INTO telegram_inbound_event (
                event_id, bot_account_id, update_id, profile, account_id,
                update_kind, session_key, priority, payload_json, payload_sha256,
                work_state, receipt_state, received_at, persisted_at, dispatch_state
            ) VALUES (
                'telegram:111:8', 111, 8, 'default', 'telegram', 'message',
                'telegram:dm:10', 10, '{}', 'legacy-hash', 'queued',
                'not_required', 0.5, 0.5, 'pending'
            );
            INSERT INTO telegram_inbound_alias (
                bot_account_id, alias_kind, scope, alias_value, event_id
            ) VALUES (111, 'message', 'chat:10', '20', 'telegram:111:8');
            UPDATE sqlite_sequence SET seq=50
            WHERE name='telegram_inbound_event';
            """
        )

    store = TelegramInboundStore(db_path)
    huge = 2**63
    update_results = [
        store.persist(
            bot_account_id=111,
            update_id=huge + offset,
            decision=decision(payload(huge + offset)),
            now=1.0 + offset,
        )
        for offset in range(2)
    ]
    bot_results = [
        store.persist(
            bot_account_id=huge + offset,
            update_id=9,
            decision=decision(payload(9)),
            now=3.0 + offset,
        )
        for offset in range(2)
    ]

    assert all(not result.duplicate for result in update_results + bot_results)
    assert len({result.event_id for result in update_results}) == 2
    assert len({result.event_id for result in bot_results}) == 2
    with sqlite3.connect(db_path) as conn:
        event_types = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(telegram_inbound_event)")
        }
        alias_types = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(telegram_inbound_alias)")
        }
        stored_types = conn.execute(
            "SELECT DISTINCT typeof(bot_account_id), typeof(update_id) "
            "FROM telegram_inbound_event"
        ).fetchall()
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        legacy_row = conn.execute(
            "SELECT bot_account_id, update_id FROM telegram_inbound_event "
            "WHERE event_id='telegram:111:8'"
        ).fetchone()
        legacy_alias = conn.execute(
            "SELECT bot_account_id, event_id FROM telegram_inbound_alias "
            "WHERE alias_value='20'"
        ).fetchone()
        custom_index = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='custom_alias_scope'"
        ).fetchone()
        first_new_seq = conn.execute(
            "SELECT seq FROM telegram_inbound_event WHERE event_id=?",
            (update_results[0].event_id,),
        ).fetchone()
    assert event_types["bot_account_id"].upper() == "TEXT"
    assert event_types["update_id"].upper() == "TEXT"
    assert alias_types["bot_account_id"].upper() == "TEXT"
    assert stored_types == [("text", "text")]
    assert foreign_key_errors == []
    assert legacy_row == ("111", "8")
    assert legacy_alias == ("111", "telegram:111:8")
    assert custom_index == (
        "CREATE INDEX custom_alias_scope ON telegram_inbound_alias(scope)",
    )
    assert first_new_seq == (51,)


def test_identifier_migration_rejects_conflicting_canonical_index(tmp_path):
    db_path = tmp_path / "telegram_inbound.db"
    conflicting_sql = (
        "CREATE INDEX telegram_inbound_event_account_update "
        "ON telegram_inbound_event(session_key)"
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            f"""
            CREATE TABLE telegram_inbound_event (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                bot_account_id INTEGER NOT NULL,
                update_id INTEGER NOT NULL,
                profile TEXT NOT NULL,
                account_id TEXT NOT NULL,
                update_kind TEXT NOT NULL,
                session_key TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100,
                work_state TEXT NOT NULL DEFAULT 'queued',
                dispatch_state TEXT NOT NULL DEFAULT 'pending',
                received_at REAL NOT NULL,
                persisted_at REAL NOT NULL
            );
            CREATE TABLE telegram_inbound_alias (
                bot_account_id INTEGER NOT NULL,
                alias_kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                alias_value TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(bot_account_id, alias_kind, scope, alias_value),
                FOREIGN KEY(event_id) REFERENCES telegram_inbound_event(event_id)
            );
            {conflicting_sql};
            INSERT INTO telegram_inbound_event (
                event_id, bot_account_id, update_id, profile, account_id,
                update_kind, session_key, received_at, persisted_at
            ) VALUES (
                'telegram:111:8', 111, 8, 'default', 'telegram', 'message',
                'telegram:dm:10', 0.5, 0.5
            );
            """
        )

    with pytest.raises(RuntimeError, match="conflicting canonical index"):
        TelegramInboundStore(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(telegram_inbound_event)")
        ]
        affinity = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(telegram_inbound_event)")
        }
        index_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='telegram_inbound_event_account_update'"
        ).fetchone()
        rows = conn.execute(
            "SELECT event_id, bot_account_id, update_id "
            "FROM telegram_inbound_event"
        ).fetchall()
    assert affinity["bot_account_id"].upper() == "INTEGER"
    assert affinity["update_id"].upper() == "INTEGER"
    assert "receipt_state" not in columns
    assert "next_attempt_at" not in columns
    assert index_sql == (conflicting_sql,)
    assert rows == [("telegram:111:8", 111, 8)]


def test_identifier_migration_rejects_global_canonical_index_collision(tmp_path):
    db_path = tmp_path / "telegram_inbound.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE unrelated(value TEXT);
            CREATE INDEX telegram_inbound_event_account_update
                ON unrelated(value);
            CREATE TABLE telegram_inbound_event (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                bot_account_id INTEGER NOT NULL,
                update_id INTEGER NOT NULL,
                profile TEXT NOT NULL,
                account_id TEXT NOT NULL,
                update_kind TEXT NOT NULL,
                session_key TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100,
                work_state TEXT NOT NULL DEFAULT 'queued',
                dispatch_state TEXT NOT NULL DEFAULT 'pending',
                received_at REAL NOT NULL,
                persisted_at REAL NOT NULL
            );
            CREATE TABLE telegram_inbound_alias (
                bot_account_id INTEGER NOT NULL,
                alias_kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                alias_value TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(bot_account_id, alias_kind, scope, alias_value),
                FOREIGN KEY(event_id) REFERENCES telegram_inbound_event(event_id)
            );
            """
        )

    with pytest.raises(RuntimeError, match="conflicting canonical index"):
        TelegramInboundStore(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(telegram_inbound_event)")
        ]
        collision_owner = conn.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type='index' "
            "AND name='telegram_inbound_event_account_update'"
        ).fetchone()
    assert "receipt_state" not in columns
    assert "next_attempt_at" not in columns
    assert collision_owner == ("unrelated",)


def test_identifier_migration_rejects_lossy_real_alias(tmp_path):
    db_path = tmp_path / "telegram_inbound.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE telegram_inbound_event (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                bot_account_id INTEGER NOT NULL,
                update_id INTEGER NOT NULL,
                profile TEXT NOT NULL,
                account_id TEXT NOT NULL,
                update_kind TEXT NOT NULL,
                session_key TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100,
                work_state TEXT NOT NULL DEFAULT 'queued',
                dispatch_state TEXT NOT NULL DEFAULT 'pending',
                received_at REAL NOT NULL,
                persisted_at REAL NOT NULL
            );
            CREATE TABLE telegram_inbound_alias (
                bot_account_id INTEGER NOT NULL,
                alias_kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                alias_value TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(bot_account_id, alias_kind, scope, alias_value),
                FOREIGN KEY(event_id) REFERENCES telegram_inbound_event(event_id)
            );
            INSERT INTO telegram_inbound_event (
                event_id, bot_account_id, update_id, profile, account_id,
                update_kind, session_key, received_at, persisted_at
            ) VALUES (
                'telegram:111:8', 111, 8, 'default', 'telegram', 'message',
                'telegram:dm:10', 0.5, 0.5
            );
            INSERT INTO telegram_inbound_alias (
                bot_account_id, alias_kind, scope, alias_value, event_id
            ) VALUES (
                9223372036854775808, 'message', 'chat:10', '20',
                'telegram:111:8'
            );
            """
        )
        assert conn.execute(
            "SELECT typeof(bot_account_id) FROM telegram_inbound_alias"
        ).fetchone() == ("real",)

    with pytest.raises(RuntimeError, match="lossy REAL identifiers"):
        TelegramInboundStore(db_path)


@pytest.mark.asyncio
async def test_cancelled_close_releases_shared_store_executor(tmp_path, monkeypatch):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        after_commit=None,
        lease_owner="gateway:test",
        active_limit=1,
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_wait_for_ingress_tasks():
        entered.set()
        await release.wait()

    monkeypatch.setattr(queue, "_wait_for_ingress_tasks", blocked_wait_for_ingress_tasks)
    close_task = asyncio.create_task(queue.close())
    await asyncio.wait_for(entered.wait(), timeout=0.5)
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert queue._store_executor_shutdown
    assert queue._store_executor_key not in inbound_store_module._STORE_EXECUTORS
    release.set()


def test_reclaim_process_leases_is_scoped_to_one_bot_account(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    account_111 = store.persist(
        bot_account_id=111,
        update_id=7,
        decision=decision(payload(7)),
        now=1.0,
    )
    account_222 = store.persist(
        bot_account_id=222,
        update_id=7,
        decision=decision(payload(7)),
        now=1.0,
    )
    store.lease_event(account_111.event_id, owner="old-111", now=2.0, lease_seconds=100.0)
    store.lease_event(account_222.event_id, owner="old-222", now=2.0, lease_seconds=100.0)

    assert store.reclaim_process_leases(
        bot_account_id=111, current_owner="new-111", now=3.0
    ) == 1
    assert store.get(account_111.event_id).work_state == "queued"
    assert store.get(account_222.event_id).work_state == "leased"
    assert store.get(account_222.event_id).lease_owner == "old-222"

    assert store.reclaim_process_leases(
        bot_account_id=222, current_owner="new-222", now=4.0
    ) == 1
    assert store.get(account_222.event_id).work_state == "queued"
    assert store.get(account_111.event_id).work_state == "queued"


def test_recover_admitted_dispatches_is_scoped_to_one_bot_account(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    account_111 = store.persist(
        bot_account_id=111,
        update_id=8,
        decision=decision(payload(8)),
        now=1.0,
    )
    account_222 = store.persist(
        bot_account_id=222,
        update_id=8,
        decision=decision(payload(8)),
        now=1.0,
    )
    assert store.mark_dispatch_admitted(account_111.event_id)
    assert store.mark_dispatch_admitted(account_222.event_id)

    assert store.recover_admitted_dispatches(bot_account_id=111, now=3.0) == 1
    assert store.get(account_111.event_id).dispatch_state == "pending"
    assert store.get(account_222.event_id).dispatch_state == "admitted"

    assert store.recover_admitted_dispatches(bot_account_id=222, now=4.0) == 1
    assert store.get(account_222.event_id).dispatch_state == "pending"
    assert store.get(account_111.event_id).dispatch_state == "pending"


def test_large_bot_account_id_round_trips_and_uses_text_safe_key(tmp_path):
    account_id = 2**63
    adjacent_id = account_id + 1
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    result = store.persist(
        bot_account_id=account_id,
        update_id=9,
        decision=decision(payload(9)),
        now=1.0,
    )
    adjacent = store.persist(
        bot_account_id=adjacent_id,
        update_id=9,
        decision=decision(payload(9)),
        now=1.0,
    )

    assert canonical_bot_account_id(account_id) == "9223372036854775808"
    assert bot_account_key(account_id) == "telegram-account:9223372036854775808"
    assert result.event_id == "telegram:9223372036854775808:9"
    assert adjacent.event_id == "telegram:9223372036854775809:9"
    assert result.event_id != adjacent.event_id
    row = store.get(result.event_id)
    assert row is not None
    assert row.bot_account_id == account_id
    assert store.event_id_for_update(account_id, 9) == result.event_id
    assert store.event_id_for_update(adjacent_id, 9) == adjacent.event_id


def test_update_identity_is_account_scoped_and_text_is_not_identity(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    first = store.persist(
        bot_account_id=111,
        update_id=10,
        decision=decision(payload(10, message_id="30", text="same")),
        now=1.0,
    )
    other_account = store.persist(
        bot_account_id=222,
        update_id=10,
        decision=decision(payload(10, message_id="30", text="same")),
        now=2.0,
    )
    second_update = store.persist(
        bot_account_id=111,
        update_id=11,
        decision=decision(payload(11, message_id="31", text="same")),
        now=3.0,
    )

    assert not first.duplicate
    assert not other_account.duplicate
    assert not second_update.duplicate
    assert len({first.event_id, other_account.event_id, second_update.event_id}) == 3
    assert [row.update_id for row in store.eligible(bot_account_id=111)] == [10, 11]
    assert [row.update_id for row in store.eligible(bot_account_id=222)] == [10]


def test_distinct_update_ids_do_not_collapse_when_message_shape_repeats(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    first = store.persist(
        bot_account_id=111,
        update_id=12,
        decision=decision(payload(12, message_id="30", text="same")),
        now=1.0,
    )
    second = store.persist(
        bot_account_id=111,
        update_id=13,
        decision=decision(payload(13, message_id="30", text="same")),
        now=2.0,
    )

    assert not first.duplicate
    assert not second.duplicate
    assert first.event_id == "telegram:111:12"
    assert second.event_id == "telegram:111:13"
    assert [row.update_id for row in store.eligible(bot_account_id=111)] == [12, 13]


@pytest.mark.asyncio
async def test_durable_admission_precedes_commit_callback_and_preserves_sentinels(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    callback_observations = []

    def classify(item):
        return decision(item, actionable=item["update_id"] != 2)

    async def after_commit(item, result):
        row = store.get(result.event_id)
        callback_observations.append(
            (item["update_id"], row is not None, row.dispatch_state if row else None)
        )

    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=classify,
        after_commit=after_commit,
        lease_owner="gateway:test",
        active_limit=4,
    )

    await queue.put(payload(1))
    assert callback_observations == [(1, True, "admitted")]
    item = await asyncio.wait_for(queue.get(), timeout=0.5)
    assert item["update_id"] == 1
    claim = queue.claim_for_update(1)
    assert claim is not None
    assert store.mark_consumed(
        claim.event_id,
        owner=claim.lease_owner,
        lease_epoch=claim.lease_epoch,
    )
    queue.task_done()
    queue.forget_claims({claim.event_id})

    sentinel = object()
    await queue.put(sentinel)
    assert await asyncio.wait_for(queue.get(), timeout=0.5) is sentinel
    queue.task_done()

    rejected = payload(2)
    await queue.put(rejected)
    assert store.get("telegram:111:2") is None
    assert await asyncio.wait_for(queue.get(), timeout=0.5) == rejected
    queue.task_done()
    assert callback_observations == [(1, True, "admitted")]


@pytest.mark.asyncio
async def test_durable_queue_recovery_pages_only_its_account(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    first = store.persist(
        bot_account_id=111,
        update_id=20,
        decision=decision(payload(20)),
        now=1.0,
    )
    second = store.persist(
        bot_account_id=222,
        update_id=20,
        decision=decision(payload(20)),
        now=1.0,
    )
    assert store.mark_dispatch_admitted(first.event_id)
    assert store.mark_dispatch_admitted(second.event_id)
    assert store.recover_admitted_dispatches(bot_account_id=222, now=2.0) == 1

    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=222,
        classifier=lambda item: decision(item),
        lease_owner="gateway:222",
    )
    assert await queue.wake_scheduler() == 1
    item = await queue.get()
    assert item["update_id"] == 20
    claim = queue.claim_for_update(20)
    assert claim is not None
    assert claim.event_id == second.event_id
    assert store.get(first.event_id).dispatch_state == "admitted"
    queue.task_done()


@pytest.mark.asyncio
async def test_projection_suspend_cancels_queued_wake_without_reprojection(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:projection-suspend",
        active_limit=1,
    )

    await queue.put(payload(29))
    assert (await queue.get())["update_id"] == 29
    assert queue.claim_for_update(29) is not None
    queue.task_done()
    assert await queue.complete_update(29, success=False)
    row = store.get("telegram:111:29")
    assert row is not None
    assert row.dispatch_state == "pending"

    await queue.suspend_projection()
    await asyncio.sleep(0)
    row = store.get("telegram:111:29")
    assert row is not None
    assert row.dispatch_state == "pending"
    task = queue._projection_retry_task
    assert task is None or task.done()


@pytest.mark.asyncio
async def test_restart_handoff_requeues_abandoned_and_transfers_live_handler_claim(
    tmp_path,
):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    abandoned = store.persist(
        bot_account_id=111,
        update_id=30,
        decision=decision(payload(30, message_id="40")),
        now=1.0,
    )
    live = store.persist(
        bot_account_id=111,
        update_id=31,
        decision=decision(payload(31, message_id="41")),
        now=1.0,
    )

    old_queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:old",
        active_limit=2,
    )
    assert await old_queue.wake_scheduler() == 2
    assert (await old_queue.get())["update_id"] == 30
    assert (await old_queue.get())["update_id"] == 31
    old_queue.mark_handler_claim(31)

    replacement = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:new",
        active_limit=2,
    )
    result = await replacement.handoff_from(old_queue)

    assert result == {"requeued": 1, "transferred": 1, "quarantined": 0}
    assert old_queue._store_executor_shutdown
    assert old_queue._store_executor is replacement._store_executor
    assert store.get(abandoned.event_id).work_state == "queued"
    assert store.get(live.event_id).lease_owner == "gateway:new"
    transferred = replacement.claim_for_update(31)
    assert transferred is not None
    assert transferred.event_id == live.event_id
    assert transferred.lease_owner == "gateway:new"

    assert await replacement.wake_scheduler() == 0
    replay = await replacement.get()
    assert replay["update_id"] == 30
    replacement.task_done()


@pytest.mark.asyncio
async def test_lifecycle_registry_unifies_symlink_parent_aliases(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    real_path = real_parent / "telegram_inbound.db"
    alias_path = alias_parent / "telegram_inbound.db"

    old_store = TelegramInboundStore(real_path)
    replacement_store = TelegramInboundStore(alias_path)
    assert os.path.samefile(real_path, alias_path)

    old_queue = DurableTelegramUpdateQueue(
        store=old_store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:real-parent",
        active_limit=1,
    )
    replacement = DurableTelegramUpdateQueue(
        store=replacement_store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:alias-parent",
        active_limit=1,
    )

    await old_queue.put(payload(310, message_id="410"))
    assert (await old_queue.get())["update_id"] == 310
    old_queue.mark_handler_claim(310)
    old_queue.task_done()
    persisted = old_store.get("telegram:111:310")
    assert persisted is not None

    TelegramQueueLifecycleRegistry.observe(old_queue)
    assert TelegramQueueLifecycleRegistry.key_for_queue(
        old_queue
    ) == TelegramQueueLifecycleRegistry.key_for_queue(replacement)
    assert await asyncio.wait_for(
        TelegramQueueLifecycleRegistry.recover(replacement), timeout=0.5
    ) == 0

    assert old_queue.handler_claim_fenced(310)
    transferred = replacement.claim_for_update(310)
    assert transferred is not None
    assert transferred.event_id == persisted.event_id
    assert transferred.lease_owner == "gateway:alias-parent"
    assert await replacement.complete_update(310, success=True)
    assert not await old_queue.complete_update(310, success=True)
    consumed = replacement_store.get(persisted.event_id)
    assert consumed is not None
    assert consumed.work_state == "consumed"


@pytest.mark.asyncio
async def test_handoff_discards_stale_old_projection_without_redirecting_to_replacement(tmp_path):
    """A retired queue must never lend replacement work to its old adapter."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    persisted = store.persist(
        bot_account_id=111,
        update_id=32,
        decision=decision(payload(32)),
        now=1.0,
    )
    old_queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:old-stale-projection",
        active_limit=1,
    )
    assert await old_queue.wake_scheduler() == 1

    replacement = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:new-stale-projection",
        active_limit=1,
    )
    assert await replacement.handoff_from(old_queue) == {
        "requeued": 0,
        "transferred": 0,
        "quarantined": 0,
    }

    sentinel = object()
    old_queue.put_nowait(sentinel)
    assert await asyncio.wait_for(old_queue.get(), timeout=0.5) is sentinel
    old_queue.task_done()

    replay = await asyncio.wait_for(replacement.get(), timeout=0.5)
    assert replay["update_id"] == 32
    claim = replacement.claim_for_update(32)
    assert claim is not None
    assert claim.event_id == persisted.event_id
    replacement.task_done()
    assert await replacement.complete_update(32, success=True)


@pytest.mark.asyncio
async def test_handoff_waits_for_inflight_get_before_transferring_claim(tmp_path, monkeypatch):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    persisted = store.persist(
        bot_account_id=111,
        update_id=33,
        decision=decision(payload(33)),
        now=1.0,
    )
    old_queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:old-inflight",
        active_limit=1,
    )
    assert await old_queue.wake_scheduler() == 1

    lease_returned = threading.Event()
    release_lease = threading.Event()
    original_lease = store.lease_event

    def delayed_lease(*args, **kwargs):
        row = original_lease(*args, **kwargs)
        lease_returned.set()
        assert release_lease.wait(timeout=2.0)
        return row

    monkeypatch.setattr(store, "lease_event", delayed_lease)
    get_task = asyncio.create_task(old_queue.get())
    await asyncio.to_thread(lease_returned.wait, 1.0)

    replacement = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:new-inflight",
        active_limit=1,
    )
    handoff_task = asyncio.create_task(replacement.handoff_from(old_queue))
    await asyncio.sleep(0.02)
    assert not handoff_task.done()

    release_lease.set()
    item = await asyncio.wait_for(get_task, timeout=1.0)
    result = await asyncio.wait_for(handoff_task, timeout=1.0)

    assert item["update_id"] == 33
    assert result["transferred"] == 1
    assert result["requeued"] == 0
    transferred = replacement.claim_for_update(33)
    assert transferred is not None
    assert transferred.event_id == persisted.event_id
    assert store.get(persisted.event_id).lease_owner == "gateway:new-inflight"


@pytest.mark.asyncio
async def test_cancelled_get_requeues_after_sqlite_lease_commit(tmp_path, monkeypatch):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    persisted = store.persist(
        bot_account_id=111,
        update_id=34,
        decision=decision(payload(34)),
        now=1.0,
    )
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:cancelled-get",
        active_limit=1,
    )
    assert await queue.wake_scheduler() == 1

    lease_returned = threading.Event()
    release_lease = threading.Event()
    original_lease = store.lease_event

    def delayed_lease(*args, **kwargs):
        row = original_lease(*args, **kwargs)
        lease_returned.set()
        assert release_lease.wait(timeout=2.0)
        return row

    monkeypatch.setattr(store, "lease_event", delayed_lease)
    get_task = asyncio.create_task(queue.get())
    await asyncio.to_thread(lease_returned.wait, 1.0)
    get_task.cancel()
    release_lease.set()

    with pytest.raises(asyncio.CancelledError):
        await get_task

    row = store.get(persisted.event_id)
    assert row is not None
    assert row.work_state == "queued"
    assert row.lease_owner is None
    assert queue.claim_for_update(34) is None


@pytest.mark.asyncio
async def test_cancelled_put_after_durable_commit_recovers_admission(tmp_path, monkeypatch):
    """Cancellation during admission must not strand a committed update."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:cancelled-put-admission",
        active_limit=1,
    )
    admission_started = asyncio.Event()
    release_admission = asyncio.Event()
    original_wake_scheduler = queue.wake_scheduler
    wake_calls = 0

    async def block_first_admission():
        nonlocal wake_calls
        wake_calls += 1
        if wake_calls == 1:
            admission_started.set()
            await release_admission.wait()
        return await original_wake_scheduler()

    monkeypatch.setattr(queue, "wake_scheduler", block_first_admission)
    put_task = asyncio.create_task(queue.put(payload(35)))
    try:
        await asyncio.wait_for(admission_started.wait(), timeout=0.5)
        row = store.get("telegram:111:35")
        assert row is not None
        assert row.dispatch_state == "pending"

        put_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await put_task

        release_admission.set()
        replay = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert replay["update_id"] == 35
        queue.task_done()
        assert await queue.complete_update(35, success=True)
    finally:
        release_admission.set()
        if not put_task.done():
            put_task.cancel()
            await asyncio.gather(put_task, return_exceptions=True)
        await queue.close()


@pytest.mark.asyncio
async def test_cancelled_handoff_put_recovers_successor_admission(tmp_path, monkeypatch):
    """A committed predecessor put must retry projection through its successor."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    predecessor = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:cancelled-handoff-predecessor",
        active_limit=1,
    )
    successor = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:cancelled-handoff-successor",
        active_limit=1,
    )
    original_persist = predecessor._persist_with_retry

    async def persist_then_publish_successor(*, update_id, decision):
        result = await original_persist(update_id=update_id, decision=decision)
        with predecessor._claim_handoff_lock:
            predecessor._handoff_target = successor
        return result

    monkeypatch.setattr(predecessor, "_persist_with_retry", persist_then_publish_successor)
    admission_started = asyncio.Event()
    release_admission = asyncio.Event()
    original_wake_scheduler = successor.wake_scheduler
    wake_calls = 0

    async def block_first_successor_admission():
        nonlocal wake_calls
        wake_calls += 1
        if wake_calls == 1:
            admission_started.set()
            await release_admission.wait()
        return await original_wake_scheduler()

    monkeypatch.setattr(successor, "wake_scheduler", block_first_successor_admission)
    put_task = asyncio.create_task(predecessor.put(payload(36)))
    try:
        await asyncio.wait_for(admission_started.wait(), timeout=0.5)
        row = store.get("telegram:111:36")
        assert row is not None
        assert row.dispatch_state == "pending"

        put_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await put_task

        release_admission.set()
        replay = await asyncio.wait_for(successor.get(), timeout=0.5)
        assert replay["update_id"] == 36
        successor.task_done()
        assert await successor.complete_update(36, success=True)
    finally:
        release_admission.set()
        if not put_task.done():
            put_task.cancel()
            await asyncio.gather(put_task, return_exceptions=True)
        await predecessor.close()
        await successor.close()


@pytest.mark.asyncio
async def test_transient_lease_failure_does_not_escape_queue_get(tmp_path, monkeypatch):
    """A local lease fault must be retried inside the PTB queue boundary."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:locked-old",
        active_limit=1,
    )
    await queue.put(payload(8001, message_id="9001"))
    original_lease = store.lease_event
    lease_calls = 0

    def transient_lease(*args, **kwargs):
        nonlocal lease_calls
        lease_calls += 1
        if lease_calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_lease(*args, **kwargs)

    monkeypatch.setattr(store, "lease_event", transient_lease)
    queue._persist_retry_initial_seconds = 0.01
    try:
        replay = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert replay["update_id"] == 8001
        assert lease_calls == 2
        queue.task_done()
        assert await queue.complete_update(8001, success=True)
    finally:
        await queue.suspend_projection()


def test_sqlite_inbox_and_wal_sidecars_are_private_under_permissive_umask(tmp_path):
    path = tmp_path / "telegram_inbound.db"
    previous_umask = os.umask(0o022)
    try:
        store = TelegramInboundStore(path)
        store.persist(
            bot_account_id=111,
            update_id=35,
            decision=decision(payload(35)),
            now=1.0,
        )
    finally:
        os.umask(previous_umask)

    candidates = [path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")]
    existing = [candidate for candidate in candidates if candidate.exists()]
    assert existing
    assert all(stat.S_IMODE(candidate.stat().st_mode) == 0o600 for candidate in existing)


def test_control_started_failure_is_terminal_and_retry_budget_is_bounded(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    control = store.persist(
        bot_account_id=111,
        update_id=36,
        decision=decision(payload(36), update_kind="callback_query"),
        now=1.0,
    )
    leased = store.lease_event(control.event_id, owner="gateway:control", now=2.0)
    assert leased is not None
    assert store.mark_control_started(
        control.event_id,
        owner="gateway:control",
        lease_epoch=leased.lease_epoch,
    )
    assert store.requeue(
        control.event_id,
        owner="gateway:control",
        lease_epoch=leased.lease_epoch,
        now=3.0,
        error_class="handler_failed",
    )
    control_row = store.get(control.event_id)
    assert control_row.work_state == "dead_letter"
    assert control_row.terminal_reason == "control_effect_failed"

    retry = store.persist(
        bot_account_id=111,
        update_id=37,
        decision=decision(payload(37)),
        now=1.0,
    )
    for attempt in range(1, 4):
        leased = store.lease_event(retry.event_id, owner="gateway:retry", now=10.0 * attempt)
        assert leased is not None
        assert store.requeue(
            retry.event_id,
            owner="gateway:retry",
            lease_epoch=leased.lease_epoch,
            now=10.0 * attempt,
            error_class="handler_failed",
        )
    retry_row = store.get(retry.event_id)
    assert retry_row.work_state == "dead_letter"
    assert retry_row.terminal_reason == "retry_budget_exhausted"


@pytest.mark.asyncio
async def test_deferred_completion_requeues_without_consuming_retry_budget(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    persisted = store.persist(
        bot_account_id=111,
        update_id=38,
        decision=decision(payload(38)),
        now=1.0,
    )
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:deferred-cap",
        active_limit=1,
    )

    assert await queue.wake_scheduler() == 1
    for defer_round in range(MAX_ATTEMPTS + 1):
        if defer_round:
            await queue.wake_scheduler()
        item = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert item["update_id"] == 38
        claim = queue.claim_for_update(38)
        assert claim is not None
        queue.task_done()

        assert await queue.complete_update(
            38,
            success=False,
            delay=0.0,
            defer=True,
        )
        row = store.get(persisted.event_id)
        assert row is not None
        assert row.work_state == "queued"
        assert row.dispatch_state == "pending"
        assert row.last_error_class == "busy_cap"
        assert row.attempt_count == 0
        assert store.pending_dispatch(now=time.time(), limit=10) == [row]

    # A real owner failure after repeated deferred projections consumes one
    # retry and remains replayable instead of inheriting the deferral count.
    await queue.wake_scheduler()
    replay = await asyncio.wait_for(queue.get(), timeout=0.5)
    assert replay["update_id"] == 38
    replay_claim = queue.claim_for_update(38)
    assert replay_claim is not None
    queue.task_done()
    assert await queue.complete_update(38, success=False)

    row = store.get(persisted.event_id)
    assert row is not None
    assert row.work_state == "queued"
    assert row.dispatch_state == "pending"
    assert row.last_error_class == "handler_failed"
    assert row.attempt_count == 1
    assert row.terminal_reason is None


@pytest.mark.asyncio
async def test_deferred_completion_wakes_projection_when_retry_becomes_due(tmp_path):
    """A busy-cap retry must re-enter PTB without an unrelated wake event."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:deferred-due-wake",
        active_limit=1,
    )

    await queue.put(payload(39))
    item = await queue.get()
    assert item["update_id"] == 39
    claim = queue.claim_for_update(39)
    assert claim is not None
    queue.task_done()

    assert await queue.complete_update(
        39,
        success=False,
        delay=0.05,
        defer=True,
    )
    await asyncio.sleep(0.08)

    replay = await asyncio.wait_for(queue.get(), timeout=0.3)
    assert replay["update_id"] == 39
    replay_claim = queue.claim_for_update(39)
    assert replay_claim is not None
    queue.task_done()
    assert await queue.complete_update(39, success=True)


def test_reconcile_consumed_is_scoped_to_the_bot_account(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    account_111 = store.persist(
        bot_account_id=111,
        update_id=32,
        decision=decision(payload(32, message_id="42")),
        now=1.0,
    )
    account_222 = store.persist(
        bot_account_id=222,
        update_id=32,
        decision=decision(payload(32, message_id="42")),
        now=1.0,
    )

    assert store.reconcile_consumed(
        bot_account_id=111,
        committed_event_ids={account_111.event_id, account_222.event_id},
        now=2.0,
    ) == 1
    assert store.get(account_111.event_id).work_state == "consumed"
    assert store.get(account_222.event_id).work_state == "queued"

    assert store.reconcile_consumed(
        bot_account_id=222,
        committed_event_ids={account_111.event_id, account_222.event_id},
        now=3.0,
    ) == 1
    assert store.get(account_222.event_id).work_state == "consumed"


def test_adapter_attaches_durable_queue_at_ptb_update_queue_boundary(tmp_path):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(
        enabled=True, token="9223372036854775808:test-token", extra={}
    )
    adapter._inbound_store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter._inbound_queue = None
    adapter._bot_account_id = None

    class Builder:
        def update_queue(self, queue):
            self.queue = queue
            return self

    builder = Builder()
    assert adapter._attach_inbound_queue(builder) is builder
    assert builder.queue.bot_account_id == 9223372036854775808
    assert builder.queue.store is adapter._inbound_store


@pytest.mark.asyncio
async def test_real_ptb_polling_survives_local_ingress_failure_and_executor_starvation(
    tmp_path, monkeypatch
):
    import importlib
    import sys

    mocked_telegram_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "telegram" or name.startswith("telegram.")
    }
    for name in mocked_telegram_modules:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        importlib.import_module("telegram")
    except ImportError:
        sys.modules.update(mocked_telegram_modules)
        pytest.skip("python-telegram-bot not installed")

    from telegram.ext import Application
    from telegram.request import BaseRequest

    class GeneralRequest(BaseRequest):
        @property
        def read_timeout(self):
            return 10

        async def initialize(self):
            return None

        async def shutdown(self):
            return None

        async def do_request(self, url, method, request_data=None, **_kwargs):
            if url.endswith("/getMe"):
                return (
                    200,
                    b'{"ok":true,"result":{"id":111,"is_bot":true,'
                    b'"first_name":"Test","username":"test_bot"}}',
                )
            return 200, b'{"ok":true,"result":true}'

    class PollingRequest(BaseRequest):
        def __init__(self):
            self.offsets = []
            self.replayed_offset_seen = asyncio.Event()

        @property
        def read_timeout(self):
            return 10

        async def initialize(self):
            return None

        async def shutdown(self):
            return None

        async def do_request(self, url, method, request_data=None, **_kwargs):
            parameters = request_data.parameters if request_data is not None else {}
            timeout = parameters.get("timeout")
            timeout_seconds = (
                timeout.total_seconds()
                if hasattr(timeout, "total_seconds")
                else timeout
            )
            if timeout_seconds == 0:
                return 200, b'{"ok":true,"result":[]}'
            offset = parameters.get("offset")
            self.offsets.append(offset)
            if offset is not None and int(offset) > 77:
                self.replayed_offset_seen.set()
                await asyncio.sleep(0.01)
                return 200, b'{"ok":true,"result":[]}'
            return (
                200,
                b'{"ok":true,"result":[{"update_id":77,"message":'
                b'{"message_id":20,"date":1,"chat":{"id":10,"type":"private"},'
                b'"from":{"id":30,"is_bot":false,"first_name":"U"},'
                b'"text":"same"}}]}',
            )

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    original_persist = store.persist
    persist_calls = 0

    def transient_persist(*args, **kwargs):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(store, "persist", transient_persist)
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item.to_dict()),
        lease_owner="gateway:ptb-boundary",
        active_limit=1,
    )
    queue._persist_retry_initial_seconds = 0.01
    queue._persist_stall_log_seconds = 0.05
    polling_request = PollingRequest()
    app = (
        Application.builder()
        .token("111:test-token")
        .request(GeneralRequest())
        .get_updates_request(polling_request)
        .update_queue(queue)
        .build()
    )
    errors = []
    initialized = False
    release_default_executor = threading.Event()
    default_executor_entered = threading.Event()
    blocked_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="blocked-default-executor"
    )
    blocker = None
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(lambda: None)
    previous_executor = getattr(loop, "_default_executor")

    def occupy_default_executor():
        default_executor_entered.set()
        release_default_executor.wait(timeout=3.0)

    try:
        await app.initialize()
        initialized = True
        loop.set_default_executor(blocked_executor)
        blocker = blocked_executor.submit(occupy_default_executor)
        assert default_executor_entered.wait(timeout=1.0)
        await app.updater.start_polling(
            poll_interval=0,
            timeout=1,
            bootstrap_retries=0,
            drop_pending_updates=False,
            error_callback=errors.append,
        )

        async def committed_and_acknowledged():
            for _ in range(100):
                row = store.get("telegram:111:77")
                if row is not None and app.updater._last_update_id == 78:
                    return row
                await asyncio.sleep(0.01)
            raise AssertionError(
                "PTB did not durably commit and acknowledge update 77 while the "
                "global executor was unavailable"
            )

        row = await committed_and_acknowledged()
        assert row.update_id == 77
        assert persist_calls == 2
        assert errors == []
        assert polling_request.offsets[0] == 0
        await asyncio.wait_for(polling_request.replayed_offset_seen.wait(), timeout=1.0)
        assert 78 in polling_request.offsets[1:]
    finally:
        release_default_executor.set()
        loop.set_default_executor(previous_executor)
        if blocker is not None:
            blocker.result(timeout=1.0)
        blocked_executor.shutdown(wait=True)
        if app.updater.running:
            await asyncio.wait_for(app.updater.stop(), timeout=2.0)
        if initialized:
            await app.shutdown()
        for name in tuple(sys.modules):
            if name == "telegram" or name.startswith("telegram."):
                sys.modules.pop(name, None)
        sys.modules.update(mocked_telegram_modules)


@pytest.mark.asyncio
async def test_real_ptb_application_fetcher_survives_transient_lease_failure(
    tmp_path, monkeypatch
):
    """PTB 22.8's running fetcher must survive a local queue lease fault."""
    import importlib
    import sys

    mocked_telegram_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "telegram" or name.startswith("telegram.")
    }
    for name in mocked_telegram_modules:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        telegram = importlib.import_module("telegram")
    except ImportError:
        sys.modules.update(mocked_telegram_modules)
        pytest.skip("python-telegram-bot not installed")
    from telegram.ext import Application, TypeHandler
    from telegram.request import BaseRequest

    class Request(BaseRequest):
        @property
        def read_timeout(self):
            return 10

        async def initialize(self):
            return None

        async def shutdown(self):
            return None

        async def do_request(self, url, method, request_data=None, **_kwargs):
            del method, request_data
            if url.endswith("/getMe"):
                return (
                    200,
                    b'{"ok":true,"result":{"id":111,"is_bot":true,'
                    b'"first_name":"Test","username":"test_bot"}}',
                )
            return 200, b'{"ok":true,"result":true}'

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item.to_dict()),
        lease_owner="gateway:ptb-fetcher",
        active_limit=1,
    )
    queue._persist_retry_initial_seconds = 0.01
    original_lease = store.lease_event
    lease_calls = 0

    def transient_lease(*args, **kwargs):
        nonlocal lease_calls
        lease_calls += 1
        if lease_calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_lease(*args, **kwargs)

    monkeypatch.setattr(store, "lease_event", transient_lease)
    app = (
        Application.builder()
        .token("111:test-token")
        .request(Request())
        .get_updates_request(Request())
        .update_queue(queue)
        .build()
    )
    handled = asyncio.Event()

    async def record(update, _context):
        assert update.update_id == 8201
        handled.set()

    app.add_handler(TypeHandler(telegram.Update, record))
    initialized = False
    started = False
    try:
        await app.initialize()
        initialized = True
        # A durable projection is rebuilt from SQLite payload, just as the
        # adapter supplies its ``_deserialize_update`` factory in production.
        queue.item_factory = lambda stored: telegram.Update.de_json(stored, app.bot)
        await app.start()
        started = True
        await queue.put(telegram.Update.de_json(payload(8201), app.bot))
        await asyncio.wait_for(handled.wait(), timeout=1.0)
        fetcher = getattr(app, "_Application__update_fetcher_task")
        assert not fetcher.done()
        assert lease_calls == 2
    finally:
        if started and app.running:
            await asyncio.wait_for(app.stop(), timeout=2.0)
        if initialized:
            await asyncio.wait_for(app.shutdown(), timeout=2.0)
        await queue.close()
        for name in tuple(sys.modules):
            if name == "telegram" or name.startswith("telegram."):
                sys.modules.pop(name, None)
        sys.modules.update(mocked_telegram_modules)


@pytest.mark.asyncio
async def test_durable_pause_suppresses_network_recovery_and_resumes(
    tmp_path, monkeypatch
):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    entered = threading.Event()
    release = threading.Event()
    original_persist = store.persist

    def blocked_persist(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2.0)
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(store, "persist", blocked_persist)
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:pause-watchdog",
        active_limit=1,
    )
    queue._persist_stall_log_seconds = 0.02
    put_task = asyncio.create_task(queue.put(payload(79)))
    for _ in range(50):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    for _ in range(50):
        if queue.ingress_paused:
            break
        await asyncio.sleep(0.01)
    snapshot = queue.ingress_pause_snapshot()
    assert snapshot["paused"] is True
    assert snapshot["stage"] == "persist_wait"
    assert snapshot["error_class"] == "TimeoutError"

    webhook_info_calls = 0

    async def get_webhook_info():
        nonlocal webhook_info_calls
        webhook_info_calls += 1
        return SimpleNamespace(pending_update_count=1)

    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="111:test-token", extra={})
    )
    adapter._webhook_mode = False
    adapter._app = SimpleNamespace(updater=SimpleNamespace(running=True))
    adapter._inbound_queue = queue
    adapter._polling_error_task = None
    adapter._polling_pending_stuck_count = 1
    adapter._polling_generation = 2
    adapter._polling_generation_started_monotonic = time.monotonic() - 500
    adapter._polling_last_progress_monotonic = time.monotonic() - 400
    bot = SimpleNamespace(get_webhook_info=get_webhook_info)

    await adapter._probe_pending_updates(bot, 0.1)
    await adapter._check_polling_stall()
    assert webhook_info_calls == 0
    assert adapter._polling_pending_stuck_count == 0
    assert adapter._polling_error_task is None

    release.set()
    await asyncio.wait_for(put_task, timeout=1.0)
    assert queue.ingress_paused is False
    assert store.get("telegram:111:79") is not None


@pytest.mark.asyncio
async def test_post_commit_callback_failure_does_not_escape_ptb_boundary(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    async def failing_callback(_item, _result):
        raise RuntimeError("post-commit failure")

    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        after_commit=failing_callback,
        lease_owner="gateway:post-commit",
        active_limit=1,
    )

    await queue.put(payload(80))

    assert store.get("telegram:111:80") is not None
    assert (await asyncio.wait_for(queue.get(), timeout=0.5))["update_id"] == 80
    queue.task_done()


@pytest.mark.asyncio
async def test_adapter_recovery_clears_stale_in_memory_dedup_keys(tmp_path):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="111:test-token", extra={})
    adapter._inbound_store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter._inbound_queue = None
    adapter._bot_account_id = None
    adapter._durable_queue_bound = True
    adapter._seen_update_ids = {("111", 44): None}
    adapter._seen_platform_update_ids = {("111", 44): None}

    queue = adapter._ensure_inbound_queue()
    try:
        await adapter._recover_inbound_queue()

        assert adapter._seen_update_ids == {}
        assert adapter._seen_platform_update_ids == {}
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_durable_backlog_above_ordinary_cap_remains_admitted(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:durable-cap-boundary",
        active_limit=32,
    )

    for update_id in range(33):
        await queue.put(payload(update_id, message_id=str(update_id)))

    rows = [store.get(f"telegram:111:{update_id}") for update_id in range(33)]
    assert all(row is not None for row in rows)
    assert queue.qsize() == 32
    assert rows[-1].dispatch_state == "pending"
    assert rows[-1].work_state == "queued"


@pytest.mark.asyncio
async def test_non_actionable_ingress_does_not_share_durable_projection_limit(tmp_path):
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    def classify(item):
        return decision(item, actionable=item["update_id"] == 1)

    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=classify,
        lease_owner="gateway:ordinary-ingress-boundary",
        active_limit=1,
    )

    await queue.put(payload(1))
    rejected = payload(2)
    await asyncio.wait_for(queue.put(rejected), timeout=0.5)

    assert queue.qsize() == 2
    assert store.get("telegram:111:2") is None


@pytest.mark.asyncio
async def test_adapter_classifier_durably_admits_authorized_update_before_put_returns(
    tmp_path,
):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="111:test-token", extra={})
    adapter._inbound_store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter._inbound_queue = None
    adapter._bot_account_id = None
    adapter._bot = SimpleNamespace(id=999)
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter._is_own_message = lambda _message: False
    adapter._should_process_message = lambda *_args, **_kwargs: True

    message = SimpleNamespace(
        message_id=41,
        text="hello",
        caption=None,
        chat=SimpleNamespace(id=7, type="private"),
        from_user=SimpleNamespace(id=8),
        location=None,
        venue=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
    )
    update = SimpleNamespace(
        update_id=41,
        message=message,
        effective_message=message,
        callback_query=None,
        to_dict=lambda: {"update_id": 41, "message": {"message_id": 41}},
    )
    adapter._deserialize_update = lambda _payload: update
    queue = adapter._ensure_inbound_queue()
    assert adapter._classify_inbound_update(update).actionable is True

    await queue.put(update)
    row = queue.store.get("telegram:111:41")
    assert row is not None
    assert row.dispatch_state == "admitted"
    assert (await queue.get()).update_id == 41
    queue.task_done()


@pytest.mark.asyncio
async def test_adapter_classifier_keeps_rejected_update_out_of_durable_store(tmp_path):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="111:test-token", extra={})
    adapter._inbound_store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter._inbound_queue = None
    adapter._bot_account_id = None
    adapter._bot = SimpleNamespace(id=999)
    adapter._is_user_authorized_from_message = lambda _message: False
    adapter._is_own_message = lambda _message: False

    message = SimpleNamespace(
        message_id=42,
        text="blocked",
        caption=None,
        chat=SimpleNamespace(id=7, type="private"),
        from_user=SimpleNamespace(id=8),
        location=None,
        venue=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
    )
    update = SimpleNamespace(
        update_id=42,
        message=message,
        effective_message=message,
        callback_query=None,
        to_dict=lambda: {"update_id": 42, "message": {"message_id": 42}},
    )
    queue = adapter._ensure_inbound_queue()
    assert adapter._classify_inbound_update(update).actionable is False

    await queue.put(update)
    assert queue.store.get("telegram:111:42") is None
    assert (await queue.get()).update_id == 42
    queue.task_done()


@pytest.mark.asyncio
async def test_reconnect_handoff_fences_old_handler_before_claim_publication(tmp_path):
    """A handler that starts after handoff must not run the old callback."""
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.telegram.adapter import TelegramAdapter

    update_id = 9223372036854776030
    update_payload = payload(update_id, message_id=str(update_id))

    class FakeUpdate:
        def __init__(self, data):
            self.update_id = data["update_id"]
            self._data = data

        def to_dict(self):
            return self._data

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    def classify(item):
        return decision(item)

    def make_queue(owner):
        return DurableTelegramUpdateQueue(
            store=store,
            bot_account_id=111,
            classifier=classify,
            lease_owner=owner,
            active_limit=1,
            item_factory=FakeUpdate,
        )

    old_queue = make_queue("gateway:old-handler-fence")
    replacement_queue = make_queue("gateway:new-handler-fence")
    await old_queue.put(update_payload)
    old_update = await old_queue.get()
    assert old_update.update_id == update_id
    assert old_queue.claim_for_update(update_id) is not None
    assert not old_queue.handler_claimed(update_id)

    # This is the exact get-to-wrapper interval: the old queue has returned its
    # item, but its registered wrapper has not published the handler claim.
    assert await replacement_queue.handoff_from(old_queue) == {
        "requeued": 1,
        "transferred": 0,
        "quarantined": 0,
    }

    def make_adapter(queue):
        adapter = object.__new__(TelegramAdapter)
        BasePlatformAdapter.__init__(
            adapter,
            PlatformConfig(enabled=True, token="111:test-token", extra={}),
            Platform.TELEGRAM,
        )
        adapter._inbound_queue = queue
        adapter._deferred_inbound_update_ids = set()
        return adapter

    old_calls = []
    replacement_calls = []

    async def old_callback(update, context):
        del context
        old_calls.append(update.update_id)

    async def replacement_callback(update, context):
        del context
        replacement_calls.append(update.update_id)

    old_adapter = make_adapter(old_queue)
    replacement_adapter = make_adapter(replacement_queue)

    # The old wrapper is deliberately invoked only after handoff. A missing
    # claim here is ownership loss, not permission to run volatile work.
    await old_adapter._wrap_inbound_handler(old_callback)(old_update, None)
    old_queue.task_done()

    replacement_update = await replacement_queue.get()
    await replacement_adapter._wrap_inbound_handler(replacement_callback)(
        replacement_update, None
    )
    replacement_queue.task_done()

    assert old_calls == []
    assert replacement_calls == [update_id]
    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "consumed"


@pytest.mark.asyncio
async def test_fresh_adapter_recovery_handoffs_delayed_old_handler(tmp_path):
    """Production adapter recovery must fence a stale handler callback."""
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.telegram.adapter import TelegramAdapter

    update_id = 9223372036854776031
    update_payload = payload(update_id, message_id=str(update_id))

    class FakeUpdate:
        def __init__(self, data):
            self.update_id = data["update_id"]
            self._data = data

        def to_dict(self):
            return self._data

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    def make_adapter():
        adapter = object.__new__(TelegramAdapter)
        BasePlatformAdapter.__init__(
            adapter,
            PlatformConfig(enabled=True, token="111:test-token", extra={}),
            Platform.TELEGRAM,
        )
        adapter._inbound_store = store
        adapter._inbound_queue = None
        adapter._bot_account_id = None
        adapter._durable_queue_bound = True
        adapter._classify_inbound_update = lambda update: decision(update)
        adapter._deserialize_update = lambda payload: FakeUpdate(payload)
        return adapter

    old_adapter = make_adapter()
    old_queue = old_adapter._ensure_inbound_queue()
    await old_queue.put(update_payload)
    old_update = await old_queue.get()
    assert old_queue.claim_for_update(update_id) is not None

    replacement_adapter = make_adapter()
    replacement_queue = replacement_adapter._ensure_inbound_queue()
    assert replacement_queue is not old_queue

    # The real fresh-adapter recovery entry point must perform the handoff
    # before it projects the replayable row into the replacement queue.
    await replacement_adapter._recover_inbound_queue()
    # Repeated replacement recovery is idempotent, and a late stale-adapter
    # recovery must not reclaim or reverse the completed handoff.
    await replacement_adapter._recover_inbound_queue()
    await old_adapter._recover_inbound_queue()
    assert replacement_queue.qsize() == 1

    old_calls = []
    replacement_calls = []

    async def old_callback(update, context):
        del context
        old_calls.append(update.update_id)

    async def replacement_callback(update, context):
        del context
        replacement_calls.append(update.update_id)

    # Simulate the old registered handler being delayed after queue claim and
    # entering its wrapper only after the replacement has recovered.
    await old_adapter._wrap_inbound_handler(old_callback)(old_update, None)
    old_queue.task_done()

    replacement_update = await asyncio.wait_for(replacement_queue.get(), timeout=0.5)
    await replacement_adapter._wrap_inbound_handler(replacement_callback)(
        replacement_update, None
    )
    replacement_queue.task_done()

    assert old_calls == []
    assert replacement_calls == [update_id]
    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "consumed"


@pytest.mark.asyncio
async def test_held_durable_overflow_requeues_before_fresh_adapter_handoff(tmp_path):
    """Overflowing a hold queue must not strand its durable claim."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class FakeUpdate:
        def __init__(self, data):
            self.update_id = data["update_id"]
            self._data = data

        def to_dict(self):
            return self._data

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    def make_adapter():
        adapter = object.__new__(TelegramAdapter)
        BasePlatformAdapter.__init__(
            adapter,
            PlatformConfig(enabled=True, token="111:test-token", extra={}),
            Platform.TELEGRAM,
        )
        adapter._inbound_store = store
        adapter._inbound_queue = None
        adapter._bot_account_id = None
        adapter._durable_queue_bound = True
        adapter._classify_inbound_update = lambda update: decision(update)
        adapter._deserialize_update = lambda payload: FakeUpdate(payload)
        adapter._drop_delayed_deliveries = True
        adapter._held_inbound_events = []
        adapter.HELD_INBOUND_MAX = 1
        return adapter

    def held_event(update_id, *, durable=True):
        metadata = (
            {"telegram_durable_update_ids": [update_id]} if durable else {}
        )
        return MessageEvent(
            text=f"held-{update_id}",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="10",
                user_id="30",
                chat_type="dm",
            ),
            message_id=str(update_id),
            platform_update_id=update_id if durable else None,
            metadata=metadata,
        )

    old_adapter = make_adapter()
    old_queue = old_adapter._ensure_inbound_queue()
    update_id = 7001
    await old_queue.put(payload(update_id))
    old_update = await old_queue.get()
    old_queue.task_done()
    assert old_update.update_id == update_id
    assert old_queue.mark_handler_claim(update_id) is not None
    old_event = held_event(update_id)
    old_adapter._hold_inbound_event(old_event, where="overflow-seed", schedule=False)
    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "leased"

    # A non-durable held event forces the durable event out of the bounded
    # process-local hold list without introducing a second durable row.
    replacement_event = held_event(7002, durable=False)
    old_adapter._hold_inbound_event(
        replacement_event, where="overflow-trigger", schedule=False
    )
    # The durable projection stays in the hold list until its asynchronous
    # store transition is confirmed. The ordinary event is retained beside it
    # while that bounded overflow cleanup is in flight.
    assert old_adapter._held_inbound_events == [old_event, replacement_event]
    await old_queue._wait_for_lifecycle_tasks()
    assert old_adapter._held_inbound_events == [replacement_event]

    replacement_adapter = make_adapter()
    replacement_queue = replacement_adapter._ensure_inbound_queue()
    await replacement_adapter._recover_inbound_queue()

    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "queued"
    replacement_update = await asyncio.wait_for(replacement_queue.get(), timeout=0.5)
    assert replacement_update.update_id == update_id
    replacement_queue.task_done()
    assert await replacement_queue.complete_update(update_id, success=True)
    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "consumed"


@pytest.mark.asyncio
async def test_started_held_durable_event_replays_after_handoff(tmp_path):
    """A started event retained on hold must be requeued, never transferred alone."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class FakeUpdate:
        def __init__(self, data):
            self.update_id = data["update_id"]
            self._data = data

        def to_dict(self):
            return self._data

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    def make_adapter():
        adapter = object.__new__(TelegramAdapter)
        BasePlatformAdapter.__init__(
            adapter,
            PlatformConfig(enabled=True, token="111:test-token", extra={}),
            Platform.TELEGRAM,
        )
        adapter._inbound_store = store
        adapter._inbound_queue = None
        adapter._bot_account_id = None
        adapter._durable_queue_bound = True
        adapter._classify_inbound_update = lambda update: decision(update)
        adapter._deserialize_update = lambda data: FakeUpdate(data)
        adapter._drop_delayed_deliveries = True
        adapter._held_inbound_events = []
        return adapter

    update_id = 7003
    old_adapter = make_adapter()
    old_queue = old_adapter._ensure_inbound_queue()
    await old_queue.put(payload(update_id))
    claimed_update = await old_queue.get()
    assert claimed_update.update_id == update_id
    old_queue.task_done()
    assert old_queue.mark_handler_claim(update_id) is not None
    held = MessageEvent(
        text="started-held",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="10", user_id="30", chat_type="dm"
        ),
        message_id=str(update_id),
        platform_update_id=update_id,
        metadata={"telegram_durable_update_ids": [update_id]},
    )
    held._telegram_processing_started = True
    old_adapter._hold_inbound_event(held, where="started-held", schedule=False)

    replacement_adapter = make_adapter()
    replacement_queue = replacement_adapter._ensure_inbound_queue()
    await replacement_adapter._recover_inbound_queue()

    replay = await asyncio.wait_for(replacement_queue.get(), timeout=0.5)
    assert replay.update_id == update_id
    replacement_queue.task_done()
    assert await replacement_queue.complete_update(update_id, success=True)


@pytest.mark.asyncio
async def test_fresh_adapter_handoff_replays_held_text_batch(tmp_path):
    """A held text batch must become a replayable replacement projection."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class FakeUpdate:
        def __init__(self, data):
            self.update_id = data["update_id"]
            message_data = data["message"]
            self.message = SimpleNamespace(
                message_id=message_data["message_id"],
                text=message_data["text"],
                caption=None,
                chat=SimpleNamespace(
                    id=message_data["chat"]["id"],
                    type="private",
                    is_forum=False,
                ),
                from_user=SimpleNamespace(id=30, is_bot=False),
                sender_chat=None,
            )
            self.effective_message = self.message
            self.callback_query = None

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    async def no_op(*_args, **_kwargs):
        return None

    def make_adapter():
        adapter = object.__new__(TelegramAdapter)
        BasePlatformAdapter.__init__(
            adapter,
            PlatformConfig(enabled=True, token="111:test-token", extra={}),
            Platform.TELEGRAM,
        )
        adapter._inbound_store = store
        adapter._inbound_queue = None
        adapter._bot_account_id = None
        adapter._durable_queue_bound = True
        adapter._classify_inbound_update = lambda update: decision(update)
        adapter._deserialize_update = lambda payload: FakeUpdate(payload)
        adapter._is_user_authorized_from_message = lambda _message: True
        adapter._should_process_message = lambda *_args, **_kwargs: True
        adapter._ensure_forum_commands = no_op
        adapter._build_message_event = lambda message, message_type, update_id=None: MessageEvent(
            text=message.text,
            message_type=message_type,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=str(message.chat.id),
                user_id="30",
                chat_type="dm",
            ),
            message_id=str(message.message_id),
            platform_update_id=update_id,
            metadata={"telegram_durable_update_ids": [update_id]},
        )
        adapter._clean_bot_trigger_text = lambda text: text
        adapter._cache_replied_media = no_op
        adapter._apply_telegram_group_observe_attribution = lambda event: event
        adapter._text_batch_key = lambda _event: "telegram:dm:10"
        adapter._text_batch_delay_seconds = 60.0
        adapter._seen_update_ids = {}
        adapter._seen_platform_update_ids = {}
        adapter._seen_update_ids_max = 4096
        adapter._pending_text_batches = {}
        adapter._pending_text_batch_tasks = {}
        adapter._pending_photo_batches = {}
        adapter._pending_photo_batch_tasks = {}
        adapter._media_group_events = {}
        adapter._media_group_tasks = {}
        adapter._drop_delayed_deliveries = False
        adapter._held_inbound_events = []
        adapter._held_inbound_redispatch_task = None
        return adapter

    update_id = 7101
    old_adapter = make_adapter()
    old_queue = old_adapter._ensure_inbound_queue()
    await old_queue.put(payload(update_id))
    old_update = await old_queue.get()
    old_queue.task_done()

    old_handler = old_adapter._wrap_inbound_handler(old_adapter._handle_text_message)
    await old_handler(old_update, None)
    assert old_adapter._pending_text_batches
    assert old_queue.handler_claimed(update_id)

    old_adapter._mark_disconnected()
    await old_adapter._cancel_pending_delivery_tasks()
    assert len(old_adapter._held_inbound_events) == 1

    replacement_adapter = make_adapter()
    replacement_queue = replacement_adapter._ensure_inbound_queue()
    await replacement_adapter._recover_inbound_queue()

    assert old_adapter._held_inbound_events == []
    replacement_update = await asyncio.wait_for(replacement_queue.get(), timeout=0.5)
    assert replacement_update.update_id == update_id
    replacement_queue.task_done()
    assert await replacement_queue.complete_update(update_id, success=True)

    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "consumed"
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(replacement_queue.get(), timeout=0.05)


@pytest.mark.asyncio
async def test_partial_batched_durable_cleanup_converges(tmp_path):
    """A partial durable requeue retains only IDs still needing cleanup."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class Queue:
        def __init__(self):
            self.calls = []
            self.outcomes = {1: [True], 2: [False, True]}

        async def complete_update(self, update_id, **kwargs):
            self.calls.append((update_id, kwargs))
            values = self.outcomes[update_id]
            return values.pop(0) if values else False

    adapter = object.__new__(TelegramAdapter)
    BasePlatformAdapter.__init__(
        adapter,
        PlatformConfig(enabled=True, token="111:test-token", extra={}),
        Platform.TELEGRAM,
    )
    queue = Queue()
    adapter._inbound_queue = queue
    adapter._deferred_inbound_update_ids = {1, 2}
    event = SimpleNamespace(
        metadata={
            "telegram_durable_update_ids": [1, 2],
            "telegram_inbound_dispatch_deferred": True,
        }
    )

    first = await adapter._complete_durable_event(
        event,
        success=False,
        defer=True,
    )
    assert first is False
    assert event.metadata["telegram_durable_update_ids"] == [2]
    assert event.metadata["telegram_inbound_dispatch_deferred"] is True
    assert adapter._deferred_inbound_update_ids == {2}

    second = await adapter._complete_durable_event(
        event,
        success=False,
        defer=True,
    )
    assert second is True
    assert "telegram_durable_update_ids" not in event.metadata
    assert "telegram_inbound_dispatch_deferred" not in event.metadata
    assert adapter._deferred_inbound_update_ids == set()
    assert [update_id for update_id, _kwargs in queue.calls] == [1, 2, 2]


@pytest.mark.asyncio
async def test_partial_batched_cleanup_retries_unresolved_real_queue_claim(
    tmp_path, monkeypatch
):
    """A transient false transition must not strand the unresolved ID."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    for update_id in (1, 2):
        store.persist(
            bot_account_id=111,
            update_id=update_id,
            decision=decision(payload(update_id)),
        )
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:partial-cleanup",
        active_limit=2,
    )
    assert await queue.wake_scheduler() == 2
    assert (await queue.get())["update_id"] == 1
    queue.task_done()
    assert (await queue.get())["update_id"] == 2
    queue.task_done()
    # Keep this deterministic: completion wakes are not part of this probe.
    queue._scheduler_loop = None

    original_requeue = store.requeue
    outcomes = {1: [True], 2: [False, True]}
    calls = []

    def transient_requeue(event_id, **kwargs):
        update_id = int(event_id.rsplit(":", 1)[-1])
        calls.append(update_id)
        if not outcomes[update_id].pop(0):
            return False
        return original_requeue(event_id, **kwargs)

    monkeypatch.setattr(store, "requeue", transient_requeue)
    adapter = object.__new__(TelegramAdapter)
    BasePlatformAdapter.__init__(
        adapter,
        PlatformConfig(enabled=True, token="111:test-token", extra={}),
        Platform.TELEGRAM,
    )
    adapter._inbound_queue = queue
    adapter._bot_account_id = "111"
    adapter._deferred_inbound_update_ids = {1, 2}
    event = MessageEvent(
        text="batched",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="10",
            user_id="30",
            chat_type="dm",
        ),
        message_id="1",
        metadata={
            "telegram_durable_update_ids": [1, 2],
            "telegram_inbound_dispatch_deferred": True,
        },
    )

    assert await adapter._complete_durable_event(event, success=False, defer=True) is False
    assert event.metadata["telegram_durable_update_ids"] == [2]
    assert adapter._deferred_inbound_update_ids == {2}
    assert store.get("telegram:111:1").work_state == "queued"
    assert store.get("telegram:111:2").work_state == "leased"

    assert await adapter._complete_durable_event(event, success=False, defer=True) is True
    assert "telegram_durable_update_ids" not in event.metadata
    assert "telegram_inbound_dispatch_deferred" not in event.metadata
    assert adapter._deferred_inbound_update_ids == set()
    assert store.get("telegram:111:2").work_state == "queued"
    assert calls == [1, 2, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ("false", "cancelled", "exception"))
async def test_failed_held_overflow_cleanup_remains_recoverable(
    tmp_path, monkeypatch, failure_mode
):
    """Failed overflow cleanup must retain its durable projection."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class FakeUpdate:
        def __init__(self, data):
            self.update_id = data["update_id"]
            self._data = data

        def to_dict(self):
            return self._data

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")

    def make_adapter():
        adapter = object.__new__(TelegramAdapter)
        BasePlatformAdapter.__init__(
            adapter,
            PlatformConfig(enabled=True, token="111:test-token", extra={}),
            Platform.TELEGRAM,
        )
        adapter._inbound_store = store
        adapter._inbound_queue = None
        adapter._bot_account_id = None
        adapter._durable_queue_bound = True
        adapter._classify_inbound_update = lambda update: decision(update)
        adapter._deserialize_update = lambda payload: FakeUpdate(payload)
        adapter._drop_delayed_deliveries = True
        adapter._held_inbound_events = []
        adapter.HELD_INBOUND_MAX = 1
        return adapter

    def held_event(update_id, *, durable=True):
        metadata = (
            {"telegram_durable_update_ids": [update_id]} if durable else {}
        )
        return MessageEvent(
            text=f"held-{update_id}",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="10",
                user_id="30",
                chat_type="dm",
            ),
            message_id=str(update_id),
            platform_update_id=update_id if durable else None,
            metadata=metadata,
        )

    old_adapter = make_adapter()
    old_queue = old_adapter._ensure_inbound_queue()
    update_id = 7003
    await old_queue.put(payload(update_id))
    old_update = await old_queue.get()
    old_queue.task_done()
    assert old_update.update_id == update_id
    assert old_queue.mark_handler_claim(update_id) is not None
    old_event = held_event(update_id)
    old_adapter._hold_inbound_event(old_event, where="overflow-seed", schedule=False)

    completion_started = None
    completion_finished = None
    requeue_started = None
    release_requeue = None
    if failure_mode == "false":

        def false_requeue(*args, **kwargs):
            del args, kwargs
            return False

        monkeypatch.setattr(store, "requeue", false_requeue)
    elif failure_mode == "cancelled":
        completion_started = asyncio.Event()
        completion_finished = asyncio.Event()
        requeue_started = threading.Event()
        release_requeue = threading.Event()
        original_complete = old_queue.complete_update

        async def tracked_complete(*args, **kwargs):
            completion_started.set()
            try:
                return await original_complete(*args, **kwargs)
            finally:
                completion_finished.set()

        def blocked_requeue(*args, **kwargs):
            del args, kwargs
            requeue_started.set()
            if not release_requeue.wait(timeout=1.0):
                raise RuntimeError("timed out waiting for cancellation probe")
            return False

        monkeypatch.setattr(old_queue, "complete_update", tracked_complete)
        monkeypatch.setattr(store, "requeue", blocked_requeue)
    elif failure_mode == "exception":

        def failed_requeue(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("forced overflow cleanup failure")

        monkeypatch.setattr(store, "requeue", failed_requeue)

    old_adapter._hold_inbound_event(
        held_event(7004, durable=False),
        where="overflow-trigger",
        schedule=False,
    )
    cleanup_tasks = tuple(old_queue._lifecycle_tasks)
    assert len(cleanup_tasks) == 1
    if failure_mode == "cancelled":
        assert completion_started is not None
        assert requeue_started is not None
        assert release_requeue is not None
        assert completion_finished is not None
        await asyncio.wait_for(completion_started.wait(), timeout=0.5)
        assert requeue_started.wait(timeout=0.5)
        cleanup_tasks[0].cancel()
        release_requeue.set()
        await asyncio.wait_for(completion_finished.wait(), timeout=0.5)
    await old_queue._wait_for_lifecycle_tasks()

    assert old_event in old_adapter._held_inbound_events
    assert update_id in old_adapter._deferred_inbound_update_ids
    # An unconfirmed requeue must retain the claim until handoff can safely
    # requeue the buffered event instead of silently releasing its capacity.
    assert old_queue.claim_for_update(update_id) is not None
    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "leased"

    replacement_adapter = make_adapter()
    replacement_queue = replacement_adapter._ensure_inbound_queue()
    await replacement_adapter._recover_inbound_queue()
    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "queued"
    replacement_update = await asyncio.wait_for(replacement_queue.get(), timeout=0.5)
    assert replacement_update.update_id == update_id
    replacement_queue.task_done()
    assert await replacement_queue.complete_update(update_id, success=True)
    row = store.get(f"telegram:111:{update_id}")
    assert row is not None
    assert row.work_state == "consumed"


@pytest.mark.asyncio
async def test_handoff_waits_for_transient_predecessor_ingress_before_retiring_executor(
    tmp_path, monkeypatch
):
    """A pre-handoff PTB put must survive its first transient SQLite failure."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    original_persist = store.persist
    first_failure = asyncio.Event()
    persist_calls = 0

    def transient_persist(*args, **kwargs):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            first_failure.set()
            raise sqlite3.OperationalError("database is locked")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(store, "persist", transient_persist)
    old_queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:handoff-ingress-old",
        active_limit=1,
    )
    old_queue._persist_retry_initial_seconds = 0.01
    replacement = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:handoff-ingress-new",
        active_limit=1,
    )
    put_task = asyncio.create_task(old_queue.put(payload(8101)))
    try:
        await asyncio.wait_for(first_failure.wait(), timeout=0.5)
        await asyncio.wait_for(replacement.handoff_from(old_queue), timeout=0.5)
        await asyncio.wait_for(put_task, timeout=0.5)
        row = store.get("telegram:111:8101")
        assert row is not None
        assert row.work_state == "queued"
        assert persist_calls == 2
    finally:
        if not put_task.done():
            put_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await put_task
        await old_queue.suspend_projection()
        await replacement.suspend_projection()


@pytest.mark.asyncio
async def test_false_consumed_transition_retries_without_a_second_handler_call(
    tmp_path, monkeypatch
):
    """A false terminal transition must retain ownership until it commits."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:false-terminal",
        active_limit=1,
    )
    original_mark_consumed = store.mark_consumed
    attempts = 0

    def false_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return False
        return original_mark_consumed(*args, **kwargs)

    monkeypatch.setattr(store, "mark_consumed", false_once)
    await queue.put(payload(8301))
    assert (await queue.get())["update_id"] == 8301
    queue.task_done()
    try:
        assert not await queue.complete_update(8301, success=True)
        for _ in range(50):
            row = store.get("telegram:111:8301")
            if (
                row is not None
                and row.work_state == "consumed"
                and queue.claim_for_update(8301) is None
            ):
                break
            await asyncio.sleep(0.01)
        row = store.get("telegram:111:8301")
        assert row is not None
        assert row.work_state == "consumed"
        assert attempts == 2
        assert queue.claim_for_update(8301) is None
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_raising_deferred_requeue_retries_without_releasing_the_claim(
    tmp_path, monkeypatch
):
    """A transient deferred requeue failure must resolve without reconnect."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:raising-requeue",
        active_limit=1,
    )
    original_requeue = store.requeue
    attempts = 0
    retry_transitioned = asyncio.Event()
    projection_scheduled = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_schedule_projection_retry = queue._schedule_projection_retry

    def record_projection_retry():
        original_schedule_projection_retry()
        projection_scheduled.set()

    def raise_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        transitioned = original_requeue(*args, **kwargs)
        loop.call_soon_threadsafe(retry_transitioned.set)
        return transitioned

    monkeypatch.setattr(store, "requeue", raise_once)
    monkeypatch.setattr(queue, "_schedule_projection_retry", record_projection_retry)
    await queue.put(payload(8302))
    assert (await queue.get())["update_id"] == 8302
    queue.task_done()
    try:
        assert not await queue.complete_update(8302, success=False, defer=True)
        await asyncio.wait_for(retry_transitioned.wait(), timeout=0.5)
        await asyncio.wait_for(projection_scheduled.wait(), timeout=0.5)
        projection_task = queue._projection_retry_task
        assert projection_task is not None
        await asyncio.wait_for(asyncio.shield(projection_task), timeout=0.5)
        row = store.get("telegram:111:8302")
        assert row is not None
        assert row.work_state == "queued"
        # The zero delay means automatic recovery is immediately projected;
        # awaiting that physical retry proves the row was not stranded leased.
        assert row.dispatch_state == "admitted"
        assert attempts == 2
        assert queue.claim_for_update(8302) is None
        replay = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert replay["update_id"] == 8302
        queue.task_done()
        assert await queue.complete_update(8302, success=True)
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_unhandled_claim_expiry_reclaims_projection_capacity(tmp_path):
    """A PTB update that never reaches a wrapper cannot permanently fill the window."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:unhandled-expiry",
        active_limit=1,
    )
    queue._prehandler_lease_seconds = 0.02
    await queue.put(payload(8303))
    first = await queue.get()
    assert first["update_id"] == 8303
    queue.task_done()
    try:
        for _ in range(50):
            row = store.get("telegram:111:8303")
            if (
                row is not None
                and row.work_state == "queued"
                and queue.claim_for_update(8303) is None
            ):
                break
            await asyncio.sleep(0.01)
        row = store.get("telegram:111:8303")
        assert row is not None
        assert row.work_state == "queued"
        assert queue.claim_for_update(8303) is None
        replay = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert replay["update_id"] == 8303
        assert queue.mark_handler_claim(8303) is not None
        queue.task_done()
        assert await queue.complete_update(8303, success=True)
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_invalid_projection_factory_output_is_requeued_not_delivered(tmp_path):
    """A projection that cannot be a Telegram update must not consume a lease."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        item_factory=lambda _payload: object(),
        lease_owner="gateway:invalid-factory",
        active_limit=1,
    )
    try:
        await queue.put(payload(8304))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.05)
        row = store.get("telegram:111:8304")
        assert row is not None
        assert row.work_state == "queued"
        assert row.dispatch_state == "pending"
        assert queue.claim_for_update(8304) is None
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_late_put_after_close_fails_without_waiting_on_handoff(tmp_path):
    """A terminally closed queue must reject ingress instead of busy-spinning."""
    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    queue = DurableTelegramUpdateQueue(
        store=store,
        bot_account_id=111,
        classifier=lambda item: decision(item),
        lease_owner="gateway:closed-ingress",
        active_limit=1,
    )
    await queue.close()
    waited = False

    async def unexpected_wait(_event):
        nonlocal waited
        waited = True
        raise AssertionError("closed ingress attempted to wait for handoff")

    queue._wait_for_claim_event = unexpected_wait
    with pytest.raises(RuntimeError, match="closed"):
        await queue.put(payload(8305))
    assert not waited


@pytest.mark.asyncio
async def test_disconnect_closes_executor_and_reconnects_with_fresh_queue(tmp_path):
    """Clean disconnect must retire SQLite work without breaking same-adapter reconnect."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    store = TelegramInboundStore(tmp_path / "telegram_inbound.db")
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="111:test-token", extra={})
    )
    adapter._inbound_store = store
    adapter._durable_queue_bound = True
    first = adapter._ensure_inbound_queue()
    replacement = None

    class FakeUpdate:
        def __init__(self, data):
            self.update_id = data["update_id"]
            self._data = data

        def to_dict(self):
            return self._data

    try:
        worker_name = await asyncio.wait_for(
            first._run_store(lambda: threading.current_thread().name), timeout=1.0
        )
        assert worker_name.startswith("telegram-inbound-111")
        first_executor = first._store_executor

        await asyncio.wait_for(adapter.disconnect(), timeout=2.0)

        assert first._store_executor_shutdown
        replacement = adapter._ensure_inbound_queue()
        assert replacement is not first
        assert replacement._store_executor is not first_executor
        assert adapter._retired_inbound_queue is first

        await asyncio.wait_for(adapter._recover_inbound_queue(), timeout=2.0)

        assert adapter._retired_inbound_queue is None
        assert first._handoff_target is replacement
        replacement.classifier = lambda item: decision(item)
        replacement.item_factory = FakeUpdate
        await asyncio.wait_for(replacement.put(payload(8307)), timeout=1.0)
        assert (
            await asyncio.wait_for(replacement.get(), timeout=1.0)
        ).update_id == 8307
        replacement.task_done()
        assert await asyncio.wait_for(
            replacement.complete_update(8307, success=True), timeout=1.0
        )
    finally:
        if replacement is not None:
            await replacement.close()
        if not first._store_executor_shutdown:
            await first.close()
