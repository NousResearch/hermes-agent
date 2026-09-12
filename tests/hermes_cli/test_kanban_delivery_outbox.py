from __future__ import annotations

import concurrent.futures
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


def _fixture(tmp_path, monkeypatch, *, chat_id="test-origin-chat"):
    db = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    kb.init_db()
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="delivery", assignee="worker")
    kbn.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id=chat_id,
                       thread_id="topic-7", notifier_profile="worker-profile", delivery_mode="notify+wake")
    kb.complete_task(conn, task_id, summary="done")
    sub = kbn.list_notify_subs(conn, task_id)[0]
    _, _, events = kbn.claim_unseen_events_for_sub(
        conn, task_id=task_id, platform="telegram", chat_id=chat_id,
        thread_id="topic-7", kinds=["completed"],
    )
    assert len(events) == 1
    return db, conn, task_id, sub, events[0]


def test_enqueue_is_idempotent_and_origin_is_exact(tmp_path, monkeypatch):
    _db, conn, task_id, sub, event = _fixture(tmp_path, monkeypatch)
    try:
        first = kbn.enqueue_delivery(conn, event=event, sub=sub)
        second = kbn.enqueue_delivery(conn, event=event, sub=sub)
        assert first["delivery_key"] == second["delivery_key"]
        rows = conn.execute("SELECT * FROM kanban_delivery_outbox").fetchall()
        assert len(rows) == 1
        assert (rows[0]["task_id"], rows[0]["chat_id"], rows[0]["thread_id"]) == (
            task_id, "test-origin-chat", "topic-7",
        )
        assert rows[0]["payload_digest"]
    finally:
        conn.close()


def test_concurrent_claim_has_one_owner(tmp_path, monkeypatch):
    db, conn, _task_id, sub, event = _fixture(tmp_path, monkeypatch)
    row = kbn.enqueue_delivery(conn, event=event, sub=sub)
    conn.close()

    def claim():
        local = kbc.connect(db_path=db)
        try:
            return kbn.claim_delivery(local, delivery_key=row["delivery_key"], now=100, lease_seconds=10)
        finally:
            local.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(lambda _: claim(), range(2)))
    assert sum(item is not None for item in claimed) == 1


def test_failure_budget_restart_receipt_and_subscription_retention(tmp_path, monkeypatch):
    db, conn, task_id, sub, event = _fixture(tmp_path, monkeypatch)
    row = kbn.enqueue_delivery(conn, event=event, sub=sub)
    key = row["delivery_key"]
    state = None
    for attempt in range(1, 13):
        claim = kbn.claim_delivery(conn, delivery_key=key, now=attempt * 1000)
        assert claim is not None
        state = kbn.fail_delivery(conn, delivery_key=key, lease_token=claim["lease_token"],
                                  error="/home/private/token detail", now=attempt * 1000,
                                  retry_base_seconds=1)
    assert state == "dead_letter"
    persisted = dict(conn.execute("SELECT * FROM kanban_delivery_outbox WHERE delivery_key=?", (key,)).fetchone())
    assert persisted["attempts"] == 12
    assert persisted["last_error"] == "delivery failed (local detail redacted)"
    assert len(kbn.list_notify_subs(conn, task_id)) == 1
    assert kbn.open_delivery_obligations(conn, task_id=task_id) == 1
    assert kbn.mark_delivery_exception_recorded(conn, delivery_key=key)
    assert not kbn.mark_delivery_exception_recorded(conn, delivery_key=key)
    conn.close()

    code = """
import os
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
c=kbc.connect(); print(kbn.open_delivery_obligations(c, task_id=os.environ['TASK'])); c.close()
"""
    env = dict(os.environ, HERMES_KANBAN_DB=str(db), TASK=task_id)
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "1"


def test_ack_requires_receipt_and_prevents_repeat(tmp_path, monkeypatch):
    _db, conn, task_id, sub, event = _fixture(tmp_path, monkeypatch)
    row = kbn.enqueue_delivery(conn, event=event, sub=sub)
    claim = kbn.claim_delivery(conn, delivery_key=row["delivery_key"])
    assert claim
    try:
        kbn.acknowledge_delivery(conn, delivery_key=row["delivery_key"],
                                 lease_token=claim["lease_token"], transport_receipt="")
    except ValueError:
        pass
    else:
        raise AssertionError("empty receipt accepted")
    assert kbn.acknowledge_delivery(conn, delivery_key=row["delivery_key"],
                                    lease_token=claim["lease_token"], transport_receipt="fake-message-42")
    assert kbn.claim_delivery(conn, delivery_key=row["delivery_key"]) is None
    assert kbn.open_delivery_obligations(conn, task_id=task_id) == 0
    conn.close()


def test_before_send_retry_and_post_send_ambiguity(tmp_path, monkeypatch):
    _db, conn, task_id, sub, event = _fixture(tmp_path, monkeypatch)
    first = kbn.enqueue_delivery(conn, event=event, sub=sub)
    claim = kbn.claim_delivery(conn, delivery_key=first["delivery_key"], now=100, lease_seconds=5)
    assert claim
    assert kbn.fail_delivery(conn, delivery_key=first["delivery_key"],
                             lease_token=claim["lease_token"], error="connect failed",
                             retry_base_seconds=1, now=100) == "retry_wait"
    retried = kbn.claim_delivery(conn, delivery_key=first["delivery_key"], now=101)
    assert retried is not None
    assert kbn.acknowledge_delivery(
        conn, delivery_key=first["delivery_key"], lease_token=retried["lease_token"],
        transport_receipt="fake-retry-receipt",
    )

    # A second origin represents a separate non-idempotent transport obligation.
    sub2 = dict(sub, chat_id="other-origin")
    second = kbn.enqueue_delivery(conn, event=event, sub=sub2)
    sent = kbn.claim_delivery(conn, delivery_key=second["delivery_key"], now=200, lease_seconds=5)
    assert sent
    assert kbn.release_expired_delivery_leases(conn, now=205) == 1
    stored = conn.execute("SELECT state FROM kanban_delivery_outbox WHERE delivery_key=?", (second["delivery_key"],)).fetchone()
    assert stored["state"] == "delivery_unknown"
    assert kbn.claim_delivery(conn, delivery_key=second["delivery_key"], now=999) is None
    assert kbn.open_delivery_obligations(conn, task_id=task_id) >= 1
    conn.close()


def test_explicit_post_send_ambiguity_is_durable_and_not_reclaimed(tmp_path, monkeypatch):
    _db, conn, task_id, sub, event = _fixture(tmp_path, monkeypatch)
    row = kbn.enqueue_delivery(conn, event=event, sub=sub)
    claim = kbn.claim_delivery(conn, delivery_key=row["delivery_key"], now=100)
    assert claim

    assert kbn.mark_delivery_ambiguous(
        conn, delivery_key=row["delivery_key"], lease_token=claim["lease_token"],
        error="local receipt write failed", transport_receipt="fake-message-42", now=101,
    )
    stored = dict(conn.execute(
        "SELECT state, attempts, lease_token, lease_expires_at, last_error, transport_receipt "
        "FROM kanban_delivery_outbox WHERE delivery_key=?", (row["delivery_key"],),
    ).fetchone())
    assert stored == {
        "state": "delivery_unknown", "attempts": 0,
        "lease_token": None, "lease_expires_at": None,
        "last_error": "local receipt write failed",
        "transport_receipt": "fake-message-42",
    }
    assert kbn.claim_delivery(conn, delivery_key=row["delivery_key"], now=999) is None
    assert kbn.open_delivery_obligations(conn, task_id=task_id) == 1
    warnings = kbn.claim_delivery_unknown_warnings(conn, task_id=task_id)
    assert [item["delivery_key"] for item in warnings] == [row["delivery_key"]]
    assert kbn.claim_delivery_unknown_warnings(conn, task_id=task_id) == []
    conn.close()


def test_due_listing_does_not_lease_later_unattempted_delivery(tmp_path, monkeypatch):
    """Waiting behind an earlier send must not make a later row ambiguous."""
    _db, conn, task_id, sub, first_event = _fixture(tmp_path, monkeypatch)
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "blocked", {"reason": "later event"})
    second_event = kb.list_events(conn, task_id)[-1]
    first = kbn.enqueue_delivery(conn, event=first_event, sub=sub)
    second = kbn.enqueue_delivery(conn, event=second_event, sub=sub)

    due = kbn.list_due_deliveries_for_sub(
        conn, task_id=task_id, platform="telegram", chat_id=sub["chat_id"],
        thread_id=sub["thread_id"], now=100,
    )
    assert [row[0]["delivery_key"] for row in due] == [
        first["delivery_key"], second["delivery_key"],
    ]
    states = {
        row["delivery_key"]: row["state"]
        for row in conn.execute("SELECT delivery_key,state FROM kanban_delivery_outbox")
    }
    assert states == {first["delivery_key"]: "pending", second["delivery_key"]: "pending"}

    claimed = kbn.claim_delivery(conn, delivery_key=first["delivery_key"], now=100, lease_seconds=5)
    assert claimed is not None
    assert kbn.release_expired_delivery_leases(conn, now=105) == 1
    states = {
        row["delivery_key"]: row["state"]
        for row in conn.execute("SELECT delivery_key,state FROM kanban_delivery_outbox")
    }
    assert states[first["delivery_key"]] == "delivery_unknown"
    assert states[second["delivery_key"]] == "pending"
    assert kbn.claim_delivery(conn, delivery_key=second["delivery_key"], now=105) is not None
    conn.close()


def test_text_event_id_migration_keeps_outbox_fk_live_and_enqueueable(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    with sqlite3.connect(db) as legacy:
        legacy.execute(
            """CREATE TABLE task_events (
                   id TEXT PRIMARY KEY,
                   task_id TEXT NOT NULL,
                   kind TEXT NOT NULL,
                   payload TEXT,
                   created_at REAL NOT NULL
               )"""
        )

    conn = kbc.connect()
    try:
        fk_targets = {
            row["table"] for row in conn.execute("PRAGMA foreign_key_list(kanban_delivery_outbox)")
        }
        assert "task_events" in fk_targets
        assert "task_events_legacy" not in fk_targets

        tid = kb.create_task(conn, title="post-migration enqueue", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary="done")
        sub = kbn.list_notify_subs(conn, tid)[0]
        event = kb.list_events(conn, tid)[-1]
        queued = kbn.enqueue_delivery(conn, event=event, sub=sub)
        assert queued["event_id"] == event.id
    finally:
        conn.close()


def test_explicit_retry_preserves_exact_ping_checkpoint(tmp_path, monkeypatch):
    _db, conn, _task_id, sub, event = _fixture(tmp_path, monkeypatch)
    row = kbn.enqueue_delivery(conn, event=event, sub=sub)
    claim = kbn.claim_delivery(conn, delivery_key=row["delivery_key"], now=100)
    assert claim is not None
    assert kbn.mark_delivery_ping_delivered(
        conn, delivery_key=row["delivery_key"], lease_token=claim["lease_token"],
        transport_receipt="transport:ping-42", now=101,
    )
    assert kbn.mark_delivery_ambiguous(
        conn, delivery_key=row["delivery_key"], lease_token=claim["lease_token"],
        error="wake outcome unknown", transport_receipt="transport:ping-42", now=102,
    )

    result = kbn.reconcile_delivery_unknown(
        conn, delivery_key=row["delivery_key"], action="retry",
        reason="operator accepts ambiguous wake retry", operator="tester",
        accept_duplicate_risk=True, now=103,
    )
    assert result["ok"] is True
    retried = kbn.claim_delivery(conn, delivery_key=row["delivery_key"], now=103)
    assert retried is not None
    assert retried["ping_delivered_at"] == 101
    assert retried["ping_receipt"] == "transport:ping-42"
    conn.close()


def test_retention_keeps_every_unsettled_event_and_exact_route(tmp_path, monkeypatch):
    db = tmp_path / "retention.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    kb.init_db()
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="retained deliveries", assignee="worker")
        open_states = (
            "pending", "retry_wait", "sending", "delivery_unknown", "dead_letter",
        )
        event_ids = {}
        for state in (*open_states, "delivered"):
            chat_id = f"route-{state}"
            kbn.add_notify_sub(
                conn, task_id=task_id, platform="telegram", chat_id=chat_id,
                thread_id="topic-7",
            )
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "completed", {"route": state})
            event = kb.list_events(conn, task_id)[-1]
            event_ids[state] = event.id
            sub = next(
                item for item in kbn.list_notify_subs(conn, task_id)
                if item["chat_id"] == chat_id
            )
            queued = kbn.enqueue_delivery(conn, event=event, sub=sub)
            conn.execute(
                "UPDATE kanban_delivery_outbox SET state=? WHERE delivery_key=?",
                (state, queued["delivery_key"]),
            )
            conn.commit()

        kbn.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="unrelated-route",
            thread_id="topic-7",
        )
        kb.complete_task(conn, task_id, summary="done")
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET created_at=1 WHERE task_id=?", (task_id,))
            conn.execute(
                "UPDATE tasks SET created_at=1, completed_at=1 WHERE id=?", (task_id,),
            )

        kb.gc_events(conn, older_than_seconds=1)
        kbn.purge_stale_done_notify_subs(conn, max_age_days=1)

        remaining_events = {
            row["id"] for row in conn.execute(
                "SELECT id FROM task_events WHERE task_id=?", (task_id,),
            )
        }
        assert {event_ids[state] for state in open_states} <= remaining_events
        assert event_ids["delivered"] not in remaining_events
        assert {
            row["state"] for row in conn.execute(
                "SELECT state FROM kanban_delivery_outbox WHERE task_id=?", (task_id,),
            )
        } == set(open_states)
        assert {
            row["chat_id"] for row in conn.execute(
                "SELECT chat_id FROM kanban_notify_subs WHERE task_id=?", (task_id,),
            )
        } == {f"route-{state}" for state in open_states}
    finally:
        conn.close()
