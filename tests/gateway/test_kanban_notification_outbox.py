"""Gateway integration tests for durable Kanban notification delivery."""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb

_REAL_ASYNCIO_SLEEP = asyncio.sleep


async def _run_one_tick(monkeypatch, runner):
    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await _REAL_ASYNCIO_SLEEP(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _setup_blocked_notification(tmp_path, monkeypatch):
    db_path = tmp_path / "gateway-outbox.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="blocked delivery", assignee="worker")
        subscription_id = kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        kb.block_task(conn, task_id, reason="needs review", kind="needs_input")
    finally:
        conn.close()
    return task_id, subscription_id


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    return runner


class ResultAdapter:
    def __init__(self, results):
        self.results = list(results)
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(content)
        return self.results.pop(0)


class NoWakeAdapter(ResultAdapter):
    def __init__(self, results):
        super().__init__(results)
        self.wake_calls = 0

    async def handle_message(self, event):
        self.wake_calls += 1
        raise AssertionError("notification delivery must not wake an agent")


class CancellingAdapter:
    def __init__(self):
        self.send_calls = 0

    async def send(self, chat_id, content, metadata=None):
        self.send_calls += 1
        raise asyncio.CancelledError()


class BlockingSuccessAdapter:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(content)
        if len(self.sent) == 1:
            self.started.set()
            await self.release.wait()
        return SendResult(success=True, message_id=f"message-{len(self.sent)}")


def test_send_result_failure_is_not_acknowledged(tmp_path, monkeypatch):
    task_id, subscription_id = _setup_blocked_notification(tmp_path, monkeypatch)
    adapter = ResultAdapter([SendResult(success=False, error="platform rejected")])

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT state, attempts, last_error FROM kanban_notification_outbox "
            "WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        assert row["state"] == "pending"
        assert row["attempts"] == 1
        assert "platform rejected" in row["last_error"]
        assert int(kb.list_notify_subs(conn, task_id)[0]["last_event_id"]) > 0
    finally:
        conn.close()


def test_none_send_result_is_an_explicit_failure(tmp_path, monkeypatch):
    _task_id, subscription_id = _setup_blocked_notification(tmp_path, monkeypatch)
    adapter = ResultAdapter([None])

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT state, attempts, last_error FROM kanban_notification_outbox "
            "WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        assert row["state"] == "pending"
        assert row["attempts"] == 1
        assert "SendResult" in row["last_error"]
    finally:
        conn.close()


def test_pre_send_lease_renewal_error_does_not_charge_an_attempt(
    tmp_path, monkeypatch,
):
    _task_id, subscription_id = _setup_blocked_notification(tmp_path, monkeypatch)
    adapter = ResultAdapter([SendResult(success=True)])
    runner = _runner(adapter)

    def fail_initial_renewal(*_args, **_kwargs):
        raise sqlite3.OperationalError("temporary database lock")

    monkeypatch.setattr(
        runner,
        "_kanban_renew_notification_lease",
        fail_initial_renewal,
    )
    asyncio.run(_run_one_tick(monkeypatch, runner))

    assert adapter.sent == []
    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT state, attempts FROM kanban_notification_outbox "
            "WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        assert (row["state"], row["attempts"]) == ("leased", 0)
    finally:
        conn.close()


def test_cancellation_does_not_charge_unsent_leased_batch(tmp_path, monkeypatch):
    db_path = tmp_path / "cancelled-batch.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="cancelled batch", assignee="worker")
        subscription_id = kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        kb._append_event(conn, task_id, "status", {"status": "one"})
        kb._append_event(conn, task_id, "status", {"status": "two"})
        kb.block_task(conn, task_id, reason="needs review", kind="needs_input")
    finally:
        conn.close()

    adapter = CancellingAdapter()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))
    assert adapter.send_calls == 1

    conn = kb.connect()
    try:
        rows = conn.execute(
            "SELECT state, attempts FROM kanban_notification_outbox "
            "WHERE subscription_id = ? ORDER BY event_id",
            (subscription_id,),
        ).fetchall()
        assert [(row["state"], row["attempts"]) for row in rows] == [
            ("leased", 0),
            ("pending", 0),
            ("pending", 0),
        ]
    finally:
        conn.close()


def test_heartbeat_blocks_a_second_watcher_from_sending_an_expired_tail(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "heartbeat-ownership.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="heartbeat ownership", assignee="worker")
        subscription_id = kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        kb._append_event(conn, task_id, "status", {"status": "first"})
        kb.block_task(conn, task_id, reason="second", kind="needs_input")
    finally:
        conn.close()

    monkeypatch.setattr(kb, "NOTIFICATION_LEASE_SECONDS", 1)
    clock = {"now": 100}
    monkeypatch.setattr(kb.time, "time", lambda: clock["now"])

    async def exercise():
        real_sleep = asyncio.sleep
        task_refs = {}
        runner_b = None

        async def controlled_sleep(delay):
            if delay == 5:
                return None
            if (
                delay == 1
                and runner_b is not None
                and asyncio.current_task() is task_refs.get("secondary")
            ):
                runner_b._running = False
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", controlled_sleep)
        loop = asyncio.get_running_loop()
        renewed = asyncio.Event()
        real_renew = kb.renew_notification_lease
        renew_count = 0

        def tracked_renew(*args, **kwargs):
            nonlocal renew_count
            result = real_renew(*args, **kwargs)
            if result:
                renew_count += 1
                if renew_count >= 2:
                    loop.call_soon_threadsafe(renewed.set)
            return result

        monkeypatch.setattr(kb, "renew_notification_lease", tracked_renew)

        blocking_adapter = BlockingSuccessAdapter()
        runner_a = _runner(blocking_adapter)
        watcher_a = asyncio.create_task(runner_a._kanban_notifier_watcher(interval=1))
        await asyncio.wait_for(blocking_adapter.started.wait(), timeout=2)

        # The initial one-second lease would be expired at this logical time.
        # A's heartbeat must renew it before B gets a chance to collect either
        # the active head or the pending tail.
        clock["now"] = 101
        await asyncio.wait_for(renewed.wait(), timeout=2)

        second_adapter = ResultAdapter([SendResult(success=True)])
        runner_b = _runner(second_adapter)
        watcher_b = asyncio.create_task(runner_b._kanban_notifier_watcher(interval=1))
        task_refs["secondary"] = watcher_b
        await asyncio.wait_for(watcher_b, timeout=2)
        assert second_adapter.sent == []

        runner_a._running = False
        blocking_adapter.release.set()
        await asyncio.wait_for(watcher_a, timeout=2)
        assert len(blocking_adapter.sent) == 2

    asyncio.run(exercise())

    conn = kb.connect()
    try:
        rows = conn.execute(
            "SELECT state FROM kanban_notification_outbox "
            "WHERE subscription_id = ? ORDER BY event_id",
            (subscription_id,),
        ).fetchall()
        assert [row["state"] for row in rows] == ["acknowledged", "acknowledged"]
    finally:
        conn.close()


def test_second_watcher_cannot_make_the_first_watcher_send_a_stale_route(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "cross-route-ownership.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        first_task_id = kb.create_task(conn, title="first route", assignee="worker")
        first_subscription_id = kb.add_notify_sub(
            conn,
            task_id=first_task_id,
            platform="telegram",
            chat_id="chat-first",
        )
        kb.block_task(conn, first_task_id, reason="first", kind="needs_input")

        second_task_id = kb.create_task(conn, title="second route", assignee="worker")
        second_subscription_id = kb.add_notify_sub(
            conn,
            task_id=second_task_id,
            platform="telegram",
            chat_id="chat-second",
        )
        kb.block_task(conn, second_task_id, reason="second", kind="needs_input")
    finally:
        conn.close()

    monkeypatch.setattr(kb, "NOTIFICATION_LEASE_SECONDS", 1)
    clock = {"now": 100}
    monkeypatch.setattr(kb.time, "time", lambda: clock["now"])

    async def exercise():
        real_sleep = asyncio.sleep
        task_refs = {}
        runner_b = None

        async def controlled_sleep(delay):
            if delay == 5:
                return None
            if (
                delay == 1
                and runner_b is not None
                and asyncio.current_task() is task_refs.get("secondary")
            ):
                runner_b._running = False
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", controlled_sleep)
        loop = asyncio.get_running_loop()
        renewed = asyncio.Event()
        real_renew = kb.renew_notification_lease
        renew_count = 0

        def tracked_renew(*args, **kwargs):
            nonlocal renew_count
            result = real_renew(*args, **kwargs)
            if result:
                renew_count += 1
                if renew_count >= 2:
                    loop.call_soon_threadsafe(renewed.set)
            return result

        monkeypatch.setattr(kb, "renew_notification_lease", tracked_renew)

        first_adapter = BlockingSuccessAdapter()
        runner_a = _runner(first_adapter)
        watcher_a = asyncio.create_task(runner_a._kanban_notifier_watcher(interval=1))
        await asyncio.wait_for(first_adapter.started.wait(), timeout=2)
        clock["now"] = 101
        await asyncio.wait_for(renewed.wait(), timeout=2)

        second_adapter = ResultAdapter([SendResult(success=True)])
        runner_b = _runner(second_adapter)
        watcher_b = asyncio.create_task(runner_b._kanban_notifier_watcher(interval=1))
        task_refs["secondary"] = watcher_b
        await asyncio.wait_for(watcher_b, timeout=2)

        assert len(second_adapter.sent) == 1

        runner_a._running = False
        first_adapter.release.set()
        await asyncio.wait_for(watcher_a, timeout=2)
        assert len(first_adapter.sent) == 1
        sent_by_a = (
            first_task_id
            if first_task_id in first_adapter.sent[0]
            else second_task_id
        )
        sent_by_b = (
            first_task_id
            if first_task_id in second_adapter.sent[0]
            else second_task_id
        )
        assert sent_by_a != sent_by_b

    asyncio.run(exercise())

    conn = kb.connect()
    try:
        first_state = conn.execute(
            "SELECT state FROM kanban_notification_outbox WHERE subscription_id = ?",
            (first_subscription_id,),
        ).fetchone()
        second_state = conn.execute(
            "SELECT state FROM kanban_notification_outbox WHERE subscription_id = ?",
            (second_subscription_id,),
        ).fetchone()
        assert first_state["state"] == "acknowledged"
        assert second_state["state"] == "acknowledged"
    finally:
        conn.close()


def test_heartbeat_renewal_error_still_acknowledges_a_valid_send(
    tmp_path, monkeypatch,
):
    _task_id, subscription_id = _setup_blocked_notification(tmp_path, monkeypatch)
    monkeypatch.setattr(kb, "NOTIFICATION_LEASE_SECONDS", 1)

    async def exercise():
        real_sleep = asyncio.sleep

        async def controlled_sleep(delay):
            if delay == 5:
                return None
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", controlled_sleep)
        heartbeat_failed = threading.Event()
        renewal_calls = 0
        adapter = BlockingSuccessAdapter()
        runner = _runner(adapter)

        real_renew = runner._kanban_renew_notification_lease

        def renew_once_then_fail(*args, **kwargs):
            nonlocal renewal_calls
            renewal_calls += 1
            if renewal_calls == 1:
                # Preserve actual token ownership for the pre-send CAS.
                return real_renew(*args, **kwargs)
            heartbeat_failed.set()
            raise sqlite3.OperationalError("transient renewal failure")

        monkeypatch.setattr(
            runner,
            "_kanban_renew_notification_lease",
            renew_once_then_fail,
        )
        watcher = asyncio.create_task(runner._kanban_notifier_watcher(interval=1))
        await asyncio.wait_for(adapter.started.wait(), timeout=2)
        assert await asyncio.to_thread(heartbeat_failed.wait, 2)

        runner._running = False
        adapter.release.set()
        await asyncio.wait_for(watcher, timeout=2)
        return adapter

    adapter = asyncio.run(exercise())

    # A later watcher pass must see the durable ACK rather than re-delivering
    # the row after the transient heartbeat renewal error.
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT state FROM kanban_notification_outbox WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        assert row["state"] == "acknowledged"
        assert adapter.sent and len(adapter.sent) == 1
    finally:
        conn.close()


def test_heartbeat_retries_a_transient_renewal_error_before_another_watcher_claims(
    tmp_path, monkeypatch,
):
    _task_id, subscription_id = _setup_blocked_notification(tmp_path, monkeypatch)
    monkeypatch.setattr(kb, "NOTIFICATION_LEASE_SECONDS", 1)
    clock = {"now": 100}
    monkeypatch.setattr(kb.time, "time", lambda: clock["now"])

    async def exercise():
        real_sleep = asyncio.sleep
        task_refs = {}
        runner_b = None

        async def controlled_sleep(delay):
            if delay == 5:
                return None
            if (
                delay == 1
                and runner_b is not None
                and asyncio.current_task() is task_refs.get("secondary")
            ):
                runner_b._running = False
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", controlled_sleep)
        loop = asyncio.get_running_loop()
        heartbeat_failed = asyncio.Event()
        heartbeat_recovered = asyncio.Event()
        renewal_calls = 0
        blocking_adapter = BlockingSuccessAdapter()
        runner_a = _runner(blocking_adapter)
        real_renew = runner_a._kanban_renew_notification_lease

        def fail_once_then_recover(*args, **kwargs):
            nonlocal renewal_calls
            renewal_calls += 1
            if renewal_calls == 2:
                loop.call_soon_threadsafe(heartbeat_failed.set)
                raise sqlite3.OperationalError("transient renewal failure")
            renewed = real_renew(*args, **kwargs)
            if renewal_calls >= 3 and renewed:
                loop.call_soon_threadsafe(heartbeat_recovered.set)
            return renewed

        monkeypatch.setattr(
            runner_a,
            "_kanban_renew_notification_lease",
            fail_once_then_recover,
        )
        watcher_a = asyncio.create_task(runner_a._kanban_notifier_watcher(interval=1))
        await asyncio.wait_for(blocking_adapter.started.wait(), timeout=2)
        await asyncio.wait_for(heartbeat_failed.wait(), timeout=2)

        # The original lease would be expired at this logical time. The next
        # heartbeat renewal must recover ownership before watcher B polls.
        clock["now"] = 101
        await asyncio.wait_for(heartbeat_recovered.wait(), timeout=2)

        second_adapter = ResultAdapter([SendResult(success=True)])
        runner_b = _runner(second_adapter)
        watcher_b = asyncio.create_task(runner_b._kanban_notifier_watcher(interval=1))
        task_refs["secondary"] = watcher_b
        await asyncio.wait_for(watcher_b, timeout=2)
        assert second_adapter.sent == []

        runner_a._running = False
        blocking_adapter.release.set()
        await asyncio.wait_for(watcher_a, timeout=2)
        assert len(blocking_adapter.sent) == 1

    asyncio.run(exercise())

    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT state FROM kanban_notification_outbox WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        assert row["state"] == "acknowledged"
    finally:
        conn.close()


def test_failures_dead_letter_across_runner_restarts(tmp_path, monkeypatch):
    _task_id, subscription_id = _setup_blocked_notification(tmp_path, monkeypatch)

    for _ in range(kb.NOTIFICATION_MAX_ATTEMPTS):
        adapter = ResultAdapter([SendResult(success=False, error="still down")])
        asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT state, attempts FROM kanban_notification_outbox "
            "WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        assert row["state"] == "dead_letter"
        assert row["attempts"] == kb.NOTIFICATION_MAX_ATTEMPTS
        assert kb.notification_outbox_counts(conn)["dead_letter"] == 1
    finally:
        conn.close()


def test_only_successful_send_result_acknowledges(tmp_path, monkeypatch):
    _task_id, subscription_id = _setup_blocked_notification(tmp_path, monkeypatch)
    adapter = ResultAdapter([SendResult(success=True, message_id="m-1")])

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT state, attempts FROM kanban_notification_outbox "
            "WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        assert row["state"] == "acknowledged"
        assert row["attempts"] == 0
    finally:
        conn.close()


def test_restart_after_ack_finishes_terminal_unsubscribe(tmp_path, monkeypatch):
    db_path = tmp_path / "post-ack-crash.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="completed delivery", assignee="worker")
        subscription_id = kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        kb.complete_task(conn, task_id, summary="done")
        kb.stage_unseen_notifications_for_sub(
            conn,
            subscription_id=subscription_id,
            kinds=["completed"],
            action_kinds=["completed"],
        )
        leased = kb.lease_notification_outbox(
            conn, subscription_ids=[subscription_id]
        )[0]
        assert kb.ack_notification_delivery(
            conn,
            subscription_id=subscription_id,
            event_id=leased["event_id"],
            action=leased["action"],
            lease_token=leased["lease_token"],
        )
        # Simulate a crash after ACK but before the old watcher could remove
        # the terminal subscription.
        assert len(kb.list_notify_subs(conn, task_id)) == 1
    finally:
        conn.close()

    restarted_adapter = ResultAdapter([])
    asyncio.run(_run_one_tick(monkeypatch, _runner(restarted_adapter)))

    conn = kb.connect()
    try:
        assert kb.list_notify_subs(conn, task_id) == []
        assert restarted_adapter.sent == []
    finally:
        conn.close()


def test_terminal_cleanup_waits_for_every_outbox_batch(tmp_path, monkeypatch):
    db_path = tmp_path / "terminal-batches.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="many notifications", assignee="worker")
        subscription_id = kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        for index in range(100):
            kb._append_event(conn, task_id, "status", {"status": f"step-{index}"})
        assert kb.complete_task(conn, task_id, summary="done")
    finally:
        conn.close()

    adapter = ResultAdapter([SendResult(success=True) for _ in range(101)])
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        assert len(adapter.sent) == 100
        assert len(kb.list_notify_subs(conn, task_id)) == 1
        counts = kb.notification_outbox_counts(conn)
        assert counts["acknowledged"] == 100
        assert counts["pending"] == 1
        remaining = conn.execute(
            "SELECT event_id FROM kanban_notification_outbox "
            "WHERE subscription_id = ? AND state = 'pending'",
            (subscription_id,),
        ).fetchall()
        assert len(remaining) == 1
    finally:
        conn.close()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        assert len(adapter.sent) == 101
        assert kb.list_notify_subs(conn, task_id) == []
        assert kb.notification_outbox_counts(conn)["pending"] == 0
    finally:
        conn.close()


def test_partial_batch_keeps_acknowledged_actions_and_recovers_later_failure(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "partial-batch.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="partial batch", assignee="worker")
        subscription_id = kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        kb._append_event(conn, task_id, "status", {"status": "ready"})
        assert kb.complete_task(conn, task_id, summary="done")
    finally:
        conn.close()

    adapter = ResultAdapter([
        SendResult(success=True),
        SendResult(success=False, error="temporary failure"),
        SendResult(success=True),
    ])
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        rows = conn.execute(
            "SELECT state FROM kanban_notification_outbox "
            "WHERE subscription_id = ? ORDER BY event_id",
            (subscription_id,),
        ).fetchall()
        assert [row["state"] for row in rows] == ["acknowledged", "pending"]
        assert len(kb.list_notify_subs(conn, task_id)) == 1
    finally:
        conn.close()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len([text for text in adapter.sent if "Kanban" in text and "→ ready" in text]) == 1
    assert len(adapter.sent) == 3
    conn = kb.connect()
    try:
        assert kb.list_notify_subs(conn, task_id) == []
    finally:
        conn.close()


def test_terminal_notification_never_injects_a_sessionless_wake(tmp_path, monkeypatch):
    db_path = tmp_path / "notify-only.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="notify only",
            assignee="worker",
            session_id="session-that-must-not-wake",
        )
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        assert kb.complete_task(conn, task_id, summary="done")
    finally:
        conn.close()

    adapter = NoWakeAdapter([SendResult(success=True)])
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert adapter.wake_calls == 0
    assert len(adapter.sent) == 1

    adapter = ResultAdapter([])
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    conn = kb.connect()
    try:
        assert kb.list_notify_subs(conn, task_id) == []
        assert adapter.sent == []
    finally:
        conn.close()
