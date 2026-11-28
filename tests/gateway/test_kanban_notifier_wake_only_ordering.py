"""Wake-only (delivery_mode='wake') push-adapter delivery ordering.

For a push-adapter subscription in wake-only mode the visible text ping is
intentionally skipped (the ``send_passive`` gate), so the wake injection IS
the sole delivery. The cursor must therefore only advance AFTER the wake
succeeds: advancing first and running the wake best-effort afterwards let a
failed wake permanently lose the event — the exact bug class the non-push
(api_server) self-post branch already guards against with rewind/retry.

Residual insight extracted from closed PR #84191 (@MaximCrabbe).
"""

import asyncio
import time

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    """Push-capable adapter (no supports_async_delivery attr => push)."""

    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


class FailingWakeAdapter(RecordingAdapter):
    """Push adapter whose wake injection (handle_message) always fails."""

    async def handle_message(self, event):
        self.handled.append(event)
        raise RuntimeError("simulated wake failure")


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _make_completed_task(delivery_mode):
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="wake ordering task",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            chat_type="dm",
            delivery_mode=delivery_mode,
        )
        kb.complete_task(conn, tid, summary="done")
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kbc.connect()
    try:
        _, events = kbn.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def _subs(tid):
    conn = kbc.connect()
    try:
        return kbn.list_notify_subs(conn, tid)
    finally:
        conn.close()


def conn_row(tid):
    conn = kbc.connect()
    try:
        return conn.execute(
            "SELECT delivery_failures, retry_after FROM kanban_notify_subs WHERE task_id = ?",
            (tid,),
        ).fetchone()
    finally:
        conn.close()


def _backdate_park(tid):
    conn = kbc.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE kanban_notify_subs SET retry_after = ? WHERE task_id = ?",
                (time.time() - 1.0, tid),
            )
    finally:
        conn.close()


def test_wake_only_success_advances_cursor_single_wake(tmp_path, monkeypatch):
    """Wake succeeds: exactly one wake, no text ping, cursor advanced."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-ok.db"))
    kb.init_db()
    tid = _make_completed_task("wake")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == [], "wake-only mode must not send a text ping"
    assert len(adapter.handled) == 1, "wake injection is the sole delivery"
    assert _unseen_terminal_events(tid) == [], (
        "cursor must advance after a successful wake-only delivery"
    )
    row = conn_row(tid)
    assert row["delivery_failures"] == 0 and row["retry_after"] is None


def test_wake_only_failure_rewinds_counts_and_parks(tmp_path, monkeypatch):
    """Wake fails: cursor rewound, failure counted durably, sub parked on
    backoff — one attempt per backoff step, never dropped."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-fail.db"))
    kb.init_db()
    tid = _make_completed_task("wake")

    adapter = FailingWakeAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.handled) == 1, "one wake attempt on the first tick"
    events = _unseen_terminal_events(tid)
    assert len(events) == 1, (
        "a failed wake-only delivery must REWIND the claim so the event "
        "is redelivered on a later tick — it was permanently lost before "
        "this fix (cursor advanced before the best-effort wake)"
    )
    row = conn_row(tid)
    assert row["delivery_failures"] == 1, "failure counter must bump on a failed wake-only delivery"
    assert row["retry_after"] is not None and row["retry_after"] > time.time()
    assert len(_subs(tid)) == 1, "one transient failure must not drop the sub"

    # Parked: the next tick claims nothing until the backoff window passes.
    runner2 = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner2))
    assert len(adapter.handled) == 1, "parked subscription must not be re-attempted"

    # Backoff expiry: the same event is claimed and the wake retried, counted.
    _backdate_park(tid)
    runner2._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner2))
    assert len(adapter.handled) == 2, "event must be redelivered after the backoff window"
    row = conn_row(tid)
    assert row["delivery_failures"] == 2


def test_notify_wake_failure_retries_without_repeating_ping(tmp_path, monkeypatch):
    """notify+wake requires both deliveries, retaining the sent-ping checkpoint."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "notify-wake.db"))
    kb.init_db()
    tid = _make_completed_task("notify+wake")

    adapter = FailingWakeAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "text ping delivered"
    assert len(adapter.handled) == 1, "wake attempted after the ping"
    assert len(_unseen_terminal_events(tid)) == 1
    assert conn_row(tid)["delivery_failures"] == 1
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1, "parked tick sends nothing"
    _backdate_park(tid)
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1, "ping checkpoint holds across parked retries"
    assert len(adapter.handled) == 2
    assert conn_row(tid)["delivery_failures"] == 2


def test_wake_only_failure_cap_parks_at_one_hour(tmp_path, monkeypatch):
    """Backoff is capped: after many consecutive failures the sub stays alive,
    parked, retried at most hourly — never deleted."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-cap.db"))
    kb.init_db()
    tid = _make_completed_task("wake")

    adapter = FailingWakeAdapter()
    runner = _make_runner(adapter)
    # Simulate 11 prior consecutive failures directly on the durable row.
    conn = kbc.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE kanban_notify_subs SET delivery_failures = 11, retry_after = ?"
                " WHERE task_id = ?",
                (time.time() - 1.0, tid),
            )
    finally:
        conn.close()
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.handled) == 1
    row = conn_row(tid)
    assert row is not None, "a dead destination is parked, never unsubscribed"
    assert row["delivery_failures"] == 12
    # 5s * 2**11 ≫ 1h — the 12th backoff is capped at the hourly ceiling.
    assert time.time() + 3599.0 <= row["retry_after"] <= time.time() + 3601.0
