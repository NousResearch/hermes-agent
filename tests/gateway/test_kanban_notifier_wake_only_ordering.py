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


class FailingWakeAdapter(RecordingAdapter):
    """Push adapter whose wake injection (handle_message) always fails."""

    async def handle_message(self, event):
        self.handled.append(event)
        raise RuntimeError("simulated wake failure")


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter, *, platform=Platform.TELEGRAM):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {platform: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _make_completed_task(delivery_mode, *, platform="telegram", chat_id="chat-1"):
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="wake ordering task",
            assignee="worker",
            session_id=f"agent:main:{platform}:dm:{chat_id}",
        )
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform=platform,
            chat_id=chat_id,
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
    assert runner._kanban_sub_fail_counts == {}


def test_wake_only_log_omits_source_recipient_id(tmp_path, monkeypatch, caplog):
    """Wake logs keep task/platform diagnostics without printing the source chat id."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-log.db"))
    kb.init_db()
    source_chat_id = "synthetic-imessage-chat-id"
    tid = _make_completed_task("wake", platform="bluebubbles", chat_id=source_chat_id)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter, platform=Platform.BLUEBUBBLES)
    with caplog.at_level("INFO", logger="gateway.run"):
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert tid in messages
    assert "bluebubbles" in messages
    assert source_chat_id not in messages


def test_passive_notify_log_omits_telegram_recipient_id(tmp_path, monkeypatch, caplog):
    """Passive raw-notify logs keep task/platform diagnostics without printing the Telegram chat id."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "notify-log.db"))
    kb.init_db()
    private_chat_id = "synthetic-private-telegram-chat-id"
    tid = _make_completed_task("notify", chat_id=private_chat_id)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    with caplog.at_level("DEBUG", logger="gateway.run"):
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert tid in messages
    assert "telegram" in messages
    assert private_chat_id not in messages


def test_wake_only_failure_rewinds_and_redelivers(tmp_path, monkeypatch):
    """Wake fails: cursor rewound, counter bumped, event retried next tick."""
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
    assert list(runner._kanban_sub_fail_counts.values()) == [1], (
        "failure counter must bump on a failed wake-only delivery"
    )
    assert len(_subs(tid)) == 1, "one transient failure must not drop the sub"

    # Next tick: the same event is claimed and the wake retried.
    runner2 = _make_runner(adapter)
    runner2._kanban_sub_fail_counts = runner._kanban_sub_fail_counts
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner2))
    assert len(adapter.handled) == 2, "event must be redelivered next tick"
    assert list(runner2._kanban_sub_fail_counts.values()) == [2]


def test_notify_wake_failure_stays_best_effort(tmp_path, monkeypatch):
    """notify+wake: text ping IS the delivery; failed wake must NOT rewind."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "notify-wake.db"))
    kb.init_db()
    tid = _make_completed_task("notify+wake")

    adapter = FailingWakeAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "text ping delivered"
    assert len(adapter.handled) == 1, "wake attempted best-effort"
    assert _unseen_terminal_events(tid) == [], (
        "notify+wake: cursor advances on text delivery; a failed wake is "
        "best-effort and must not rewind"
    )
    assert runner._kanban_sub_fail_counts == {}, (
        "best-effort wake failure must not bump the send-failure counter"
    )
    # (The sub itself unsubscribes because the task reached 'done' —
    # pre-existing task_terminal behavior, unrelated to the wake outcome.)


def test_wake_only_failure_cap_drops_subscription(tmp_path, monkeypatch):
    """After MAX_SEND_FAILURES consecutive wake failures the sub is dropped."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-cap.db"))
    kb.init_db()
    tid = _make_completed_task("wake")

    adapter = FailingWakeAdapter()
    runner = _make_runner(adapter)
    # Simulate 11 prior consecutive failures (MAX_SEND_FAILURES = 12).
    runner._kanban_sub_fail_counts = {
        (tid, "telegram", "chat-1", ""): 11,
    }
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.handled) == 1
    assert _subs(tid) == [], (
        "subscription must drop after MAX_SEND_FAILURES consecutive "
        "wake-only delivery failures, like text sends do"
    )
    assert runner._kanban_sub_fail_counts == {}, (
        "counter entry must clear when the subscription is dropped"
    )
