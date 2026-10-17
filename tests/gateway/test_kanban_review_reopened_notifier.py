"""`review_reopened` must reach subscribed channels like the other review kinds.

`reopen_review_task` (review -> ready/todo) appends the event, but the notifier
never claimed it: neither TERMINAL_KINDS nor _WAKE_KINDS nor the formatter map
named it, so the subscription cursor stalled on the `review_requested` handoff
and a channel kept showing "ready for review" while the card was already back
for rework (#135738). These tests pin the producer/consumer pair end to end.
"""

import asyncio

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    def __init__(self, *, fail_send=False):
        self.sent = []
        self.handled = []
        self.fail_send = fail_send

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        if self.fail_send:
            raise RuntimeError("transient send failure")

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


async def _run_one_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _review_task_reopened(delivery_mode, *, payload=True, reason=None):
    """A card that went to review and was reopened by the operator.

    The real pair (`request_review` then `reopen_review_task`) is used so the
    payload shape — landing status plus the implementer restored from the
    handoff event — is the one production writes, not a hand-copied dict.
    """
    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="implementation under review",
            assignee="codex-cua",
            session_id="agent:main:telegram:thread:chat-1:topic-7",
        )
        kb.request_review(
            conn, task_id, summary="handoff summary", reviewer="claude-qa", force=True
        )
        kbn.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-7",
            chat_type="thread",
            delivery_mode=delivery_mode,
            delivery_metadata={"thread_id": "topic-7", "chat_type": "thread"},
        )
        if payload:
            assert kb.reopen_review_task(conn, task_id, reason=reason)
        else:
            # `reopen_review_task` omits the payload entirely when the card lands
            # on `ready` with no implementer to restore; the formatter must not
            # require one.
            kb._append_event(conn, task_id, kind="review_reopened", payload=None)
        return task_id
    finally:
        conn.close()


def _unseen(task_id):
    conn = kbc.connect()
    try:
        _, events = kbn.unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            thread_id="topic-7",
            kinds=["review_reopened"],
        )
        return events
    finally:
        conn.close()


def test_review_reopened_notify_wake_is_actionable_and_exactly_routed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-reopen.db"))
    kb.init_db()
    task_id = _review_task_reopened("notify+wake")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert text.startswith(
        f"🔁 [default] Kanban {task_id} review reopened — returned to"
    )
    assert "for rework → implementer @codex-cua" in text
    assert adapter.sent[0]["metadata"]["thread_id"] == "topic-7"
    assert len(adapter.handled) == 1
    wake = adapter.handled[0]
    assert wake.source.chat_id == "chat-1"
    assert wake.source.thread_id == "topic-7"
    assert "review reopened" in wake.text
    assert _unseen(task_id) == []

    # A fresh watcher after restart cannot replay an event whose cursor advanced.
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1


def test_review_reopened_without_payload_notifies_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "no-payload.db"))
    kb.init_db()
    _review_task_reopened("notify", payload=False)
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 1
    assert "returned to ready for rework" in adapter.sent[0]["text"]
    assert adapter.handled == []


def test_review_reopened_reason_reaches_ping_and_wake(tmp_path, monkeypatch):
    """The operator's reopen reason rides on the event payload into both the
    channel ping and the wake turn's review detail (#135738)."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "reopen-reason.db"))
    kb.init_db()
    task_id = _review_task_reopened(
        "notify+wake", reason="Device verification failed; return to implementer"
    )
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert (
        f"review reopened — returned to ready for rework → implementer @codex-cua"
        ": Device verification failed; return to implementer" in text
    )
    assert len(adapter.handled) == 1
    assert "Device verification failed" in adapter.handled[0].text
    assert _unseen(task_id) == []
