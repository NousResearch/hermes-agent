"""A reopened review must reach the subscription side.

Kernel receipt (``hermes_cli/kanban_db.py``, ``reopen_review_for_rework``): a
review/controller flow that sends a review card back to its implementer records
``review_reopened`` — and when the payload would be exactly ``{"status":
"ready"}`` the kernel records the event with ``payload=None``, so the formatter
must tolerate a missing payload.

The notifier claimed only the kinds in ``TERMINAL_KINDS``, which did not list
``review_reopened``: the subscriber's last message stayed on "ready for review"
and the cursor never advanced past the event, so it stayed unseen forever.
These tests pin the delivered behaviour (a ping, and a wake for
notify+wake subscribers), not the source shape of the kind list.
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


def _create_reopened_review(delivery_mode, *, reopen_payload=None):
    """One subscription that saw the handoff, then the reopen the kernel writes."""
    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="existing implementation card",
            assignee="implementer",
            session_id="agent:main:telegram:thread:chat-1:topic-7",
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
        kb._append_event(
            conn,
            task_id,
            kind="review_requested",
            payload={"summary": "ready for review"},
        )
        kb._append_event(conn, task_id, kind="review_reopened", payload=reopen_payload)
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


def test_review_reopened_without_payload_is_delivered_and_settles_cursor(tmp_path, monkeypatch):
    """The kernel's exact ready-shape reopen (payload omitted) must be pinged."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "reopen.db"))
    kb.init_db()
    task_id = _create_reopened_review("notify+wake")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 2
    assert "ready for review" in adapter.sent[0]["text"]
    reopen_ping = adapter.sent[1]["text"]
    assert task_id in reopen_ping
    assert "reopen" in reopen_ping.lower()
    assert adapter.sent[1]["metadata"]["thread_id"] == "topic-7"
    # The event is claimed, so a fresh watcher cannot replay it.
    assert _unseen(task_id) == []
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))
    assert len(adapter.sent) == 2


def test_review_reopened_wakes_the_origin_session(tmp_path, monkeypatch):
    """notify+wake hands the decision back to the origin, like changes_requested."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "reopen-wake.db"))
    kb.init_db()
    _create_reopened_review("notify+wake")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.handled) == 1
    wake = adapter.handled[0]
    assert wake.source.chat_id == "chat-1"
    assert wake.source.thread_id == "topic-7"
    assert "reopen" in wake.text.lower()


def test_review_reopened_passive_notify_does_not_wake(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "notify.db"))
    kb.init_db()
    task_id = _create_reopened_review("notify")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 2
    assert adapter.handled == []
    assert _unseen(task_id) == []


def test_review_reopened_carries_the_implementer_from_the_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "impl.db"))
    kb.init_db()
    _create_reopened_review(
        "notify", reopen_payload={"status": "ready", "implementer": "reworker"}
    )
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 2
    assert "implementer @reworker" in adapter.sent[1]["text"]


def test_review_reopened_send_failure_retries_without_event_loss(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "retry.db"))
    kb.init_db()
    task_id = _create_reopened_review("notify")
    failing = RecordingAdapter(fail_send=True)

    asyncio.run(_run_one_tick(monkeypatch, _runner(failing)))
    # The batch aborts on the first failed send; the cursor rewinds so nothing is lost.
    assert len(failing.sent) == 1
    assert len(_unseen(task_id)) == 1

    healthy = RecordingAdapter()
    asyncio.run(_run_one_tick(monkeypatch, _runner(healthy)))
    # The rewound cursor replays the whole batch, the reopen ping included.
    assert len(healthy.sent) == 2
    assert "reopen" in healthy.sent[1]["text"].lower()
    assert _unseen(task_id) == []
