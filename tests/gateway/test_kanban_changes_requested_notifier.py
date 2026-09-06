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


async def _run_one_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
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


def _create_review_block(delivery_mode, *, reason="Tests need updates"):
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
            kind="changes_requested",
            payload={
                "reason": reason,
                "reviewer": "claude-qa",
                "implementer": "codex-cua",
                "status": "ready",
            },
        )
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
            kinds=["changes_requested"],
        )
        return events
    finally:
        conn.close()


def test_changes_requested_notify_wake_is_actionable_and_exactly_routed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-block.db"))
    kb.init_db()
    task_id = _create_review_block("notify+wake")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert text.startswith(f"🛑 [default] Kanban {task_id} review requested changes/BLOCK: Tests need updates")
    assert "reviewer @claude-qa → implementer @codex-cua" in text
    assert adapter.sent[0]["metadata"]["thread_id"] == "topic-7"
    assert len(adapter.handled) == 1
    wake = adapter.handled[0]
    assert wake.source.chat_id == "chat-1"
    assert wake.source.chat_type == "thread"
    assert wake.source.thread_id == "topic-7"
    assert "implementation is not approved" in wake.text
    assert "Inspect the existing card and its current review run" in wake.text
    assert "do not create a duplicate task" in wake.text
    assert wake.text.count("do not create a duplicate task") == 1
    assert _unseen(task_id) == []

    # A fresh watcher after restart cannot replay an event whose cursor advanced.
    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1


def test_changes_requested_notify_is_passive_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "notify.db"))
    kb.init_db()
    task_id = _create_review_block("notify")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert len(adapter.sent) == 1
    assert adapter.handled == []
    assert _unseen(task_id) == []


def test_changes_requested_wake_only_has_no_passive_post(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake.db"))
    kb.init_db()
    task_id = _create_review_block("wake")
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    assert adapter.sent == []
    assert len(adapter.handled) == 1
    assert _unseen(task_id) == []


def test_changes_requested_send_failure_retries_without_event_loss(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "retry.db"))
    kb.init_db()
    task_id = _create_review_block("notify")
    failing = RecordingAdapter(fail_send=True)

    asyncio.run(_run_one_tick(monkeypatch, _runner(failing)))
    assert len(failing.sent) == 1
    assert len(_unseen(task_id)) == 1

    healthy = RecordingAdapter()
    asyncio.run(_run_one_tick(monkeypatch, _runner(healthy)))
    assert len(healthy.sent) == 1
    assert _unseen(task_id) == []


def test_changes_requested_reason_is_redacted_path_safe_and_truncated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "redact.db"))
    kb.init_db()
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    reason = f"See /Users/alice/private/review.log token={secret} " + ("x" * 300)
    _create_review_block("notify", reason=reason)
    adapter = RecordingAdapter()

    asyncio.run(_run_one_tick(monkeypatch, _runner(adapter)))

    text = adapter.sent[0]["text"]
    assert "/Users/alice" not in text
    assert "abcdefghijklmnopqrstuvwxyz" not in text
    assert "[local path]" in text
    assert "… — reviewer @claude-qa" in text
