"""Delivery contracts for Telegram todo snapshots (#31001, PR #31106)."""
import asyncio
import queue
from contextlib import suppress
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class ChecklistAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent, self.edits = [], []
        self.pending = {topic: asyncio.Event() for topic in ("a", "b")}
        self.completed = {topic: asyncio.Event() for topic in ("a", "b")}

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        mid = str(len(self.sent) + 1)
        self.sent.append(dict(chat_id=chat_id, content=content, message_id=mid, metadata=metadata))
        topic = (metadata or {}).get("thread_id")
        if "[ ] Work" in content:
            self.pending[topic].set()
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content, *, metadata=None):
        self.edits.append(dict(chat_id=chat_id, message_id=message_id, content=content, metadata=metadata))
        topic = (metadata or {}).get("thread_id")
        if "[x] Work" in content:
            self.completed[topic].set()
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def stop_typing(self, chat_id):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def make_turn(adapter, source, *, grouping="accumulate", enabled=True):
    ctx = TurnContext(source=source, progress_queue=queue.Queue(),
                      _run_still_current=lambda: True, tool_progress_enabled=enabled,
                      progress_mode="all", progress_grouping=grouping,
                      _progress_metadata={"thread_id": source.thread_id},
                      long_tool_hint_fired=[True])
    return TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter), ctx)


def snapshot(topic, status):
    return {"todos": [{"content": f"Work {topic}", "id": topic, "status": status}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("grouping", ["accumulate", "separate"])
async def test_snapshots_reuse_scoped_message_and_final_is_separate(grouping):
    adapter = ChecklistAdapter()

    async def handler(event):
        topic = event.source.thread_id
        turn = make_turn(adapter, event.source, grouping=grouping)
        sender = asyncio.create_task(turn.send_progress_messages())
        try:
            turn.progress_callback("tool.completed", "todo", result=snapshot(topic, "pending"))
            await asyncio.wait_for(adapter.pending[topic].wait(), 5)
            # Interim assistant delivery closes ordinary progress bubbles, not the checklist.
            turn._ctx.progress_queue.put(("__reset__",))
            turn.progress_callback("tool.started", "terminal", "ordinary progress", {})
            turn.progress_callback("tool.completed", "todo_list", result=snapshot(topic, "completed"))
            await asyncio.wait_for(adapter.completed[topic].wait(), 5)
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender
        return f"Final {topic}"

    adapter.set_message_handler(handler)
    events = [MessageEvent(text="work", message_id=f"input-{topic}",
                          source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", thread_id=topic))
              for topic in ("a", "b")]
    await asyncio.gather(*(adapter._process_message_background(event, f"session-{event.source.thread_id}")
                           for event in events))
    ids = []
    for topic in ("a", "b"):
        checklist = [call for call in adapter.sent if f"[ ] Work {topic}" in call["content"]]
        assert len(checklist) == 1
        mid = checklist[0]["message_id"]
        ids.append(mid)
        edits = [call for call in adapter.edits if call["message_id"] == mid]
        assert edits and f"[x] Work {topic}" in edits[-1]["content"]
        assert "[ ] Work" not in edits[-1]["content"]
        assert all(call["metadata"]["thread_id"] == topic for call in edits)
        final = [call for call in adapter.sent if call["content"] == f"Final {topic}"]
        assert len(final) == 1 and final[0]["message_id"] != mid
    assert ids[0] != ids[1]


@pytest.mark.asyncio
async def test_turn_end_flushes_latest_queued_snapshot():
    adapter = ChecklistAdapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", thread_id="a")
    turn = make_turn(adapter, source)
    sender = asyncio.create_task(turn.send_progress_messages())
    turn.progress_callback("tool.completed", "todo", result=snapshot("a", "pending"))
    await asyncio.wait_for(adapter.pending["a"].wait(), 5)
    turn.progress_callback("tool.completed", "todo", result=snapshot("a", "completed"))
    sender.cancel()
    with suppress(asyncio.CancelledError):
        await sender
    assert len(adapter.sent) == 1
    assert adapter.edits[-1]["message_id"] == adapter.sent[0]["message_id"]
    assert "[x] Work a" in adapter.edits[-1]["content"]
    assert "[ ] Work a" not in adapter.edits[-1]["content"]


@pytest.mark.parametrize("platform,enabled,result", [
    (Platform.TELEGRAM, False, snapshot("a", "pending")),
    (Platform.DISCORD, True, snapshot("a", "pending")),
    (Platform.TELEGRAM, True, "invalid"),
])
def test_completion_fallback_and_visibility_gates(monkeypatch, platform, enabled, result):
    from unittest.mock import Mock

    source = SessionSource(platform=platform, chat_id="chat", thread_id="a")
    turn = make_turn(None, source, enabled=enabled)
    turn._ctx.long_tool_hint_fired[0] = False
    fallback = Mock()
    monkeypatch.setattr(turn, "_progress_onboarding_hint", fallback)
    turn.progress_callback("tool.completed", "todo", result=result)
    assert turn._ctx.progress_queue.empty()
    fallback.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transient", "permanent", "exception", "flood"])
async def test_delivery_failure_keeps_binding_and_rate_limit(monkeypatch, failure):
    from gateway.todo_progress import TodoChecklist, TodoProgress

    adapter = ChecklistAdapter()
    turn = make_turn(adapter, SessionSource(platform=Platform.TELEGRAM, chat_id="chat", thread_id="a"))
    status = TodoProgress(turn, adapter)
    clock = [10.0]
    monkeypatch.setattr("gateway.todo_progress.time", SimpleNamespace(monotonic=lambda: clock[0]))
    status.absorb(TodoChecklist("📋 Task List\n[ ] Work a"))
    await status.flush()
    mid = adapter.sent[0]["message_id"]
    status.absorb(TodoChecklist("📋 Task List\n[x] Work a"))
    original_edit = adapter.edit_message
    attempts = []

    async def fail_once(**kwargs):
        attempts.append(kwargs)
        if len(attempts) > 1:
            return await original_edit(**kwargs)
        if failure == "exception":
            raise ConnectionError("offline")
        return SendResult(success=False, retryable=failure == "transient",
                          retry_after=10 if failure == "flood" else None)

    monkeypatch.setattr(adapter, "edit_message", fail_once)
    clock[0] = 12
    await status.flush()
    await status.flush(force=True)
    assert len(attempts) == 1
    clock[0] = 23
    await status.flush()
    assert len(adapter.sent) == 1
    assert all(call["message_id"] == mid for call in attempts)
    if failure == "permanent":
        assert len(attempts) == 1
    else:
        assert "[x] Work a" in adapter.edits[-1]["content"]


@pytest.mark.asyncio
async def test_large_snapshot_stays_within_platform_length_limit():
    from gateway.todo_progress import TodoChecklist, TodoProgress, _render_todo_checklist

    adapter = ChecklistAdapter()
    turn = make_turn(adapter, SessionSource(platform=Platform.TELEGRAM, chat_id="chat", thread_id="a"))
    status = TodoProgress(turn, adapter)
    status.state._progress_len_fn = lambda text: len(text.encode("utf-16-le")) // 2
    status.state._PROGRESS_TEXT_LIMIT = 180
    text = _render_todo_checklist({"todos": [{"content": "😀" * 80} for _ in range(256)]})
    status.absorb(TodoChecklist(text))
    await status.flush()
    content = adapter.sent[0]["content"]
    assert status.state._progress_len_fn(content) <= 180
    assert "more tasks" in content


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [True, False])
async def test_stopped_or_superseded_run_does_not_flush(cancelled):
    from gateway.todo_progress import TodoChecklist, TodoProgress

    adapter = ChecklistAdapter()
    turn = make_turn(adapter, SessionSource(platform=Platform.TELEGRAM, chat_id="chat", thread_id="a"))
    if cancelled:
        turn._ctx.agent_holder[0] = SimpleNamespace(is_interrupted=lambda: True)
    else:
        turn._ctx._run_still_current = lambda: False
    status = TodoProgress(turn, adapter)
    status.absorb(TodoChecklist("📋 Task List\n[x] Work a"))
    await status.flush(force=True)
    assert not adapter.sent


@pytest.mark.asyncio
async def test_refused_status_edit_does_not_block_final_reply(monkeypatch):
    adapter = ChecklistAdapter()

    async def refuse(**kwargs):
        return SendResult(success=False, error="Not permitted")

    monkeypatch.setattr(adapter, "edit_message", refuse)

    async def handler(event):
        turn = make_turn(adapter, event.source)
        sender = asyncio.create_task(turn.send_progress_messages())
        try:
            turn.progress_callback("tool.completed", "todo", result=snapshot("a", "pending"))
            await asyncio.wait_for(adapter.pending["a"].wait(), 5)
            turn.progress_callback("tool.completed", "todo", result=snapshot("a", "completed"))
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender
        return "Final despite edit failure"

    adapter.set_message_handler(handler)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", thread_id="a")
    await adapter._process_message_background(MessageEvent(text="work", source=source), "session-a")
    assert adapter.sent[-1]["content"] == "Final despite edit failure"
    assert len(adapter.sent) == 2
