"""Full-progress edits preserve a consumed, redacted tool entry across transient failures."""

import asyncio
from queue import Queue
from gateway.session import SessionSource
from gateway.config import Platform

import pytest

from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from tests.gateway.test_consolidated_lifecycle_progress import Capture, rail
from tests.gateway.test_run_progress_topics import MetadataEditProgressCaptureAdapter, _make_runner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_retryable_edit_retains_tool_arguments_and_order_through_completion():
    class Adapter(MetadataEditProgressCaptureAdapter):
        def __init__(self):
            super().__init__()
            self.visible = {}
            self.fail_next = True

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            result = await super().send(chat_id, content, reply_to, metadata)
            self.visible[result.message_id] = content
            return result

        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.edits.append(content)
            if self.fail_next:
                self.fail_next = False
                return SendResult(success=False, message_id=message_id, retryable=True, error="transient timeout")
            self.visible[message_id] = content
            return SendResult(success=True, message_id=message_id)

    adapter = Adapter()
    ctx = TurnContext(source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="group"), _run_still_current=lambda: True,
                      progress_mode="full", tool_progress_enabled=True, progress_grouping="accumulate", progress_queue=Queue(),
                      _progress_reply_to="anchor", _progress_metadata={"thread_id": "topic"})
    turn = TurnRunner(_make_runner(adapter), ctx)
    for name, arguments in (("first", {"query": "safe-one"}), ("second", {"query": "safe-two"})):
        turn.progress_callback("tool.started", name, "preview", arguments)
    task = asyncio.create_task(turn.send_progress_messages())
    try:
        await async_wait_until(lambda: bool(adapter.edits), timeout=2, what="transient progress edit")
    finally:
        task.cancel()
        await asyncio.wait_for(task, 2)
    assert len(adapter.sent) == 1
    assert "safe-one" in adapter.visible["progress-1"]
    assert "safe-two" in adapter.visible["progress-1"]
    assert adapter.visible["progress-1"].index("safe-one") < adapter.visible["progress-1"].index("safe-two")


@pytest.mark.asyncio
async def test_skipped_final_edit_publishes_only_unseen_tool_entry_as_continuation():
    class Adapter(MetadataEditProgressCaptureAdapter):
        def __init__(self):
            super().__init__()
            self.visible = []

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            result = await super().send(chat_id, content, reply_to, metadata)
            self.visible.append(content)
            return result

        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.edits.append(content)
            return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})

    adapter = Adapter()
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="group"),
        _run_still_current=lambda: True,
        progress_mode="full",
        tool_progress_enabled=True,
        progress_grouping="accumulate",
        progress_queue=Queue(),
        _progress_reply_to="anchor",
        _progress_metadata={"thread_id": "topic"},
    )
    turn = TurnRunner(_make_runner(adapter), ctx)
    turn.progress_callback("tool.started", "first", "preview", {"query": "safe-one"})
    task = asyncio.create_task(turn.send_progress_messages())
    try:
        await async_wait_until(lambda: len(adapter.visible) == 1, timeout=2, what="initial progress send")
        turn.progress_callback("tool.started", "second", "preview", {"query": "safe-two"})
        await async_wait_until(lambda: bool(adapter.edits), timeout=2, what="skipped progress edit")
    finally:
        task.cancel()
        await asyncio.wait_for(task, 2)

    assert len(adapter.visible) == 2
    assert "safe-one" in adapter.visible[0]
    assert "safe-one" not in adapter.visible[1]
    assert "safe-two" in adapter.visible[1]


@pytest.mark.asyncio
async def test_skipped_terminal_edit_does_not_discard_definitely_unseen_tool_entry():
    class Adapter(Capture):
        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.edits.append(content)
            return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})

    _, turn, _, adapter = rail(adapter=Adapter())
    st = turn._progress_edit_state(adapter)
    assert await turn._deliver_full_progress_entry(st, "safe-one") is True
    assert await turn._deliver_full_progress_entry(st, "safe-two") is False
    assert await turn._deliver_lifecycle_progress(st, ("__terminal__", "ordinary final", asyncio.get_running_loop().create_future())) is False
    assert "safe-one" in adapter.visible["p1"]
    assert "safe-two" not in adapter.visible["p1"]
    assert sum("safe-two" in text for text in adapter.visible.values()) == 1
    assert len(adapter.visible) == 2


@pytest.mark.asyncio
async def test_two_skipped_edits_and_skipped_final_edit_publish_each_unseen_entry_once():
    class Adapter(Capture):
        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.edits.append(content)
            return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})

    _, turn, _, adapter = rail(adapter=Adapter())
    st = turn._progress_edit_state(adapter)
    assert await turn._deliver_full_progress_entry(st, "safe-one") is True
    assert await turn._deliver_full_progress_entry(st, "safe-two") is False
    assert await turn._deliver_full_progress_entry(st, "safe-three") is False
    await turn._finalize_full_progress(st)
    visible = list(adapter.visible.values())
    assert len(adapter.edits) == 3
    assert len(visible) == 3
    for entry in ("safe-one", "safe-two", "safe-three"):
        assert sum(entry in bubble for bubble in visible) == 1


@pytest.mark.asyncio
async def test_skipped_terminal_edit_does_not_continue_after_run_becomes_stale():
    current = [True]

    class Adapter(Capture):
        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.edits.append(content)
            if "ordinary final" in content:
                current[0] = False
            return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})

    _, turn, ctx, adapter = rail(adapter=Adapter())
    ctx._run_still_current = lambda: current[0]
    st = turn._progress_edit_state(adapter)
    assert await turn._deliver_full_progress_entry(st, "safe-one") is True
    assert await turn._deliver_full_progress_entry(st, "safe-two") is False
    assert await turn._deliver_lifecycle_progress(st, ("__terminal__", "ordinary final", asyncio.get_running_loop().create_future())) is False
    assert list(adapter.visible.values()) == ["safe-one"]


@pytest.mark.asyncio
async def test_retryable_then_skipped_edit_never_replays_delivery_uncertain_entry():
    class Adapter(MetadataEditProgressCaptureAdapter):
        def __init__(self):
            super().__init__()
            self.visible = []

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            result = await super().send(chat_id, content, reply_to, metadata)
            self.visible.append(content)
            return result

        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.edits.append(content)
            if len(self.edits) == 1:
                return SendResult(success=False, message_id=message_id, retryable=True, error="timeout")
            return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})

    adapter = Adapter()
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="group"),
        _run_still_current=lambda: True,
        progress_mode="full",
        tool_progress_enabled=True,
        progress_grouping="accumulate",
        progress_queue=Queue(),
        _progress_reply_to="anchor",
        _progress_metadata={"thread_id": "topic"},
    )
    turn = TurnRunner(_make_runner(adapter), ctx)
    turn.progress_callback("tool.started", "first", "preview", {"query": "safe-one"})
    task = asyncio.create_task(turn.send_progress_messages())
    try:
        await async_wait_until(lambda: len(adapter.visible) == 1, timeout=2, what="initial progress send")
        turn.progress_callback("tool.started", "second", "preview", {"query": "safe-two"})
        await async_wait_until(lambda: bool(adapter.edits), timeout=2, what="retryable progress edit")
    finally:
        task.cancel()
        await asyncio.wait_for(task, 2)

    assert len(adapter.edits) == 2
    assert len(adapter.visible) == 1
    assert "safe-one" in adapter.visible[0]


@pytest.mark.asyncio
async def test_skipped_then_retryable_combined_edit_revokes_safe_continuation():
    class Adapter(MetadataEditProgressCaptureAdapter):
        def __init__(self):
            super().__init__()
            self.visible = []

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            result = await super().send(chat_id, content, reply_to, metadata)
            self.visible.append(content)
            return result

        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.edits.append(content)
            if len(self.edits) == 1:
                return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})
            if len(self.edits) == 2:
                return SendResult(success=False, message_id=message_id, retryable=True, error="timeout")
            return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})

    adapter = Adapter()
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="group"),
        _run_still_current=lambda: True,
        progress_mode="full",
        tool_progress_enabled=True,
        progress_grouping="accumulate",
        progress_queue=Queue(),
        _progress_reply_to="anchor",
        _progress_metadata={"thread_id": "topic"},
    )
    turn = TurnRunner(_make_runner(adapter), ctx)
    turn.progress_callback("tool.started", "first", "preview", {"query": "safe-one"})
    task = asyncio.create_task(turn.send_progress_messages())
    try:
        await async_wait_until(lambda: len(adapter.visible) == 1, timeout=2, what="initial progress send")
        turn.progress_callback("tool.started", "second", "preview", {"query": "safe-two"})
        await async_wait_until(lambda: len(adapter.edits) == 1, timeout=2, what="skipped progress edit")
        turn.progress_callback("tool.started", "third", "preview", {"query": "safe-three"})
        await async_wait_until(lambda: len(adapter.edits) == 2, timeout=2, what="retryable combined edit")
    finally:
        task.cancel()
        await asyncio.wait_for(task, 2)

    assert len(adapter.edits) == 3
    assert "safe-two" in adapter.edits[0]  # skipped A
    assert "safe-two" in adapter.edits[1] and "safe-three" in adapter.edits[1]  # uncertain A+B
    assert "safe-two" in adapter.edits[2] and "safe-three" in adapter.edits[2]  # skipped final edit
    assert len(adapter.visible) == 1  # no continuation-send of A
    assert "safe-one" in adapter.visible[0]
