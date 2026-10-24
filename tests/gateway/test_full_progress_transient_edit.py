"""Full-progress edits preserve a consumed, redacted tool entry across transient failures."""

import asyncio
from queue import Queue
from gateway.session import SessionSource
from gateway.config import Platform

import pytest

from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
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
