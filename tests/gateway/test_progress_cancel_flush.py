"""Progress lines held back by the edit throttle must be flushed when the turn ends.

``send_progress_messages`` idles in ``except queue.Empty: await asyncio.sleep(0.3)``.
A cancellation landing in that sleep — the loop's idle state, so the common case —
is raised from inside the ``queue.Empty`` handler, where the sibling
``except asyncio.CancelledError`` clause cannot catch it. The end-of-turn drain and
final edit were then skipped, so tool lines that arrived within the 1.5 s edit
throttle of the previous edit were never shown.
"""
import asyncio
import queue

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent, self.edits = [], []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content, **kwargs) -> SendResult:
        self.edits.append(content)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_runner(adapter, ctx):
    from gateway.run_turn_runner import TurnRunner

    class _StubGatewayRunner:
        adapters = {adapter.platform: adapter}

        def _delivery_adapter_for(self, source):
            return adapter

    return TurnRunner(_StubGatewayRunner(), ctx)


@pytest.mark.asyncio
async def test_throttled_line_is_flushed_when_cancelled_while_idle():
    adapter = _CaptureAdapter()
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm"),
        progress_queue=queue.Queue(),
        progress_grouping="accumulate",
        tool_progress_enabled=True,
        _run_still_current=lambda: True,
    )
    task = asyncio.create_task(_make_runner(adapter, ctx).send_progress_messages())

    ctx.progress_queue.put("⚙️ first tool")
    for _ in range(50):                       # wait for the bubble to be sent
        if adapter.sent:
            break
        await asyncio.sleep(0.05)
    assert adapter.sent == ["⚙️ first tool"]

    ctx.progress_queue.put("⚙️ second tool")  # lands inside the 1.5 s edit throttle
    await asyncio.sleep(2.2)                  # throttle elapses; loop now idles on queue.Empty

    task.cancel()                             # the turn ends
    await asyncio.gather(task, return_exceptions=True)

    shown = (adapter.sent + adapter.edits)[-1]
    assert "⚙️ second tool" in shown, (adapter.sent, adapter.edits)
