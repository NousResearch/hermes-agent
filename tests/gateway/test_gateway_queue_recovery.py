"""Regression tests for restored gateway FIFO recursion behavior."""

from unittest.mock import MagicMock

from gateway.platforms.base import MessageEvent, MessageType, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.platforms.base import BasePlatformAdapter


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.SLACK)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=MagicMock(platform=Platform.SLACK),
        message_id=text,
    )


def test_depth_cap_does_not_stage_another_fifo_item():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    adapter = _Adapter()
    session_key = "slack:T1:D1:thread"
    runner._enqueue_fifo(session_key, _event("first"), adapter)
    runner._enqueue_fifo(session_key, _event("second"), adapter)

    pending = adapter._pending_messages[session_key]
    returned = runner._promote_queued_event(
        session_key, adapter, pending, stage_next=False
    )

    assert returned is pending
    assert adapter._pending_messages[session_key] is pending
    assert [event.text for event in runner._queued_events[session_key]] == ["second"]
