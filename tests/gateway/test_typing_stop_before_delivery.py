"""Regression tests for typing indicator shutdown around final delivery."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _TypingOrderAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.DISCORD)
        self.events: list[str] = []
        self._stop_seen = asyncio.Event()

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def stop_typing(self, chat_id: str) -> None:
        self.events.append("stop_typing")
        self._stop_seen.set()

    async def get_chat_info(self, chat_id):
        return {}


def _make_event(text="hi", chat_id="42"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id=chat_id, chat_type="dm"),
    )


def _session_key(chat_id="42"):
    return build_session_key(
        SessionSource(platform=Platform.DISCORD, chat_id=chat_id, chat_type="dm")
    )


@pytest.mark.asyncio
async def test_typing_refresh_stops_before_final_response_delivery():
    """Final delivery must not race with a still-running typing refresh.

    Discord has no explicit "typing stop" endpoint.  A typing event sent just
    before the final answer can keep "bot is typing" visible for Discord's TTL
    after the answer appears.  The base gateway therefore stops the refresh task
    before it calls the platform send path, not only in the finally cleanup.
    """
    adapter = _TypingOrderAdapter()

    async def handler(event):
        adapter.events.append("handler_done")
        return "final answer"

    async def send_with_retry(*args, **kwargs):
        adapter.events.append("send")
        assert adapter._stop_seen.is_set(), (
            "typing refresh was still active when final response delivery began"
        )
        return None

    adapter._message_handler = handler
    adapter._send_with_retry = AsyncMock(side_effect=send_with_retry)

    await adapter.handle_message(_make_event())

    for _ in range(50):
        if _session_key() not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)

    await adapter.cancel_background_tasks()

    assert "send" in adapter.events
    assert adapter.events.index("stop_typing") < adapter.events.index("send")
