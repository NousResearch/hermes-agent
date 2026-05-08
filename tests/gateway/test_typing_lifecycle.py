"""Regression tests for platform typing lifecycle around final delivery."""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.telegram import TelegramAdapter
from gateway.session import SessionSource, build_session_key


class _OneShotTypingAdapter(BasePlatformAdapter):
    """Adapter that behaves like Telegram for typing lifecycle tests."""

    TYPING_CLEARED_BY_OUTBOUND_MESSAGE = True

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.events: list[str] = []

    async def connect(self):
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.events.append("send")
        return SendResult(success=True, message_id="m1")

    async def send_typing(self, chat_id, metadata=None):
        self.events.append("typing")

    async def stop_typing(self, chat_id):
        self.events.append("stop")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def _keep_typing(self, chat_id, interval=2.0, metadata=None, stop_event=None):
        # Tight interval makes the race deterministic.  Before the lifecycle
        # fix, this loop could tick again after the final send while
        # on_processing_complete was still running.
        await super()._keep_typing(
            chat_id,
            interval=0.01,
            metadata=metadata,
            stop_event=stop_event,
        )

    async def on_processing_complete(self, event, outcome):
        self.events.append("complete-start")
        await asyncio.sleep(0.05)
        self.events.append("complete-end")


def _make_event(text="hi", chat_id="42"):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="inbound-1",
    )


@pytest.mark.asyncio
async def test_one_shot_typing_refresh_stops_before_final_response():
    adapter = _OneShotTypingAdapter()
    event = _make_event()
    session_key = build_session_key(event.source)

    async def handler(_event):
        # Let the typing loop emit at least one tick while work is active.
        await asyncio.sleep(0.03)
        return "done"

    adapter._message_handler = handler

    await adapter._process_message_background(event, session_key)

    assert "send" in adapter.events
    send_index = adapter.events.index("send")
    assert "typing" in adapter.events[:send_index], adapter.events
    assert "typing" not in adapter.events[send_index + 1 :], adapter.events


@pytest.mark.asyncio
async def test_base_platform_keeps_default_post_response_typing_policy():
    assert BasePlatformAdapter.TYPING_CLEARED_BY_OUTBOUND_MESSAGE is False


def test_telegram_stops_typing_refresh_before_response_delivery():
    assert TelegramAdapter.TYPING_CLEARED_BY_OUTBOUND_MESSAGE is True
