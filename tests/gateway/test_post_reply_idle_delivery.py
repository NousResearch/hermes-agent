"""The idle deadline starts after a delivered reply, not after model completion."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class Adapter(BasePlatformAdapter):
    def __init__(self, delivered=True):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.SIGNAL)
        self.delivered = delivered

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=self.delivered, message_id="reply" if self.delivered else None)

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


async def _hold_typing(chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()


@pytest.mark.asyncio
async def test_unauthorized_queued_input_does_not_cancel_idle_job():
    adapter = Adapter()
    source = SessionSource(platform=Platform.SIGNAL, chat_id="group-1", chat_type="group")
    event = MessageEvent(text="untrusted", message_type=MessageType.TEXT, source=source)
    key = build_session_key(source)
    adapter._active_sessions[key] = asyncio.Event()
    async def forbidden(*args):
        raise AssertionError("unauthorized input must not invalidate")
    adapter.gateway_runner = SimpleNamespace(_is_user_authorized=lambda source: False,
                                             _invalidate_post_reply_idle_for_turn=forbidden)
    await adapter._handle_message_while_active(event, key)


@pytest.mark.asyncio
async def test_queued_followup_invalidates_before_runner_drains():
    adapter = Adapter()
    source = SessionSource(platform=Platform.SIGNAL, chat_id="group-1", chat_type="group")
    event = MessageEvent(text="follow up", message_type=MessageType.TEXT, source=source, message_id="later")
    key = build_session_key(source)
    adapter._active_sessions[key] = asyncio.Event()
    calls = []
    async def invalidate(event, source, key):
        calls.append(key)
    adapter.gateway_runner = SimpleNamespace(_is_user_authorized=lambda source: True,
                                             _invalidate_post_reply_idle_for_turn=invalidate)
    await adapter._handle_message_while_active(event, key)
    assert calls == [key]
    assert event._gateway_accepted


@pytest.mark.asyncio
async def test_streamed_reply_arms_without_duplicate_send():
    adapter = Adapter()
    adapter._keep_typing = _hold_typing
    calls = []

    async def arm(event, key):
        calls.append(key)

    adapter.gateway_runner = SimpleNamespace(_arm_post_reply_idle=arm, _clear_durable_active_turn=lambda e: asyncio.sleep(0))

    async def handler(event):
        event._post_reply_session_id = "session-1"
        event._post_reply_streamed = True
        return None

    adapter.set_message_handler(handler)
    event = MessageEvent(text="question", message_type=MessageType.TEXT,
                         source=SessionSource(platform=Platform.SIGNAL, chat_id="group-1", chat_type="group"),
                         message_id="inbound")
    key = build_session_key(event.source)
    await adapter._process_message_background(event, key)
    assert calls == [key]


@pytest.mark.asyncio
@pytest.mark.parametrize("delivered", [True, False])
async def test_only_delivered_reply_arms_idle_deadline(delivered):
    adapter = Adapter(delivered=delivered)
    adapter._keep_typing = _hold_typing
    calls = []

    async def arm(event, key):
        calls.append((event._post_reply_session_id, key))

    adapter.gateway_runner = SimpleNamespace(_arm_post_reply_idle=arm, _clear_durable_active_turn=lambda e: asyncio.sleep(0))

    async def handler(event):
        event._post_reply_session_id = "session-1"
        assert calls == []
        return "done"

    adapter.set_message_handler(handler)
    event = MessageEvent(text="question", message_type=MessageType.TEXT,
                         source=SessionSource(platform=Platform.SIGNAL, chat_id="group-1", chat_type="group"),
                         message_id="inbound")
    key = build_session_key(event.source)
    await adapter._process_message_background(event, key)
    assert calls == ([("session-1", key)] if delivered else [])
