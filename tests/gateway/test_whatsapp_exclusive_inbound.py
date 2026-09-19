from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


class Response:
    status = 200

    async def __aenter__(self): return self
    async def __aexit__(self, *_args): return None
    async def json(self): return [{"messageId": "m1"}]


def adapter():
    value = object.__new__(WhatsAppAdapter)
    value._running = True
    value._http_session = object()
    value._report_bridge_exit = AsyncMock(return_value=False)
    value._bridge_req = MagicMock(return_value=Response())
    value._send_read_receipt = AsyncMock()
    value._enqueue_text_event = MagicMock()
    value.handle_message = AsyncMock()
    return value


def event(native_type=""):
    return MessageEvent(text="hello", message_type=MessageType.TEXT, source=SessionSource(
        platform=Platform.WHATSAPP, chat_id="claimed", chat_type="group", user_id="sender"),
        message_id="m1", metadata={"whatsapp_native_type": native_type} if native_type else {})


@pytest.mark.asyncio
async def test_claim_precedes_whatsapp_text_debounce_and_normal_dispatch():
    value = adapter()
    value._build_message_event = AsyncMock(return_value=event())

    async def claim(_event):
        value._running = False
        return True

    value.dispatch_exclusive_inbound = claim
    await value._poll_messages()
    value._enqueue_text_event.assert_not_called()
    value.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native_type", ["pollUpdateMessage", "pollCreationMessageV3", "reactionMessage"])
async def test_native_poll_and_reaction_envelopes_bypass_claim(native_type):
    value = adapter()

    async def build(_data):
        value._running = False
        return event(native_type)

    value._build_message_event = build
    value.dispatch_exclusive_inbound = AsyncMock(return_value=True)
    await value._poll_messages()
    value.dispatch_exclusive_inbound.assert_not_awaited()
    value._enqueue_text_event.assert_called_once()
