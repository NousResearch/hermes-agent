from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.whatsapp import WhatsAppAdapter
from gateway.session import Platform, SessionSource
from gateway.platforms.base import MessageEvent, MessageType


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def text(self):
        return ""


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()


@pytest.mark.asyncio
async def test_whatsapp_processing_start_reacts_with_customer_service_emoji():
    adapter = WhatsAppAdapter(PlatformConfig(extra={"reaction_emoji": "👩🏻"}))
    adapter._running = True
    fake_session = _FakeSession()
    adapter._http_session = fake_session
    adapter._bridge_port = 3000

    event = MessageEvent(
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="584128034216@s.whatsapp.net",
            user_id="584128034216@s.whatsapp.net",
            chat_type="dm",
        ),
        message_id="ABC123",
        text="Hola",
        message_type=MessageType.TEXT,
    )

    await adapter.on_processing_start(event)

    assert fake_session.calls == [
        {
            "url": "http://127.0.0.1:3000/react",
            "json": {
                "chatId": "584128034216@s.whatsapp.net",
                "messageId": "ABC123",
                "emoji": "👩🏻",
                "senderId": "584128034216@s.whatsapp.net",
            },
            "timeout": fake_session.calls[0]["timeout"],
        }
    ]


def test_whatsapp_reactions_default_enabled():
    adapter = WhatsAppAdapter(PlatformConfig(extra={}))

    assert adapter._reactions_enabled() is True
    assert adapter._reaction_emoji() == "👩🏻"
