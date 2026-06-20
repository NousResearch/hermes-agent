from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return {"messageId": "MSG1"}

    async def text(self):
        return ""


class _FakeChatResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return {
            "name": "Reto de movilidad Garita Otay a SF",
            "isGroup": True,
            "participants": ["20130595618881@lid", "63045875281992@lid"],
        }


class _FakeSession:
    def __init__(self):
        self.posts = []
        self.gets = []

    def post(self, url, *, json, timeout):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    def get(self, url, *, timeout):
        self.gets.append({"url": url, "timeout": timeout})
        return _FakeChatResponse()


async def _no_bridge_exit():
    return None


@pytest.mark.asyncio
async def test_whatsapp_send_forwards_mentions_metadata_on_first_chunk():
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._running = True
    adapter._bridge_port = 3000
    fake_session = _FakeSession()
    adapter._http_session = fake_session
    adapter._check_managed_bridge_exit = _no_bridge_exit
    adapter._outgoing_chunk_limit = lambda: 4096

    result = await adapter.send(
        "120363406770604689@g.us",
        "Falta confirmar fecha y hora con @20130595618881.",
        metadata={"mentions": ["20130595618881@lid"]},
    )

    assert result.success is True
    assert fake_session.posts[0]["json"] == {
        "chatId": "120363406770604689@g.us",
        "message": "Falta confirmar fecha y hora con @20130595618881.",
        "mentions": ["20130595618881@lid"],
    }


@pytest.mark.asyncio
async def test_whatsapp_send_auto_resolves_visible_group_mentions():
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._running = True
    adapter._bridge_port = 3000
    fake_session = _FakeSession()
    adapter._http_session = fake_session
    adapter._check_managed_bridge_exit = _no_bridge_exit
    adapter._outgoing_chunk_limit = lambda: 4096

    result = await adapter.send(
        "120363406770604689@g.us",
        "Falta confirmar fecha y hora con @20130595618881.",
    )

    assert result.success is True
    assert fake_session.gets
    assert fake_session.posts[0]["json"] == {
        "chatId": "120363406770604689@g.us",
        "message": "Falta confirmar fecha y hora con @20130595618881.",
        "mentions": ["20130595618881@lid"],
    }


@pytest.mark.asyncio
async def test_whatsapp_send_omits_empty_mentions():
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._running = True
    adapter._bridge_port = 3000
    fake_session = _FakeSession()
    adapter._http_session = fake_session
    adapter._check_managed_bridge_exit = _no_bridge_exit
    adapter._outgoing_chunk_limit = lambda: 4096

    result = await adapter.send(
        "120363406770604689@g.us",
        "Plain message",
        metadata={"mentions": ["", "   "]},
    )

    assert result.success is True
    assert "mentions" not in fake_session.posts[0]["json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("blankish", [" ", "\u200b", "\u200b \n\u2060", "[SILENT]"])
async def test_whatsapp_send_suppresses_non_visible_content(blankish):
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._running = True
    adapter._bridge_port = 3000
    fake_session = _FakeSession()
    adapter._http_session = fake_session
    adapter._check_managed_bridge_exit = _no_bridge_exit
    adapter._outgoing_chunk_limit = lambda: 4096

    result = await adapter.send("120363406770604689@g.us", blankish)

    assert result.success is True
    assert result.message_id is None
    assert fake_session.posts == []
