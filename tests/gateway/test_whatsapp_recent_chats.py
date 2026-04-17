"""Tests for WhatsApp recent-chat discovery helpers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.whatsapp import WhatsAppAdapter


class _AsyncCM:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter():
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter._bridge_port = 3000
    adapter._http_session = MagicMock()
    adapter._bridge_process = None
    adapter._running = True
    return adapter


@pytest.mark.asyncio
async def test_list_recent_chats_returns_bridge_payload():
    adapter = _make_adapter()
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value=[
        {"chatId": "34600111222@s.whatsapp.net", "name": "Movistar", "isGroup": False, "lastMessageTimestamp": 1700000000},
        {"chatId": "12345@g.us", "name": "Family", "isGroup": True, "lastMessageTimestamp": 1690000000},
    ])
    adapter._http_session.get.return_value = _AsyncCM(response)

    chats = await adapter.list_recent_chats(limit=10)

    assert chats[0]["chatId"] == "34600111222@s.whatsapp.net"
    call = adapter._http_session.get.call_args
    assert call.kwargs["params"] == {"limit": 10}


@pytest.mark.asyncio
async def test_list_recent_chats_returns_empty_on_non_200():
    adapter = _make_adapter()
    response = MagicMock(status=503)
    response.json = AsyncMock(return_value={"error": "nope"})
    adapter._http_session.get.return_value = _AsyncCM(response)

    chats = await adapter.list_recent_chats()

    assert chats == []


@pytest.mark.asyncio
async def test_list_recent_chats_filters_non_dict_entries():
    adapter = _make_adapter()
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value=[None, {"name": "missing id"}, {"chatId": "12345@g.us", "name": "Family"}])
    adapter._http_session.get.return_value = _AsyncCM(response)

    chats = await adapter.list_recent_chats()

    assert chats == [{"chatId": "12345@g.us", "name": "Family"}]
