"""Baileys standalone and live-adapter delivery of outbound mentions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.whatsapp.adapter import _standalone_send
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter
from tests.tools.test_whatsapp_send_message_media import _resp, _session_with


def test_standalone_sender_posts_mentions_to_bridge():
    session_ctx, calls = _session_with([_resp(200, {"messageId": "m1"})])
    pconfig = SimpleNamespace(token="", extra={"bridge_port": 3000})

    with patch("aiohttp.ClientSession", return_value=session_ctx):
        result = asyncio.run(_standalone_send(
            pconfig,
            "120363408391911677@g.us",
            "hello group",
            mentions=["15551234567@s.whatsapp.net", "149606612619433@lid"],
        ))

    assert result["success"] is True
    assert calls == [("http://localhost:3000/send", {
        "chatId": "120363408391911677@g.us",
        "message": "hello group",
        "mentions": ["15551234567@s.whatsapp.net", "149606612619433@lid"],
    })]


@pytest.mark.asyncio
async def test_live_adapter_forwards_metadata_mentions_to_bridge():
    adapter = _make_adapter()
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"messageId": "m1"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(response))

    result = await adapter.send(
        "120363408391911677@g.us",
        "hello group",
        metadata={"mentions": ["15551234567@s.whatsapp.net", "149606612619433@lid"]},
    )

    assert result.success
    payload = adapter._http_session.post.call_args.kwargs["json"]
    assert payload == {
        "chatId": "120363408391911677@g.us",
        "message": "hello group",
        "mentions": ["15551234567@s.whatsapp.net", "149606612619433@lid"],
    }


@pytest.mark.asyncio
async def test_live_adapter_no_mentions_preserves_bridge_payload_shape():
    adapter = _make_adapter()
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"messageId": "m1"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(response))

    result = await adapter.send("120363408391911677@g.us", "hello group")

    assert result.success
    assert adapter._http_session.post.call_args.kwargs["json"] == {
        "chatId": "120363408391911677@g.us",
        "message": "hello group",
    }
