"""Tests for WhatsApp Baileys outbound media reply_to propagation (#80064).

The Cloud adapter already forwards reply_to; this covers the Baileys adapter's
outbound HTTP contract with the bridge, without requiring a WhatsApp account.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _resp(json_data=None, status=200):
    r = AsyncMock()
    r.status = status
    r.json = AsyncMock(return_value=json_data or {"messageId": "m1"})
    r.text = AsyncMock(return_value="")
    return r


def _session_with():
    calls = []

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_resp())
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    return session, calls


def _make_adapter():
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True))
    adapter._running = True
    adapter._bridge_port = 3000
    adapter._check_managed_bridge_exit = AsyncMock(return_value=False)
    return adapter


@pytest.fixture
def tmp_media():
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(b"x")
    f.close()
    try:
        yield f.name
    finally:
        os.unlink(f.name)


@pytest.mark.parametrize(
    "method,media_type",
    [("send_image", "image"), ("send_image_file", "image"),
     ("send_video", "video"), ("send_voice", "audio"),
     ("send_document", "document")],
)
@pytest.mark.parametrize("reply_to", ["msg-123", None, ""])
def test_media_send_reply_contract(tmp_media, method, media_type, reply_to):
    adapter = _make_adapter()
    session, calls = _session_with()
    adapter._http_session = session

    media_path = "https://example.com/media.png" if method == "send_image" else tmp_media
    with patch("plugins.platforms.whatsapp.adapter.cache_image_from_url", new=AsyncMock(return_value=tmp_media)):
        result = asyncio.run(getattr(adapter, method)("12345", media_path, caption="caption", reply_to=reply_to))

    assert result.success is True
    assert result.message_id == "m1"
    assert len(calls) == 1
    url, payload = calls[0]
    assert url.endswith("/send-media")
    assert payload["chatId"] == "12345@s.whatsapp.net"
    assert payload["filePath"] == tmp_media
    assert payload["mediaType"] == media_type
    assert payload["caption"] == "caption"
    if method == "send_document":
        assert payload["fileName"] == os.path.basename(tmp_media)
    if reply_to:
        assert payload["replyTo"] == reply_to
    else:
        assert "replyTo" not in payload
