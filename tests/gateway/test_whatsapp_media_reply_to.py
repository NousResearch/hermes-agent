"""Tests for WhatsApp Baileys outbound media reply_to propagation (#80064).

The Cloud adapter already forwards reply_to; this covers the Baileys bridge
Python adapter and the Node.js bridge /send-media endpoint.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
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
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True)
    adapter._running = True
    adapter._bridge_port = 3000
    adapter._check_managed_bridge_exit = AsyncMock(return_value=False)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._mention_patterns = []
    adapter._free_response_chats = set()
    adapter._whatsapp_free_response_chats = lambda: set()
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


def test_send_image_file_includes_reply_to(tmp_media):
    adapter = _make_adapter()
    session, calls = _session_with()
    adapter._http_session = session

    result = asyncio.run(
        adapter.send_image_file("12345", tmp_media, reply_to="msg-123")
    )

    assert result.success is True
    assert len(calls) == 1
    url, payload = calls[0]
    assert url.endswith("/send-media")
    assert payload["replyTo"] == "msg-123"


def test_send_video_includes_reply_to(tmp_media):
    adapter = _make_adapter()
    session, calls = _session_with()
    adapter._http_session = session

    result = asyncio.run(
        adapter.send_video("12345", tmp_media, reply_to="msg-456")
    )

    assert result.success is True
    assert calls[0][1]["replyTo"] == "msg-456"


def test_send_voice_includes_reply_to(tmp_media):
    adapter = _make_adapter()
    session, calls = _session_with()
    adapter._http_session = session

    result = asyncio.run(
        adapter.send_voice("12345", tmp_media, reply_to="msg-789")
    )

    assert result.success is True
    assert calls[0][1]["replyTo"] == "msg-789"


def test_send_document_includes_reply_to(tmp_media):
    adapter = _make_adapter()
    session, calls = _session_with()
    adapter._http_session = session

    result = asyncio.run(
        adapter.send_document("12345", tmp_media, reply_to="msg-abc")
    )

    assert result.success is True
    assert calls[0][1]["replyTo"] == "msg-abc"


def test_send_image_file_omits_reply_to_when_not_given(tmp_media):
    adapter = _make_adapter()
    session, calls = _session_with()
    adapter._http_session = session

    result = asyncio.run(adapter.send_image_file("12345", tmp_media))

    assert result.success is True
    assert "replyTo" not in calls[0][1]
