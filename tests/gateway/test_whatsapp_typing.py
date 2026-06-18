"""Tests for WhatsApp typing presence lifecycle."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.whatsapp import WhatsAppAdapter


class _AsyncCM:
    def __init__(self, value=None):
        self.value = value or MagicMock()

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, *, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return _AsyncCM()


def _make_adapter():
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = Path("/tmp/test-wa-session")
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._running = True
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    session = _FakeSession()
    adapter._http_session = session
    return adapter, session


@pytest.mark.asyncio
async def test_send_typing_posts_composing_state(monkeypatch):
    adapter, session = _make_adapter()
    monkeypatch.setattr(adapter, "_check_managed_bridge_exit", AsyncMock(return_value=False))

    await adapter.send_typing("chat-123")

    assert session.posts == [
        {
            "url": "http://127.0.0.1:19876/typing",
            "json": {"chatId": "chat-123", "state": "composing"},
            "timeout": session.posts[0]["timeout"],
        }
    ]


@pytest.mark.asyncio
async def test_stop_typing_posts_paused_state(monkeypatch):
    adapter, session = _make_adapter()
    monkeypatch.setattr(adapter, "_check_managed_bridge_exit", AsyncMock(return_value=False))

    await adapter.stop_typing("chat-123")

    assert session.posts == [
        {
            "url": "http://127.0.0.1:19876/typing",
            "json": {"chatId": "chat-123", "state": "paused"},
            "timeout": session.posts[0]["timeout"],
        }
    ]
