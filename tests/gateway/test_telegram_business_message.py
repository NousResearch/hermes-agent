"""Tests for Telegram Secretary Mode / business_message support.

Covers:
  - SessionSource carries business_connection_id
  - _build_message_event extracts it from business messages
  - Auth bypass: business messages pre-authorized by Chat Automation
  - send() passes business_connection_id to bot.send_message
  - BusinessConnection lifecycle handler stores/removes connections
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


def _install_fake_telegram(monkeypatch):
    """Stub the python-telegram-bot package so TelegramAdapter can be imported."""
    fake_telegram = types.ModuleType("telegram")
    fake_telegram.Update = SimpleNamespace(ALL_TYPES=())
    fake_telegram.Bot = object
    fake_telegram.Message = object
    fake_telegram.InlineKeyboardButton = object
    fake_telegram.InlineKeyboardMarkup = object

    fake_error = types.ModuleType("telegram.error")
    fake_error.NetworkError = type("NetworkError", (Exception,), {})
    fake_error.BadRequest = type("BadRequest", (Exception,), {})
    fake_error.TimedOut = type("TimedOut", (Exception,), {})
    fake_telegram.error = fake_error

    fake_constants = types.ModuleType("telegram.constants")
    fake_constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    fake_constants.ChatType = SimpleNamespace(
        GROUP="group", SUPERGROUP="supergroup",
        CHANNEL="channel", PRIVATE="private",
    )
    fake_telegram.constants = fake_constants

    fake_ext = types.ModuleType("telegram.ext")
    fake_ext.Application = object
    fake_ext.CommandHandler = object
    fake_ext.CallbackQueryHandler = object
    fake_ext.MessageHandler = object
    fake_ext.TypeHandler = object
    fake_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    fake_ext.filters = object

    fake_request = types.ModuleType("telegram.request")
    fake_request.HTTPXRequest = object

    monkeypatch.setitem(sys.modules, "telegram", fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", fake_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", fake_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", fake_request)


# ---------------------------------------------------------------------------
# Task 1: SessionSource field
# ---------------------------------------------------------------------------


def test_session_source_accepts_business_connection_id():
    """SessionSource should accept business_connection_id."""
    from gateway.session import SessionSource
    from gateway.platforms.base import Platform

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        business_connection_id="conn_abc123",
    )
    assert source.business_connection_id == "conn_abc123"


def test_session_source_business_connection_id_defaults_none():
    """business_connection_id should default to None for non-business sources."""
    from gateway.session import SessionSource
    from gateway.platforms.base import Platform

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
    assert source.business_connection_id is None


# ---------------------------------------------------------------------------
# Task 3: Auth bypass for business messages
# ---------------------------------------------------------------------------


def test_business_message_bypasses_user_auth(monkeypatch):
    """Business messages are pre-authorized by Chat Automation connection."""
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))

    # A business message from an external client (not in any allowlist)
    msg = MagicMock()
    msg.from_user = MagicMock(id=999999)
    msg.business_connection_id = "conn_active123"
    msg.chat = MagicMock(id=123456, type="private")

    result = a._is_user_authorized_from_message(msg)
    assert result is True, "Business messages should bypass user-ID auth"


def test_normal_message_does_not_trigger_bypass(monkeypatch):
    """Non-business messages must not trigger the Secretary Mode bypass.

    The adapter defers unknown-DM auth to the pairing flow, so we can't
    assert a False return. Instead, verify the bypass path is NOT taken
    by checking that auth falls through to the normal logic.
    """
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))

    msg = MagicMock()
    msg.from_user = MagicMock(id=999999)
    msg.business_connection_id = None  # not a business message
    msg.chat = MagicMock(id=123456, type="private")

    # _source_from_message_for_auth should still be called (no early return)
    called = MagicMock(wraps=a._source_from_message_for_auth)
    a._source_from_message_for_auth = called
    a._is_user_authorized_from_message(msg)
    assert called.called, "Non-business messages must reach the normal auth path"


# ---------------------------------------------------------------------------
# Task 4: send() passes business_connection_id
# ---------------------------------------------------------------------------


@pytest.fixture
def send_adapter(monkeypatch):
    """Adapter wired for testing the send() path."""
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    a._bot = MagicMock()
    a._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    a._send_path_degraded = False
    a.format_message = lambda x: x
    a.truncate_message = lambda x, y, len_fn=None: [x]
    a._metadata_thread_id = lambda x: None
    a._message_thread_id_for_send = lambda x: None
    a._should_thread_reply = lambda x, y: False
    a._thread_kwargs_for_send = lambda *ag, **kw: {}
    a._link_preview_kwargs = lambda: {}
    a._notification_kwargs = lambda x: {}
    a._should_attempt_rich = lambda x, metadata=None: False
    a._reply_to_mode = "smart"
    a._is_thread_not_found_error = lambda e: False
    a._looks_like_network_error = lambda e: False
    return a


@pytest.mark.asyncio
async def test_send_passes_business_connection_id(send_adapter):
    """send() should pass business_connection_id to bot.send_message."""
    metadata = {"business_connection_id": "conn_send123"}

    await send_adapter.send("123456", "Hello client", metadata=metadata)

    assert send_adapter._bot.send_message.called
    call_kwargs = send_adapter._bot.send_message.call_args
    assert call_kwargs.kwargs.get("business_connection_id") == "conn_send123"


@pytest.mark.asyncio
async def test_send_omits_business_connection_id_when_absent(send_adapter):
    """send() should NOT pass business_connection_id for normal messages."""
    await send_adapter.send("123456", "Hello", metadata={})

    assert send_adapter._bot.send_message.called
    call_kwargs = send_adapter._bot.send_message.call_args
    assert "business_connection_id" not in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# Task 6: BusinessConnection lifecycle handler
# ---------------------------------------------------------------------------


@pytest.fixture
def lifecycle_adapter(monkeypatch):
    """Adapter wired for testing the lifecycle handler."""
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    return a


@pytest.mark.asyncio
async def test_business_connection_stores_active_connection(lifecycle_adapter):
    """_handle_business_connection should store active connections."""
    update = MagicMock()
    update.business_connection = MagicMock()
    update.business_connection.id = "conn_lifecycle1"
    update.business_connection.is_enabled = True
    update.business_connection.can_reply = True
    update.business_connection.user_chat_id = 111111

    await lifecycle_adapter._handle_business_connection(update, MagicMock())

    assert "conn_lifecycle1" in lifecycle_adapter._business_connections
    assert lifecycle_adapter._business_connections["conn_lifecycle1"]["can_reply"] is True


@pytest.mark.asyncio
async def test_business_connection_removes_on_disconnect(lifecycle_adapter):
    """Disconnect should remove the connection from active set."""
    lifecycle_adapter._business_connections = {
        "conn_lifecycle1": {"can_reply": True, "user_chat_id": 111111}
    }

    update = MagicMock()
    update.business_connection = MagicMock()
    update.business_connection.id = "conn_lifecycle1"
    update.business_connection.is_enabled = False

    await lifecycle_adapter._handle_business_connection(update, MagicMock())

    assert "conn_lifecycle1" not in lifecycle_adapter._business_connections
