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


# ---------------------------------------------------------------------------
# #42400: bot-echo classification — _build_message_event returns None
# ---------------------------------------------------------------------------


@pytest.fixture
def classify_adapter(monkeypatch):
    """Adapter wired for testing the bot-echo classification."""
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    return a


def test_build_message_event_skips_bot_echo(classify_adapter):
    """_build_message_event should return None for bot-echo business messages.

    When the bot sends via business_connection, Telegram relays the message
    back with sender_business_bot set. The gateway must not re-process its
    own output (#42400).
    """
    msg = MagicMock()
    msg.business_connection_id = "conn_echo1"
    msg.sender_business_bot = MagicMock(id=8712662056)
    msg.message_id = 123
    msg.chat = MagicMock(id=456, type="private", full_name="Test")
    msg.from_user = MagicMock(id=999, full_name="Client", is_bot=False)

    result = classify_adapter._build_message_event(msg, MagicMock())
    assert result is None, "Bot-echo business messages must return None"


def test_build_message_event_keeps_customer_inbound(classify_adapter):
    """_build_message_event should process customer inbound business messages."""
    msg = MagicMock()
    msg.business_connection_id = "conn_inbound1"
    msg.sender_business_bot = None  # not a bot echo
    msg.message_id = 124
    msg.message_thread_id = None
    msg.chat = MagicMock(id=456, type="private", full_name="Test")
    msg.chat.message_thread_id = None
    msg.from_user = MagicMock(id=999, full_name="Client", is_bot=False)
    msg.reply_to_message = None
    msg.quote = None

    result = classify_adapter._build_message_event(msg, MagicMock())
    assert result is not None, "Customer inbound business messages must be processed"


# ---------------------------------------------------------------------------
# Review follow-up: SessionSource persistence round-trip
# ---------------------------------------------------------------------------


def test_session_source_serialization_round_trips_business_connection_id():
    """Restored sources must keep send-as-owner routing across persistence."""
    from gateway.session import SessionSource
    from gateway.platforms.base import Platform

    src = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        business_connection_id="conn_persist1",
    )
    data = src.to_dict()
    assert data["business_connection_id"] == "conn_persist1"
    restored = SessionSource.from_dict(data)
    assert restored.business_connection_id == "conn_persist1"


def test_session_source_serialization_omits_field_when_absent():
    from gateway.session import SessionSource
    from gateway.platforms.base import Platform

    src = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
    data = src.to_dict()
    assert "business_connection_id" not in data
    assert SessionSource.from_dict(data).business_connection_id is None


# ---------------------------------------------------------------------------
# Review follow-up (#42400): owner/bot/customer classification
# ---------------------------------------------------------------------------


def _biz_msg(conn="conn_c1", from_id=999, chat_id=999, sender_business_bot=None):
    msg = MagicMock()
    msg.business_connection_id = conn
    msg.sender_business_bot = sender_business_bot
    msg.from_user = SimpleNamespace(id=from_id) if from_id is not None else None
    msg.chat = SimpleNamespace(id=chat_id, type="private")
    return msg


class TestBusinessMessageClassification:
    def test_non_business_returns_none(self, classify_adapter):
        assert classify_adapter._classify_business_message(_biz_msg(conn=None)) is None

    def test_bot_echo(self, classify_adapter):
        m = _biz_msg(sender_business_bot=SimpleNamespace(id=42))
        assert classify_adapter._classify_business_message(m) == "bot_echo"

    def test_customer_inbound(self, classify_adapter):
        # In a private business chat the peer's from_user.id equals chat.id.
        m = _biz_msg(from_id=999, chat_id=999)
        assert classify_adapter._classify_business_message(m) == "customer"

    def test_owner_via_connection_state(self, classify_adapter):
        classify_adapter._business_connections["conn_c1"] = {
            "can_reply": True,
            "user_chat_id": 111,
        }
        m = _biz_msg(from_id=111, chat_id=999)
        assert classify_adapter._classify_business_message(m) == "owner_outgoing"

    def test_owner_via_structural_fallback_after_restart(self, classify_adapter):
        # No tracked connection (e.g. gateway restarted; Telegram does not
        # replay BusinessConnection updates): the owner is still detected
        # because their from_user.id differs from the chat's peer id.
        m = _biz_msg(from_id=111, chat_id=999)
        assert classify_adapter._classify_business_message(m) == "owner_outgoing"


# ---------------------------------------------------------------------------
# Review follow-up (#42400): handler wiring — owner observed, not enqueued
# ---------------------------------------------------------------------------


@pytest.fixture
def intake_adapter(monkeypatch):
    """Adapter wired for testing handler intake classification."""
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    a._ensure_forum_commands = AsyncMock()
    a._cache_replied_media = AsyncMock()
    a._cache_observed_media = AsyncMock()
    a._clean_bot_trigger_text = lambda t: t
    a._enqueue_text_event = MagicMock()
    a._observe_business_owner_message = MagicMock()
    a.handle_message = AsyncMock()
    return a


def _update_for(msg, update_id=7):
    """A business update: update.message is None, effective_message is set."""
    return SimpleNamespace(update_id=update_id, message=None, effective_message=msg)


def _full_biz_text_msg(text="hello", from_id=456, chat_id=456, conn="conn_t1"):
    msg = MagicMock()
    msg.text = text
    msg.business_connection_id = conn
    msg.sender_business_bot = None
    msg.message_id = 321
    msg.message_thread_id = None
    msg.chat = MagicMock(id=chat_id, type="private", full_name="Client Chat")
    msg.chat.message_thread_id = None
    msg.from_user = MagicMock(id=from_id, full_name="Someone", is_bot=False)
    msg.reply_to_message = None
    msg.quote = None
    return msg


@pytest.mark.asyncio
async def test_text_handler_enqueues_customer_message(intake_adapter):
    msg = _full_biz_text_msg(from_id=456, chat_id=456)
    await intake_adapter._handle_text_message(_update_for(msg), MagicMock())
    intake_adapter._enqueue_text_event.assert_called_once()
    intake_adapter._observe_business_owner_message.assert_not_called()


@pytest.mark.asyncio
async def test_text_handler_observes_owner_message_without_enqueue(intake_adapter):
    from gateway.platforms.base import MessageType

    msg = _full_biz_text_msg(from_id=111, chat_id=456)
    await intake_adapter._handle_text_message(_update_for(msg), MagicMock())
    intake_adapter._observe_business_owner_message.assert_called_once()
    assert intake_adapter._observe_business_owner_message.call_args.args[1] == MessageType.TEXT
    intake_adapter._enqueue_text_event.assert_not_called()


@pytest.mark.asyncio
async def test_text_handler_skips_bot_echo_entirely(intake_adapter):
    msg = _full_biz_text_msg(from_id=111, chat_id=456)
    msg.sender_business_bot = MagicMock(id=8712662056)
    await intake_adapter._handle_text_message(_update_for(msg), MagicMock())
    intake_adapter._enqueue_text_event.assert_not_called()
    intake_adapter._observe_business_owner_message.assert_not_called()


@pytest.mark.asyncio
async def test_command_from_business_customer_routed_as_text(intake_adapter):
    """A customer must never drive operator slash commands: the business
    connection pre-authorizes the conversation, not operator authority."""
    msg = _full_biz_text_msg(text="/restart", from_id=456, chat_id=456)
    await intake_adapter._handle_command(_update_for(msg), MagicMock())
    intake_adapter.handle_message.assert_not_called()
    intake_adapter._enqueue_text_event.assert_called_once()


@pytest.mark.asyncio
async def test_command_from_owner_observed_not_executed(intake_adapter):
    msg = _full_biz_text_msg(text="/note done", from_id=111, chat_id=456)
    await intake_adapter._handle_command(_update_for(msg), MagicMock())
    intake_adapter.handle_message.assert_not_called()
    intake_adapter._enqueue_text_event.assert_not_called()
    intake_adapter._observe_business_owner_message.assert_called_once()


# ---------------------------------------------------------------------------
# Review follow-up: media handler must use the effective-message path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_handler_reaches_business_media(intake_adapter):
    """Business media arrives as update.business_message (update.message is
    None); reading update.message directly dropped it before any handling."""
    msg = MagicMock()
    msg.business_connection_id = "conn_m1"
    msg.sender_business_bot = None
    auth = MagicMock(return_value=False)
    intake_adapter._is_user_authorized_from_message = auth

    await intake_adapter._handle_media_message(_update_for(msg), MagicMock())

    auth.assert_called_once_with(msg)


@pytest.mark.asyncio
async def test_media_handler_observes_owner_media(intake_adapter):
    from gateway.platforms.base import MessageType

    msg = _full_biz_text_msg(from_id=111, chat_id=456)
    msg.caption = None
    intake_adapter._media_message_type = MagicMock(return_value=MessageType.PHOTO)
    stub_event = MagicMock(message_type=MessageType.PHOTO)
    intake_adapter._build_message_event = MagicMock(return_value=stub_event)

    await intake_adapter._handle_media_message(_update_for(msg), MagicMock())

    intake_adapter._cache_observed_media.assert_awaited_once_with(msg, stub_event)
    intake_adapter._observe_business_owner_message.assert_called_once()
    intake_adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_media_handler_skips_bot_echo(intake_adapter):
    msg = _full_biz_text_msg(from_id=111, chat_id=456)
    msg.sender_business_bot = MagicMock(id=1)
    intake_adapter._build_message_event = MagicMock()

    await intake_adapter._handle_media_message(_update_for(msg), MagicMock())

    intake_adapter._build_message_event.assert_not_called()
    intake_adapter.handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# Review follow-up: owner observation appends to the customer-chat session
# ---------------------------------------------------------------------------


def test_observe_business_owner_message_appends_to_chat_session(monkeypatch):
    from gateway.platforms.base import MessageType

    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    store = MagicMock()
    store.get_or_create_session.return_value = SimpleNamespace(session_id="sess-1")
    a._session_store = store

    event = MagicMock()
    event.source = SimpleNamespace(user_name="Olga Owner", user_id="111")
    event.text = "handled it, no need to reply"
    event.message_id = 777

    a._observe_business_owner_message(MagicMock(), MessageType.TEXT, event=event)

    store.get_or_create_session.assert_called_once_with(event.source)
    store.append_to_transcript.assert_called_once()
    sid, entry = store.append_to_transcript.call_args.args
    assert sid == "sess-1"
    assert entry["role"] == "user"
    assert entry["observed"] is True
    assert entry["message_id"] == "777"
    assert "Olga Owner" in entry["content"]
    assert "handled it, no need to reply" in entry["content"]


def test_observe_business_owner_message_without_store_is_noop(monkeypatch):
    from gateway.platforms.base import MessageType

    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    # No _session_store set: must not raise.
    a._observe_business_owner_message(MagicMock(), MessageType.TEXT, event=MagicMock())


# ---------------------------------------------------------------------------
# Review follow-up: media/draft send paths carry the connection ID
# ---------------------------------------------------------------------------


def test_business_kwargs_helper(monkeypatch):
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    assert a._business_kwargs({"business_connection_id": "c9"}) == {
        "business_connection_id": "c9"
    }
    assert a._business_kwargs({}) == {}
    assert a._business_kwargs(None) == {}


@pytest.mark.asyncio
async def test_send_document_carries_business_connection_id(send_adapter, tmp_path):
    doc = tmp_path / "invoice.pdf"
    doc.write_bytes(b"%PDF-1.4 test")
    send_adapter._send_with_dm_topic_reply_anchor_retry = AsyncMock(
        return_value=SimpleNamespace(message_id=5)
    )
    send_adapter._reply_to_message_id_for_send = lambda *a, **k: None

    await send_adapter.send_document(
        "123456", str(doc), metadata={"business_connection_id": "conn_doc1"}
    )

    kwargs_dict = send_adapter._send_with_dm_topic_reply_anchor_retry.call_args.args[1]
    assert kwargs_dict["business_connection_id"] == "conn_doc1"


def test_draft_streaming_disabled_for_business_sends(monkeypatch):
    """Drafts cannot target a business chat; business replies must take the
    plain send path (which carries business_connection_id end to end)."""
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    a._bot = MagicMock()  # MagicMock exposes send_message_draft

    assert a.supports_draft_streaming("dm", metadata={}) is True
    assert (
        a.supports_draft_streaming(
            "dm", metadata={"business_connection_id": "conn_d1"}
        )
        is False
    )
