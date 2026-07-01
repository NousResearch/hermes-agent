"""Tests for Telegram username (non-numeric) chat_id handling (#13206).

When ``TELEGRAM_HOME_CHANNEL`` is an ``@username`` rather than a numeric chat
ID, webhook/cron deliveries that fall back to the home channel used to crash
with ``ValueError: invalid literal for int()`` because the adapter coerced
every chat_id with ``int()``. Telegram's Bot API accepts both forms, so the
adapter now normalizes instead of force-casting.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform
from plugins.platforms.telegram.telegram_ids import (
    looks_like_telegram_username,
    normalize_telegram_chat_id,
    parse_telegram_username_target,
    telegram_chat_id_key,
)


# ---------------------------------------------------------------------------
# Helper-level behavior (no telegram import needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("123456789", 123456789),             # positive numeric DM id
        ("-1001234567890", -1001234567890),   # negative channel/supergroup id
        (123456789, 123456789),               # already int
        ("  42 ", 42),                        # surrounding whitespace
        ("@some_user", "@some_user"),         # username passes through as str
        ("@a_channel", "@a_channel"),
        ("not_numeric", "not_numeric"),       # any other non-numeric string
    ],
)
def test_normalize_returns_int_or_passthrough_string(value, expected):
    assert normalize_telegram_chat_id(value) == expected


def test_normalize_never_raises_on_username():
    # A bare int() here would raise ValueError; normalize must not.
    assert normalize_telegram_chat_id("@some_user") == "@some_user"


def test_numeric_normalizes_to_int_type():
    assert isinstance(normalize_telegram_chat_id("123"), int)


def test_username_normalizes_to_str_type():
    assert isinstance(normalize_telegram_chat_id("@some_user"), str)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("@some_user", True),
        ("@a_chan", True),
        ("@abcd", True),       # 4-char minimum
        ("@abc", False),       # too short
        ("123456", False),     # numeric
        ("-100123", False),
        ("@with space", False),
        ("plain", False),
    ],
)
def test_looks_like_username(value, expected):
    assert looks_like_telegram_username(value) is expected


def test_parse_username_target():
    assert parse_telegram_username_target("@some_user") == "@some_user"
    assert parse_telegram_username_target("  @some_user  ") == "@some_user"
    assert parse_telegram_username_target("123456") is None
    assert parse_telegram_username_target("-1001234567890") is None


def test_chat_id_key_is_stable_string():
    assert telegram_chat_id_key("123") == "123"
    assert telegram_chat_id_key(123) == "123"
    assert telegram_chat_id_key("@some_user") == "@some_user"


# ---------------------------------------------------------------------------
# Fake telegram module tree (mirrors test_telegram_thread_fallback.py)
# ---------------------------------------------------------------------------

class FakeNetworkError(Exception):
    pass


class FakeBadRequest(FakeNetworkError):
    pass


class FakeTimedOut(FakeNetworkError):
    pass


class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


_fake_telegram = types.ModuleType("telegram")
_fake_telegram.Update = object
_fake_telegram.Bot = object
_fake_telegram.Message = object
_fake_telegram.InlineKeyboardButton = _FakeInlineKeyboardButton
_fake_telegram.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup
_fake_telegram.InputMediaPhoto = object
_fake_telegram_error = types.ModuleType("telegram.error")
_fake_telegram_error.NetworkError = FakeNetworkError
_fake_telegram_error.BadRequest = FakeBadRequest
_fake_telegram_error.TimedOut = FakeTimedOut
_fake_telegram.error = _fake_telegram_error
_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(
    MARKDOWN_V2="MarkdownV2", MARKDOWN="Markdown", HTML="HTML",
)
_fake_telegram_constants.ChatType = SimpleNamespace(
    GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel", PRIVATE="private",
)
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram_ext = types.ModuleType("telegram.ext")
for _attr in (
    "Application", "CommandHandler", "CallbackQueryHandler",
    "MessageHandler", "TypeHandler",
):
    setattr(_fake_telegram_ext, _attr, object)
_fake_telegram_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
_fake_telegram_ext.filters = object
_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = object


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    monkeypatch.setitem(sys.modules, "telegram", _fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", _fake_telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", _fake_telegram_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", _fake_telegram_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", _fake_telegram_request)


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    adapter._reply_to_mode = "first"
    adapter._fallback_ips = []
    adapter._polling_conflict_count = 0
    adapter._polling_network_error_count = 0
    adapter._polling_error_callback_ref = None
    adapter._pending_guest_queries = {}
    adapter._guest_only_chats = set()
    adapter._guest_reply_buffer = {}
    adapter._guest_inline_message_ids = {}
    adapter._guest_turn_media = {}
    adapter._guest_file_id_cache = {}
    adapter._seen_guest_update_ids = set()
    adapter._last_guest_update_id = 0
    adapter.platform = Platform.TELEGRAM
    return adapter


# ---------------------------------------------------------------------------
# Adapter send path: username chat_id reaches the Bot API without int() crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_passes_username_chat_id_through_unchanged():
    """adapter.send(@username) calls the Bot API with the username string
    rather than crashing on int() coercion (the #13206 regression)."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=99)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(chat_id="@some_user", content="hello world")

    assert result.success is True
    assert call_log, "send_message was never called"
    assert call_log[0]["chat_id"] == "@some_user"


@pytest.mark.asyncio
async def test_send_passes_numeric_chat_id_as_int():
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=1)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(chat_id="123456789", content="hi")

    assert result.success is True
    assert call_log[0]["chat_id"] == 123456789
    assert isinstance(call_log[0]["chat_id"], int)
