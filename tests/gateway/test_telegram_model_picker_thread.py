"""Regression tests for `/model` picker thread-id handling on Telegram.

Background
----------
`TelegramAdapter.send_model_picker` builds an inline-keyboard picker for the
`/model` slash command. Until this fix, it computed `message_thread_id` with::

    message_thread_id=int(thread_id) if thread_id else None

bypassing `_message_thread_id_for_send`, which is the helper every other send
path in the file routes through. The helper maps thread_id "1" (the forum
General topic synthetic id) to `None` because Telegram's `sendMessage` rejects
`message_thread_id=1` with `BadRequest: Message thread not found`.

Symptom: in any forum-group General topic (the default thread for many users),
`/model` silently fails with::

    [Telegram] send_model_picker failed: Message thread not found

…and the gateway falls through to the plain-text model list — instead of the
inline-keyboard picker users expect.

The bug was reintroduced after every `hermes update` because the local
post-update patches were never landed upstream. These tests guard against
future regressions.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import SendResult


# ── Fake telegram.error hierarchy (mirrors python-telegram-bot) ──────────


class FakeNetworkError(Exception):
    pass


class FakeBadRequest(FakeNetworkError):
    pass


class FakeTimedOut(FakeNetworkError):
    pass


# Build a fake telegram module tree so the adapter's internal imports work
_fake_telegram = types.ModuleType("telegram")
_fake_telegram.Update = object
_fake_telegram.Bot = object
_fake_telegram.Message = object


class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data


class _FakeInlineKeyboardMarkup:
    def __init__(self, rows):
        self.rows = rows


_fake_telegram.InlineKeyboardButton = _FakeInlineKeyboardButton
_fake_telegram.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup
_fake_telegram_error = types.ModuleType("telegram.error")
_fake_telegram_error.NetworkError = FakeNetworkError
_fake_telegram_error.BadRequest = FakeBadRequest
_fake_telegram_error.TimedOut = FakeTimedOut
_fake_telegram.error = _fake_telegram_error
_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(
    MARKDOWN="Markdown", MARKDOWN_V2="MarkdownV2"
)
_fake_telegram_constants.ChatType = SimpleNamespace(
    GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel"
)
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.Application = object
_fake_telegram_ext.CommandHandler = object
_fake_telegram_ext.CallbackQueryHandler = object
_fake_telegram_ext.MessageHandler = object
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
    from gateway.platforms.telegram import TelegramAdapter

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
    adapter._model_picker_state = {}
    adapter.platform = Platform.TELEGRAM
    return adapter


_PROVIDERS_FIXTURE = [
    {
        "slug": "anthropic",
        "name": "Anthropic",
        "models": ["claude-opus-4.7"],
        "total_models": 1,
        "is_current": True,
    },
    {
        "slug": "openai",
        "name": "OpenAI",
        "models": ["gpt-5"],
        "total_models": 1,
        "is_current": False,
    },
]


@pytest.mark.asyncio
async def test_model_picker_omits_general_topic_thread_id():
    """Picker in forum General topic must NOT pass message_thread_id=1.

    This is the regression: prior to the fix, the picker passed
    `message_thread_id=int("1")=1` which Telegram rejects with
    BadRequest('Message thread not found'), causing the picker to fail and
    the gateway to fall through to plain-text output.
    """
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result: SendResult = await adapter.send_model_picker(
        chat_id="-1003972169801",
        providers=_PROVIDERS_FIXTURE,
        current_model="claude-opus-4.7",
        current_provider="anthropic",
        session_key="session-1",
        on_model_selected=lambda *_a, **_k: None,
        metadata={"thread_id": "1"},  # forum General-topic synthetic id
    )

    assert result.success is True
    assert result.message_id == "42"
    assert len(call_log) == 1
    assert call_log[0]["chat_id"] == -1003972169801
    # Critical assertion: General-topic id "1" must collapse to None.
    assert call_log[0]["message_thread_id"] is None


@pytest.mark.asyncio
async def test_model_picker_passes_real_thread_id_through():
    """Real (non-General) topic ids must still be forwarded to Telegram."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send_model_picker(
        chat_id="-100123",
        providers=_PROVIDERS_FIXTURE,
        current_model="claude-opus-4.7",
        current_provider="anthropic",
        session_key="session-1",
        on_model_selected=lambda *_a, **_k: None,
        metadata={"thread_id": "17585"},
    )

    assert result.success is True
    assert call_log[0]["message_thread_id"] == 17585


@pytest.mark.asyncio
async def test_model_picker_retries_without_thread_on_thread_not_found():
    """A real thread that disappears mid-conversation should retry without it.

    Mirrors the retry contract of `send()` so the picker still reaches the
    user when a topic is deleted/closed between cache and send.
    """
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        if kwargs.get("message_thread_id") is not None:
            raise FakeBadRequest("Message thread not found")
        return SimpleNamespace(message_id=99)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send_model_picker(
        chat_id="-100123",
        providers=_PROVIDERS_FIXTURE,
        current_model="claude-opus-4.7",
        current_provider="anthropic",
        session_key="session-1",
        on_model_selected=lambda *_a, **_k: None,
        metadata={"thread_id": "99999"},
    )

    assert result.success is True
    assert result.message_id == "99"
    assert len(call_log) == 2
    assert call_log[0]["message_thread_id"] == 99999
    assert call_log[1]["message_thread_id"] is None


@pytest.mark.asyncio
async def test_model_picker_no_metadata_sends_without_thread():
    """Picker with no metadata behaves like a plain DM/group send."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=7)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send_model_picker(
        chat_id="123",
        providers=_PROVIDERS_FIXTURE,
        current_model="claude-opus-4.7",
        current_provider="anthropic",
        session_key="session-1",
        on_model_selected=lambda *_a, **_k: None,
        metadata=None,
    )

    assert result.success is True
    assert call_log[0]["message_thread_id"] is None
