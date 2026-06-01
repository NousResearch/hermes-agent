"""Tests for auto-discovery of Telegram supergroup forum topic names.

The Bot API has no endpoint that enumerates existing forum topics, so the
adapter must learn topic names from ``forum_topic_created`` /
``forum_topic_edited`` service messages on incoming traffic and persist
them across restarts.  The agent's session context surfaces the resolved
name as ``Channel Topic`` so a human-readable label like "Rustbelt"
appears instead of a bare ``thread: 5`` number.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform


# ── Fake telegram module tree (mirrors test_telegram_thread_fallback.py) ──


class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


class _FakeInputMediaPhoto:
    def __init__(self, media, caption=None, **kwargs):
        self.media = media
        self.caption = caption


_fake_telegram = types.ModuleType("telegram")
_fake_telegram.Update = object
_fake_telegram.Bot = object
_fake_telegram.Message = object
_fake_telegram.InlineKeyboardButton = _FakeInlineKeyboardButton
_fake_telegram.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup
_fake_telegram.InputMediaPhoto = _FakeInputMediaPhoto

_fake_telegram_error = types.ModuleType("telegram.error")
_fake_telegram_error.NetworkError = type("NetworkError", (Exception,), {})
_fake_telegram_error.BadRequest = type("BadRequest", (Exception,), {})
_fake_telegram_error.TimedOut = type("TimedOut", (Exception,), {})
_fake_telegram.error = _fake_telegram_error

_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(
    MARKDOWN_V2="MarkdownV2", MARKDOWN="Markdown", HTML="HTML"
)
_fake_telegram_constants.ChatType = SimpleNamespace(
    GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel", PRIVATE="private"
)
_fake_telegram.constants = _fake_telegram_constants

_fake_telegram_ext = types.ModuleType("telegram.ext")
for _attr in (
    "Application",
    "CommandHandler",
    "CallbackQueryHandler",
    "MessageHandler",
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


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point get_hermes_home() at a tmp dir so the cache file is sandboxed."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def _make_adapter():
    """Construct a minimal TelegramAdapter without running __init__."""
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    adapter._group_topics_cache = {}
    adapter._group_topics_cache_path = None
    adapter._reply_to_mode = "first"
    return adapter


def _make_message(
    *,
    chat_id: int = -100123,
    thread_id: int = 5,
    text: str = "hello",
    forum_topic_created_name: str | None = None,
    forum_topic_edited_name: str | None = None,
    reply_to: object | None = None,
):
    from gateway.platforms import telegram as telegram_mod

    chat = SimpleNamespace(
        id=chat_id,
        type=telegram_mod.ChatType.SUPERGROUP,
        is_forum=True,
        title="Forum group",
    )
    msg = SimpleNamespace(
        text=text,
        caption=None,
        chat=chat,
        from_user=SimpleNamespace(id=456, full_name="Alice"),
        message_thread_id=thread_id,
        is_topic_message=True,
        reply_to_message=reply_to,
        message_id=10,
        date=None,
        forum_topic_created=(
            SimpleNamespace(name=forum_topic_created_name)
            if forum_topic_created_name
            else None
        ),
        forum_topic_edited=(
            SimpleNamespace(name=forum_topic_edited_name)
            if forum_topic_edited_name
            else None
        ),
        quote=None,
    )
    return msg


# ── Tests ─────────────────────────────────────────────────────────────────


def test_forum_topic_created_populates_cache_and_chat_topic(isolated_hermes_home):
    """A ``forum_topic_created`` service message names its own topic."""
    adapter = _make_adapter()
    message = _make_message(
        thread_id=5,
        forum_topic_created_name="Rustbelt",
    )

    event = adapter._build_message_event(
        message, msg_type=SimpleNamespace(value="text")
    )

    assert event.source.thread_id == "5"
    assert event.source.chat_topic == "Rustbelt"
    assert adapter._group_topics_cache == {"-100123": {"5": "Rustbelt"}}


def test_followup_message_resolves_topic_from_runtime_cache(isolated_hermes_home):
    """Subsequent messages in the same topic see the cached name."""
    adapter = _make_adapter()

    # Discovery message
    adapter._build_message_event(
        _make_message(thread_id=5, forum_topic_created_name="Rustbelt"),
        msg_type=SimpleNamespace(value="text"),
    )

    # Plain follow-up — no service message, just thread id
    follow_up = _make_message(thread_id=5, text="follow-up")
    event = adapter._build_message_event(
        follow_up, msg_type=SimpleNamespace(value="text")
    )

    assert event.source.chat_topic == "Rustbelt"


def test_reply_to_forum_topic_created_backfills_topic_name(isolated_hermes_home):
    """First post in a pre-existing topic carries the create event via reply_to."""
    adapter = _make_adapter()

    reply_to = SimpleNamespace(
        message_id=1,
        text=None,
        caption=None,
        forum_topic_created=SimpleNamespace(name="XCS"),
        forum_topic_edited=None,
    )
    msg = _make_message(thread_id=7, text="first post", reply_to=reply_to)

    event = adapter._build_message_event(msg, msg_type=SimpleNamespace(value="text"))

    assert event.source.chat_topic == "XCS"
    assert adapter._group_topics_cache["-100123"]["7"] == "XCS"


def test_forum_topic_edited_updates_cached_name(isolated_hermes_home):
    """Renaming a topic via Telegram clients propagates to the cache."""
    adapter = _make_adapter()
    adapter._group_topics_cache = {"-100123": {"5": "OldName"}}

    msg = _make_message(thread_id=5, forum_topic_edited_name="NewName")
    event = adapter._build_message_event(msg, msg_type=SimpleNamespace(value="text"))

    assert event.source.chat_topic == "NewName"
    assert adapter._group_topics_cache["-100123"]["5"] == "NewName"


def test_general_topic_falls_back_to_implicit_name(isolated_hermes_home):
    """thread_id=1 is Telegram's General topic — surface it as 'General'."""
    adapter = _make_adapter()
    msg = _make_message(thread_id=1, text="hi from General")

    event = adapter._build_message_event(msg, msg_type=SimpleNamespace(value="text"))

    assert event.source.thread_id == "1"
    assert event.source.chat_topic == "General"


def test_static_config_wins_over_runtime_cache(isolated_hermes_home):
    """Operator-set group_topics config overrides auto-discovered names."""
    adapter = _make_adapter()
    adapter.config.extra["group_topics"] = [
        {
            "chat_id": -100123,
            "topics": [{"thread_id": 5, "name": "Operator Override"}],
        }
    ]
    adapter._group_topics_cache = {"-100123": {"5": "Auto-Discovered"}}

    msg = _make_message(thread_id=5, text="hi")
    event = adapter._build_message_event(msg, msg_type=SimpleNamespace(value="text"))

    assert event.source.chat_topic == "Operator Override"


def test_cache_is_persisted_and_reloaded(isolated_hermes_home, tmp_path):
    """Discovered topic names survive an adapter restart via the JSON sidecar."""
    adapter = _make_adapter()
    adapter._build_message_event(
        _make_message(thread_id=5, forum_topic_created_name="Job Hunt"),
        msg_type=SimpleNamespace(value="text"),
    )

    # Discovery should have written to disk.
    cache_files = list(isolated_hermes_home.glob("telegram_group_topics.*.json"))
    assert len(cache_files) == 1
    on_disk = json.loads(cache_files[0].read_text())
    assert on_disk == {"-100123": {"5": "Job Hunt"}}

    # Simulate a restart by spinning up a fresh adapter — it should load
    # the persisted cache.
    fresh = object.__new__(type(adapter))
    fresh.config = adapter.config
    fresh._config = adapter._config
    fresh._platform = Platform.TELEGRAM
    fresh.platform = Platform.TELEGRAM
    fresh._dm_topics = {}
    fresh._dm_topics_config = []
    fresh._group_topics_cache_path = None
    fresh._group_topics_cache = {}
    fresh._load_group_topics_cache()

    assert fresh._group_topics_cache == {"-100123": {"5": "Job Hunt"}}
    assert fresh._lookup_group_topic_name("-100123", "5") == "Job Hunt"
