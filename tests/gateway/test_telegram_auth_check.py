"""Tests for Telegram adapter early authorization check.

Verifies that unauthorized users are blocked before any text batching,
event building, or response generation occurs.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


def _make_adapter(allow_from=None, allowed_chats=None, group_allowed_chats=None, callback_auth=None, **extra_overrides):
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
    except ModuleNotFoundError:  # PR branch before Telegram plugin extraction
        from gateway.platforms.telegram import TelegramAdapter

    extra = {}
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if allowed_chats is not None:
        extra["allowed_chats"] = allowed_chats
    if group_allowed_chats is not None:
        extra["group_allowed_chats"] = group_allowed_chats
    extra.update(extra_overrides)

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    if callback_auth is not None:
        adapter._is_callback_user_authorized = callback_auth
    return adapter


def _make_message(text="hello", *, from_user_id=111, chat_id=-100, chat_type="group"):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="Test", is_forum=False),
        from_user=SimpleNamespace(id=from_user_id, full_name="Test User", first_name="Test"),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )


@pytest.mark.asyncio
async def test_unauthorized_user_blocked_before_event_building():
    """Unauthorized user's message should be blocked before _build_message_event."""
    adapter = _make_adapter(allow_from=["222"])  # Only user 222 allowed

    build_called = False
    original_build = adapter._build_message_event

    def track_build(*a, **kw):
        nonlocal build_called
        build_called = True
        return original_build(*a, **kw)

    adapter._build_message_event = track_build

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(from_user_id=111),  # User 111 NOT in allow_from
        effective_message=None,
    )

    await adapter._handle_text_message(update, SimpleNamespace())

    assert build_called is False, "build_message_event should not be called for unauthorized user"


@pytest.mark.asyncio
async def test_authorized_user_processed_normally():
    """Authorized user's message should pass the auth check and build an event."""
    adapter = _make_adapter(allow_from=["111"])

    build_called = False
    original_build = adapter._build_message_event

    def track_build(*a, **kw):
        nonlocal build_called
        build_called = True
        return original_build(*a, **kw)

    adapter._build_message_event = track_build

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(from_user_id=111),
        effective_message=None,
    )

    await adapter._handle_text_message(update, SimpleNamespace())

    assert build_called is True, "build_message_event should be called for authorized user"


@pytest.mark.asyncio
async def test_channel_post_passes_auth():
    """Messages with no from_user (channel posts) should pass user-level auth."""
    adapter = _make_adapter(allow_from=["111"])

    build_called = False
    original_build = adapter._build_message_event

    def track_build(*a, **kw):
        nonlocal build_called
        build_called = True
        return original_build(*a, **kw)

    adapter._build_message_event = track_build

    msg = _make_message()
    msg.from_user = None  # Channel post has no sender

    update = SimpleNamespace(
        update_id=1,
        message=msg,
        effective_message=None,
    )

    await adapter._handle_text_message(update, SimpleNamespace())

    assert build_called is True, "Channel posts should pass user-level auth"


@pytest.mark.asyncio
async def test_command_from_unauthorized_user_blocked():
    """Commands from unauthorized users should be blocked."""
    adapter = _make_adapter(allow_from=["222"])
    adapter.handle_message = AsyncMock()

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(text="/start", from_user_id=111),
        effective_message=None,
    )

    await adapter._handle_command(update, SimpleNamespace())

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_command_from_authorized_user_processed():
    """Commands from authorized users should be processed."""
    adapter = _make_adapter(allow_from=["111"])
    adapter.handle_message = AsyncMock()

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(text="/start", from_user_id=111),
        effective_message=None,
    )

    await adapter._handle_command(update, SimpleNamespace())

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_location_from_unauthorized_user_blocked():
    """Location messages from unauthorized users should be blocked."""
    adapter = _make_adapter(allow_from=["222"])

    msg = _make_message(from_user_id=111)
    msg.text = None
    msg.location = SimpleNamespace(latitude=53.3498, longitude=-6.2603)

    update = SimpleNamespace(
        update_id=1,
        message=msg,
        effective_message=None,
    )

    # Should not raise — just silently return
    await adapter._handle_location_message(update, SimpleNamespace())


def test_is_user_authorized_from_message_allow_from():
    """_is_user_authorized_from_message should respect adapter-level allow_from."""
    adapter = _make_adapter(allow_from=["111", "222"])

    msg = _make_message(from_user_id=111)
    assert adapter._is_user_authorized_from_message(msg) is True

    msg = _make_message(from_user_id=333)
    assert adapter._is_user_authorized_from_message(msg) is False


def test_is_user_authorized_from_message_allow_from_scalar_int():
    """allow_from can be a YAML scalar when set via `hermes config set`."""
    adapter = _make_adapter(allow_from=111)

    msg = _make_message(from_user_id=111)
    assert adapter._is_user_authorized_from_message(msg) is True

    msg = _make_message(from_user_id=222)
    assert adapter._is_user_authorized_from_message(msg) is False


def test_is_user_authorized_from_message_allow_from_csv_string():
    """allow_from should accept comma-separated config/env style values."""
    adapter = _make_adapter(allow_from="111,222")

    msg = _make_message(from_user_id=222)
    assert adapter._is_user_authorized_from_message(msg) is True

    msg = _make_message(from_user_id=333)
    assert adapter._is_user_authorized_from_message(msg) is False


def test_is_user_authorized_from_message_wildcard():
    """_is_user_authorized_from_message should accept wildcard '*'."""
    adapter = _make_adapter(allow_from=["*"])

    msg = _make_message(from_user_id=999)
    assert adapter._is_user_authorized_from_message(msg) is True


def test_is_user_authorized_from_message_no_from_user():
    """_is_user_authorized_from_message should return True for messages without from_user."""
    adapter = _make_adapter(allow_from=["111"])

    msg = _make_message()
    msg.from_user = None
    assert adapter._is_user_authorized_from_message(msg) is True


def test_is_user_authorized_from_message_callback():
    """_is_user_authorized_from_message should use _is_callback_user_authorized."""
    adapter = _make_adapter(callback_auth=lambda uid, **_kw: uid == "555")

    msg = _make_message(from_user_id=555)
    assert adapter._is_user_authorized_from_message(msg) is True

    msg = _make_message(from_user_id=666)
    assert adapter._is_user_authorized_from_message(msg) is False


def test_unknown_dm_with_no_allowlist_passes_to_pairing(monkeypatch):
    """Unknown DMs must still reach the gateway pairing flow when no allowlist exists."""
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = _make_adapter()
    msg = _make_message(from_user_id=111, chat_id=111, chat_type="private")

    assert adapter._is_user_authorized_from_message(msg) is True


def test_runner_auth_gets_group_user_allowlist_context(monkeypatch):
    """Group user allowlists need a group-shaped source, not a DM-shaped one."""
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "111")
    seen_sources = []

    class Runner:
        def _is_user_authorized(self, source):
            seen_sources.append(source)
            return source.chat_type == "group" and source.chat_id == "-100" and source.user_id == "111"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=111, chat_id=-100, chat_type="group")

    assert adapter._is_user_authorized_from_message(msg) is True
    assert seen_sources
    assert seen_sources[0].chat_type == "group"
    assert seen_sources[0].chat_id == "-100"


def test_runner_auth_gets_group_chat_allowlist_context(monkeypatch):
    """Group chat allowlists need the real chat id before intake drops updates."""
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-222")
    seen_sources = []

    class Runner:
        def _is_user_authorized(self, source):
            seen_sources.append(source)
            return source.chat_type == "group" and source.chat_id == "-222"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=111, chat_id=-222, chat_type="group")

    assert adapter._is_user_authorized_from_message(msg) is True
    assert seen_sources
    assert seen_sources[0].chat_type == "group"
    assert seen_sources[0].chat_id == "-222"


def test_removed_dm_user_blocked_before_pairing_when_allowlist_exists(monkeypatch):
    """A user removed from TELEGRAM_ALLOWED_USERS should be blocked at intake."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "222")
    adapter = _make_adapter()
    msg = _make_message(from_user_id=111, chat_id=111, chat_type="private")

    assert adapter._is_user_authorized_from_message(msg) is False


@pytest.mark.asyncio
async def test_media_from_removed_user_blocked_before_event_building(monkeypatch):
    """Removed users must not inject prompt-bearing documents via media handlers."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "222")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()

    build_called = False

    def track_build(*_args, **_kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("media handler built an event for an unauthorized user")

    adapter._build_message_event = track_build
    document = SimpleNamespace(
        file_name="payload.txt",
        mime_type="text/plain",
        file_size=42,
        get_file=AsyncMock(side_effect=AssertionError("unauthorized document was downloaded")),
    )
    msg = _make_message(text=None, from_user_id=111, chat_id=111, chat_type="private")
    msg.caption = "please process this caption"
    msg.document = document

    update = SimpleNamespace(update_id=1, message=msg, effective_message=None)

    await adapter._handle_media_message(update, SimpleNamespace())

    assert build_called is False
    adapter.handle_message.assert_not_awaited()
    document.get_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_unmentioned_group_text_from_removed_user_not_observed():
    """Removed users must not persist unmentioned group text into observed context."""
    adapter = _make_adapter(
        allow_from=["222"],
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
        require_mention=True,
        observe_unmentioned_group_messages=True,
    )
    observed = []
    adapter._observe_unmentioned_group_message = lambda *args, **kwargs: observed.append((args, kwargs))

    msg = _make_message(text="side chatter", from_user_id=111, chat_id=-100, chat_type="group")
    update = SimpleNamespace(update_id=1, message=msg, effective_message=None)

    await adapter._handle_text_message(update, SimpleNamespace())

    assert observed == []


@pytest.mark.asyncio
async def test_unmentioned_group_location_from_removed_user_not_observed():
    """Removed users must not persist unmentioned group locations into observed context."""
    adapter = _make_adapter(
        allow_from=["222"],
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
        require_mention=True,
        observe_unmentioned_group_messages=True,
    )
    observed = []
    adapter._observe_unmentioned_group_message = lambda *args, **kwargs: observed.append((args, kwargs))

    msg = _make_message(text=None, from_user_id=111, chat_id=-100, chat_type="group")
    msg.location = SimpleNamespace(latitude=53.3498, longitude=-6.2603)
    update = SimpleNamespace(update_id=1, message=msg, effective_message=None)

    await adapter._handle_location_message(update, SimpleNamespace())

    assert observed == []
