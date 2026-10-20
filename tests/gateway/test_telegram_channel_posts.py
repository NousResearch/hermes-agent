"""Regression tests for Telegram channel_post updates.

Telegram channel broadcasts are delivered as ``Update.channel_post`` rather than
``Update.message``.  The adapter should use ``effective_message`` so channel
posts are converted into Hermes gateway events instead of being silently
ignored.
"""

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType


def _build_telegram_stubs():
    telegram_mod = types.ModuleType("telegram")
    telegram_mod.Update = object
    telegram_mod.Bot = object
    telegram_mod.Message = object
    telegram_mod.InlineKeyboardButton = object
    telegram_mod.InlineKeyboardMarkup = object
    telegram_mod.LinkPreviewOptions = object

    telegram_ext_mod = types.ModuleType("telegram.ext")
    telegram_ext_mod.Application = object
    telegram_ext_mod.CommandHandler = object
    telegram_ext_mod.CallbackQueryHandler = object
    telegram_ext_mod.InlineQueryHandler = object
    telegram_ext_mod.MessageHandler = object
    telegram_ext_mod.ContextTypes = SimpleNamespace(DEFAULT_TYPE=type(None))
    telegram_ext_mod.filters = SimpleNamespace()

    telegram_constants_mod = types.ModuleType("telegram.constants")
    telegram_constants_mod.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    telegram_constants_mod.ChatType = SimpleNamespace(
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
        PRIVATE="private",
    )

    telegram_request_mod = types.ModuleType("telegram.request")
    telegram_request_mod.HTTPXRequest = object

    telegram_mod.ext = telegram_ext_mod
    telegram_mod.constants = telegram_constants_mod
    telegram_mod.request = telegram_request_mod

    return {
        "telegram": telegram_mod,
        "telegram.ext": telegram_ext_mod,
        "telegram.constants": telegram_constants_mod,
        "telegram.request": telegram_request_mod,
    }


@pytest.fixture
def telegram_adapter_cls(monkeypatch):
    """Import TelegramAdapter without leaking temporary telegram stubs."""
    module_name = "plugins.platforms.telegram.adapter"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        yield existing_module.TelegramAdapter
        return

    telegram_pkg = sys.modules.get("telegram")
    installed = isinstance(getattr(telegram_pkg, "__file__", None), str)
    if telegram_pkg is None:
        try:
            installed = importlib.util.find_spec("telegram") is not None
        except ValueError:
            installed = False

    if not installed:
        for name, module in _build_telegram_stubs().items():
            monkeypatch.setitem(sys.modules, name, module)

    module = importlib.import_module(module_name)
    try:
        yield module.TelegramAdapter
    finally:
        if not installed:
            sys.modules.pop(module_name, None)


def _make_adapter(telegram_adapter_cls):
    a = telegram_adapter_cls(PlatformConfig(enabled=True, token="***", extra={}))
    # Channel posts have from_user=None.  After PR #28494's fail-closed
    # auth, the empty-allowlist adapter rejects all messages including
    # channel posts.  These tests focus on routing, not auth gating.
    a._is_callback_user_authorized = lambda user_id, **_kw: True
    return a


def _make_channel_message(text="channel id test @hermes_bot"):
    chat = SimpleNamespace(
        id=-1003950368353,
        type="channel",
        title="wzrd",
        full_name=None,
        is_forum=False,
    )
    return SimpleNamespace(
        chat=chat,
        from_user=None,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=11,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
    )


def _make_channel_update(msg):
    return SimpleNamespace(
        update_id=12345,
        message=None,
        channel_post=msg,
        effective_message=msg,
    )


def test_build_message_event_uses_channel_identity_for_channel_posts(telegram_adapter_cls):
    adapter = _make_adapter(telegram_adapter_cls)
    msg = _make_channel_message()

    event = adapter._build_message_event(msg, MessageType.TEXT, update_id=12345)

    assert event.source.chat_type == "channel"
    assert event.source.chat_id == "-1003950368353"
    # Channel posts often have no from_user.  Preserve an identity so the
    # gateway authorization layer can allowlist the channel by numeric ID.
    assert event.source.user_id == "-1003950368353"
    assert event.source.user_name == "wzrd"
    assert event.platform_update_id == 12345


def _make_channel_photo_message(caption="look at this @hermes_bot"):
    """A channel post carrying a photo instead of text."""
    msg = _make_channel_message(text=None)
    file_obj = SimpleNamespace(
        file_path="photos/file_1.jpg",
        download_as_bytearray=AsyncMock(return_value=bytearray(b"\xff\xd8\xff\xd9")),
    )
    photo_size = SimpleNamespace(
        file_id="AgACphoto123",
        file_unique_id="uniq123",
        width=1280,
        height=720,
        file_size=2048,
        get_file=AsyncMock(return_value=file_obj),
    )
    msg.photo = [photo_size]
    msg.caption = caption
    msg.sticker = None
    msg.document = None
    msg.video = None
    msg.audio = None
    msg.voice = None
    msg.media_group_id = None
    return msg


@pytest.mark.asyncio
async def test_handle_media_message_processes_channel_post_photo(telegram_adapter_cls, monkeypatch):
    """A channel-post photo must reach the photo routing path, not be dropped.

    ``_handle_media_message`` previously read ``update.message`` directly, so a
    channel post (delivered as ``update.channel_post``) hit the ``if not msg:
    return`` guard and was consumed before any event was built, even though the
    text handlers already resolved ``effective_message``.
    """
    adapter = _make_adapter(telegram_adapter_cls)
    msg = _make_channel_photo_message()
    update = _make_channel_update(msg)

    # Let the payload through auth/trigger gating; this test is about whether
    # the media handler resolves the channel-post payload at all.
    adapter._is_user_authorized_from_message = lambda _m: True
    adapter._should_process_message = lambda _m: True
    monkeypatch.setattr(
        sys.modules["plugins.platforms.telegram.adapter"],
        "cache_image_from_bytes_async",
        AsyncMock(return_value="/tmp/cached-photo.jpg"),
    )
    routed = AsyncMock()
    adapter._route_photo_event = routed

    await adapter._handle_media_message(update, None)

    assert routed.await_count == 1, "channel-post photo was dropped before photo routing"
    event = routed.await_args.args[1]
    assert event.message_type == MessageType.PHOTO
    assert event.source.chat_id == "-1003950368353"
    assert event.platform_update_id == 12345
    assert "look at this" in (event.text or "")


@pytest.mark.asyncio
async def test_handle_media_message_still_ignores_empty_update(telegram_adapter_cls):
    """No message-like payload at all must remain a no-op, not an exception."""
    adapter = _make_adapter(telegram_adapter_cls)
    routed = AsyncMock()
    adapter._route_photo_event = routed
    adapter.handle_message = AsyncMock()
    empty = SimpleNamespace(update_id=1, message=None, channel_post=None, effective_message=None)

    await adapter._handle_media_message(empty, None)

    assert routed.await_count == 0
    assert adapter.handle_message.await_count == 0
