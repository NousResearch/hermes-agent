"""#68502: Telegram inbound update_id dedup prevents duplicate agent turns.

Covers the private cache helper and handler-level guards on the four
registered message handlers so a redelivered update_id never reaches
enqueue / observe / dispatch a second time.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._seen_update_ids = {}
    adapter._seen_update_ids_lock = threading.Lock()
    adapter._seen_update_ids_max = 4096
    # Downstream side effects — assert these are NOT called on the second hop.
    adapter._enqueue_text_event = MagicMock()
    adapter.handle_message = AsyncMock()
    adapter._observe_unmentioned_group_message = MagicMock()
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._cache_observed_media = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = MagicMock(
        side_effect=lambda event: event
    )
    adapter._clean_bot_trigger_text = MagicMock(side_effect=lambda t: t)
    adapter._is_user_authorized_from_message = MagicMock(return_value=True)
    adapter._should_process_message = MagicMock(return_value=True)
    adapter._should_observe_unmentioned_group_message = MagicMock(return_value=False)
    adapter._SPLIT_THRESHOLD = 4000
    return adapter


def _msg(*, text="hello", chat_id=123, user_id=456, message_id=1, **extra):
    msg = MagicMock()
    msg.text = text
    msg.caption = extra.get("caption")
    msg.chat = MagicMock(id=chat_id, type="private")
    msg.from_user = MagicMock(id=user_id)
    msg.message_id = message_id
    msg.photo = extra.get("photo")
    msg.video = extra.get("video")
    msg.audio = extra.get("audio")
    msg.voice = extra.get("voice")
    msg.document = extra.get("document")
    msg.sticker = extra.get("sticker")
    msg.location = extra.get("location")
    msg.venue = extra.get("venue")
    msg.media_group_id = extra.get("media_group_id")
    return msg


def _update(update_id: int, msg=None, **msg_kw):
    if msg is None:
        msg = _msg(**msg_kw)
    return SimpleNamespace(
        update_id=update_id,
        message=msg,
        effective_message=msg,
        channel_post=None,
    )


def _event(text="hello"):
    return SimpleNamespace(text=text, media_urls=None, media_types=None, message_type=None)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_duplicate_update_id_is_suppressed():
    adapter = _make_adapter()
    update = _update(1001)
    assert adapter._is_duplicate_update(update) is False
    assert adapter._is_duplicate_update(update) is True
    assert adapter._is_duplicate_update(_update(1002)) is False


def test_none_update_id_is_not_duplicate():
    adapter = _make_adapter()
    update = SimpleNamespace(update_id=None)
    assert adapter._is_duplicate_update(update) is False
    assert adapter._is_duplicate_update(update) is False


def test_dedup_evicts_oldest_when_cap_exceeded():
    adapter = _make_adapter()
    adapter._seen_update_ids_max = 3
    for i in range(3):
        adapter._is_duplicate_update(_update(i))
    assert len(adapter._seen_update_ids) == 3
    adapter._is_duplicate_update(_update(3))
    assert len(adapter._seen_update_ids) == 3
    assert 0 not in adapter._seen_update_ids
    assert 3 in adapter._seen_update_ids


# ---------------------------------------------------------------------------
# Handler-level: second delivery with same update_id must not dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_text_message_suppresses_second_delivery():
    adapter = _make_adapter()
    fake_event = _event("hello")
    adapter._build_message_event = MagicMock(return_value=fake_event)
    update = _update(2001, text="hello")
    ctx = MagicMock()

    await adapter._handle_text_message(update, ctx)
    await adapter._handle_text_message(update, ctx)

    assert adapter._enqueue_text_event.call_count == 1
    adapter._build_message_event.assert_called_once()


@pytest.mark.asyncio
async def test_handle_command_suppresses_second_delivery():
    adapter = _make_adapter()
    fake_event = _event("/status")
    adapter._build_message_event = MagicMock(return_value=fake_event)
    update = _update(2002, text="/status")
    ctx = MagicMock()

    await adapter._handle_command(update, ctx)
    await adapter._handle_command(update, ctx)

    assert adapter.handle_message.await_count == 1
    adapter._build_message_event.assert_called_once()


@pytest.mark.asyncio
async def test_handle_location_message_suppresses_second_delivery():
    adapter = _make_adapter()
    location = SimpleNamespace(latitude=37.77, longitude=-122.42)
    msg = _msg(text=None, location=location)
    msg.text = None
    fake_event = _event("[location]")
    adapter._build_message_event = MagicMock(return_value=fake_event)
    adapter.handle_message = AsyncMock()
    update = _update(2003, msg=msg)
    ctx = MagicMock()

    await adapter._handle_location_message(update, ctx)
    first_builds = adapter._build_message_event.call_count

    await adapter._handle_location_message(update, ctx)
    second_builds = adapter._build_message_event.call_count

    assert first_builds == 1, "first location delivery must build an event"
    assert second_builds == 1, "second delivery must not rebuild/dispatch"


@pytest.mark.asyncio
async def test_handle_media_message_suppresses_second_delivery():
    adapter = _make_adapter()
    sticker = MagicMock()
    msg = _msg(text=None, sticker=sticker)
    msg.text = None
    msg.caption = None
    msg.photo = None
    msg.video = None
    msg.audio = None
    msg.voice = None
    msg.document = None
    fake_event = _event("[sticker]")
    adapter._build_message_event = MagicMock(return_value=fake_event)
    adapter._media_message_type = MagicMock(return_value=MagicMock(name="STICKER"))
    adapter._handle_sticker = AsyncMock()
    update = _update(2004, msg=msg)
    update.message = msg
    ctx = MagicMock()

    await adapter._handle_media_message(update, ctx)
    await adapter._handle_media_message(update, ctx)

    assert adapter.handle_message.await_count == 1
    adapter._handle_sticker.assert_awaited_once()
    adapter._build_message_event.assert_called_once()


@pytest.mark.asyncio
async def test_cross_handler_same_update_id_suppressed():
    """A redelivered update must be suppressed even across handler entry points."""
    adapter = _make_adapter()
    fake_event = _event("hi")
    adapter._build_message_event = MagicMock(return_value=fake_event)
    update = _update(3001, text="hi")
    ctx = MagicMock()

    await adapter._handle_text_message(update, ctx)
    await adapter._handle_command(update, ctx)

    assert adapter._enqueue_text_event.call_count == 1
    assert adapter.handle_message.await_count == 0


def test_dedup_keeps_current_id_when_max_non_positive():
    """A mis-set max<=0 must not disable dedup for the just-seen update."""
    adapter = _make_adapter()
    adapter._seen_update_ids_max = 0
    update = _update(9001)
    assert adapter._is_duplicate_update(update) is False
    assert adapter._is_duplicate_update(update) is True
    assert 9001 in adapter._seen_update_ids
