"""Telegram: updates held across a disconnect are replayed in the order they arrived.

Updates that arrive after the delivery fence goes up are held at once, while batches buffered before
the fence are held only later, by their own flush or by teardown salvage. The reconnect drain must
still replay the earlier update first.
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 60
    adapter._media_batch_delay_seconds = 60
    return adapter


def _event(text, kind=MessageType.TEXT):
    event = MessageEvent(
        text=text, message_type=kind,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="1"))
    if kind == MessageType.PHOTO:
        event.media_urls, event.media_types = [f"{text}.jpg"], ["image/jpeg"]
    return event


async def _buffer(adapter, kind, text):
    if kind == "text":
        adapter._enqueue_text_event(_event(text))
    elif kind == "photo":
        adapter._enqueue_photo_event("photo-burst", _event(text, MessageType.PHOTO))
    else:
        await adapter._queue_media_group_event("album", _event(text, MessageType.PHOTO))


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["text", "photo", "album"])
@pytest.mark.parametrize("late", ["text", "photo", "album"])
async def test_update_held_after_the_fence_replays_after_the_earlier_buffered_one(first, late):
    adapter = _adapter()
    await _buffer(adapter, first, "earlier")
    adapter._mark_disconnected()  # fence: delayed delivery is now held instead of dispatched
    await _buffer(adapter, late, "later")
    await adapter._cancel_pending_delivery_tasks()  # teardown salvages what was still buffered

    assert [e.text for e in adapter._held_inbound_events] == ["earlier", "later"]


@pytest.mark.asyncio
async def test_teardown_salvage_keeps_arrival_order_across_batch_kinds():
    adapter = _adapter()
    await _buffer(adapter, "album", "first")
    await _buffer(adapter, "text", "second")
    await _buffer(adapter, "photo", "third")
    adapter._mark_disconnected()
    await adapter._cancel_pending_delivery_tasks()

    assert [e.text for e in adapter._held_inbound_events] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_interrupted_replay_keeps_its_remaining_updates_ahead_of_a_newer_one():
    adapter = _adapter()
    adapter._mark_disconnected()
    for text in ("a", "b", "c"):
        adapter._enqueue_text_event(_event(text))  # held on arrival
    replayed = []

    async def handle(event):
        replayed.append(event.text)
        adapter._mark_disconnected()  # the link drops again mid-replay ...
        adapter._enqueue_text_event(_event("d"))  # ... and a newer update arrives behind the fence

    adapter.handle_message = handle
    adapter._mark_connected()
    await adapter._held_inbound_redispatch_task

    assert replayed == ["a"]
    assert [e.text for e in adapter._held_inbound_events] == ["b", "c", "d"]
