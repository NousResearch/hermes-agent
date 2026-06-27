"""Album (media-group) merge: a slow/sequential per-photo download must NOT
split a Telegram album so the model sees only the first image.

Regression for "I sent 2 photos as one album, the model saw 1 photo + my text".
"""
import asyncio
import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.platforms.base import MessageEvent, MessageType


def _adapter():
    a = object.__new__(TelegramAdapter)
    a._media_group_events = {}
    a._media_group_tasks = {}
    a._media_group_pending = {}
    a.MEDIA_GROUP_WAIT_SECONDS = 0.2
    a.MEDIA_GROUP_MAX_WAIT_SECONDS = 2.0
    return a


def _photo_event(url, text=""):
    e = MessageEvent(text=text, message_type=MessageType.PHOTO)
    e.media_urls = [url]
    e.media_types = ["image/jpeg"]
    return e


@pytest.mark.asyncio
async def test_album_basic_two_photos_merge_into_one_event():
    a = _adapter()
    calls = []
    async def _handle(ev):
        calls.append(list(ev.media_urls))
    a.handle_message = _handle

    a._note_media_group_arrival("g")
    a._note_media_group_arrival("g")
    await a._queue_media_group_event("g", _photo_event("p1", "состав сока"))
    await a._queue_media_group_event("g", _photo_event("p2"))
    await asyncio.sleep(0.5)

    assert len(calls) == 1, f"album split into {len(calls)} messages"
    assert calls[0] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_album_survives_slow_sibling_download():
    """The 2nd photo finishes downloading AFTER the debounce window — the album
    must still arrive whole (the exact failure mode of the original bug)."""
    a = _adapter()
    calls = []

    async def handle(ev):
        calls.append(list(ev.media_urls))
    a.handle_message = handle

    # Both album updates arrive ~together (before their downloads).
    a._note_media_group_arrival("g")
    a._note_media_group_arrival("g")
    # Item 1 downloads fast and is buffered.
    await a._queue_media_group_event("g", _photo_event("p1", "состав сока"))
    # Item 2's download is SLOW — longer than MEDIA_GROUP_WAIT_SECONDS.
    await asyncio.sleep(0.4)
    await a._queue_media_group_event("g", _photo_event("p2"))
    await asyncio.sleep(0.5)

    assert len(calls) == 1, f"slow sibling split the album into {len(calls)} messages"
    assert calls[0] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_album_five_photos_all_reach_model():
    a = _adapter()
    calls = []
    async def _handle(ev):
        calls.append(list(ev.media_urls))
    a.handle_message = _handle

    for _ in range(5):
        a._note_media_group_arrival("g")
    for i in range(5):
        await a._queue_media_group_event("g", _photo_event(f"p{i}"))
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)

    assert len(calls) == 1
    assert calls[0] == [f"p{i}" for i in range(5)]
