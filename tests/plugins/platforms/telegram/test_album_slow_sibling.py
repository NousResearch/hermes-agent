"""A Telegram album stays one turn while a sibling's download is still running (#74965, #150720).

Updates of one chat are handled one after another, and each album item is downloaded before it is
merged, so a slow sibling used to land after the first item's debounce had already dispatched it
alone. Driven through the real ``_handle_media_message`` in production order: item 2's handler
starts as soon as item 1's returns.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.plugins.platforms.telegram import telegram_media_group
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_documents import _make_file_obj, _make_message, _make_photo
from tests.waiting import async_wait_until

DEBOUNCE_S = 0.2


def _make_update(msg):
    """PTB shape for a private/group message: ``effective_message`` is the message itself."""
    return SimpleNamespace(update_id=1, message=msg, effective_message=msg, channel_post=None,
                           business_message=None, edited_business_message=None)


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={"media_group_wait_seconds": DEBOUNCE_S}))
    adapter.handle_message = AsyncMock()
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    return adapter


def _slow_photo(release: asyncio.Event):
    file_obj = _make_file_obj(b"second")

    async def download():
        await release.wait()
        return bytearray(b"second")
    file_obj.download_as_bytearray = AsyncMock(side_effect=download)
    return _make_photo(file_obj)


async def _send_album(adapter, release, *, stall_s: float) -> None:
    first = _make_message(caption="front and back", photo=[_make_photo(_make_file_obj(b"first"))], media_group_id="mg")
    second = _make_message(photo=[_slow_photo(release)], media_group_id="mg")
    await adapter._handle_media_message(_make_update(first), MagicMock())
    slow = asyncio.create_task(adapter._handle_media_message(_make_update(second), MagicMock()))
    await asyncio.sleep(stall_s)  # health: allow HX013 -- the stalled sibling download IS the behaviour under test
    release.set()
    await slow


@pytest.mark.asyncio
async def test_album_waits_for_a_sibling_still_downloading():
    adapter = _adapter()
    release = asyncio.Event()
    with patch("hermes_agent.plugins.platforms.telegram.adapter.cache_image_from_bytes_async",
               new=AsyncMock(side_effect=["/tmp/front.jpg", "/tmp/back.jpg"])):
        await _send_album(adapter, release, stall_s=1.2)  # past both the old 0.8 s window and the debounce
        await async_wait_until(lambda: adapter.handle_message.await_count, timeout=5)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == ["/tmp/front.jpg", "/tmp/back.jpg"]
    assert event.text == "front and back"


@pytest.mark.asyncio
async def test_a_stuck_sibling_does_not_hold_the_album_forever(monkeypatch):
    monkeypatch.setattr(telegram_media_group, "MEDIA_GROUP_MAX_WAIT_SECONDS", 0.3)
    adapter = _adapter()
    release = asyncio.Event()
    with patch("hermes_agent.plugins.platforms.telegram.adapter.cache_image_from_bytes_async",
               new=AsyncMock(side_effect=["/tmp/front.jpg", "/tmp/back.jpg"])):
        first = _make_message(caption="front", photo=[_make_photo(_make_file_obj(b"first"))], media_group_id="mg")
        await adapter._handle_media_message(_make_update(first), MagicMock())
        stuck = asyncio.create_task(adapter._handle_media_message(
            _make_update(_make_message(photo=[_slow_photo(release)], media_group_id="mg")), MagicMock()))
        await async_wait_until(lambda: adapter.handle_message.await_count, timeout=5)
        assert adapter.handle_message.await_args.args[0].media_urls == ["/tmp/front.jpg"]
        release.set()
        await stuck
        await async_wait_until(lambda: adapter.handle_message.await_count == 2, timeout=5)
    assert adapter.handle_message.await_args.args[0].media_urls == ["/tmp/back.jpg"]  # late, but not lost


@pytest.mark.parametrize("raw, expected", [
    (0.2, 0.2),
    ("soon", TelegramAdapter.MEDIA_GROUP_WAIT_SECONDS),
    (-1, TelegramAdapter.MEDIA_GROUP_WAIT_SECONDS),
    (50, 10.0),
])
def test_album_debounce_reads_media_group_wait_seconds(raw, expected):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={"media_group_wait_seconds": raw}))
    assert telegram_media_group.media_group_wait_seconds(adapter) == expected
