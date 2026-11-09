"""A short Telegram question and the photo sent right after it reach the agent as one turn (#33270).

The text batch flushes after a short quiet period, while the photo is still downloading and then
waits in its own burst batch, so the agent used to answer the question without the image. Driven
through the real ``_handle_text_message`` and ``_handle_media_message`` in production order: the
photo's handler starts once the text's has returned, and its download outlasts the text delay.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_documents import _make_file_obj, _make_message, _make_photo
from tests.waiting import async_wait_until


def _update(msg):
    return SimpleNamespace(update_id=1, message=msg, effective_message=msg, channel_post=None,
                           business_message=None, edited_business_message=None)


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t"))
    adapter.handle_message = AsyncMock()
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    adapter._media_batch_delay_seconds = 0.1
    return adapter


def _text(text: str, user_id: int = 1):
    msg = _make_message()
    msg.text, msg.caption, msg.from_user.id = text, None, user_id
    return msg


def _slow_photo(release: asyncio.Event, user_id: int = 1):
    file_obj = _make_file_obj(b"post")

    async def download():
        await release.wait()
        return bytearray(b"post")
    file_obj.download_as_bytearray = AsyncMock(side_effect=download)
    msg = _make_message(photo=[_make_photo(file_obj)])
    msg.from_user.id = user_id
    return msg


async def _text_then_photo(adapter, *, photo_user: int):
    release = asyncio.Event()
    with patch("plugins.platforms.telegram.adapter.cache_image_from_bytes_async", new=AsyncMock(return_value="/tmp/post.jpg")):
        await adapter._handle_text_message(_update(_text("Is this a real study?")), MagicMock())
        photo = asyncio.create_task(adapter._handle_media_message(_update(_slow_photo(release, photo_user)), MagicMock()))
        await asyncio.sleep(0.6)  # health: allow HX013 -- a download outlasting the 0.18 s text delay IS the race
        release.set()
        await photo
        await async_wait_until(lambda: any(c.args[0].media_urls for c in adapter.handle_message.await_args_list), timeout=5)
    return [c.args[0] for c in adapter.handle_message.await_args_list]


@pytest.mark.asyncio
async def test_question_then_photo_is_one_turn_with_the_image():
    events = await _text_then_photo(_adapter(), photo_user=1)
    assert len(events) == 1
    assert events[0].media_urls == ["/tmp/post.jpg"]
    assert events[0].text == "Is this a real study?"


@pytest.mark.asyncio
async def test_another_senders_photo_does_not_take_the_question():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = lambda _msg: True
    events = await _text_then_photo(adapter, photo_user=2)
    assert [e.text for e in events] == ["Is this a real study?", ""]
    assert [e.media_urls for e in events] == [[], ["/tmp/post.jpg"]]
