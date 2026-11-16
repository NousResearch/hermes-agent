"""A short Telegram comment and the video sent right after it reach the agent as one turn (#105050).

Forward-with-comment, or "here's the clip:" then a clip, sends the TEXT update first. The image
case is pinned in ``test_text_then_photo_one_turn``; this covers the generic media dispatch path
(video, voice, audio, documents), driven through the real ``_handle_text_message`` and
``_handle_media_message`` in production order with a download that outlasts the text delay.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.test_telegram_documents import _make_document, _make_file_obj, _make_message, _make_video
from tests.plugins.platforms.telegram.test_text_then_photo_one_turn import _adapter, _text, _update
from tests.waiting import async_wait_until


def _slow_video(release: asyncio.Event, user_id: int = 1):
    file_obj = _make_file_obj(b"clip")
    file_obj.file_path = "videos/clip.mp4"

    async def download():
        await release.wait()
        return bytearray(b"clip")
    file_obj.download_as_bytearray = AsyncMock(side_effect=download)
    msg = _make_message()
    msg.video, msg.from_user.id, msg.reply_to_message = _make_video(file_obj), user_id, None
    return msg


async def _comment_then_video(adapter, *, video_user: int):
    release = asyncio.Event()
    with patch("plugins.platforms.telegram.adapter.cache_video_from_bytes_async", new=AsyncMock(return_value="/v/clip.mp4")):
        await adapter._handle_text_message(_update(_text("Look at this clip")), MagicMock())
        video = asyncio.create_task(adapter._handle_media_message(_update(_slow_video(release, video_user)), MagicMock()))
        await asyncio.sleep(0.6)  # health: allow HX013 -- a download outlasting the 0.18 s text delay IS the race
        release.set()
        await video
        await async_wait_until(lambda: any(c.args[0].media_urls for c in adapter.handle_message.await_args_list), timeout=5)
    return [c.args[0] for c in adapter.handle_message.await_args_list]


@pytest.mark.asyncio
async def test_comment_then_video_is_one_turn_with_the_clip():
    events = await _comment_then_video(_adapter(), video_user=1)
    assert [(e.text, e.media_urls) for e in events] == [("Look at this clip", ["/v/clip.mp4"])]


@pytest.mark.asyncio
async def test_another_senders_video_does_not_take_the_comment():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = lambda _msg: True
    events = await _comment_then_video(adapter, video_user=2)
    assert [(e.text, e.media_urls) for e in events] == [("Look at this clip", []), ("", ["/v/clip.mp4"])]


@pytest.mark.asyncio
async def test_comment_before_a_document_album_with_a_command_caption_reaches_the_turn(tmp_path, monkeypatch):
    # The comment joins an uncaptioned text file that inlined its content; the album's /goal caption
    # then rebuilds the text from the saved captions, which must carry the comment too.
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path)
    adapter = _adapter()
    adapter.config.extra["media_group_wait_seconds"] = 0.05
    notes = _make_message(document=_make_document("notes.txt", "text/plain", file_obj=_make_file_obj(b"alpha")),
                          media_group_id="mg")
    goal = _make_message(document=_make_document("plan.txt", "text/plain", file_obj=_make_file_obj(b"beta")),
                         caption="/goal summarize these", media_group_id="mg")
    for msg in (notes, goal):
        msg.reply_to_message = None
    await adapter._handle_text_message(_update(_text("Look at these")), MagicMock())
    await adapter._handle_media_message(_update(notes), MagicMock())
    await adapter._handle_media_message(_update(goal), MagicMock())
    await async_wait_until(lambda: adapter.handle_message.await_count, timeout=5)
    await async_wait_until(lambda: not adapter._media_group_events and not adapter._pending_text_batches, timeout=5)

    events = [c.args[0] for c in adapter.handle_message.await_args_list]
    assert len(events) == 1, [e.text for e in events]
    assert events[0].get_command() == "goal", events[0].text
    assert "Look at these" in events[0].get_command_args(), events[0].text
