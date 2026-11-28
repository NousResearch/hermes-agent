"""A photo or file that arrives only inside ``rich_message`` blocks is found, at any nesting, and its caption is
message text (#155902): before, the adapter flattened block text alone and the agent got the caption, not the image."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import CachedMedia
from gateway.platforms.event import MessageType
from tests.waiting import async_wait_until
from plugins.platforms.telegram.telegram_entities import rich_message_text
from plugins.platforms.telegram.telegram_rich_media import rich_media_items


def _photo(fid):
    return {"type": "photo", "photo": [{"file_id": f"{fid}-small"}, {"file_id": fid, "file_size": 9}],
            "caption": {"text": f"caption {fid}"}}


def test_every_media_block_is_found_in_order_with_its_caption_as_text():
    msg = SimpleNamespace(text=None, api_kwargs={"rich_message": {"blocks": [
        _photo("p1"),
        {"type": "paragraph", "text": "look at these"},
        {"type": "collage", "blocks": [_photo("p2"), {"type": "video", "video": {"file_id": "v1"}}]},
        {"type": "list", "items": [{"label": "-", "blocks": [
            {"type": "document", "document": {"file_id": "d1", "file_name": "a.pdf", "mime_type": "application/pdf"}}]}]},
        {"type": "photo", "photo": []},  # no sizes: nothing to fetch
    ]}})
    assert [(fid, mime, kind) for _block, fid, _media, mime, kind in rich_media_items(msg)] == [
        ("p1", "", "image"), ("p2", "", "image"), ("v1", "video/mp4", "video"), ("d1", "application/pdf", None)]
    assert rich_message_text(msg).splitlines()[:2] == ["caption p1", "look at these"]


@pytest.mark.asyncio
async def test_a_question_then_a_rich_photo_is_one_turn_with_the_image():
    """The question typed just before a rich photo waits for its download, as for a top-level photo."""
    from tests.plugins.platforms.telegram.test_text_then_photo_one_turn import _adapter, _slow_photo, _text, _update
    adapter, release = _adapter(), asyncio.Event()
    file_obj = _slow_photo(release).photo[0].get_file.return_value
    adapter._bot = SimpleNamespace(get_file=AsyncMock(return_value=file_obj))
    rich = _text(None)
    rich.rich_message, rich.api_kwargs = None, {"rich_message": {"blocks": [_photo("p1")]}}
    cached = CachedMedia("/cache/p1.jpg", "image/jpeg", "image", "p1.jpg")
    with patch("gateway.platforms.base.cache_media_bytes_async", new=AsyncMock(return_value=cached)):
        await adapter._handle_text_message(_update(_text("Is this a real study?")), MagicMock())
        photo = asyncio.create_task(adapter._handle_text_message(_update(rich), MagicMock()))
        await asyncio.sleep(0.6)  # health: allow HX013 -- a download outlasting the 0.18 s text delay IS the race
        release.set()
        await photo
        await async_wait_until(lambda: adapter.handle_message.await_args_list, timeout=5)
    [event] = [c.args[0] for c in adapter.handle_message.await_args_list]
    assert event.media_urls == ["/cache/p1.jpg"] and event.message_type == MessageType.PHOTO
    assert event.text.startswith("Is this a real study?")
