"""One Telegram forward gesture must reach the first agent boundary whole."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import Bot, Update

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def make_update(bot, mid, *, text=None, media=None, album=None):
    message = {
        "message_id": mid, "date": 1789043677,
        "chat": {"id": 123, "type": "private", "first_name": "Maxim"},
        "from": {"id": 123, "is_bot": False, "first_name": "Maxim"},
        "message_thread_id": 453221, "is_topic_message": True,
    }
    if text is not None:
        message["text"] = text
    if media:
        message[media] = {
            "file_id": str(mid), "file_unique_id": str(mid), "file_size": 12,
            "duration": 2, "width": 20, "height": 20,
            "file_name": f"{mid}.json", "mime_type": "application/json",
        }
        message["caption"] = f"caption {mid}"
        message["forward_origin"] = {
            "type": "hidden_user", "sender_user_name": f"Source {mid}", "date": 1789043600,
        }
    if album:
        message["media_group_id"] = album
    return Update.de_json({"update_id": mid, "message": message}, bot)


@pytest.fixture
def adapter(monkeypatch):
    a = TelegramAdapter(PlatformConfig(enabled=True, token="123:test"))
    a._is_user_authorized_from_message = lambda _: True
    a._should_process_message = lambda *args, **kwargs: True
    a._ensure_forum_commands = AsyncMock()
    a._cache_replied_media = AsyncMock()
    a._text_batch_delay_seconds = 0.01
    a._media_batch_delay_seconds = 0.03
    a.MEDIA_GROUP_WAIT_SECONDS = 0.03
    a._ingress_coordinator().ADMISSION_SECONDS = 0.1
    a._ingress_coordinator().DRAIN_SECONDS = 0.7
    a.set_message_handler(AsyncMock())
    a.seen = []
    a._start_session_processing = lambda e, k, **kw: a.seen.append((e, k))
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.cache_video_from_bytes",
        lambda data, **kw: "/tmp/ingress-video.mp4",
    )
    return a


@pytest.mark.asyncio
@pytest.mark.parametrize("media,album", [("video", None), ("document", "docs")])
@pytest.mark.parametrize("media_first", [False, True])
async def test_first_dispatch_contains_comment_forward_caption_and_media(adapter, monkeypatch, media, album, media_first):
    started, release = asyncio.Event(), asyncio.Event()

    async def get_file(_self, file_id, **kw):
        started.set()
        await release.wait()
        return SimpleNamespace(file_path="/a.mp4", download_as_bytearray=AsyncMock(return_value=b"{}"))

    monkeypatch.setattr(Bot, "get_file", get_file)
    bot = Bot("123:test")
    comment = make_update(bot, 1 if not media_first else 3, text="Это реально работает?")
    attachment = make_update(bot, 2, media=media, album=album)
    if not media_first:
        await adapter._handle_text_message(comment, None)
    download = asyncio.create_task(adapter._handle_media_message(attachment, None))
    try:
        await asyncio.wait_for(started.wait(), 2)
        if media_first:
            await adapter._handle_text_message(comment, None)
        await asyncio.sleep(0.08)
        assert adapter.seen == [], "comment escaped while its accepted media was downloading"
        release.set()
        await download
        await asyncio.sleep(1.0)
        assert len(adapter.seen) == 1
        event, _ = adapter.seen[0]
        assert "Это реально работает?" in event.text
        assert "caption 2" in event.text
        assert "Source 2" in event.text
        assert len(event.media_urls) == 1
    finally:
        release.set()
        await download
        await adapter._cancel_pending_delivery_tasks()
