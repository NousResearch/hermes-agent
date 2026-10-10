"""Replied-to Telegram media that cannot be cached must leave a note in the transcript.

``_cache_replied_media`` used to act only on the ``"ok"`` download status. A reply to a message
whose attachment is oversized, fails to download or is unreadable produced no media and no note, so
the agent saw a bare caption and answered as if the attachment were the caption itself.
``_cache_observed_media`` already annotates those cases; this covers the replied-to path.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("telegram")

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter

CAPTION = "what does the attached file say?"


def _make_adapter(status, cached=None):
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})
    adapter._max_doc_bytes = 20 * 1024 * 1024
    calls = []

    async def fake_download(msg, what):
        calls.append((msg, what))
        return status, cached

    adapter._download_observed_media = fake_download
    adapter._download_calls = calls
    return adapter


def _reply_message():
    return SimpleNamespace(reply_to_message=SimpleNamespace(message_id=7))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,cached,expected",
    [
        ("oversized", 33 * 1024 * 1024, "too large or unverifiable, not cached. Maximum: 20 MB."),
        ("failed", None, "could not be read, not cached."),
        ("unreadable", None, "could not be read, not cached."),
    ],
)
async def test_uncacheable_replied_media_is_noted(status, cached, expected):
    adapter = _make_adapter(status, cached)
    event = MessageEvent(text=CAPTION)

    await adapter._cache_replied_media(_reply_message(), event)

    assert event.text.startswith(CAPTION)
    assert "[Replied-to Telegram attachment " in event.text
    assert expected in event.text
    assert event.media_urls == [] and event.media_types == []
    assert event.message_type is MessageType.TEXT


@pytest.mark.asyncio
async def test_cached_replied_media_is_still_attached():
    cached = SimpleNamespace(kind="image", display_name="photo.jpg", path="/tmp/cache/photo.jpg", media_type="image/jpeg")
    adapter = _make_adapter("ok", cached)
    event = MessageEvent(text=CAPTION)

    await adapter._cache_replied_media(_reply_message(), event)

    assert event.media_urls == [cached.path]
    assert event.media_types == [cached.media_type]
    assert event.message_type is MessageType.PHOTO
    assert "[Replied-to image 'photo.jpg' saved at: /tmp/cache/photo.jpg]" in event.text
    assert "not cached" not in event.text


@pytest.mark.asyncio
async def test_reply_without_media_leaves_text_untouched():
    adapter = _make_adapter("none")
    event = MessageEvent(text=CAPTION)

    await adapter._cache_replied_media(_reply_message(), event)

    assert event.text == CAPTION
    assert event.media_urls == []


@pytest.mark.asyncio
async def test_message_without_reply_does_not_download():
    adapter = _make_adapter("oversized", 1)
    event = MessageEvent(text=CAPTION)

    await adapter._cache_replied_media(SimpleNamespace(reply_to_message=None), event)

    assert event.text == CAPTION
    assert adapter._download_calls == []
