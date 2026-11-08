"""An album item that skips the normal cache (a video sent as a document, a file too large to
download, an oversized video) still joins its album: one album, one turn. Driven through the real
``_handle_media_message`` in production order, one item after another.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_documents import _make_document, _make_file_obj, _make_message
from tests.waiting import async_wait_until

TOO_BIG = 500 * 1024 * 1024


def _make_update(msg):
    return SimpleNamespace(update_id=1, message=msg, effective_message=msg, channel_post=None,
                           business_message=None, edited_business_message=None)


def _video_as_document():
    return _make_message(document=_make_document("clip.mp4", "video/mp4", file_obj=_make_file_obj(b"vid")),
                         media_group_id="mg")


def _too_large_document():
    return _make_message(document=_make_document("huge.zip", "application/zip", file_size=TOO_BIG),
                         media_group_id="mg")


def _oversized_video():
    msg = _make_message(media_group_id="mg")
    msg.video = MagicMock(file_size=TOO_BIG)
    return msg


@pytest.fixture(autouse=True)
def _cache_dirs(tmp_path, monkeypatch):
    for name in ("DOCUMENT_CACHE_DIR", "VIDEO_CACHE_DIR", "AUDIO_CACHE_DIR"):
        monkeypatch.setattr(f"gateway.platforms.base.{name}", tmp_path / name.lower())


@pytest.mark.asyncio
@pytest.mark.parametrize("second_item", [_video_as_document, _too_large_document, _oversized_video])
async def test_album_item_on_an_early_exit_path_joins_the_album_turn(second_item):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={"media_group_wait_seconds": 0.05}))
    adapter.handle_message = AsyncMock()
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    pdf = _make_message(document=_make_document("report.pdf"), caption="see both", media_group_id="mg")

    await adapter._handle_media_message(_make_update(pdf), MagicMock())
    await adapter._handle_media_message(_make_update(second_item()), MagicMock())
    await async_wait_until(lambda: adapter.handle_message.await_count, timeout=5)
    await async_wait_until(lambda: not adapter._media_group_events, timeout=5)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "see both" in event.text
    assert any(url.endswith("report.pdf") for url in event.media_urls), event.media_urls
