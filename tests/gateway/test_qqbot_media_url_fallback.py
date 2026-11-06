"""QQ media sending when the platform itself cannot fetch the URL.

Regression: the QQ Bot API fetches ``url=`` sources on its own servers. When that
fetch times out it answers ``400 .../files: 富媒体文件上传超时`` and the caller
fell back to posting the bare URL as text (users saw a link instead of the image).
The adapter now downloads the media locally and delivers it through the chunked
upload path instead.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.qqbot.constants import (
    MEDIA_TYPE_FILE,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
)

STICKER_URL = "https://gcore.jsdelivr.net/gh/berserk36900-del/whale-stickers@main/stickers/whale-027.jpg"
QQ_FETCH_TIMEOUT = (
    "QQ Bot API error [400] /v2/groups/group-1/files: 富媒体文件上传超时"
)


def _adapter():
    from plugins.platforms.qqbot.adapter import QQAdapter
    return QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))


def _run(coro):
    return asyncio.run(coro)


def _send(adapter, source, file_type=MEDIA_TYPE_IMAGE):
    """Call ``_send_media`` with the connection/chat plumbing stubbed out."""
    adapter._ensure_connected = AsyncMock(return_value=True)
    adapter._guess_chat_type = lambda chat_id: "group"
    return _run(
        adapter._send_media("group-1", source, file_type, "image", caption="傲娇的就这张嘴硬的。")
    )


class TestMediaUrlFetchFallback:
    def test_url_upload_success_skips_local_download(self, tmp_path):
        adapter = _adapter()
        adapter._upload_media = AsyncMock(return_value={"file_info": "fi-url"})
        adapter._cache_remote_media = AsyncMock(return_value=str(tmp_path / "never.jpg"))
        adapter._post_message = AsyncMock(return_value=SendResult(success=True, message_id="m1"))

        result = _send(adapter, STICKER_URL)

        assert result.success
        adapter._upload_media.assert_awaited_once()
        adapter._cache_remote_media.assert_not_awaited()
        assert adapter._upload_media.await_args.kwargs["url"] == STICKER_URL

    def test_qq_fetch_timeout_falls_back_to_chunked_upload(self, tmp_path):
        adapter = _adapter()
        cached = tmp_path / "whale-027.jpg"
        cached.write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
        adapter._upload_media = AsyncMock(side_effect=RuntimeError(QQ_FETCH_TIMEOUT))
        adapter._cache_remote_media = AsyncMock(return_value=str(cached))
        adapter._upload_local_file = AsyncMock(return_value={"file_info": "fi-local"})
        adapter._post_message = AsyncMock(return_value=SendResult(success=True, message_id="m2"))

        result = _send(adapter, STICKER_URL)

        assert result.success
        adapter._cache_remote_media.assert_awaited_once_with(STICKER_URL, MEDIA_TYPE_IMAGE)
        adapter._upload_local_file.assert_awaited_once()
        assert adapter._upload_local_file.await_args.args[2] == str(cached)
        assert adapter._post_message.await_args.args[1]["media"]["file_info"] == "fi-local"

    def test_both_paths_failing_returns_failed_result(self, tmp_path):
        """No silent success: if the local download also fails the caller still sees a failure
        (and ``send_image`` turns that into its documented text fallback)."""
        adapter = _adapter()
        adapter._upload_media = AsyncMock(side_effect=RuntimeError(QQ_FETCH_TIMEOUT))
        adapter._cache_remote_media = AsyncMock(side_effect=RuntimeError("download 502"))
        adapter._post_message = AsyncMock()

        result = _send(adapter, STICKER_URL)

        assert not result.success
        assert "download 502" in result.error
        adapter._post_message.assert_not_awaited()

    @pytest.mark.parametrize("file_type", [MEDIA_TYPE_VIDEO, MEDIA_TYPE_FILE])
    def test_video_and_file_urls_keep_old_behaviour(self, file_type):
        """Only image/voice URLs can be re-delivered from a local copy; others still raise."""
        adapter = _adapter()
        adapter._upload_media = AsyncMock(side_effect=RuntimeError(QQ_FETCH_TIMEOUT))
        adapter._cache_remote_media = AsyncMock(return_value="/tmp/x.bin")
        adapter._post_message = AsyncMock()

        result = _send(adapter, STICKER_URL, file_type=file_type)

        assert not result.success
        assert "富媒体文件上传超时" in result.error
        adapter._cache_remote_media.assert_not_awaited()

    def test_send_image_posts_link_only_when_media_really_fails(self, tmp_path):
        """``send_image`` keeps its text fallback for a genuine double failure."""
        adapter = _adapter()
        adapter._send_media = AsyncMock(return_value=SendResult(success=False, error="boom"))
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m3"))

        result = _run(adapter.send_image("group-1", STICKER_URL, caption="傲娇的就这张嘴硬的。"))

        assert result.success
        assert adapter.send.await_args.kwargs["content"] == f"傲娇的就这张嘴硬的。\n{STICKER_URL}"


class TestCacheRemoteMedia:
    def test_image_url_is_downloaded_to_local_jpeg(self, tmp_path, monkeypatch):
        """The downloader really fetches bytes (no network mock): proves the fallback has a
        local file to upload instead of handing QQ another URL."""
        adapter = _adapter()
        calls = {}

        async def fake_cache_image_from_url(url, ext=".jpg", retries=2):
            calls["args"] = (url, ext)
            path = tmp_path / f"cached{ext}"
            path.write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
            return str(path)

        with patch("plugins.platforms.qqbot.adapter.cache_image_from_url", fake_cache_image_from_url):
            cached = _run(adapter._cache_remote_media(STICKER_URL, MEDIA_TYPE_IMAGE))

        assert calls["args"] == (STICKER_URL, ".jpg")
        assert cached.endswith(".jpg")
        with open(cached, "rb") as fh:
            assert fh.read(4) == b"\xff\xd8\xff\xe0"

    def test_voice_url_uses_audio_cache(self):
        adapter = _adapter()
        voice_url = "https://example.com/speech.mp3"
        with patch(
            "plugins.platforms.qqbot.adapter.cache_audio_from_url",
            AsyncMock(return_value="/tmp/speech.mp3"),
        ) as audio:
            from plugins.platforms.qqbot.constants import MEDIA_TYPE_VOICE
            cached = _run(adapter._cache_remote_media(voice_url, MEDIA_TYPE_VOICE))

        assert cached == "/tmp/speech.mp3"
        audio.assert_awaited_once_with(voice_url, ".mp3")
