"""
Tests for media download retry logic added in PR #2982.

Covers:
- gateway/platforms/base.py:       cache_image_from_url
- gateway/platforms/slack.py:      SlackAdapter._download_slack_file
                                    SlackAdapter._download_slack_file_bytes
- gateway/platforms/mattermost.py: MattermostAdapter._send_url_as_file

All async tests use asyncio.run() directly — pytest-asyncio is not installed
in this environment.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

# ---------------------------------------------------------------------------
# Helpers for building httpx exceptions
# ---------------------------------------------------------------------------

def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://example.com/img.jpg")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=request, response=response
    )


def _make_timeout_error() -> httpx.TimeoutException:
    return httpx.TimeoutException("timed out")


# ---------------------------------------------------------------------------
# cache_image_from_url (base.py)
# ---------------------------------------------------------------------------

class TestCacheImageFromUrl:
    """Tests for gateway.platforms.base.cache_image_from_url"""

    def test_success_on_first_attempt(self, tmp_path, monkeypatch):
        """A clean 200 response caches the image and returns a path."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")

        fake_response = MagicMock()
        fake_response.content = b"\xff\xd8\xff fake jpeg"
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                from gateway.platforms.base import cache_image_from_url
                return await cache_image_from_url(
                    "http://example.com/img.jpg", ext=".jpg"
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        mock_client.get.assert_called_once()

    def test_retries_on_timeout_then_succeeds(self, tmp_path, monkeypatch):
        """A timeout on the first attempt is retried; second attempt succeeds."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")

        fake_response = MagicMock()
        fake_response.content = b"image data"
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_make_timeout_error(), fake_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_sleep = AsyncMock()

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", mock_sleep):
                from gateway.platforms.base import cache_image_from_url
                return await cache_image_from_url(
                    "http://example.com/img.jpg", ext=".jpg", retries=2
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        assert mock_client.get.call_count == 2
        mock_sleep.assert_called_once()

    def test_retries_on_429_then_succeeds(self, tmp_path, monkeypatch):
        """A 429 response on the first attempt is retried; second attempt succeeds."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")

        ok_response = MagicMock()
        ok_response.content = b"image data"
        ok_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_make_http_status_error(429), ok_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                from gateway.platforms.base import cache_image_from_url
                return await cache_image_from_url(
                    "http://example.com/img.jpg", ext=".jpg", retries=2
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        assert mock_client.get.call_count == 2

    def test_raises_after_max_retries_exhausted(self, tmp_path, monkeypatch):
        """Timeout on every attempt raises after all retries are consumed."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_make_timeout_error())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                from gateway.platforms.base import cache_image_from_url
                await cache_image_from_url(
                    "http://example.com/img.jpg", ext=".jpg", retries=2
                )

        with pytest.raises(httpx.TimeoutException):
            asyncio.run(run())

        # 3 total calls: initial + 2 retries
        assert mock_client.get.call_count == 3

    def test_non_retryable_4xx_raises_immediately(self, tmp_path, monkeypatch):
        """A 404 (non-retryable) is raised immediately without any retry."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")

        mock_sleep = AsyncMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_make_http_status_error(404))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", mock_sleep):
                from gateway.platforms.base import cache_image_from_url
                await cache_image_from_url(
                    "http://example.com/img.jpg", ext=".jpg", retries=2
                )

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(run())

        # Only 1 attempt, no sleep
        assert mock_client.get.call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# cache_audio_from_url (base.py)
# ---------------------------------------------------------------------------

class TestCacheAudioFromUrl:
    """Tests for gateway.platforms.base.cache_audio_from_url"""

    def test_success_on_first_attempt(self, tmp_path, monkeypatch):
        """A clean 200 response caches the audio and returns a path."""
        monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio")

        fake_response = MagicMock()
        fake_response.content = b"\x00\x01 fake audio"
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                from gateway.platforms.base import cache_audio_from_url
                return await cache_audio_from_url(
                    "http://example.com/voice.ogg", ext=".ogg"
                )

        path = asyncio.run(run())
        assert path.endswith(".ogg")
        mock_client.get.assert_called_once()

    def test_retries_on_timeout_then_succeeds(self, tmp_path, monkeypatch):
        """A timeout on the first attempt is retried; second attempt succeeds."""
        monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio")

        fake_response = MagicMock()
        fake_response.content = b"audio data"
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_make_timeout_error(), fake_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_sleep = AsyncMock()

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", mock_sleep):
                from gateway.platforms.base import cache_audio_from_url
                return await cache_audio_from_url(
                    "http://example.com/voice.ogg", ext=".ogg", retries=2
                )

        path = asyncio.run(run())
        assert path.endswith(".ogg")
        assert mock_client.get.call_count == 2
        mock_sleep.assert_called_once()

    def test_retries_on_429_then_succeeds(self, tmp_path, monkeypatch):
        """A 429 response on the first attempt is retried; second attempt succeeds."""
        monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio")

        ok_response = MagicMock()
        ok_response.content = b"audio data"
        ok_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_make_http_status_error(429), ok_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                from gateway.platforms.base import cache_audio_from_url
                return await cache_audio_from_url(
                    "http://example.com/voice.ogg", ext=".ogg", retries=2
                )

        path = asyncio.run(run())
        assert path.endswith(".ogg")
        assert mock_client.get.call_count == 2

    def test_retries_on_500_then_succeeds(self, tmp_path, monkeypatch):
        """A 500 response on the first attempt is retried; second attempt succeeds."""
        monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio")

        ok_response = MagicMock()
        ok_response.content = b"audio data"
        ok_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_make_http_status_error(500), ok_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                from gateway.platforms.base import cache_audio_from_url
                return await cache_audio_from_url(
                    "http://example.com/voice.ogg", ext=".ogg", retries=2
                )

        path = asyncio.run(run())
        assert path.endswith(".ogg")
        assert mock_client.get.call_count == 2

    def test_raises_after_max_retries_exhausted(self, tmp_path, monkeypatch):
        """Timeout on every attempt raises after all retries are consumed."""
        monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_make_timeout_error())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                from gateway.platforms.base import cache_audio_from_url
                await cache_audio_from_url(
                    "http://example.com/voice.ogg", ext=".ogg", retries=2
                )

        with pytest.raises(httpx.TimeoutException):
            asyncio.run(run())

        # 3 total calls: initial + 2 retries
        assert mock_client.get.call_count == 3

    def test_non_retryable_4xx_raises_immediately(self, tmp_path, monkeypatch):
        """A 404 (non-retryable) is raised immediately without any retry."""
        monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio")

        mock_sleep = AsyncMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_make_http_status_error(404))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", mock_sleep):
                from gateway.platforms.base import cache_audio_from_url
                await cache_audio_from_url(
                    "http://example.com/voice.ogg", ext=".ogg", retries=2
                )

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(run())

        # Only 1 attempt, no sleep
        assert mock_client.get.call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Slack mock setup (mirrors existing test_slack.py approach)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import gateway.platforms.slack as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter  # noqa: E402
from gateway.config import Platform, PlatformConfig  # noqa: E402


def _make_slack_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter._running = True
    return adapter


# ---------------------------------------------------------------------------
# SlackAdapter._download_slack_file
# ---------------------------------------------------------------------------

class TestSlackDownloadSlackFile:
    """Tests for SlackAdapter._download_slack_file"""

    def test_success_on_first_attempt(self, tmp_path, monkeypatch):
        """Successful download on first try returns a cached file path."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        adapter = _make_slack_adapter()

        fake_response = MagicMock()
        fake_response.content = b"fake image bytes"
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                return await adapter._download_slack_file(
                    "https://files.slack.com/img.jpg", ext=".jpg"
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        mock_client.get.assert_called_once()

    def test_retries_on_timeout_then_succeeds(self, tmp_path, monkeypatch):
        """Timeout on first attempt triggers retry; success on second."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        adapter = _make_slack_adapter()

        fake_response = MagicMock()
        fake_response.content = b"image bytes"
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_make_timeout_error(), fake_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_sleep = AsyncMock()

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", mock_sleep):
                return await adapter._download_slack_file(
                    "https://files.slack.com/img.jpg", ext=".jpg"
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        assert mock_client.get.call_count == 2
        mock_sleep.assert_called_once()

    def test_raises_after_max_retries(self, tmp_path, monkeypatch):
        """Timeout on every attempt eventually raises after 3 total tries."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        adapter = _make_slack_adapter()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_make_timeout_error())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await adapter._download_slack_file(
                    "https://files.slack.com/img.jpg", ext=".jpg"
                )

        with pytest.raises(httpx.TimeoutException):
            asyncio.run(run())

        assert mock_client.get.call_count == 3

    def test_non_retryable_403_raises_immediately(self, tmp_path, monkeypatch):
        """A 403 is not retried; it raises immediately."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        adapter = _make_slack_adapter()

        mock_sleep = AsyncMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_make_http_status_error(403))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", mock_sleep):
                await adapter._download_slack_file(
                    "https://files.slack.com/img.jpg", ext=".jpg"
                )

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(run())

        assert mock_client.get.call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# SlackAdapter._download_slack_file_bytes
# ---------------------------------------------------------------------------

class TestSlackDownloadSlackFileBytes:
    """Tests for SlackAdapter._download_slack_file_bytes"""

    def test_success_returns_bytes(self):
        """Successful download returns raw bytes."""
        adapter = _make_slack_adapter()

        fake_response = MagicMock()
        fake_response.content = b"raw bytes here"
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                return await adapter._download_slack_file_bytes(
                    "https://files.slack.com/file.bin"
                )

        result = asyncio.run(run())
        assert result == b"raw bytes here"

    def test_retries_on_429_then_succeeds(self):
        """429 on first attempt is retried; raw bytes returned on second."""
        adapter = _make_slack_adapter()

        ok_response = MagicMock()
        ok_response.content = b"final bytes"
        ok_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_make_http_status_error(429), ok_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                return await adapter._download_slack_file_bytes(
                    "https://files.slack.com/file.bin"
                )

        result = asyncio.run(run())
        assert result == b"final bytes"
        assert mock_client.get.call_count == 2

    def test_raises_after_max_retries(self):
        """Persistent timeouts raise after all 3 attempts are exhausted."""
        adapter = _make_slack_adapter()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_make_timeout_error())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await adapter._download_slack_file_bytes(
                    "https://files.slack.com/file.bin"
                )

        with pytest.raises(httpx.TimeoutException):
            asyncio.run(run())

        assert mock_client.get.call_count == 3


# ---------------------------------------------------------------------------
# MattermostAdapter._send_url_as_file
# ---------------------------------------------------------------------------

def _make_mm_adapter():
    """Build a minimal MattermostAdapter with mocked internals."""
    from gateway.platforms.mattermost import MattermostAdapter
    config = PlatformConfig(
        enabled=True, token="mm-token-fake",
        extra={"url": "https://mm.example.com"},
    )
    adapter = MattermostAdapter(config)
    adapter._session = MagicMock()
    adapter._upload_file = AsyncMock(return_value="file-id-123")
    adapter._api_post = AsyncMock(return_value={"id": "post-id-abc"})
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    return adapter


def _make_aiohttp_resp(status: int, content: bytes = b"file bytes",
                       content_type: str = "image/jpeg"):
    """Build a context-manager mock for an aiohttp response."""
    resp = MagicMock()
    resp.status = status
    resp.content_type = content_type
    resp.read = AsyncMock(return_value=content)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


class TestMattermostSendUrlAsFile:
    """Tests for MattermostAdapter._send_url_as_file"""

    def test_success_on_first_attempt(self):
        """200 on first attempt → file uploaded and post created."""
        adapter = _make_mm_adapter()
        resp = _make_aiohttp_resp(200)
        adapter._session.get = MagicMock(return_value=resp)

        async def run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await adapter._send_url_as_file(
                    "C123", "http://cdn.example.com/img.png", "caption", None
                )

        result = asyncio.run(run())
        assert result.success
        adapter._upload_file.assert_called_once()
        adapter._api_post.assert_called_once()

    def test_retries_on_429_then_succeeds(self):
        """429 on first attempt is retried; 200 on second attempt succeeds."""
        adapter = _make_mm_adapter()

        resp_429 = _make_aiohttp_resp(429)
        resp_200 = _make_aiohttp_resp(200)
        adapter._session.get = MagicMock(side_effect=[resp_429, resp_200])

        mock_sleep = AsyncMock()

        async def run():
            with patch("asyncio.sleep", mock_sleep):
                return await adapter._send_url_as_file(
                    "C123", "http://cdn.example.com/img.png", None, None
                )

        result = asyncio.run(run())
        assert result.success
        assert adapter._session.get.call_count == 2
        mock_sleep.assert_called_once()

    def test_retries_on_500_then_succeeds(self):
        """5xx on first attempt is retried; 200 on second attempt succeeds."""
        adapter = _make_mm_adapter()

        resp_500 = _make_aiohttp_resp(500)
        resp_200 = _make_aiohttp_resp(200)
        adapter._session.get = MagicMock(side_effect=[resp_500, resp_200])

        async def run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await adapter._send_url_as_file(
                    "C123", "http://cdn.example.com/img.png", None, None
                )

        result = asyncio.run(run())
        assert result.success
        assert adapter._session.get.call_count == 2

    def test_falls_back_to_text_after_max_retries_on_5xx(self):
        """Three consecutive 500s exhaust retries; falls back to send() with URL text."""
        adapter = _make_mm_adapter()

        resp_500 = _make_aiohttp_resp(500)
        adapter._session.get = MagicMock(return_value=resp_500)

        async def run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await adapter._send_url_as_file(
                    "C123", "http://cdn.example.com/img.png", "my caption", None
                )

        asyncio.run(run())

        adapter.send.assert_called_once()
        text_arg = adapter.send.call_args[0][1]
        assert "http://cdn.example.com/img.png" in text_arg

    def test_falls_back_on_client_error(self):
        """aiohttp.ClientError on every attempt falls back to send() with URL."""
        import aiohttp

        adapter = _make_mm_adapter()

        error_resp = MagicMock()
        error_resp.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientConnectionError("connection refused")
        )
        error_resp.__aexit__ = AsyncMock(return_value=False)
        adapter._session.get = MagicMock(return_value=error_resp)

        async def run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await adapter._send_url_as_file(
                    "C123", "http://cdn.example.com/img.png", None, None
                )

        asyncio.run(run())

        adapter.send.assert_called_once()
        text_arg = adapter.send.call_args[0][1]
        assert "http://cdn.example.com/img.png" in text_arg

    def test_non_retryable_404_falls_back_immediately(self):
        """404 is non-retryable (< 500, != 429); send() is called right away."""
        adapter = _make_mm_adapter()

        resp_404 = _make_aiohttp_resp(404)
        adapter._session.get = MagicMock(return_value=resp_404)

        mock_sleep = AsyncMock()

        async def run():
            with patch("asyncio.sleep", mock_sleep):
                return await adapter._send_url_as_file(
                    "C123", "http://cdn.example.com/img.png", None, None
                )

        asyncio.run(run())

        adapter.send.assert_called_once()
        # No sleep — fell back on first attempt
        mock_sleep.assert_not_called()
        assert adapter._session.get.call_count == 1
