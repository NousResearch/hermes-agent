"""QQ media/voice downloads must be bounded (#81046 class, sibling sites).

`_download_qq_media` and the STT voice fetch both read `resp.content`, which
buffers the whole remote body before any size check. A server that omits or
understates Content-Length can therefore stream an arbitrarily large payload
into the gateway process. #81046 fixed the same shape for vision downloads;
these two adapters were never covered.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform


def _adapter():
    from gateway.platforms.qqbot.adapter import QQAdapter

    a = QQAdapter.__new__(QQAdapter)
    a.platform = Platform.QQBOT
    a._app_id = "test-app"   # _log_tag is a derived property
    return a


class _FakeStream:
    """Minimal httpx streaming-response context manager."""

    def __init__(self, chunks, headers=None, status_ok=True):
        self._chunks = chunks
        self.headers = headers or {}
        self._status_ok = status_ok

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("bad status")

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


def _client(stream):
    client = MagicMock()
    client.stream = MagicMock(return_value=stream)
    return client


class TestDownloadCap:
    @pytest.mark.asyncio
    async def test_streams_and_returns_body_under_the_cap(self):
        a = _adapter()
        a._http_client = _client(_FakeStream([b"ab", b"cd"], {"content-type": "image/png"}))

        body, ctype = await a._download_media_with_cap("https://x/y.png", headers={})

        assert body == b"abcd"
        assert ctype == "image/png"

    @pytest.mark.asyncio
    async def test_aborts_when_the_stream_crosses_the_cap(self):
        """No Content-Length at all: the running byte count is the guard."""
        a = _adapter()
        a._http_client = _client(_FakeStream([b"x" * 600] * 10))  # 6000 bytes

        with patch.dict(os.environ, {"QQBOT_MAX_MEDIA_BYTES": "1000"}):
            with pytest.raises(ValueError, match="cap"):
                await a._download_media_with_cap("https://x/big", headers={})

    @pytest.mark.asyncio
    async def test_declared_oversize_is_refused_before_reading(self):
        a = _adapter()
        stream = _FakeStream([b"x"], {"content-length": "999999999"})
        a._http_client = _client(stream)

        with patch.dict(os.environ, {"QQBOT_MAX_MEDIA_BYTES": "1000"}):
            with pytest.raises(ValueError, match="declared"):
                await a._download_media_with_cap("https://x/big", headers={})

    @pytest.mark.asyncio
    async def test_malformed_content_length_falls_through_to_the_stream_guard(self):
        """A junk header is not a size verdict; the byte count still bounds it."""
        a = _adapter()
        a._http_client = _client(_FakeStream([b"y" * 5000], {"content-length": "not-a-number"}))

        with patch.dict(os.environ, {"QQBOT_MAX_MEDIA_BYTES": "1000"}):
            with pytest.raises(ValueError, match="cap"):
                await a._download_media_with_cap("https://x/junk", headers={})

    @pytest.mark.asyncio
    async def test_exact_cap_is_allowed(self):
        a = _adapter()
        a._http_client = _client(_FakeStream([b"z" * 1000]))

        with patch.dict(os.environ, {"QQBOT_MAX_MEDIA_BYTES": "1000"}):
            body, _ = await a._download_media_with_cap("https://x/exact", headers={})

        assert len(body) == 1000


class TestCapConfig:
    def test_default_cap(self):
        a = _adapter()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QQBOT_MAX_MEDIA_BYTES", None)
            assert a._qq_max_media_bytes() == 100 * 1024 * 1024

    def test_env_override(self):
        a = _adapter()
        with patch.dict(os.environ, {"QQBOT_MAX_MEDIA_BYTES": "2048"}):
            assert a._qq_max_media_bytes() == 2048

    @pytest.mark.parametrize("raw", ["abc", "-5", "0", ""])
    def test_invalid_override_falls_back_to_default(self, raw):
        """A bad or non-positive value must not disarm the cap."""
        a = _adapter()
        with patch.dict(os.environ, {"QQBOT_MAX_MEDIA_BYTES": raw}):
            assert a._qq_max_media_bytes() == 100 * 1024 * 1024
