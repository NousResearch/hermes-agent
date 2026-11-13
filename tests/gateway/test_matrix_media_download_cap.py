"""Inbound ``mxc://`` media is streamed under ``MATRIX_MAX_MEDIA_BYTES`` (#55041).

mautrix's ``download_media`` buffers the whole body (``await response.read()``), so the cap must
hold at the read boundary even when the event omits ``info.size`` and the server omits
``Content-Length``.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


class _Content:
    def __init__(self, chunks):
        self._chunks = chunks
        self.yielded = 0

    async def iter_chunked(self, _size):
        for chunk in self._chunks:
            self.yielded += 1
            yield chunk


class _Response:
    status = 200

    def __init__(self, chunks, headers=None):
        self.headers = headers or {}
        self.content = _Content(chunks)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        self.closed = True

    def raise_for_status(self):
        pass


class _Versions:
    def __init__(self, authenticated):
        self._authenticated = authenticated

    def supports(self, version):
        return self._authenticated and str(version) == "v1.11"


class _Api:
    token = "syt_secret"
    as_user_id = None

    def __init__(self, response):
        self.response = response
        self.requests = []
        self.session = SimpleNamespace(get=self._get)

    def _get(self, url, params=None, headers=None):
        self.requests.append((url, params, headers))
        return self.response

    def get_download_url(self, mxc, authenticated=False):
        base = "_matrix/client/v1/media" if authenticated else "_matrix/media/v3"
        return f"https://hs.example.org/{base}/download/{str(mxc)[6:]}"

    def log_download_request(self, _url, _params):
        return 1

    def log_download_request_done(self, *_args):
        pass


class _Client:
    def __init__(self, response, authenticated=False):
        self.api = _Api(response)
        self._authenticated = authenticated
        self.download_media = AsyncMock(side_effect=AssertionError("unbounded download_media used"))

    async def versions(self):
        return _Versions(self._authenticated)


@pytest.mark.asyncio
async def test_under_cap_body_is_returned_with_auth_header():
    from plugins.platforms.matrix.media_download import download_mxc_capped

    client = _Client(_Response([b"abc", b"def"]), authenticated=True)
    assert await download_mxc_capped(client, "mxc://example.org/ok", 10) == b"abcdef"
    url, params, headers = client.api.requests[0]
    assert "/_matrix/client/v1/media/download/example.org/ok" in url
    assert headers == {"Authorization": "Bearer syt_secret"}
    assert params == {"allow_redirect": "true"}


@pytest.mark.asyncio
async def test_stream_without_content_length_stops_once_cap_is_crossed():
    from plugins.platforms.matrix.media_download import MatrixMediaTooLarge, download_mxc_capped

    response = _Response([b"12345", b"67890", b"!", b"never-read"])
    with pytest.raises(MatrixMediaTooLarge):
        await download_mxc_capped(_Client(response), "mxc://example.org/big", 10)
    assert response.content.yielded == 3
    assert response.closed


@pytest.mark.asyncio
async def test_oversized_content_length_is_rejected_before_reading():
    from plugins.platforms.matrix.media_download import MatrixMediaTooLarge, download_mxc_capped

    response = _Response([b"x"], headers={"Content-Length": "11"})
    with pytest.raises(MatrixMediaTooLarge):
        await download_mxc_capped(_Client(response), "mxc://example.org/big", 10)
    assert response.content.yielded == 0


def _adapter(monkeypatch, client, max_bytes):
    monkeypatch.setenv("MATRIX_MAX_MEDIA_BYTES", str(max_bytes))
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="syt_secret", extra={
        "homeserver": "https://hs.example.org", "user_id": "@hermes:example.org", "require_mention": False}))
    adapter._startup_ts = time.time() - 10
    adapter._client = client
    adapter.handle_message = AsyncMock()
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
        display_name="DM", room_topic=None, server_name="example.org", chat_type="dm"))
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = lambda *a, **k: None
    return adapter


def _event(content):
    return SimpleNamespace(sender="@alice:example.org", event_id="$m1", room_id="!dm:example.org",
                           timestamp=int(time.time() * 1000), content=content)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    {"msgtype": "m.file", "body": "huge.bin", "url": "mxc://example.org/huge"},
    {"msgtype": "m.file", "body": "huge.bin", "info": {"size": 1},
     "file": {"url": "mxc://example.org/huge", "key": {"k": "k"}, "iv": "iv", "hashes": {"sha256": "h"}}},
], ids=["plain-no-size", "encrypted-underreported-size"])
async def test_oversized_inbound_media_is_dropped_not_cached(monkeypatch, content):
    from gateway.platforms import base

    cache = AsyncMock(return_value="/cache/huge.bin")
    monkeypatch.setattr(base, "cache_document_from_bytes_async", cache)
    response = _Response([b"x" * 8] * 4)
    adapter = _adapter(monkeypatch, _Client(response), max_bytes=16)

    await adapter._on_room_message(_event(content))

    assert response.content.yielded == 3
    cache.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_under_cap_inbound_media_is_cached_and_dispatched(monkeypatch):
    from gateway.platforms import base

    cache = AsyncMock(return_value="/cache/small.bin")
    monkeypatch.setattr(base, "cache_document_from_bytes_async", cache)
    adapter = _adapter(monkeypatch, _Client(_Response([b"small"])), max_bytes=16)

    await adapter._on_room_message(_event({"msgtype": "m.file", "body": "small.bin", "url": "mxc://example.org/s"}))

    cache.assert_awaited_once_with(b"small", "small.bin")
    assert adapter.handle_message.await_args.args[0].media_urls == ["/cache/small.bin"]


class _HttpSession:
    """aiohttp.ClientSession stand-in serving one 200 response to the outbound image fetch."""

    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    def get(self, *_args, **_kwargs):
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize("headers, yielded", [({}, 3), ({"Content-Length": "17"}, 0)],
                         ids=["streamed", "declared"])
async def test_outbound_image_fetch_uses_the_same_cap(monkeypatch, headers, yielded):
    """send_image's external fetch shares read_body_capped with the inbound mxc read."""
    import aiohttp
    from tools import url_safety

    response = _Response([b"x" * 8] * 4, headers=headers)
    response.content_type = "image/png"
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_kw: _HttpSession(response))
    monkeypatch.setattr(url_safety, "is_safe_url", lambda *_a, **_k: True)
    adapter = _adapter(monkeypatch, _Client(response), max_bytes=16)

    with pytest.raises(ValueError, match="exceeds Matrix limit"):
        await adapter._download_external_media_with_cap("https://cdn.example.com/huge.png")
    assert response.content.yielded == yielded
