"""Tests for bounded HTTP response reads in Discord image/download paths.

Companion to #60122, #60112 (REST body bounding) — extends the same
resource-limiting pattern to image/animation/attachment downloads
in the Discord adapter that were left unbounded.
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import plugins.platforms.discord.adapter as discord_adapter
from plugins.platforms.discord.adapter import (
    _create_discord_image_http_client,
    _read_response_bytes_bounded,
    _DISCORD_IMAGE_DOWNLOAD_MAX_BYTES,
    DiscordAdapter,
)
from plugins.platforms.discord.outbound_image_fetch import (
    _DISCORD_IMAGE_DECODED_READ_CHUNK_MAX_BYTES,
)
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from tools.url_safety import (
    SSRFConnectionBlocked,
    _reset_allow_private_cache,
)


class _CaseInsensitiveHeaders(dict[str, str]):
    """Small case-insensitive header mapping for the fake response."""

    def __init__(self, values: dict[str, str]):
        super().__init__((str(key).lower(), value) for key, value in values.items())

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(str(key).lower(), default)


class _FakeResponseContent:
    """Deterministic byte source that returns one available chunk per read."""

    def __init__(self, chunks: tuple[bytes, ...]):
        self._chunks = chunks
        self._next_index = 0
        self.consumed_chunks = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self._next_index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._next_index]
        self._next_index += 1
        self.consumed_chunks += 1
        return chunk

    async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
        while self._next_index < len(self._chunks):
            yield await self.read()


class _FakeResponse:
    status = 200

    def __init__(
        self,
        chunks: tuple[bytes, ...],
        headers: dict[str, str] | None = None,
        close_error: Exception | None = None,
    ):
        self.headers = _CaseInsensitiveHeaders(headers or {})
        self.content = _FakeResponseContent(chunks)
        self.read_called = False
        self.close_called = 0
        self._close_error = close_error

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def aiter_bytes(
        self, *, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        async for chunk in self.content.iter_chunked(64 * 1024):
            yield chunk

    async def read(self) -> bytes:
        """Model an unbounded response read for the regression assertion."""
        self.read_called = True
        chunks = []
        async for chunk in self.content.iter_chunked(64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        self.close_called += 1
        if self._close_error is not None:
            raise self._close_error


class _ChunkSizeAwareResponse:
    """Response fake that records the decoded iterator chunk-size request."""

    def __init__(self, chunks: tuple[bytes, ...]):
        self.headers: dict[str, str] = {}
        self._chunks = chunks
        self.requested_chunk_size: int | None = None
        self.close_called = 0

    async def aiter_bytes(self, *, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        self.requested_chunk_size = chunk_size
        for chunk in self._chunks:
            yield chunk

    def close(self) -> None:
        self.close_called += 1


class _ReleaseOnlyResponse:
    """Response variant exposing release() but no close()."""

    def __init__(self, chunks: tuple[bytes, ...]):
        self.headers: dict[str, str] = {}
        self.content = _FakeResponseContent(chunks)
        self.release_called = 0

    async def aiter_bytes(
        self, *, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        async for chunk in self.content.iter_chunked(64 * 1024):
            yield chunk

    def release(self) -> None:
        self.release_called += 1


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
        assert method == "GET"
        self.calls.append((url, kwargs))
        return self.response


_IMAGE_URL = "https://cdn.example.test/image.png"


async def _allow_safe_url(_url: str) -> bool:
    return True


def _read_url_image(response: _FakeResponse) -> tuple[int, bytes, dict[str, str]]:
    session = _FakeSession(response)
    timeout = object()
    result = asyncio.run(
        discord_adapter._read_url_image_with_redirect_guard(
            session,
            _IMAGE_URL,
            timeout=timeout,
            request_kwargs={},
        )
    )
    assert len(session.calls) == 1
    requested_url, request_kwargs = session.calls[0]
    assert requested_url == _IMAGE_URL
    assert request_kwargs["timeout"] is timeout
    assert request_kwargs["follow_redirects"] is False
    return result


class _ImageBodyResponse:
    status_code = 200

    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self.body = body

    async def __aenter__(self) -> "_ImageBodyResponse":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def aiter_bytes(
        self, *, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        yield self.body


class _ImageBodyClient:
    def __init__(self, response: _ImageBodyResponse):
        self.response = response

    def stream(self, method: str, url: str, **kwargs: Any) -> _ImageBodyResponse:
        assert method == "GET"
        assert kwargs["follow_redirects"] is False
        return self.response

    async def __aenter__(self) -> "_ImageBodyClient":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def aclose(self) -> None:
        return None


def _make_send_image_adapter(client: _ImageBodyClient, *, forum: bool = False):
    channel = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(id="sent")),
    )
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(return_value=channel),
    )
    adapter._is_forum_parent = MagicMock(return_value=forum)
    return adapter, channel


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "extension"),
    [
        (b"\x89PNG\r\n\x1a\npng-body", "png"),
        (b"\xff\xd8\xff\xe0jpeg-body", "jpg"),
        (b"GIF87agif-body", "gif"),
        (b"GIF89agif-body", "gif"),
        (b"RIFF\x00\x00\x00\x00WEBPwebp-body", "webp"),
    ],
)
async def test_send_image_uses_image_magic_bytes_for_filename(
    monkeypatch, body: bytes, extension: str
):
    response = _ImageBodyResponse(body, {"Content-Type": "image/png"})
    client = _ImageBodyClient(response)
    adapter, channel = _make_send_image_adapter(client)
    file_cls = MagicMock()

    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    monkeypatch.setattr(
        discord_adapter,
        "_create_discord_image_http_client",
        lambda _proxy: client,
    )
    monkeypatch.setattr(discord_adapter.discord, "File", file_cls)

    result = await adapter.send_image("123", "https://cdn.example.test/asset.png")

    assert result.success
    assert file_cls.call_args.kwargs["filename"] == f"image.{extension}"
    channel.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"<html>not an image</html>", b'{"error":"nope"}'])
async def test_send_image_does_not_upload_non_image_bytes_despite_image_content_type(
    monkeypatch, body: bytes
):
    response = _ImageBodyResponse(body, {"Content-Type": "image/png"})
    client = _ImageBodyClient(response)
    adapter, channel = _make_send_image_adapter(client)
    file_cls = MagicMock()

    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    monkeypatch.setattr(
        discord_adapter,
        "_create_discord_image_http_client",
        lambda _proxy: client,
    )
    monkeypatch.setattr(discord_adapter.discord, "File", file_cls)

    result = await adapter.send_image("123", "https://cdn.example.test/asset.png")

    assert result.success
    file_cls.assert_not_called()
    assert all(
        "file" not in call.kwargs and "files" not in call.kwargs
        for call in channel.send.await_args_list
    )


@pytest.mark.asyncio
async def test_send_animation_requires_gif_magic(monkeypatch):
    response = _ImageBodyResponse(
        b"\x89PNG\r\n\x1a\nnot-a-gif",
        {"Content-Type": "image/gif"},
    )
    client = _ImageBodyClient(response)
    adapter, channel = _make_send_image_adapter(client)
    file_cls = MagicMock()

    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    monkeypatch.setattr(
        discord_adapter,
        "_create_discord_image_http_client",
        lambda _proxy: client,
    )
    monkeypatch.setattr(discord_adapter.discord, "File", file_cls)

    result = await adapter.send_animation("123", "https://cdn.example.test/asset.gif")

    assert result.success
    file_cls.assert_not_called()
    assert all(
        "file" not in call.kwargs and "files" not in call.kwargs
        for call in channel.send.await_args_list
    )


@pytest.mark.asyncio
async def test_forum_starter_does_not_receive_invalid_image_body(monkeypatch):
    response = _ImageBodyResponse(
        b"<html>not an image</html>",
        {"Content-Type": "image/jpeg"},
    )
    client = _ImageBodyClient(response)
    adapter, _channel = _make_send_image_adapter(client, forum=True)
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    adapter._forum_post_file = AsyncMock()
    file_cls = MagicMock()

    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    monkeypatch.setattr(
        discord_adapter,
        "_create_discord_image_http_client",
        lambda _proxy: client,
    )
    monkeypatch.setattr(discord_adapter.discord, "File", file_cls)

    result = await adapter.send_image("123", "https://cdn.example.test/asset.jpg")

    assert result.success
    file_cls.assert_not_called()
    adapter._forum_post_file.assert_not_awaited()
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_url_fetch_blocks_dns_rebinding_before_raw_connect(monkeypatch):
    """The Discord fetch path must validate the IP immediately before connect."""
    from httpcore._backends.auto import AutoBackend

    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(proxy_var, raising=False)
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    _reset_allow_private_cache()

    resolution_results: list[str] = []

    def rebinding_getaddrinfo(host, port, *args, **kwargs):
        del host, args, kwargs
        ip = "93.184.216.34" if not resolution_results else "10.0.0.8"
        resolution_results.append(ip)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 443))]

    raw_connect_attempts: list[tuple[str, int]] = []

    async def raw_connect_tcp(
        self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        del self, timeout, local_address, socket_options
        raw_connect_attempts.append((host, port))
        raise AssertionError("raw network backend must not be called")

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)
    monkeypatch.setattr(AutoBackend, "connect_tcp", raw_connect_tcp)

    async with discord_adapter._create_discord_image_http_client() as client:
        with pytest.raises(SSRFConnectionBlocked, match="private/internal"):
            await discord_adapter._read_url_image_with_redirect_guard(
                client,
                _IMAGE_URL,
                timeout=1.0,
                request_kwargs={},
            )

    assert resolution_results == ["93.184.216.34", "10.0.0.8"]
    assert raw_connect_attempts == []


def test_image_client_passes_explicit_proxy_to_ssrf_safe_client(monkeypatch):
    captured_kwargs: dict[str, Any] = {}
    sentinel_client = object()

    def fake_create(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return sentinel_client

    monkeypatch.setattr(discord_adapter, "create_ssrf_safe_async_client", fake_create)

    result = _create_discord_image_http_client("http://discord-proxy.test:8080")

    assert result is sentinel_client
    assert captured_kwargs["proxy"] == "http://discord-proxy.test:8080"
    assert captured_kwargs["trust_env"] is False
    assert captured_kwargs["follow_redirects"] is False
    assert captured_kwargs["timeout"] == 30.0


class TestReadResponseBytesBounded:
    def test_requests_bounded_chunk_size_and_closes_on_overflow(self):
        response_limit = 4
        response = _ChunkSizeAwareResponse((b"abc", b"de"))
        result = None

        with pytest.raises(ValueError, match="exceeded 4 bytes"):
            result = asyncio.run(
                _read_response_bytes_bounded(response, response_limit)
            )

        assert response.requested_chunk_size is not None
        assert response.requested_chunk_size > 0
        assert (
            response.requested_chunk_size
            <= _DISCORD_IMAGE_DECODED_READ_CHUNK_MAX_BYTES
        )
        assert response.requested_chunk_size <= response_limit
        assert response.close_called == 1
        assert result is None

    def test_reads_all_chunks_within_limit(self):
        resp = _FakeResponse((b"xx", b"yyy"))

        result = asyncio.run(_read_response_bytes_bounded(resp, 6))

        assert result == b"xxyyy"
        assert resp.content.consumed_chunks == 2
        assert resp.close_called == 0

    def test_raises_on_aggregate_overflow(self):
        resp = _FakeResponse((b"xxx", b"y"))

        with pytest.raises(ValueError, match="exceeded 3 bytes"):
            asyncio.run(_read_response_bytes_bounded(resp, 3))

        assert resp.close_called == 1
        assert resp.content.consumed_chunks == 2

    def test_close_failure_does_not_mask_size_error(self):
        resp = _FakeResponse((b"xxxx",), close_error=RuntimeError("close failed"))

        with pytest.raises(ValueError, match="exceeded 3 bytes"):
            asyncio.run(_read_response_bytes_bounded(resp, 3))

        assert resp.close_called == 1

    def test_release_is_used_when_close_is_unavailable(self):
        resp = _ReleaseOnlyResponse((b"xxxx",))

        with pytest.raises(ValueError, match="exceeded 3 bytes"):
            asyncio.run(_read_response_bytes_bounded(resp, 3))

        assert resp.release_called == 1

    def test_exact_limit_passes(self):
        resp = _FakeResponse((b"x" * 2, b"y" * 3))

        result = asyncio.run(_read_response_bytes_bounded(resp, 5))

        assert result == b"xxyyy"
        assert resp.content.consumed_chunks == 2


def test_content_length_over_limit_is_rejected_before_body_consumption(monkeypatch):
    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    monkeypatch.setattr(discord_adapter, "_DISCORD_IMAGE_DOWNLOAD_MAX_BYTES", 4)
    response = _FakeResponse(
        (b"body must not be consumed",),
        {"Content-Length": "5"},
    )

    with pytest.raises(ValueError, match="exceeded 4 bytes"):
        _read_url_image(response)

    assert response.content.consumed_chunks == 0
    assert response.read_called is False
    assert response.close_called == 1


def test_missing_content_length_rejects_aggregate_overflow(monkeypatch):
    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    monkeypatch.setattr(discord_adapter, "_DISCORD_IMAGE_DOWNLOAD_MAX_BYTES", 4)
    response = _FakeResponse((b"AA", b"BBB"))

    with pytest.raises(ValueError, match="exceeded 4 bytes"):
        _read_url_image(response)

    assert response.content.consumed_chunks == 2
    assert response.close_called == 1


def test_underreported_content_length_rejects_later_chunk_overflow(monkeypatch):
    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    monkeypatch.setattr(discord_adapter, "_DISCORD_IMAGE_DOWNLOAD_MAX_BYTES", 4)
    response = _FakeResponse(
        (b"AAA", b"BB"),
        {"Content-Length": "3"},
    )

    with pytest.raises(ValueError, match="exceeded 4 bytes"):
        _read_url_image(response)

    assert response.content.consumed_chunks == 2
    assert response.close_called == 1


def test_multi_chunk_response_at_exact_limit_is_returned_intact(monkeypatch):
    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    monkeypatch.setattr(discord_adapter, "_DISCORD_IMAGE_DOWNLOAD_MAX_BYTES", 4)
    response = _FakeResponse(
        (b"AA", b"BB"),
        {"Content-Length": "4"},
    )

    status, body, headers = _read_url_image(response)

    assert status == 200
    assert body == b"AABB"
    assert headers["content-length"] == "4"
    assert response.content.consumed_chunks == 2
    assert response.read_called is False
    assert response.close_called == 0


def test_redirect_guard_keeps_ten_redirect_limit_and_disables_auto_follow(monkeypatch):
    monkeypatch.setattr(discord_adapter, "async_is_safe_url", _allow_safe_url)
    response = _FakeResponse((b"must not be read",), {"Location": _IMAGE_URL})
    response.status = 302
    session = _FakeSession(response)

    with pytest.raises(ValueError, match="Too many image URL redirects"):
        asyncio.run(
            discord_adapter._read_url_image_with_redirect_guard(
                session,
                _IMAGE_URL,
                timeout=30.0,
                request_kwargs={},
            )
        )

    assert len(session.calls) == 11
    assert all(
        request_kwargs["follow_redirects"] is False
        for _url, request_kwargs in session.calls
    )


class TestImageDownloadLimits:
    def test_outbound_image_limit_is_50_mib(self):
        assert _DISCORD_IMAGE_DOWNLOAD_MAX_BYTES == 50 * 1024 * 1024
