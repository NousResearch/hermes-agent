"""End-to-end redirect tests for BlueBubbles attachment downloads."""

from __future__ import annotations

import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter


class _RedirectServer(ThreadingHTTPServer):
    redirect_target: str
    payload: bytes
    requests: list[str]


class _RedirectHandler(BaseHTTPRequestHandler):
    server: _RedirectServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.server.requests.append(self.path)
        if self.path.startswith("/api/v1/attachment/"):
            self.send_response(302)
            self.send_header("Location", self.server.redirect_target)
            self.end_headers()
            return

        body = self.server.payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def _serve(*, redirect_target: str = "/payload") -> Iterator[_RedirectServer]:
    server = _RedirectServer(("127.0.0.1", 0), _RedirectHandler)
    server.redirect_target = redirect_target
    server.payload = b"bluebubbles-attachment"
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _origin(server: _RedirectServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def _adapter(origin: str) -> BlueBubblesAdapter:
    return BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={"server_url": origin, "password": "test-password"},
        )
    )


@pytest.mark.asyncio
async def test_private_same_origin_redirect_reaches_attachment_cache(monkeypatch):
    """A legitimate loopback BlueBubbles deployment must continue to work."""
    with _serve() as server:
        adapter = _adapter(_origin(server))
        adapter.client = httpx.AsyncClient(timeout=5)
        cached: list[bytes] = []
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes",
            lambda data, _ext: cached.append(data) or "/tmp/attachment.png",
        )

        try:
            result = await adapter._download_attachment(
                "download",
                {"mimeType": "image/png", "transferName": "attachment.png"},
            )
        finally:
            await adapter.client.aclose()

    assert result == "/tmp/attachment.png"
    assert cached == [b"bluebubbles-attachment"]
    assert len(server.requests) == 2


@pytest.mark.asyncio
async def test_cross_origin_private_redirect_is_blocked_before_connect(monkeypatch):
    """A redirect may not turn the adapter into a private-network client."""
    with _serve() as foreign_server:
        foreign_url = f"{_origin(foreign_server)}/payload"
        with _serve(redirect_target=foreign_url) as bluebubbles_server:
            adapter = _adapter(_origin(bluebubbles_server))
            adapter.client = httpx.AsyncClient(timeout=5)
            monkeypatch.setattr(
                "gateway.platforms.bluebubbles.cache_image_from_bytes",
                lambda *_args: pytest.fail("blocked response reached the cache"),
            )

            try:
                result = await adapter._download_attachment(
                    "download",
                    {"mimeType": "image/png", "transferName": "attachment.png"},
                )
            finally:
                await adapter.client.aclose()

    assert result is None
    assert foreign_server.requests == []


@pytest.mark.asyncio
async def test_cross_origin_public_redirect_uses_guarded_client(monkeypatch):
    """A legitimate public attachment host remains reachable through the guard."""

    class _GuardedClient:
        def __init__(self) -> None:
            self.requests: list[str] = []
            self.closed = False

        async def get(self, url: str, **kwargs: object) -> httpx.Response:
            self.requests.append(url)
            assert kwargs["follow_redirects"] is False
            request = httpx.Request("GET", url)
            return httpx.Response(
                200,
                content=b"public-attachment",
                request=request,
            )

        async def aclose(self) -> None:
            self.closed = True

    guarded = _GuardedClient()
    public_url = "https://cdn.example.test/attachment.png"
    with _serve(redirect_target=public_url) as bluebubbles_server:
        adapter = _adapter(_origin(bluebubbles_server))
        adapter.client = httpx.AsyncClient(timeout=5)
        cached: list[bytes] = []
        monkeypatch.setattr(
            "tools.url_safety.create_ssrf_safe_async_client",
            lambda **_kwargs: guarded,
        )
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes",
            lambda data, _ext: cached.append(data) or "/tmp/public.png",
        )

        try:
            result = await adapter._download_attachment(
                "download",
                {"mimeType": "image/png", "transferName": "attachment.png"},
            )
        finally:
            await adapter.client.aclose()

    assert result == "/tmp/public.png"
    assert cached == [b"public-attachment"]
    assert guarded.requests == [public_url]
    assert guarded.closed is True
