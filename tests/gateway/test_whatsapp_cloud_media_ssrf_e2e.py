"""Network-level regression tests for WhatsApp Cloud media URL handling."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from gateway.platforms import whatsapp_cloud as wac
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter


ResponseSpec = tuple[int, dict[str, str], bytes]
Responder = Callable[[str], ResponseSpec]


@contextmanager
def _http_server(
    responder: Responder,
) -> Iterator[tuple[str, list[dict[str, str | None]]]]:
    requests: list[dict[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
            })
            status, headers, body = responder(self.path)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = str(server.server_address[0])
    port = int(server.server_address[1])
    try:
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _adapter(client: httpx.AsyncClient) -> WhatsAppCloudAdapter:
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._access_token = "e2e-test-token"
    adapter._api_version = "v20.0"
    adapter._http_client = client
    return adapter


def _metadata_response(temp_url: str) -> ResponseSpec:
    body = json.dumps({"url": temp_url, "mime_type": "image/jpeg"}).encode("utf-8")
    return 200, {"Content-Type": "application/json"}, body


@pytest.mark.asyncio
async def test_private_media_url_is_rejected_before_second_request(
    tmp_path, monkeypatch
):
    """A real metadata response cannot make the adapter call loopback."""

    from tools import url_safety

    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "false")
    monkeypatch.setattr(url_safety, "_allow_private_resolved", False)
    monkeypatch.setattr(url_safety, "_cached_allow_private", False)

    with _http_server(
        lambda _path: (200, {"Content-Type": "image/jpeg"}, b"private")
    ) as (media_base, media_requests):
        with _http_server(
            lambda _path: _metadata_response(f"{media_base}/media.jpg")
        ) as (graph_base, graph_requests):
            monkeypatch.setattr(wac, "GRAPH_API_BASE", graph_base)
            monkeypatch.setattr(wac, "_INBOUND_MEDIA_CACHE", tmp_path)

            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                local_path, mime = await _adapter(client)._download_media_to_cache(
                    "MEDIA123"
                )

    assert local_path is None and mime is None
    assert [request["path"] for request in graph_requests] == ["/v20.0/MEDIA123"]
    assert media_requests == []


@pytest.mark.asyncio
async def test_safe_media_url_uses_real_two_step_download(tmp_path, monkeypatch):
    """The safety boundary preserves an allowed Graph media download."""

    async def allow_test_server(_url: str) -> bool:
        return True

    monkeypatch.setattr(wac, "async_is_safe_url", allow_test_server)
    image = b"\xff\xd8\xff\xe0e2e-jpeg"

    with _http_server(lambda _path: (200, {"Content-Type": "image/jpeg"}, image)) as (
        media_base,
        media_requests,
    ):
        with _http_server(
            lambda _path: _metadata_response(f"{media_base}/media.jpg")
        ) as (graph_base, graph_requests):
            monkeypatch.setattr(wac, "GRAPH_API_BASE", graph_base)
            monkeypatch.setattr(wac, "_INBOUND_MEDIA_CACHE", tmp_path)

            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                local_path, mime = await _adapter(client)._download_media_to_cache(
                    "MEDIA123"
                )

    assert local_path is not None
    assert mime == "image/jpeg"
    assert (tmp_path / "MEDIA123.jpg").read_bytes() == image
    assert [request["path"] for request in graph_requests] == ["/v20.0/MEDIA123"]
    assert [request["path"] for request in media_requests] == ["/media.jpg"]
    assert media_requests[0]["authorization"] == "Bearer e2e-test-token"


@pytest.mark.asyncio
async def test_media_redirect_is_not_followed_with_bearer_token(tmp_path, monkeypatch):
    """A media redirect cannot carry Meta credentials to another origin."""

    async def allow_test_server(_url: str) -> bool:
        return True

    monkeypatch.setattr(wac, "async_is_safe_url", allow_test_server)

    with _http_server(
        lambda _path: (200, {"Content-Type": "text/plain"}, b"capture")
    ) as (capture_base, capture_requests):
        with _http_server(
            lambda _path: (
                302,
                {"Location": f"{capture_base}/capture"},
                b"",
            )
        ) as (media_base, media_requests):
            with _http_server(
                lambda _path: _metadata_response(f"{media_base}/redirect")
            ) as (graph_base, _graph_requests):
                monkeypatch.setattr(wac, "GRAPH_API_BASE", graph_base)
                monkeypatch.setattr(wac, "_INBOUND_MEDIA_CACHE", tmp_path)

                async with httpx.AsyncClient(
                    timeout=5.0, trust_env=False, follow_redirects=True
                ) as client:
                    local_path, mime = await _adapter(client)._download_media_to_cache(
                        "MEDIA123"
                    )

    assert local_path is None and mime is None
    assert media_requests[0]["authorization"] == "Bearer e2e-test-token"
    assert capture_requests == []
