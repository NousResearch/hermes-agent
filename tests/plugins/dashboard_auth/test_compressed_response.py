"""Compressed OIDC responses are decoded once while retaining the body budget."""
from __future__ import annotations

import gzip
import zlib

import httpx
import pytest

from hermes_cli.dashboard_auth import ProviderError
from plugins.dashboard_auth import _shared as shared


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "identity"])
def test_compressed_response_is_readable(encoding: str, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b'{"issuer":"https://auth.example.com"}'
    wire = gzip.compress(body) if encoding == "gzip" else zlib.compress(body) if encoding == "deflate" else body

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={
            "Content-Encoding": encoding, "Content-Length": str(len(wire)),
            "Content-Type": "application/json", "Cache-Control": "max-age=300",
        }, stream=httpx.ByteStream(wire), request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(shared.httpx, "stream", client.stream)
        response = shared._request_limited_response("GET", "https://auth.example.com/discovery")

    assert response.json() == {"issuer": "https://auth.example.com"}
    assert response.content == body
    assert "content-encoding" not in response.headers
    assert int(response.headers["content-length"]) == len(body)
    assert response.headers["cache-control"] == "max-age=300"
    assert str(response.request.url) == "https://auth.example.com/discovery"


def test_compressed_response_still_bounds_decoded_body(monkeypatch: pytest.MonkeyPatch) -> None:
    wire = gzip.compress(b"x" * (shared._OIDC_RESPONSE_BODY_LIMIT_BYTES + 1))

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={
            "Content-Encoding": "gzip", "Content-Length": str(len(wire)),
        }, stream=httpx.ByteStream(wire), request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(shared.httpx, "stream", client.stream)
        with pytest.raises(ProviderError, match="endpoint response exceeds"):
            shared._request_limited_response("GET", "https://auth.example.com/discovery")
