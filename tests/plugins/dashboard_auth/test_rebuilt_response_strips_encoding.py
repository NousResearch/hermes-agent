"""#134462 — a rebuilt ``_request_limited_response`` must not carry the streamed
response's content-encoding over to the already-decompressed body.

The helper reads the IdP response via ``iter_bytes()``, which transparently
decompresses (gzip/deflate/br), and repacks the chunks into a fresh
``httpx.Response``. Keeping ``content-encoding: gzip`` from the streamed
response made httpx decompress the plain bytes a second time on ``.json()`` —
the Portal's gzipped token responses surfaced as "Provider unreachable:
Portal token endpoint unreachable: Error -3 while decompressing data:
incorrect header check" and locked users out of the dashboard. A compressed
CDN answer defeats the ``Accept-Encoding: identity`` default too, so the
rebuild itself must be correct for any encoding.

All HTTP is mocked: nothing in this file talks to a real IdP.
"""

from __future__ import annotations

import gzip
import json
from typing import Any, Callable

import httpx
import pytest

from plugins.dashboard_auth._shared import _request_limited_response

_TOKEN_URL = "https://portal.example/api/oauth/token"
_TOKEN_BODY = {"access_token": "t", "token_type": "bearer"}


def _mock_stream(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., Any]:
    """``httpx.stream`` stand-in backed by ``MockTransport``, so the helper's
    real size-cap / rebuild logic runs against a scripted response."""

    def stream(method: str, url: str, **kwargs: Any) -> Any:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return client.stream(method, url, **kwargs)

    return stream


def _gzipped_token_response() -> httpx.Response:
    return httpx.Response(
        200,
        content=gzip.compress(json.dumps(_TOKEN_BODY).encode("utf-8")),
        headers={"content-encoding": "gzip"},
    )


def test_rebuilt_gzipped_response_decodes_without_double_decompression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        httpx, "stream", _mock_stream(lambda r: _gzipped_token_response())
    )

    response = _request_limited_response("POST", _TOKEN_URL, data={})

    # Before the fix this raised zlib.error "Error -3 ... incorrect header check".
    assert response.json() == _TOKEN_BODY
    assert response.text == json.dumps(_TOKEN_BODY)


def test_rebuilt_response_drops_transfer_encoding_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        httpx, "stream", _mock_stream(lambda r: _gzipped_token_response())
    )

    response = _request_limited_response("POST", _TOKEN_URL, data={})

    # The body is decompressed bytes; the headers must not claim otherwise.
    assert "content-encoding" not in response.headers
    # content-length described the compressed size; httpx recomputes it.
    assert response.headers["content-length"] == str(len(response.content))


def test_uncompressed_response_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_TOKEN_BODY,
            headers={"content-type": "application/json", "x-custom": "kept"},
        )

    monkeypatch.setattr(httpx, "stream", _mock_stream(handler))

    response = _request_limited_response("POST", _TOKEN_URL, data={})

    assert response.json() == _TOKEN_BODY
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-custom"] == "kept"
    assert response.headers["content-length"] == str(len(response.content))
