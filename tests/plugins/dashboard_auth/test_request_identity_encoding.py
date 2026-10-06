"""Wire-level tests for ``_request_limited_response`` (#134222).

The provider suites patch this helper out entirely; these tests pin its
transport behaviour — in particular the default ``Accept-Encoding: identity``
that keeps a CDN-served body httpx cannot decompress from surfacing as
"token endpoint unreachable: Error -3 while decompressing data".

All HTTP is mocked: nothing in this file talks to a real IdP.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx
import pytest

from plugins.dashboard_auth._shared import _request_limited_response


def _mock_stream(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., Any]:
    """``httpx.stream`` stand-in backed by ``MockTransport``, so the helper's
    real header-merge / size-cap logic runs against a scripted response."""

    def stream(method: str, url: str, **kwargs: Any) -> Any:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return client.stream(method, url, **kwargs)

    return stream


def _handler(
    seen: dict, response: httpx.Response | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        return response or httpx.Response(
            200, json={"access_token": "t", "token_type": "bearer"}
        )

    return handler


def test_requests_identity_accept_encoding_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}
    monkeypatch.setattr(httpx, "stream", _mock_stream(_handler(seen)))

    response = _request_limited_response(
        "POST", "https://portal.example/api/oauth/token", data={}
    )

    assert response.status_code == 200
    assert seen["headers"]["accept-encoding"] == "identity"


def test_caller_supplied_accept_encoding_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # Case variants must override the default too — a plain dict merge would emit
    # two Accept-Encoding headers and leave the decompression bug in place.
    for caller_headers in ({"Accept-Encoding": "gzip"}, {"ACCEPT-ENCODING": "gzip"}):
        seen: dict = {}
        monkeypatch.setattr(httpx, "stream", _mock_stream(_handler(seen)))

        _request_limited_response(
            "POST",
            "https://portal.example/api/oauth/token",
            data={},
            headers=caller_headers,
        )

        assert seen["headers"].get_list("accept-encoding") == ["gzip"]


def test_caller_headers_survive_the_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(httpx, "stream", _mock_stream(_handler(seen)))

    _request_limited_response(
        "POST",
        "https://portal.example/api/oauth/token",
        data={},
        headers={"Accept": "application/json", "Authorization": "Basic abc"},
    )

    assert seen["headers"]["accept"] == "application/json"
    assert seen["headers"]["authorization"] == "Basic abc"
    assert seen["headers"]["accept-encoding"] == "identity"
