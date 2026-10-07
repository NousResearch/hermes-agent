"""Wire-level tests for ``_request_limited_response`` (#134222): gzip bodies are decoded once
and ``Accept-Encoding: identity`` is requested by default."""

from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from unittest.mock import patch

import httpx

from plugins.dashboard_auth import _shared


def _stream_via(handler):
    @contextmanager
    def fake_stream(method, url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)).stream(method, url) as response:
            yield response

    return fake_stream


def _stream_via_kwargs(handler):
    """Like ``_stream_via`` but forwards kwargs so the helper's headers reach the wire."""

    @contextmanager
    def fake_stream(method, url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)).stream(method, url, **kwargs) as response:
            yield response

    return fake_stream


def test_gzip_response_is_decoded_once():
    body = gzip.compress(json.dumps({"access_token": "t"}).encode())

    def handler(request):
        return httpx.Response(
            200, headers={"content-encoding": "gzip", "content-type": "application/json"}, content=body)

    with patch("plugins.dashboard_auth._shared.httpx.stream", _stream_via(handler)):
        response = _shared._request_limited_response("POST", "https://portal.example/api/oauth/token")

    assert response.status_code == 200
    assert response.json() == {"access_token": "t"}
    assert "content-encoding" not in response.headers


def _capture_headers(seen):
    def handler(request):
        seen["headers"] = request.headers
        return httpx.Response(200, json={"access_token": "t"})

    return handler


def test_requests_identity_accept_encoding_by_default():
    seen: dict = {}
    with patch("plugins.dashboard_auth._shared.httpx.stream", _stream_via_kwargs(_capture_headers(seen))):
        _shared._request_limited_response("POST", "https://portal.example/api/oauth/token", data={})

    assert seen["headers"]["accept-encoding"] == "identity"


def test_caller_supplied_accept_encoding_wins_in_any_case():
    for caller_headers in ({"Accept-Encoding": "gzip"}, {"ACCEPT-ENCODING": "gzip"}):
        seen: dict = {}
        with patch("plugins.dashboard_auth._shared.httpx.stream", _stream_via_kwargs(_capture_headers(seen))):
            _shared._request_limited_response(
                "POST", "https://portal.example/api/oauth/token", data={}, headers=caller_headers)

        assert seen["headers"].get_list("accept-encoding") == ["gzip"]


def test_caller_headers_survive_the_merge():
    seen: dict = {}
    with patch("plugins.dashboard_auth._shared.httpx.stream", _stream_via_kwargs(_capture_headers(seen))):
        _shared._request_limited_response(
            "POST", "https://portal.example/api/oauth/token", data={},
            headers={"Accept": "application/json", "x-nous-refresh-token": "rt"})

    assert seen["headers"]["accept"] == "application/json"
    assert seen["headers"]["x-nous-refresh-token"] == "rt"
