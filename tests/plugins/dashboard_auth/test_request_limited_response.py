"""Regression: a gzip IdP response must not be decoded twice (#134222)."""

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
