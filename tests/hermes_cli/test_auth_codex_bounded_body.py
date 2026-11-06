"""Codex OAuth/device-auth responses are read through a 1 MiB body cap (#55253).

A hostile or broken auth endpoint/proxy answering 200 with megabytes of "JSON" used to be
fully buffered and parsed by every ``client.post(...).json()`` in the CLI device-code flow,
the dashboard login worker and ``refresh_codex_oauth_pure``. The CLI builds its client via
``auth_codex._codex_http_client`` and the dashboard via ``web_routers.oauth._codex_client``; both
install the response hook that cuts the read off at the cap.
"""
from __future__ import annotations

import functools
import json

import httpx
import pytest

from hermes_cli import auth_codex
from hermes_cli.auth import AuthError
from hermes_cli.web_routers import oauth as web_oauth


class _LazyBody(httpx.SyncByteStream):
    """Like a socket: chunks are only produced when pulled, and the pull count is observable."""

    def __init__(self, data: bytes, pulled: list) -> None:
        self._data, self._pulled = data, pulled

    def __iter__(self):
        for i in range(0, len(self._data), 65536):
            self._pulled[0] += len(self._data[i:i + 65536])
            yield self._data[i:i + 65536]

    def close(self) -> None:
        pass


def _serve(monkeypatch, payload: dict) -> list:
    pulled = [0]
    body = json.dumps(payload).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=_LazyBody(body, pulled))

    monkeypatch.setattr(httpx, "Client", functools.partial(httpx.Client, transport=httpx.MockTransport(handler)))
    return pulled


def test_oversized_200_auth_body_is_rejected_before_being_buffered(monkeypatch):
    pulled = _serve(monkeypatch, {"access_token": "at", "refresh_token": "rt", "pad": "a" * (3 * 1024 * 1024)})

    with pytest.raises(AuthError) as excinfo:
        auth_codex.refresh_codex_oauth_pure("old-at", "old-rt")
    assert excinfo.value.code == "codex_auth_response_too_large"
    # Stopped within one chunk of the cap, not the full 3 MiB.
    assert auth_codex._CODEX_AUTH_BODY_MAX_BYTES < pulled[0] <= auth_codex._CODEX_AUTH_BODY_MAX_BYTES + 65536

    with pytest.raises(AuthError, match="exceeded 1024 KiB"):
        web_oauth._codex_exchange_tokens(httpx, {"authorization_code": "c", "code_verifier": "v"})

    # The dashboard poll loop holds its own long-lived client; it must be built with the same cap.
    monkeypatch.setattr(web_oauth.time, "sleep", lambda *_: None)
    sess = {"expires_in": 900, "device_auth_id": "dev", "user_code": "ABCD-EFGH", "interval": 3}
    with pytest.raises(AuthError, match="exceeded 1024 KiB"):
        web_oauth._codex_poll_authorization(httpx, sess, "sid")


def test_normal_auth_body_still_parses(monkeypatch):
    _serve(monkeypatch, {"access_token": "at-new", "refresh_token": "rt-new"})

    refreshed = auth_codex.refresh_codex_oauth_pure("old-at", "old-rt")
    assert refreshed["access_token"] == "at-new"
    assert web_oauth._codex_exchange_tokens(httpx, {"authorization_code": "c", "code_verifier": "v"}) == {
        "access_token": "at-new", "refresh_token": "rt-new"}
