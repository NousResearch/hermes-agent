"""Codex device-login survives transient transport blips.

A single dropped connection (e.g. an SSL EOF between polls) used to abort the whole
device-code flow and waste the browser approval the user had already completed. Transport
errors are now retried — the poll loop tolerates a bounded run of consecutive blips and the
one-shot device-login POSTs retry twice — while the terminal contract is unchanged: a typed
``AuthError`` with the original cause chained and the TLS hint, and non-transport errors
still fail immediately without burning retries.
"""

import ssl

import httpx
import pytest

from hermes_cli import auth_codex
from hermes_cli.auth import AuthError


_SSL_EOF_MESSAGE = (
    "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)")


class _ScriptedClient:
    """Context-manager ``httpx.Client`` stand-in: each ``post`` pops the next scripted step —
    a ``BaseException`` is raised, anything else is returned as the response."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def post(self, url, **kwargs):
        self.calls += 1
        step = self._script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _install(monkeypatch, client):
    monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **kw: client)
    monkeypatch.setattr(auth_codex.time, "sleep", lambda *_: None)


def _login_post():
    return auth_codex._codex_login_post(
        "https://auth.openai.com/api/accounts/deviceauth/usercode",
        failure=("Failed to request device code", "device_code_request_failed"))


def _poll():
    return auth_codex._codex_poll_authorization_code(
        "https://auth.openai.com", device_auth_id="da", user_code="uc", poll_interval=0)


def test_poll_survives_transport_blips_but_not_other_errors(monkeypatch):
    approval = _Response(200, {"authorization_code": "ac", "code_verifier": "cv"})
    client = _ScriptedClient([
        ssl.SSLEOFError(8, _SSL_EOF_MESSAGE), httpx.ConnectError("connection reset"), approval])
    _install(monkeypatch, client)

    assert _poll() == {"authorization_code": "ac", "code_verifier": "cv"}
    assert client.calls == 3

    bug = ValueError("decode bug")
    client = _ScriptedClient([bug, approval])
    _install(monkeypatch, client)
    with pytest.raises(AuthError) as excinfo:
        _poll()
    assert excinfo.value.__cause__ is bug
    assert client.calls == 1  # a non-transport exception is a bug, never retried


def test_login_post_retries_bounded_and_keeps_typed_failure(monkeypatch):
    client = _ScriptedClient(
        [ssl.SSLEOFError(8, _SSL_EOF_MESSAGE), _Response(200, {"user_code": "uc"})])
    _install(monkeypatch, client)

    assert _login_post().status_code == 200
    assert client.calls == 2

    exc = ssl.SSLEOFError(8, _SSL_EOF_MESSAGE)
    client = _ScriptedClient([exc] * 4)
    _install(monkeypatch, client)
    with pytest.raises(AuthError) as excinfo:
        _login_post()
    err = excinfo.value
    assert err.code == "device_code_request_failed"
    assert "UNEXPECTED_EOF_WHILE_READING" in str(err) and "OPENSSL_CONF" in str(err)
    assert err.__cause__ is exc
    assert client.calls == 3  # capped, not an endless retry
