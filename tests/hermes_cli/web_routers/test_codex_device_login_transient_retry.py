"""Dashboard/Desktop Codex device-code login survives transient transport blips.

The web-surface worker is a deliberate copy of the CLI poll loop (``auth_codex``); it must
share the same retry contract: the poll tolerates a bounded run of consecutive transport
errors (reset by any response) and the one-shot POSTs retry twice, while non-transport
errors still surface immediately without burning retries.
"""

import ssl
from types import SimpleNamespace

import pytest

from hermes_cli.web_routers import oauth as rt_oauth


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


def _fake_httpx(monkeypatch, client):
    """The worker helpers take the ``httpx`` module as a parameter; hand them a scripted one."""
    monkeypatch.setattr(rt_oauth.time, "sleep", lambda *_: None)
    return SimpleNamespace(Client=lambda **kw: client, Timeout=lambda t: t)


def test_poll_survives_transport_blips_and_returns_approval(monkeypatch):
    approved = {"authorization_code": "code", "code_verifier": "verifier"}
    client = _ScriptedClient([
        ssl.SSLEOFError(8, _SSL_EOF_MESSAGE),
        _Response(404),  # still pending — any response resets the blip counter
        ssl.SSLEOFError(8, _SSL_EOF_MESSAGE),
        _Response(200, approved),
    ])
    sess = {"expires_in": 900, "device_auth_id": "dev", "user_code": "ABCD-EFGH", "interval": 3}

    result = rt_oauth._codex_poll_authorization(_fake_httpx(monkeypatch, client), sess, "sid")

    assert result == approved
    assert client.calls == 4


def test_one_shot_posts_retry_blips_but_not_other_errors(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.CODEX_OAUTH_CLIENT_ID", "client-id", raising=False)
    device = {"user_code": "ABCD-EFGH", "device_auth_id": "dev", "interval": "5"}
    client = _ScriptedClient([ssl.SSLEOFError(8, _SSL_EOF_MESSAGE), _Response(200, dict(device))])

    assert rt_oauth._codex_request_user_code(_fake_httpx(monkeypatch, client))["device_auth_id"] == "dev"
    assert client.calls == 2

    client = _ScriptedClient([ValueError("not a network blip"), _Response(200, {"access_token": "t"})])
    with pytest.raises(ValueError):
        rt_oauth._codex_exchange_tokens(
            _fake_httpx(monkeypatch, client), {"authorization_code": "code", "code_verifier": "verifier"})
    assert client.calls == 1
