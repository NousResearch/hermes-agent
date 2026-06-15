"""Coverage for the xAI OAuth device-code login flow (RFC 8628).

NousResearch/hermes-agent previously supported xAI Grok OAuth only via the
PKCE loopback flow (``--manual-paste`` for browser-only remotes). That flow
binds 127.0.0.1 on the box and/or expects the operator to copy a callback URL
out of a browser — neither works cleanly on a headless, SSH-only VPS.

This adds a device-authorization-grant path (``hermes auth add xai-oauth
--device-code`` / ``hermes model --device-code``) mirroring:

* Hermes's own OpenAI-Codex device-code login (``_codex_device_code_login``).
* OpenClaw's ``xai-device-code`` method (extensions/xai/xai-oauth.ts), which
  hits ``device_authorization_endpoint`` then polls ``token_endpoint`` with
  ``grant_type=urn:ietf:params:oauth:grant-type:device_code``.

These tests pin the wire contract (no PKCE on the device path), the RFC 8628
polling semantics (authorization_pending / slow_down / terminal errors), the
loopback-compatible return shape, and the mutually-exclusive flag guard. None
of them touch the network.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import httpx
import pytest

from hermes_cli.auth import (
    AuthError,
    XAI_DEVICE_CODE_GRANT_TYPE,
    XAI_OAUTH_CLIENT_ID,
    XAI_OAUTH_SCOPE,
    _login_xai_oauth,
    _xai_oauth_device_code_login,
    _xai_oauth_poll_device_token,
    _xai_oauth_request_device_code,
)


# ---------------------------------------------------------------------------
# httpx recorders (no network)
# ---------------------------------------------------------------------------


class _PostRecorder:
    """Capture ``httpx.post`` calls and replay a fixed response."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url, *, headers=None, data=None, timeout=None, **kw):
        self.calls.append(
            {"url": url, "headers": headers or {}, "data": data or {}}
        )
        return self.response


class _ClientPostSequence:
    """A fake ``httpx.Client`` whose ``.post`` replays a fixed sequence.

    Used to drive ``_xai_oauth_poll_device_token`` through its
    authorization_pending / slow_down / success states deterministically.
    """

    def __init__(self, responses: List[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __enter__(self) -> "_ClientPostSequence":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def post(self, url, *, headers=None, data=None, **kw):
        self.calls.append({"url": url, "headers": headers or {}, "data": data or {}})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _err(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, json=payload)


# ---------------------------------------------------------------------------
# Device-code request: no PKCE, just client_id + scope
# ---------------------------------------------------------------------------


@pytest.fixture
def device_code_post(monkeypatch):
    recorder = _PostRecorder(
        _ok(
            {
                "device_code": "DC-123",
                "user_code": "WXYZ-1234",
                "verification_uri": "https://accounts.x.ai/device",
                "verification_uri_complete": "https://accounts.x.ai/device?code=WXYZ-1234",
                "expires_in": 900,
                "interval": 5,
            }
        )
    )
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)
    return recorder


def test_device_code_request_sends_client_id_and_scope(device_code_post):
    """The device-authorization request is a public-client call — only
    ``client_id`` + ``scope`` go on the wire, never a PKCE verifier."""
    _xai_oauth_request_device_code(
        device_authorization_endpoint="https://auth.x.ai/oauth2/device/code",
    )
    sent = device_code_post.calls[-1]["data"]
    assert sent["client_id"] == XAI_OAUTH_CLIENT_ID
    assert sent["scope"] == XAI_OAUTH_SCOPE
    assert "code_verifier" not in sent
    assert "code_challenge" not in sent


def test_device_code_request_uses_form_urlencoded(device_code_post):
    _xai_oauth_request_device_code(
        device_authorization_endpoint="https://auth.x.ai/oauth2/device/code",
    )
    headers = device_code_post.calls[-1]["headers"]
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"


def test_device_code_request_rejects_incomplete_response(monkeypatch):
    recorder = _PostRecorder(_ok({"user_code": "ABCD"}))  # no device_code
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)
    with pytest.raises(AuthError) as ei:
        _xai_oauth_request_device_code(
            device_authorization_endpoint="https://auth.x.ai/oauth2/device/code",
        )
    assert ei.value.code == "xai_device_code_incomplete"


# ---------------------------------------------------------------------------
# Token poll: device-code grant + RFC 8628 §3.5 error handling
# ---------------------------------------------------------------------------


def test_poll_uses_device_code_grant(monkeypatch):
    """The poll body must carry the device_code grant + client_id +
    device_code, and never a PKCE code/verifier."""
    seq = _ClientPostSequence([_ok({"access_token": "AT", "refresh_token": "RT"})])
    monkeypatch.setattr("hermes_cli.auth.httpx.Client", lambda *a, **k: seq)
    # Patch sleep so the test doesn't wait for the poll interval.
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    payload = _xai_oauth_poll_device_token(
        token_endpoint="https://auth.x.ai/oauth2/token",
        device_code="DC-123",
        expires_in=900,
        poll_interval=5,
    )
    assert payload["access_token"] == "AT"
    sent = seq.calls[-1]["data"]
    assert sent["grant_type"] == XAI_DEVICE_CODE_GRANT_TYPE
    assert sent["client_id"] == XAI_OAUTH_CLIENT_ID
    assert sent["device_code"] == "DC-123"
    assert "code_verifier" not in sent


def test_poll_waits_through_authorization_pending(monkeypatch):
    """``authorization_pending`` keeps polling; success on a later attempt
    is returned."""
    seq = _ClientPostSequence(
        [
            _err(400, {"error": "authorization_pending"}),
            _err(400, {"error": "authorization_pending"}),
            _ok({"access_token": "AT", "refresh_token": "RT"}),
        ]
    )
    monkeypatch.setattr("hermes_cli.auth.httpx.Client", lambda *a, **k: seq)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    payload = _xai_oauth_poll_device_token(
        token_endpoint="https://auth.x.ai/oauth2/token",
        device_code="DC-123",
        expires_in=900,
        poll_interval=1,
    )
    assert payload["refresh_token"] == "RT"
    assert len(seq.calls) == 3


def test_poll_raises_on_access_denied(monkeypatch):
    seq = _ClientPostSequence([_err(400, {"error": "access_denied"})])
    monkeypatch.setattr("hermes_cli.auth.httpx.Client", lambda *a, **k: seq)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    with pytest.raises(AuthError) as ei:
        _xai_oauth_poll_device_token(
            token_endpoint="https://auth.x.ai/oauth2/token",
            device_code="DC-123",
            expires_in=900,
            poll_interval=1,
        )
    assert ei.value.code == "xai_device_access_denied"


def test_poll_raises_on_expired_token(monkeypatch):
    seq = _ClientPostSequence([_err(400, {"error": "expired_token"})])
    monkeypatch.setattr("hermes_cli.auth.httpx.Client", lambda *a, **k: seq)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    with pytest.raises(AuthError) as ei:
        _xai_oauth_poll_device_token(
            token_endpoint="https://auth.x.ai/oauth2/token",
            device_code="DC-123",
            expires_in=900,
            poll_interval=1,
        )
    assert ei.value.code == "xai_device_code_expired"


# ---------------------------------------------------------------------------
# Orchestrator: loopback-compatible return shape
# ---------------------------------------------------------------------------


def test_device_code_login_returns_loopback_compatible_shape(monkeypatch):
    """``_xai_oauth_device_code_login`` must return the same dict shape as
    ``_xai_oauth_loopback_login`` so the shared persistence path
    (``_save_xai_oauth_tokens``) works unchanged."""
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_device_discovery",
        lambda *_a, **_k: {
            "device_authorization_endpoint": "https://auth.x.ai/oauth2/device/code",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_request_device_code",
        lambda *_a, **_k: {
            "device_code": "DC-123",
            "user_code": "WXYZ-1234",
            "verification_uri": "https://accounts.x.ai/device",
            "expires_in": 900,
            "interval": 5,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_poll_device_token",
        lambda *_a, **_k: {
            "access_token": "AT-dev",
            "refresh_token": "RT-dev",
            "id_token": "ID",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    # Never try to open a browser in the test env.
    monkeypatch.setattr("hermes_cli.auth._is_remote_session", lambda: True)

    creds = _xai_oauth_device_code_login(open_browser=False)
    assert creds["tokens"]["access_token"] == "AT-dev"
    assert creds["tokens"]["refresh_token"] == "RT-dev"
    assert creds["source"] == "device-code"
    # discovery carries the validated token_endpoint for refresh reuse.
    assert creds["discovery"]["token_endpoint"] == "https://auth.x.ai/oauth2/token"
    # Same top-level keys the loopback login returns.
    for key in ("tokens", "discovery", "redirect_uri", "base_url", "last_refresh"):
        assert key in creds


def test_device_code_login_requires_refresh_token(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_device_discovery",
        lambda *_a, **_k: {
            "device_authorization_endpoint": "https://auth.x.ai/oauth2/device/code",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_request_device_code",
        lambda *_a, **_k: {
            "device_code": "DC-123",
            "user_code": "WXYZ-1234",
            "verification_uri": "https://accounts.x.ai/device",
            "expires_in": 900,
            "interval": 5,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_poll_device_token",
        lambda *_a, **_k: {"access_token": "AT-dev"},  # no refresh_token
    )
    monkeypatch.setattr("hermes_cli.auth._is_remote_session", lambda: True)
    with pytest.raises(AuthError) as ei:
        _xai_oauth_device_code_login(open_browser=False)
    assert ei.value.code == "xai_token_exchange_invalid"


# ---------------------------------------------------------------------------
# Flag guard: --device-code and --manual-paste are mutually exclusive
# ---------------------------------------------------------------------------


def test_login_rejects_device_code_with_manual_paste(monkeypatch):
    """``_login_xai_oauth`` must refuse both flags together — the device-code
    flow has no loopback callback to paste."""
    # Force a fresh login path (skip the "reuse existing creds" branch).
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AuthError("none", provider="xai-oauth", code="missing")
        ),
    )
    args = argparse.Namespace(
        device_code=True, manual_paste=True, no_browser=True, timeout=None
    )
    with pytest.raises(AuthError) as ei:
        _login_xai_oauth(args, None, force_new_login=True)
    assert ei.value.code == "xai_login_flag_conflict"
