"""Regression tests for the OAuth dispatcher in hermes_cli.web_server.

Bug history (2026-05-09): the `_OAUTH_PROVIDER_CATALOG` had two entries
flagged ``flow: "pkce"`` — anthropic and minimax-oauth — and the
dispatcher ``start_oauth_login`` hardcoded ``_start_anthropic_pkce()``
for any pkce-flagged provider. So clicking "Login" next to MiniMax in
the dashboard's Keys tab silently launched the Anthropic/Claude OAuth
flow.

The fix:
  1. Catalog entry for minimax-oauth changed from ``flow: "pkce"`` to
     ``flow: "device_code"`` (the actual UX is verification URI + user
     code + background poll, with PKCE as a security extension).
  2. New MiniMax branch added to ``_start_device_code_flow``.
  3. Dispatcher tightened: pkce branch now requires
     ``provider_id == "anthropic"``, so any future PKCE provider added
     without an explicit branch gets a clean ``400 Unsupported flow``
     instead of silently launching Anthropic OAuth.

These tests pin the corrected behavior.
"""
import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def _fake_nous_device_data():
    return {
        "device_code": "device-code",
        "user_code": "NOUS-1234",
        "verification_uri": "https://portal.nousresearch.com/device",
        "verification_uri_complete": (
            "https://portal.nousresearch.com/device?user_code=NOUS-1234"
        ),
        "expires_in": 600,
        "interval": 5,
    }


def _invoke_scope_refusal():
    request = httpx.Request("POST", "https://portal.nousresearch.com/oauth/device/code")
    response = httpx.Response(
        400,
        json={
            "error": "invalid_scope",
            "error_description": "unsupported scope inference:invoke",
        },
        request=request,
    )
    return httpx.HTTPStatusError("invalid scope", request=request, response=response)


def test_minimax_login_does_not_launch_anthropic_flow():
    """Click 'Login' on MiniMax → MUST NOT return claude.ai auth_url."""
    fake_user_code_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://api.minimax.io/oauth/verify",
        # `expired_in` < 1e12 so the heuristic treats it as seconds.
        "expired_in": 600,
        "interval": 2000,
        "state": "stub-state",
    }
    with patch(
        "hermes_cli.auth._minimax_request_user_code",
        return_value=fake_user_code_resp,
    ), patch(
        "hermes_cli.auth._minimax_pkce_pair",
        return_value=("verifier-stub", "challenge-stub", "stub-state"),
    ), patch(
        "hermes_cli.web_server._minimax_poller",
        return_value=None,
    ):
        resp = client.post(
            "/api/providers/oauth/minimax-oauth/start",
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()

    # The bug used to return Anthropic's auth_url — make sure the response
    # references neither the auth_url field nor anything Claude-related.
    assert "auth_url" not in body
    assert "claude.ai" not in str(body).lower()

    # And the response IS the device-code shape pointing at MiniMax.
    assert body["flow"] == "device_code"
    assert "minimax" in body["verification_url"].lower()
    assert body["user_code"] == "ABCD-1234"
    assert body["expires_in"] == 600


def test_nous_dashboard_device_flow_ignores_legacy_scope_override(monkeypatch):
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    requested_scopes = []

    def fake_request_device_code(**kwargs):
        requested_scopes.append(kwargs["scope"])
        return _fake_nous_device_data()

    monkeypatch.setenv("HERMES_AGENT_USE_LEGACY_SESSION_KEYS", "true")
    monkeypatch.setattr(auth_mod, "_request_device_code", fake_request_device_code)
    monkeypatch.setattr(ws, "_nous_poller", lambda sid: None)

    result = asyncio.run(ws._start_device_code_flow("nous"))
    try:
        assert requested_scopes == [auth_mod.DEFAULT_NOUS_SCOPE]
        assert result["flow"] == "device_code"
        assert result["user_code"] == "NOUS-1234"
        assert (
            ws._oauth_sessions[result["session_id"]]["scope"]
            == auth_mod.DEFAULT_NOUS_SCOPE
        )
    finally:
        ws._oauth_sessions.pop(result["session_id"], None)


def test_nous_dashboard_device_flow_does_not_retry_legacy_scope_on_invoke_refusal(monkeypatch):
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    requested_scopes = []

    def fake_request_device_code(**kwargs):
        requested_scopes.append(kwargs["scope"])
        raise _invoke_scope_refusal()

    monkeypatch.delenv("HERMES_AGENT_USE_LEGACY_SESSION_KEYS", raising=False)
    monkeypatch.setattr(auth_mod, "_request_device_code", fake_request_device_code)
    monkeypatch.setattr(ws, "_nous_poller", lambda sid: None)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(ws._start_device_code_flow("nous"))
    assert requested_scopes == [auth_mod.DEFAULT_NOUS_SCOPE]


def test_codex_dashboard_worker_persists_runtime_provider(tmp_path, monkeypatch):
    from hermes_cli import web_server as ws
    from hermes_cli.auth import get_active_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider

    access_token = "h.eyJleHAiOjk5OTk5OTk5OTl9.s"

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if url.endswith("/deviceauth/usercode"):
                return _Resp(200, {
                    "device_auth_id": "device-auth-id",
                    "interval": 3,
                    "user_code": "CODEX-1234",
                })
            if url.endswith("/deviceauth/token"):
                return _Resp(200, {
                    "authorization_code": "authorization-code",
                    "code_verifier": "code-verifier",
                })
            return _Resp(200, {
                "access_token": access_token,
                "refresh_token": "codex-refresh",
            })

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(ws.time, "sleep", lambda _: None)

    sid, _ = ws._new_oauth_session("openai-codex", "device_code")
    try:
        ws._codex_full_login_worker(sid)

        assert ws._oauth_sessions[sid]["status"] == "approved"
        assert get_active_provider() == "openai-codex"

        runtime = resolve_runtime_provider(requested=None)
        assert runtime["provider"] == "openai-codex"
        assert runtime["api_key"] == access_token
        assert runtime["api_mode"] == "codex_responses"
    finally:
        ws._oauth_sessions.pop(sid, None)


def test_nous_dashboard_poller_preserves_effective_scope_when_token_omits_scope(monkeypatch):
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    session_id = "nous-effective-scope-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "nous",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "hermes-cli",
        "device_code": "device-code",
        "interval": 5,
        "expires_at": time.time() + 600,
        "scope": auth_mod.DEFAULT_NOUS_SCOPE,
    }
    captured_state = {}

    def fake_refresh_nous_oauth_from_state(state, **kwargs):
        captured_state.update(state)
        return {**state, "agent_key": "jwt-agent-key"}

    monkeypatch.setattr(
        auth_mod,
        "_poll_for_token",
        lambda **kwargs: {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    monkeypatch.setattr(
        auth_mod,
        "refresh_nous_oauth_from_state",
        fake_refresh_nous_oauth_from_state,
    )
    monkeypatch.setattr(auth_mod, "persist_nous_credentials", lambda state: None)

    try:
        ws._nous_poller(session_id)
        assert captured_state["scope"] == auth_mod.DEFAULT_NOUS_SCOPE
        assert ws._oauth_sessions[session_id]["status"] == "approved"
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_minimax_dashboard_poller_accepts_absolute_ms_expired_in():
    """Dashboard MiniMax completion must accept unix-ms token expiry values."""
    from hermes_cli import web_server as ws

    now = datetime.now(timezone.utc)
    abs_ms = int((now.timestamp() + 1800) * 1000)
    session_id = "minimax-absolute-ms-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "minimax-oauth",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "portal_base_url": "https://api.minimax.io",
        "client_id": "client-id",
        "user_code": "ABCD-1234",
        "code_verifier": "verifier",
        "interval_ms": 2000,
        "expired_in_raw": abs_ms,
        "region": "global",
    }
    captured_state = {}

    try:
        with patch(
            "hermes_cli.auth._minimax_poll_token",
            return_value={
                "status": "success",
                "access_token": "access",
                "refresh_token": "refresh",
                "expired_in": abs_ms,
                "token_type": "Bearer",
            },
        ), patch(
            "hermes_cli.auth._minimax_save_auth_state",
            side_effect=lambda state: captured_state.update(state),
        ):
            ws._minimax_poller(session_id)
    finally:
        ws._oauth_sessions.pop(session_id, None)

    assert captured_state["access_token"] == "access"
    assert 1790 <= captured_state["expires_in"] <= 1810
    assert datetime.fromisoformat(captured_state["expires_at"]).year < 9999


def test_anthropic_pkce_branch_still_works():
    """Sanity: the dispatcher tightening doesn't break the legitimate Anthropic PKCE path."""
    fake_anthropic_response = {
        "session_id": "stub-session",
        "flow": "pkce",
        "auth_url": "https://claude.ai/oauth/authorize?code=true&...",
        "expires_in": 600,
    }
    with patch(
        "hermes_cli.web_server._start_anthropic_pkce",
        return_value=fake_anthropic_response,
    ):
        resp = client.post(
            "/api/providers/oauth/anthropic/start",
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["flow"] == "pkce"
    assert "claude.ai" in body["auth_url"]


def test_xai_oauth_listed_as_loopback_flow():
    """xAI Grok OAuth must surface in the catalog as a first-class loopback flow."""
    resp = client.get("/api/providers/oauth", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    providers = {p["id"]: p for p in resp.json()["providers"]}
    assert "xai-oauth" in providers
    assert providers["xai-oauth"]["flow"] == "loopback"
    assert "grok" in providers["xai-oauth"]["name"].lower()


def test_xai_loopback_start_returns_authorize_url(monkeypatch):
    """Start MUST bind the loopback listener and hand back an xAI authorize URL."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    class _FakeServer:
        def shutdown(self):
            pass

        def server_close(self):
            pass

    class _FakeThread:
        def join(self, timeout=None):
            pass

    redirect_uri = (
        f"http://{auth_mod.XAI_OAUTH_REDIRECT_HOST}:{auth_mod.XAI_OAUTH_REDIRECT_PORT}"
        f"{auth_mod.XAI_OAUTH_REDIRECT_PATH}"
    )

    monkeypatch.setattr(
        auth_mod,
        "_xai_oauth_discovery",
        lambda *a, **k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/auth",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )
    monkeypatch.setattr(
        auth_mod,
        "_xai_start_callback_server",
        lambda *a, **k: (_FakeServer(), _FakeThread(), {"code": None, "error": None}, redirect_uri),
    )
    # Don't let the background worker run a real callback wait/exchange.
    monkeypatch.setattr(ws, "_xai_loopback_worker", lambda sid: None)

    resp = client.post("/api/providers/oauth/xai-oauth/start", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    try:
        assert body["flow"] == "loopback"
        assert "user_code" not in body  # loopback has nothing to paste/show
        assert body["auth_url"].startswith("https://auth.x.ai/oauth2/auth?")
        assert "code_challenge" in body["auth_url"]
        sess = ws._oauth_sessions[body["session_id"]]
        assert sess["provider"] == "xai-oauth"
        assert sess["flow"] == "loopback"
    finally:
        ws._oauth_sessions.pop(body["session_id"], None)


def test_xai_loopback_worker_persists_tokens_on_success(monkeypatch):
    """The worker exchanges the callback code and marks the session approved."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    saved = {}
    session_id = "xai-loopback-success-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "xai-oauth",
        "flow": "loopback",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "server": object(),
        "thread": object(),
        "callback_result": {"code": "auth-code", "state": "st"},
        "redirect_uri": "http://127.0.0.1:56121/callback",
        "verifier": "verifier",
        "challenge": "challenge",
        "state": "st",
        "token_endpoint": "https://auth.x.ai/oauth2/token",
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth2/token"},
    }

    monkeypatch.setattr(
        auth_mod,
        "_xai_wait_for_callback",
        lambda *a, **k: {"code": "auth-code", "state": "st"},
    )
    monkeypatch.setattr(
        auth_mod,
        "_xai_oauth_exchange_code_for_tokens",
        lambda **k: {
            "access_token": "xai-access",
            "refresh_token": "xai-refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    monkeypatch.setattr(
        auth_mod,
        "_save_xai_oauth_tokens",
        lambda tokens, **k: saved.update(tokens),
    )
    monkeypatch.setattr(ws, "_add_xai_oauth_pool_entry", lambda *a, **k: None)

    try:
        ws._xai_loopback_worker(session_id)
        assert ws._oauth_sessions[session_id]["status"] == "approved"
        assert saved["access_token"] == "xai-access"
        assert saved["refresh_token"] == "xai-refresh"
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_xai_loopback_worker_fails_on_state_mismatch(monkeypatch):
    """A mismatched OAuth state must fail the session, not persist tokens."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    session_id = "xai-loopback-state-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "xai-oauth",
        "flow": "loopback",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "server": object(),
        "thread": object(),
        "callback_result": {},
        "redirect_uri": "http://127.0.0.1:56121/callback",
        "verifier": "verifier",
        "challenge": "challenge",
        "state": "expected-state",
        "token_endpoint": "https://auth.x.ai/oauth2/token",
        "discovery": {},
    }

    monkeypatch.setattr(
        auth_mod,
        "_xai_wait_for_callback",
        lambda *a, **k: {"code": "auth-code", "state": "ATTACKER-state"},
    )

    def _boom(**kwargs):
        raise AssertionError("token exchange must not run on state mismatch")

    monkeypatch.setattr(auth_mod, "_xai_oauth_exchange_code_for_tokens", _boom)

    try:
        ws._xai_loopback_worker(session_id)
        sess = ws._oauth_sessions[session_id]
        assert sess["status"] == "error"
        assert "state mismatch" in sess["error_message"].lower()
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_unknown_pkce_provider_rejected_cleanly():
    """A future PKCE provider without an explicit branch must NOT silently route to Anthropic.

    Simulates a hypothetical catalog entry with ``flow: "pkce"`` and an
    id other than "anthropic". The dispatcher should fall through past
    the pkce branch (now gated on provider_id) and the device_code
    branch, then hit "Unsupported flow" — proving the bug class is
    structurally prevented.
    """
    from hermes_cli import web_server as ws

    # Inject a hypothetical catalog entry that's pkce-flagged but isn't
    # anthropic. This shape mirrors what would happen if a developer
    # added a new provider entry without remembering to wire up its
    # start function.
    fake_entry = {
        "id": "hypothetical-pkce-provider",
        "name": "Hypothetical PKCE Provider",
        "flow": "pkce",
        "cli_command": "hermes auth add hypothetical-pkce-provider",
        "docs_url": "https://example.com",
        "status_fn": None,
    }
    original_catalog = ws._OAUTH_PROVIDER_CATALOG
    try:
        ws._OAUTH_PROVIDER_CATALOG = original_catalog + (fake_entry,)
        resp = client.post(
            "/api/providers/oauth/hypothetical-pkce-provider/start",
            headers=HEADERS,
        )
    finally:
        ws._OAUTH_PROVIDER_CATALOG = original_catalog

    # Either 400 "Unsupported flow" (the explicit fall-through) or any
    # 4xx — what we MUST NOT see is a 200 with claude.ai in the body.
    assert resp.status_code >= 400, resp.text
    assert "claude.ai" not in resp.text.lower()
