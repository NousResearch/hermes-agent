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
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


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
