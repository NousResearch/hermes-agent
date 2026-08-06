"""Tests for MiniMax OAuth provider (hermes_cli/auth.py).

Covers:
- PKCE pair generation (S256 challenge)
- _minimax_request_user_code happy path and state-mismatch error
- _minimax_poll_token: pending→success flow, error status, timeout
- _refresh_minimax_oauth_state: skip when not expired, update on success,
  re-login required on invalid_grant
- resolve_minimax_oauth_runtime_credentials: error when not logged in
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    AuthError,
    MINIMAX_OAUTH_CLIENT_ID,
    MINIMAX_OAUTH_GLOBAL_BASE,
    MINIMAX_OAUTH_GLOBAL_INFERENCE,
    MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
    _minimax_pkce_pair,
    _minimax_request_user_code,
    _minimax_poll_token,
    _minimax_resolve_token_expiry_unix,
    _refresh_minimax_oauth_state,
    resolve_minimax_oauth_runtime_credentials,
    get_minimax_oauth_auth_status,
    get_auth_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_httpx_response(status_code: int, body: dict | None = None, text: str = ""):
    """Return a minimal mock that quacks like httpx.Response.

    Includes the streamed-read surface used by ``_minimax_post_form`` /
    ``_minimax_response_error_text``: ``is_stream_consumed`` is False and
    ``iter_bytes()`` yields the body/text bytes, so non-200 paths exercise
    the real bounded-read code instead of a truthy MagicMock attribute.
    """
    resp = MagicMock()
    resp.status_code = status_code
    if body is not None:
        resp.json.return_value = body
        resp.text = json.dumps(body)
    else:
        resp.json.side_effect = Exception("No body")
        resp.text = text
    resp.reason_phrase = "OK" if status_code == 200 else "Error"
    resp.is_stream_consumed = False
    resp.encoding = "utf-8"
    resp.iter_bytes.return_value = iter([resp.text.encode("utf-8")] if resp.text else [])
    return resp


def _future_iso(seconds_from_now: int = 3600) -> str:
    ts = time.time() + seconds_from_now
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _past_iso(seconds_ago: int = 3600) -> str:
    ts = time.time() - seconds_ago
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 0. test_resolve_token_expiry_unix_ttl_vs_absolute_ms
# ---------------------------------------------------------------------------

def test_resolve_token_expiry_unix_ttl_seconds():
    now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    got = _minimax_resolve_token_expiry_unix(3600, now=now)
    assert abs(got - (now.timestamp() + 3600)) < 0.01




# ---------------------------------------------------------------------------
# 1. test_pkce_pair_produces_valid_s256
# ---------------------------------------------------------------------------

def test_pkce_pair_produces_valid_s256():
    verifier, challenge, state = _minimax_pkce_pair()

    # Verifier must be non-empty and URL-safe
    assert isinstance(verifier, str)
    assert len(verifier) >= 32

    # Challenge must be URL-safe base64 without trailing "="
    assert isinstance(challenge, str)
    assert "=" not in challenge

    # Re-compute challenge from verifier and verify it matches
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    assert challenge == expected

    # State must be non-empty
    assert isinstance(state, str)
    assert len(state) >= 8

    # Two calls must return different values (randomness)
    v2, c2, s2 = _minimax_pkce_pair()
    assert verifier != v2
    assert state != s2


# ---------------------------------------------------------------------------
# 2. test_request_user_code_happy_path
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 3. test_request_user_code_state_mismatch_raises
# ---------------------------------------------------------------------------

def test_request_user_code_state_mismatch_raises():
    mock_response = _make_httpx_response(200, {
        "user_code": "XYZ",
        "verification_uri": "https://minimax.io/verify",
        "expired_in": 300,
        "state": "wrong-state",  # Mismatched!
    })

    client = MagicMock()
    client.post.return_value = mock_response
    client.send.return_value = mock_response

    with pytest.raises(AuthError) as exc_info:
        _minimax_request_user_code(
            client,
            portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
            client_id=MINIMAX_OAUTH_CLIENT_ID,
            code_challenge="challenge",
            state="correct-state",
        )

    assert exc_info.value.code == "state_mismatch"
    assert "CSRF" in str(exc_info.value) or "mismatch" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 4. test_request_user_code_non_200_raises
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 5. test_poll_token_pending_then_success
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 6. test_poll_token_error_raises
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 7. test_poll_token_timeout_raises
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 8. test_refresh_skip_when_not_expired
# ---------------------------------------------------------------------------

def test_refresh_skip_when_not_expired():
    """When token is far from expiry, refresh should return the same state."""
    state = {
        "access_token": "old-access",
        "refresh_token": "refresh-token",
        "portal_base_url": MINIMAX_OAUTH_GLOBAL_BASE,
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "expires_at": _future_iso(3600),  # 1 hour in the future
    }

    result = _refresh_minimax_oauth_state(state)
    assert result["access_token"] == "old-access"
    assert result is state  # Same object returned (no refresh)


# ---------------------------------------------------------------------------
# 9. test_refresh_updates_access_token
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# 10. test_refresh_reuse_triggers_relogin_required
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 11. test_resolve_credentials_requires_login
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 11b. Terminal refresh failure quarantines dead tokens (#28003)
# ---------------------------------------------------------------------------

def test_resolve_credentials_quarantines_dead_tokens_on_terminal_refresh_failure():
    """Terminal refresh failure (relogin_required + refresh_token present) must
    clear access_token/refresh_token/expires_* from auth.json and write a
    last_auth_error marker, so subsequent calls fail fast with not_logged_in
    instead of replaying the dead refresh token over the network.
    Mirrors Nous / xAI-OAuth / Codex-OAuth quarantine pattern.
    """
    stale_state = {
        "access_token": "dead-access-token",
        "refresh_token": "dead-refresh-token",
        "expires_at": "2026-01-01T00:00:00Z",
        "expires_in": 3600,
        "obtained_at": "2026-01-01T00:00:00Z",
        "inference_base_url": "https://api.minimax.io/v1",
        "portal_base_url": "https://portal.minimax.io",
        "client_id": "test-client",
        "region": "global",
    }
    saved_states = []

    def _capture_save(s):
        saved_states.append(dict(s))

    def _terminal_refresh(_state):
        raise AuthError(
            "invalid_grant",
            provider="minimax-oauth",
            code="invalid_grant",
            relogin_required=True,
        )

    with patch("hermes_cli.auth.get_provider_auth_state", return_value=stale_state), \
         patch("hermes_cli.auth._refresh_minimax_oauth_state", side_effect=_terminal_refresh), \
         patch("hermes_cli.auth._minimax_save_auth_state", side_effect=_capture_save):
        with pytest.raises(AuthError) as exc_info:
            resolve_minimax_oauth_runtime_credentials()

    # The original AuthError is re-raised so callers get the right error surface.
    assert exc_info.value.code == "invalid_grant"
    assert exc_info.value.relogin_required is True

    # A quarantine save must have happened.
    assert len(saved_states) == 1
    quarantined = saved_states[0]

    # Dead OAuth fields cleared.
    assert "access_token" not in quarantined
    assert "refresh_token" not in quarantined
    assert "expires_at" not in quarantined
    assert "expires_in" not in quarantined
    assert "obtained_at" not in quarantined

    # Routing/identity metadata preserved.
    assert quarantined["inference_base_url"] == "https://api.minimax.io/v1"
    assert quarantined["portal_base_url"] == "https://portal.minimax.io"
    assert quarantined["client_id"] == "test-client"
    assert quarantined["region"] == "global"

    # Structured diagnostic blob written.
    err = quarantined.get("last_auth_error")
    assert isinstance(err, dict)
    assert err["provider"] == "minimax-oauth"
    assert err["code"] == "invalid_grant"
    assert err["reason"] == "runtime_refresh_failure"
    assert err["relogin_required"] is True
    assert "at" in err




# ---------------------------------------------------------------------------
# 12. test_provider_registry_contains_minimax_oauth
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 13. test_minimax_oauth_alias_resolves
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 14. test_get_minimax_oauth_auth_status_not_logged_in
# ---------------------------------------------------------------------------

def test_get_minimax_oauth_auth_status_not_logged_in():
    with patch("hermes_cli.auth.get_provider_auth_state", return_value=None):
        status = get_minimax_oauth_auth_status()

    assert status["logged_in"] is False
    assert status["provider"] == "minimax-oauth"


# ---------------------------------------------------------------------------
# 15. test_get_minimax_oauth_auth_status_logged_in
# ---------------------------------------------------------------------------



def test_generic_auth_status_dispatches_minimax_oauth():
    state = {
        "access_token": "tok",
        "expires_at": _future_iso(3600),
        "region": "global",
    }

    with patch("hermes_cli.auth.get_provider_auth_state", return_value=state):
        status = get_auth_status("minimax-oauth")

    assert status["logged_in"] is True
    assert status["provider"] == "minimax-oauth"
    assert status["region"] == "global"


# ---------------------------------------------------------------------------
# build_minimax_oauth_token_provider — per-request callable bearer
# ---------------------------------------------------------------------------
# These tests verify the fix for short-lived (~15-min) MiniMax access tokens
# expiring mid-session. The callable is invoked by the Anthropic SDK on every
# outbound request via the existing Entra-style bearer hook.


def test_token_provider_returns_current_access_token_when_fresh():
    """When token is far from expiry, callable just returns the cached token."""
    from hermes_cli.auth import build_minimax_oauth_token_provider

    state = {
        "access_token": "still-fresh",
        "refresh_token": "rt",
        "portal_base_url": MINIMAX_OAUTH_GLOBAL_BASE,
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "expires_at": _future_iso(3600),
    }

    provider = build_minimax_oauth_token_provider()

    with patch("hermes_cli.auth.get_provider_auth_state", return_value=state), \
         patch("httpx.Client") as mock_client_class:
        token = provider()
        # No network call should happen — token is fresh.
        mock_client_class.assert_not_called()

    assert token == "still-fresh"


def test_token_provider_refreshes_when_near_expiry():
    """When token is within the skew window, callable mints a fresh one."""
    from hermes_cli.auth import build_minimax_oauth_token_provider

    state = {
        "access_token": "about-to-die",
        "refresh_token": "rt",
        "portal_base_url": MINIMAX_OAUTH_GLOBAL_BASE,
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "expires_at": _future_iso(MINIMAX_OAUTH_REFRESH_SKEW_SECONDS - 1),
    }

    refreshed_body = {
        "status": "success",
        "access_token": "fresh-bearer",
        "refresh_token": "rt2",
        "expired_in": 900,
    }
    mock_resp = _make_httpx_response(200, refreshed_body)

    provider = build_minimax_oauth_token_provider()

    with patch("hermes_cli.auth.get_provider_auth_state", return_value=state), \
         patch("httpx.Client") as mock_client_class, \
         patch("hermes_cli.auth._minimax_save_auth_state"):
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.post.return_value = mock_resp
        mock_instance.send.return_value = mock_resp
        mock_client_class.return_value = mock_instance

        token = provider()

    assert token == "fresh-bearer"




def test_token_provider_raises_not_logged_in_when_state_missing():
    """No state in auth.json → AuthError(not_logged_in, relogin_required=True)."""
    from hermes_cli.auth import build_minimax_oauth_token_provider

    provider = build_minimax_oauth_token_provider()
    with patch("hermes_cli.auth.get_provider_auth_state", return_value=None):
        with pytest.raises(AuthError) as exc_info:
            provider()

    assert exc_info.value.code == "not_logged_in"
    assert exc_info.value.relogin_required is True


def test_token_provider_quarantines_state_on_terminal_refresh():
    """When refresh returns invalid_grant, callable raises AuthError AND
    wipes the dead tokens so subsequent calls fail fast without network."""
    from hermes_cli.auth import build_minimax_oauth_token_provider

    state = {
        "access_token": "expired",
        "refresh_token": "burned-rt",
        "portal_base_url": MINIMAX_OAUTH_GLOBAL_BASE,
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "expires_at": _past_iso(100),
    }

    bad_resp = _make_httpx_response(400, text="invalid_grant")
    bad_resp.json.side_effect = Exception("no json")
    bad_resp.text = "invalid_grant"
    bad_resp.reason_phrase = "Bad Request"

    saved_states: list[dict] = []

    provider = build_minimax_oauth_token_provider()
    with patch("hermes_cli.auth.get_provider_auth_state", return_value=state), \
         patch("httpx.Client") as mock_client_class, \
         patch(
             "hermes_cli.auth._minimax_save_auth_state",
             side_effect=lambda s: saved_states.append(dict(s)),
         ):
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.post.return_value = bad_resp
        mock_instance.send.return_value = bad_resp
        mock_client_class.return_value = mock_instance

        with pytest.raises(AuthError) as exc_info:
            provider()

    assert exc_info.value.relogin_required is True
    # Quarantine wrote a state with tokens removed.
    assert len(saved_states) == 1
    quarantined = saved_states[0]
    assert "access_token" not in quarantined
    assert "refresh_token" not in quarantined
    assert quarantined["last_auth_error"]["relogin_required"] is True


def test_resolve_returns_callable_when_as_token_provider_true():
    """Explicit opt-in path: resolve_minimax_oauth_runtime_credentials(as_token_provider=True)
    returns a callable api_key."""
    state = {
        "access_token": "tok",
        "refresh_token": "rt",
        "portal_base_url": MINIMAX_OAUTH_GLOBAL_BASE,
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "expires_at": _future_iso(3600),
    }

    with patch("hermes_cli.auth.get_provider_auth_state", return_value=state):
        creds = resolve_minimax_oauth_runtime_credentials(as_token_provider=True)

    assert callable(creds["api_key"])
    assert not isinstance(creds["api_key"], str)
    assert creds["base_url"] == MINIMAX_OAUTH_GLOBAL_INFERENCE.rstrip("/")


# ---------------------------------------------------------------------------
# Bounded error-body reads (#56548 / PR #56549)
# ---------------------------------------------------------------------------

def test_refresh_error_body_bounded_and_readable_with_real_client():
    """Refresh non-200 path over a REAL socket transport.

    The error body is obtained via a streamed response; the bounded read
    must happen while the client context is still open.  A real socket is
    required to bind this contract: closing the client tears the connection
    down, so a read after the ``with httpx.Client(...)`` block raises
    ReadError/StreamClosed.  (MockTransport buffers in memory and would NOT
    catch the regression.)
    """
    import http.server
    import socketserver
    import threading

    import httpx

    from hermes_cli.auth import _refresh_minimax_oauth_state

    big_body = b"invalid_grant " + b"x" * (64 * 1024)  # 64KB error body

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(400)
            self.send_header("Content-Length", str(len(big_body)))
            self.end_headers()
            self.wfile.write(big_body)

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            state = {
                "access_token": "expired",
                "refresh_token": "burned-rt",
                "portal_base_url": f"http://127.0.0.1:{port}",
                "client_id": MINIMAX_OAUTH_CLIENT_ID,
                "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
                "expires_at": _past_iso(100),
            }
            with pytest.raises(AuthError) as exc_info:
                _refresh_minimax_oauth_state(state, force=True)
        finally:
            server.shutdown()

    msg = str(exc_info.value)
    assert "invalid_grant" in msg
    assert exc_info.value.relogin_required is True
    # Bounded: 16KB limit + truncation marker, never the full 64KB body.
    assert len(msg) < 20 * 1024
    assert "...[truncated]" in msg


def test_minimax_response_error_text_truncates_above_limit():
    """Bodies above the 16KB bound are cut and marked truncated."""
    import httpx

    from hermes_cli.auth import (
        _MINIMAX_OAUTH_ERROR_BODY_LIMIT,
        _minimax_response_error_text,
    )

    big = "e" * (_MINIMAX_OAUTH_ERROR_BODY_LIMIT * 4)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=big)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        request = client.build_request("POST", "https://api.minimax.io/oauth/token")
        response = client.send(request, stream=True)
        text = _minimax_response_error_text(response)

    assert text.endswith("...[truncated]")
    assert len(text) <= _MINIMAX_OAUTH_ERROR_BODY_LIMIT + len("...[truncated]")
