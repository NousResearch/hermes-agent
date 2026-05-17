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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    AuthError,
    MINIMAX_OAUTH_CLIENT_ID,
    MINIMAX_OAUTH_GLOBAL_BASE,
    MINIMAX_OAUTH_GLOBAL_INFERENCE,
    MINIMAX_OAUTH_CN_BASE,
    MINIMAX_OAUTH_CN_INFERENCE,
    MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
    _minimax_pkce_pair,
    _minimax_request_user_code,
    _minimax_poll_token,
    _minimax_resolve_token_expiry_unix,
    _refresh_minimax_oauth_state,
    resolve_minimax_oauth_runtime_credentials,
    get_minimax_oauth_auth_status,
    get_auth_status,
    get_provider_auth_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_httpx_response(status_code: int, body: dict | None = None, text: str = ""):
    """Return a minimal mock that quacks like httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if body is not None:
        resp.json.return_value = body
        resp.text = json.dumps(body)
    else:
        resp.json.side_effect = Exception("No body")
        resp.text = text
    resp.reason_phrase = "OK" if status_code == 200 else "Error"
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


def test_resolve_token_expiry_unix_absolute_ms():
    now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    abs_ms = int((now.timestamp() + 7200) * 1000)
    got = _minimax_resolve_token_expiry_unix(abs_ms, now=now)
    assert abs(got - (now.timestamp() + 7200)) < 0.01


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

def test_request_user_code_happy_path():
    state = "test-state-abc"
    mock_response = _make_httpx_response(200, {
        "user_code": "ABC-123",
        "verification_uri": "https://minimax.io/verify",
        "expired_in": int(time.time() * 1000) + 300_000,
        "state": state,
    })

    client = MagicMock()
    client.post.return_value = mock_response

    result = _minimax_request_user_code(
        client,
        portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
        client_id=MINIMAX_OAUTH_CLIENT_ID,
        code_challenge="test-challenge",
        state=state,
    )

    assert result["user_code"] == "ABC-123"
    assert result["verification_uri"] == "https://minimax.io/verify"
    assert result["state"] == state

    # Verify correct endpoint was called
    call_args = client.post.call_args
    assert "/oauth/code" in call_args[0][0]
    headers = call_args[1].get("headers", {})
    assert "x-request-id" in headers


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

def test_request_user_code_non_200_raises():
    mock_response = _make_httpx_response(400, text="Bad Request")
    mock_response.json.side_effect = Exception("no json")
    mock_response.text = "Bad Request"

    client = MagicMock()
    client.post.return_value = mock_response

    with pytest.raises(AuthError) as exc_info:
        _minimax_request_user_code(
            client,
            portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
            client_id=MINIMAX_OAUTH_CLIENT_ID,
            code_challenge="challenge",
            state="state",
        )

    assert exc_info.value.code == "authorization_failed"


# ---------------------------------------------------------------------------
# 5. test_poll_token_pending_then_success
# ---------------------------------------------------------------------------

def test_poll_token_pending_then_success():
    # Set a deadline far enough in the future for polling
    deadline_ms = int(time.time() * 1000) + 60_000  # 60 seconds from now

    pending_body = {"status": "pending"}
    success_body = {
        "status": "success",
        "access_token": "access-abc",
        "refresh_token": "refresh-xyz",
        "expired_in": 3600,
        "token_type": "Bearer",
    }

    pending_resp = _make_httpx_response(200, pending_body)
    success_resp = _make_httpx_response(200, success_body)

    client = MagicMock()
    client.post.side_effect = [pending_resp, pending_resp, success_resp]

    with patch("time.sleep"):  # don't actually sleep
        result = _minimax_poll_token(
            client,
            portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
            client_id=MINIMAX_OAUTH_CLIENT_ID,
            user_code="USER-CODE",
            code_verifier="verifier",
            expired_in=deadline_ms,
            interval_ms=2000,
        )

    assert result["status"] == "success"
    assert result["access_token"] == "access-abc"
    assert result["refresh_token"] == "refresh-xyz"
    assert client.post.call_count == 3


# ---------------------------------------------------------------------------
# 6. test_poll_token_error_raises
# ---------------------------------------------------------------------------

def test_poll_token_error_raises():
    deadline_ms = int(time.time() * 1000) + 60_000
    error_body = {"status": "error"}
    error_resp = _make_httpx_response(200, error_body)

    client = MagicMock()
    client.post.return_value = error_resp

    with pytest.raises(AuthError) as exc_info:
        _minimax_poll_token(
            client,
            portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
            client_id=MINIMAX_OAUTH_CLIENT_ID,
            user_code="U",
            code_verifier="v",
            expired_in=deadline_ms,
            interval_ms=2000,
        )

    assert exc_info.value.code == "authorization_denied"


# ---------------------------------------------------------------------------
# 7. test_poll_token_timeout_raises
# ---------------------------------------------------------------------------

def test_poll_token_timeout_raises():
    # expired_in is a small duration (treated as seconds from now, already expired)
    expired_in = 1  # 1 second from now
    # Make sleep a no-op and time.time advance quickly by using a small deadline
    # We use a duration-style expired_in (small enough to not be a unix timestamp)
    # duration mode: deadline = time.time() + max(1, expired_in)
    # We need time() to exceed deadline immediately.

    fixed_now = time.time()
    call_count = [0]

    def fake_time():
        call_count[0] += 1
        # After 2 calls, return a time past the deadline
        if call_count[0] > 2:
            return fixed_now + 10  # past deadline
        return fixed_now

    client = MagicMock()
    pending_resp = _make_httpx_response(200, {"status": "pending"})
    client.post.return_value = pending_resp

    import hermes_cli.auth as auth_module
    with patch.object(auth_module, "time") as mock_time_mod:
        # We need to patch the 'time' module used inside _minimax_poll_token
        # The function imports 'import time as _time' locally.
        # Patch time.sleep and time.time in the auth module's local scope.
        pass

    # Use a simpler approach: expired_in as past timestamp (already expired)
    past_deadline_ms = int((time.time() - 1) * 1000)  # 1 second ago

    with pytest.raises(AuthError) as exc_info:
        _minimax_poll_token(
            client,
            portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
            client_id=MINIMAX_OAUTH_CLIENT_ID,
            user_code="U",
            code_verifier="v",
            expired_in=past_deadline_ms,
            interval_ms=2000,
        )

    assert exc_info.value.code == "timeout"


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

def test_refresh_updates_access_token():
    """When token is close to expiry, refresh should update the state."""
    # expires_at just MINIMAX_OAUTH_REFRESH_SKEW_SECONDS - 1 from now (close to expiry)
    state = {
        "access_token": "old-access",
        "refresh_token": "my-refresh",
        "portal_base_url": MINIMAX_OAUTH_GLOBAL_BASE,
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "expires_at": _future_iso(MINIMAX_OAUTH_REFRESH_SKEW_SECONDS - 1),
    }

    new_token_body = {
        "status": "success",
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expired_in": 7200,
    }

    mock_resp = _make_httpx_response(200, new_token_body)

    with patch("httpx.Client") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post.return_value = mock_resp
        mock_client_class.return_value = mock_client_instance

        # Patch _minimax_save_auth_state to avoid touching the auth store
        with patch("hermes_cli.auth._minimax_save_auth_state"):
            result = _refresh_minimax_oauth_state(state)

    assert result["access_token"] == "new-access"
    assert result["refresh_token"] == "new-refresh"
    assert result["expires_in"] == 7200


def test_refresh_updates_access_token_absolute_ms_expired_in():
    """Refresh payload may use unix-ms absolute ``expired_in`` (same as device-code)."""
    now0 = datetime.now(timezone.utc)
    abs_ms = int((now0.timestamp() + 1800) * 1000)

    state = {
        "access_token": "old-access",
        "refresh_token": "my-refresh",
        "portal_base_url": MINIMAX_OAUTH_GLOBAL_BASE,
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "expires_at": _future_iso(MINIMAX_OAUTH_REFRESH_SKEW_SECONDS - 1),
    }

    new_token_body = {
        "status": "success",
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expired_in": abs_ms,
    }

    mock_resp = _make_httpx_response(200, new_token_body)

    with patch("httpx.Client") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post.return_value = mock_resp
        mock_client_class.return_value = mock_client_instance

        with patch("hermes_cli.auth._minimax_save_auth_state"):
            result = _refresh_minimax_oauth_state(state)

    assert result["access_token"] == "new-access"
    assert 1790 <= result["expires_in"] <= 1810
    exp = datetime.fromisoformat(result["expires_at"].replace("Z", "+00:00"))
    skew = exp.timestamp() - datetime.now(timezone.utc).timestamp()
    assert 1790 <= skew <= 1810


# ---------------------------------------------------------------------------
# 10. test_refresh_reuse_triggers_relogin_required
# ---------------------------------------------------------------------------

def test_refresh_reuse_triggers_relogin_required():
    """On 400 + invalid_grant body, relogin_required should be set."""
    state = {
        "access_token": "old-access",
        "refresh_token": "old-refresh",
        "portal_base_url": MINIMAX_OAUTH_GLOBAL_BASE,
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "expires_at": _past_iso(100),  # already expired
    }

    bad_resp = _make_httpx_response(400, text="invalid_grant")
    bad_resp.json.side_effect = Exception("no json")
    bad_resp.text = "invalid_grant"
    bad_resp.reason_phrase = "Bad Request"

    with patch("httpx.Client") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post.return_value = bad_resp
        mock_client_class.return_value = mock_client_instance

        with pytest.raises(AuthError) as exc_info:
            _refresh_minimax_oauth_state(state)

    assert exc_info.value.code == "refresh_failed"
    assert exc_info.value.relogin_required is True


# ---------------------------------------------------------------------------
# 11. test_resolve_credentials_requires_login
# ---------------------------------------------------------------------------

def test_resolve_credentials_requires_login():
    """When no state is stored, resolve_minimax_oauth_runtime_credentials raises."""
    with patch("hermes_cli.auth.get_provider_auth_state", return_value=None):
        with pytest.raises(AuthError) as exc_info:
            resolve_minimax_oauth_runtime_credentials()

    assert exc_info.value.code == "not_logged_in"
    assert exc_info.value.relogin_required is True


# ---------------------------------------------------------------------------
# 12. test_provider_registry_contains_minimax_oauth
# ---------------------------------------------------------------------------

def test_provider_registry_contains_minimax_oauth():
    assert "minimax-oauth" in PROVIDER_REGISTRY
    pconfig = PROVIDER_REGISTRY["minimax-oauth"]
    assert pconfig.auth_type == "oauth_minimax"
    assert pconfig.client_id == MINIMAX_OAUTH_CLIENT_ID
    assert MINIMAX_OAUTH_GLOBAL_BASE in pconfig.portal_base_url
    assert MINIMAX_OAUTH_GLOBAL_INFERENCE in pconfig.inference_base_url
    assert "cn_portal_base_url" in pconfig.extra
    assert "cn_inference_base_url" in pconfig.extra


# ---------------------------------------------------------------------------
# 13. test_minimax_oauth_alias_resolves
# ---------------------------------------------------------------------------

def test_minimax_oauth_alias_resolves():
    from hermes_cli.auth import resolve_provider
    # Only test that minimax-oauth itself resolves (alias resolution is tested in models)
    result = resolve_provider("minimax-oauth")
    assert result == "minimax-oauth"


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

def test_get_minimax_oauth_auth_status_logged_in():
    state = {
        "access_token": "tok",
        "expires_at": _future_iso(3600),
        "region": "global",
    }

    with patch("hermes_cli.auth.get_provider_auth_state", return_value=state):
        status = get_minimax_oauth_auth_status()

    assert status["logged_in"] is True
    assert status["region"] == "global"


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
