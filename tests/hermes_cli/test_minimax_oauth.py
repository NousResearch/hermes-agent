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
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.auth import (
    AuthError,
    MINIMAX_OAUTH_CLIENT_ID,
    MINIMAX_OAUTH_GLOBAL_BASE,
    MINIMAX_OAUTH_GLOBAL_INFERENCE,
    MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
    _minimax_pkce_pair,
    _minimax_request_user_code,
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

    def _capture_save(s, **_kwargs):
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
             side_effect=lambda s, **_k: saved_states.append(dict(s)),
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


# ---------------------------------------------------------------------------
# Concurrent refresh + active-provider invariants (teknium1 review of #133534)
# ---------------------------------------------------------------------------

def _minimax_state(tmp_path, *, expires_in: int = 30, refresh_token: str = "r1",
                   access_token: str = "tok1") -> dict:
    from hermes_cli.auth import MINIMAX_OAUTH_CLIENT_ID

    state = {
        "provider": "minimax-oauth",
        "region": "global",
        "portal_base_url": "https://portal.minimax.io",
        "inference_base_url": "https://api.minimax.io/anthropic",
        "client_id": MINIMAX_OAUTH_CLIENT_ID,
        "token_type": "Bearer",
        "access_token": access_token,
        "refresh_token": refresh_token,
        **_minimax_expiry_fields_for_test(expires_in),
    }
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "nous",
        "providers": {"minimax-oauth": state},
    }), encoding="utf-8")
    return state


def _minimax_expiry_fields_for_test(expires_in: int) -> dict:
    from hermes_cli.auth_minimax import _minimax_expiry_fields

    return _minimax_expiry_fields(expires_in)


class _RotatingPortal:
    """Loopback MiniMax portal: rotates the refresh token on each successful
    refresh and rejects reuse of a consumed token (``refresh_token_reused``),
    the upstream behavior our quarantine treats as relogin-required."""

    def __init__(self):
        import http.server
        import socketserver

        self.refresh_calls: list[str] = []
        self.reuse_rejections = 0
        self._current_refresh = "r1"
        self._lock = threading.Lock()
        portal = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
                used = (body.get("refresh_token") or [""])[0]
                with portal._lock:
                    portal.refresh_calls.append(used)
                    if used != portal._current_refresh:
                        portal.reuse_rejections += 1
                        status, payload = 400, {"base_resp": {"status_msg": "refresh_token_reused"}}
                    else:
                        portal._current_refresh = "r" + str(int(portal._current_refresh[1:]) + 1)
                        status, payload = 200, {
                            "status": "success",
                            "access_token": f"tok-{portal._current_refresh}",
                            "refresh_token": portal._current_refresh,
                            "expired_in": 900,
                        }
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):
                pass

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


def test_concurrent_refresh_rotates_token_once_and_keeps_login(tmp_path, monkeypatch):
    """Two token providers racing on a near-expiry token must produce exactly one
    refresh POST, no errors, and intact tokens in auth.json.

    MiniMax refresh tokens are single-use and rotate; without a lock across
    read → refresh → save in ``_minimax_fresh_state``, both racers POST the
    same still-valid ``r1``, the portal rejects the second (``refresh_token_reused``),
    and the quarantine wipes the winner's fresh tokens (the review's case D).
    """
    from hermes_cli.auth import build_minimax_oauth_token_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _minimax_state(tmp_path, expires_in=30, refresh_token="r1")
    portal = _RotatingPortal().start()
    try:
        # Point the persisted state at the loopback portal.
        store = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
        store["providers"]["minimax-oauth"]["portal_base_url"] = portal.url
        (tmp_path / "auth.json").write_text(json.dumps(store), encoding="utf-8")

        # Both callers must observe the same near-expiry state before either
        # refreshes (the race window): release them together, no sleeps.
        barrier = threading.Barrier(3, timeout=30)
        results: dict = {}

        def _provider_call(name: str):
            provider = build_minimax_oauth_token_provider()
            barrier.wait()
            try:
                results[name] = ("ok", provider())
            except Exception as exc:  # noqa: BLE001 -- test witness, recorded not swallowed
                results[name] = ("error", repr(exc))

        threads = [threading.Thread(target=_provider_call, args=(f"t{i}",)) for i in range(2)]
        for t in threads:
            t.start()
        barrier.wait()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads), "a provider call hung"

        # Exactly one refresh POST, zero refresh_token_reused, zero errors.
        assert len(portal.refresh_calls) == 1, (
            f"expected exactly 1 refresh POST, got {portal.refresh_calls}"
        )
        assert portal.refresh_calls[0] == "r1"
        assert portal.reuse_rejections == 0, (
            "a second caller replayed the single-use refresh token (unlocked race)"
        )
        statuses = {name: outcome for name, (outcome, _) in results.items()}
        assert statuses == {"t0": "ok", "t1": "ok"}, (
            f"a concurrent provider call failed: {results}"
        )
        tokens = sorted(str(v) for outcome, v in results.values() if outcome == "ok")
        assert tokens == ["tok-r2", "tok-r2"], (
            f"both callers must see the winner's fresh token: {results}"
        )

        # auth.json kept the rotated pair, not a quarantine wipe.
        final = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
        final_state = final["providers"]["minimax-oauth"]
        assert final_state["access_token"] == "tok-r2", (
            f"auth.json lost the fix's fresh token: {final_state}"
        )
        assert final_state["refresh_token"] == "r2"
        assert "last_auth_error" not in final_state
    finally:
        portal.stop()


def test_aux_refresh_preserves_active_provider(tmp_path, monkeypatch):
    """An aux-triggered refresh rewrites credentials, not the user's provider
    choice: ``auth.json.active_provider`` stays ``nous`` (review case B'; the
    rule ``_save_provider_state_to_source`` already documents)."""
    from hermes_cli.auth import build_minimax_oauth_token_provider, get_active_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _minimax_state(tmp_path, expires_in=30, refresh_token="r1")
    portal = _RotatingPortal().start()
    try:
        store = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
        store["providers"]["minimax-oauth"]["portal_base_url"] = portal.url
        (tmp_path / "auth.json").write_text(json.dumps(store), encoding="utf-8")

        assert get_active_provider() == "nous"
        token = build_minimax_oauth_token_provider()()
        assert token == "tok-r2", "the refresh must actually run over the loopback portal"

        final = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
        assert final["active_provider"] == "nous", (
            "an aux-side refresh must not flip the user's active provider"
        )
        assert final["providers"]["minimax-oauth"]["access_token"] == "tok-r2"
    finally:
        portal.stop()


def test_minimax_oauth_login_sets_active_provider(tmp_path, monkeypatch):
    """``_minimax_oauth_login`` is the one path that legitimately makes
    minimax-oauth the active provider (the user just chose it)."""
    from hermes_cli import auth_minimax

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = {"version": 1, "active_provider": "", "providers": {}}
    (tmp_path / "auth.json").write_text(json.dumps(store), encoding="utf-8")

    token_data = {"access_token": "tok", "refresh_token": "r1", "expired_in": 900}
    with patch("hermes_cli.auth._minimax_pkce_pair", return_value=("v", "c", "s")), \
         patch("hermes_cli.auth._minimax_request_user_code", return_value={
             "verification_uri": "https://portal.minimax.io/device", "user_code": "ABCD",
             "expired_in": 600, "interval": 1000}), \
         patch("hermes_cli.auth._print_device_code_instructions"), \
         patch.object(auth_minimax, "_minimax_poll_token", return_value=token_data):
        auth_minimax._minimax_oauth_login(region="global", open_browser=False)

    final = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
    assert final["active_provider"] == "minimax-oauth"
    assert final["providers"]["minimax-oauth"]["access_token"] == "tok"
