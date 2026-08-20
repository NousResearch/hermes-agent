"""Regression coverage for Spotify refresh-token flow selection.

Hermes mints refresh tokens via PKCE by default, where the refresh request
sends ``client_id`` in the body and no ``Authorization`` header. But users
can legitimately end up with a refresh token minted via the Authorization
Code flow (e.g. brought in from Bitwarden Secrets Manager, an external
script, or another tool) — those tokens are bound to ``client_secret`` +
HTTP Basic auth at the token endpoint and reject PKCE-style refreshes
with ``400 invalid_request``.

``_refresh_spotify_oauth_state`` therefore branches on the presence of a
``SPOTIFY_CLIENT_SECRET`` / ``HERMES_SPOTIFY_CLIENT_SECRET`` env var:

* secret present → HTTP Basic ``Authorization`` header, no ``client_id``
  in body. Compatible with Auth Code flow refresh tokens.
* secret absent → original PKCE behavior (``client_id`` in body, no
  ``Authorization`` header). Compatible with PKCE refresh tokens.

These tests pin the wire format for both branches so the selection
logic can't silently regress.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List

import httpx
import pytest

from hermes_cli import auth as auth_mod


class _PostRecorder:
    """Capture every ``httpx.post`` call without touching the network."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url, *, headers=None, data=None, timeout=None, **kw):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers or {}),
                "data": dict(data or {}),
                "timeout": timeout,
                "extra": kw,
            }
        )
        return self.response


def _ok_refresh_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "access_token": "AT-fresh",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "user-library-read",
            # Spotify refresh responses commonly OMIT refresh_token — that
            # means "keep using the existing one". The state-builder must
            # fall back to previous_state.refresh_token. We rely on that
            # behavior in the assertions below.
        },
    )


def _state(**overrides: Any) -> Dict[str, Any]:
    base = {
        "client_id": "spotify-client",
        "redirect_uri": "http://127.0.0.1:43827/spotify/callback",
        "accounts_base_url": auth_mod.DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
        "api_base_url": auth_mod.DEFAULT_SPOTIFY_API_BASE_URL,
        "scope": auth_mod.DEFAULT_SPOTIFY_SCOPE,
        "access_token": "AT-stale",
        "refresh_token": "RT-existing",
        "token_type": "Bearer",
        "expires_at": "2000-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient Spotify env vars so tests are deterministic.

    A developer's shell may have ``SPOTIFY_CLIENT_ID`` set (Bitwarden /
    direnv / .env), which ``_spotify_client_id`` reads with higher
    priority than the state dict. Clear it (and the secret variants)
    so each test sees a clean slate.
    """
    for key in (
        "SPOTIFY_CLIENT_ID",
        "HERMES_SPOTIFY_CLIENT_ID",
        "SPOTIFY_CLIENT_SECRET",
        "HERMES_SPOTIFY_CLIENT_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Branch 1: PKCE refresh (no client_secret in env)
# ---------------------------------------------------------------------------


def test_refresh_uses_pkce_body_when_no_client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default path — client_id in body, no Authorization header."""
    monkeypatch.delenv("HERMES_SPOTIFY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)

    recorder = _PostRecorder(_ok_refresh_response())
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)

    new_state = auth_mod._refresh_spotify_oauth_state(_state())

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["url"] == f"{auth_mod.DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL}/api/token"
    assert call["data"]["grant_type"] == "refresh_token"
    assert call["data"]["refresh_token"] == "RT-existing"
    # PKCE: client_id in body
    assert call["data"]["client_id"] == "spotify-client"
    # PKCE: NO Basic auth header
    assert "Authorization" not in call["headers"]
    assert call["headers"]["Content-Type"] == "application/x-www-form-urlencoded"

    # State refresh succeeded
    assert new_state["access_token"] == "AT-fresh"
    # Refresh token preserved across the call (server omitted it in response)
    assert new_state["refresh_token"] == "RT-existing"


# ---------------------------------------------------------------------------
# Branch 2: Basic auth refresh (client_secret in env)
# ---------------------------------------------------------------------------


def test_refresh_uses_basic_auth_when_client_secret_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auth Code flow path — Basic auth header, no client_id in body."""
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "shhh-secret")
    monkeypatch.delenv("HERMES_SPOTIFY_CLIENT_SECRET", raising=False)

    recorder = _PostRecorder(_ok_refresh_response())
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)

    new_state = auth_mod._refresh_spotify_oauth_state(_state())

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["url"] == f"{auth_mod.DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL}/api/token"
    # Basic auth: NO client_id in body (would conflict with the spec)
    assert "client_id" not in call["data"]
    assert call["data"]["grant_type"] == "refresh_token"
    assert call["data"]["refresh_token"] == "RT-existing"
    # Basic auth: Authorization header with base64-encoded creds
    expected_creds = base64.b64encode(b"spotify-client:shhh-secret").decode()
    assert call["headers"]["Authorization"] == f"Basic {expected_creds}"
    assert call["headers"]["Content-Type"] == "application/x-www-form-urlencoded"

    # State refresh succeeded; refresh token preserved
    assert new_state["access_token"] == "AT-fresh"
    assert new_state["refresh_token"] == "RT-existing"


def test_hermes_prefixed_client_secret_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """``HERMES_SPOTIFY_CLIENT_SECRET`` wins over ``SPOTIFY_CLIENT_SECRET``."""
    monkeypatch.setenv("HERMES_SPOTIFY_CLIENT_SECRET", "hermes-wins")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "should-be-ignored")

    recorder = _PostRecorder(_ok_refresh_response())
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)

    auth_mod._refresh_spotify_oauth_state(_state())

    expected_creds = base64.b64encode(b"spotify-client:hermes-wins").decode()
    assert recorder.calls[0]["headers"]["Authorization"] == f"Basic {expected_creds}"


def test_blank_client_secret_falls_back_to_pkce(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only secret is treated as absent — don't half-set Basic auth."""
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "   ")
    monkeypatch.delenv("HERMES_SPOTIFY_CLIENT_SECRET", raising=False)

    recorder = _PostRecorder(_ok_refresh_response())
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)

    auth_mod._refresh_spotify_oauth_state(_state())

    call = recorder.calls[0]
    assert "Authorization" not in call["headers"]
    assert call["data"]["client_id"] == "spotify-client"


# ---------------------------------------------------------------------------
# Failure-mode coverage (unchanged from pre-patch behavior)
# ---------------------------------------------------------------------------


def test_refresh_token_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("HERMES_SPOTIFY_CLIENT_SECRET", raising=False)

    with pytest.raises(auth_mod.AuthError) as excinfo:
        auth_mod._refresh_spotify_oauth_state(_state(refresh_token=""))
    assert excinfo.value.code == "spotify_refresh_token_missing"


def test_400_response_raises_with_relogin_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("HERMES_SPOTIFY_CLIENT_SECRET", raising=False)

    err_response = httpx.Response(400, text='{"error":"invalid_request"}')
    monkeypatch.setattr("hermes_cli.auth.httpx.post", _PostRecorder(err_response))

    with pytest.raises(auth_mod.AuthError) as excinfo:
        auth_mod._refresh_spotify_oauth_state(_state())
    assert excinfo.value.code == "spotify_refresh_failed"
    assert excinfo.value.relogin_required is True
