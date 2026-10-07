from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import auth as auth_mod
import hermes_cli.auth_spotify as auth_spotify
from hermes_cli.auth import AuthError, resolve_spotify_runtime_credentials




def test_resolve_spotify_runtime_credentials_refreshes_without_changing_active_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        store["active_provider"] = "nous"
        auth_mod._store_provider_state(
            store,
            "spotify",
            {
                "client_id": "spotify-client",
                "redirect_uri": "http://127.0.0.1:43827/spotify/callback",
                "api_base_url": auth_mod.DEFAULT_SPOTIFY_API_BASE_URL,
                "accounts_base_url": auth_mod.DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
                "scope": auth_mod.DEFAULT_SPOTIFY_SCOPE,
                "access_token": "expired-token",
                "refresh_token": "refresh-token",
                "token_type": "Bearer",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
            set_active=False,
        )
        auth_mod._save_auth_store(store)

    monkeypatch.setattr(
        auth_mod,
        "_refresh_spotify_oauth_state",
        lambda state, timeout_seconds=20.0: {
            **state,
            "access_token": "fresh-token",
            "expires_at": "2099-01-01T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        auth_spotify,
        "_refresh_spotify_oauth_state",
        lambda state, timeout_seconds=20.0: {
            **state,
            "access_token": "fresh-token",
            "expires_at": "2099-01-01T00:00:00+00:00",
        },
    )

    creds = auth_mod.resolve_spotify_runtime_credentials()

    assert creds["access_token"] == "fresh-token"
    persisted = auth_mod.get_provider_auth_state("spotify")
    assert persisted is not None
    assert persisted["access_token"] == "fresh-token"
    assert auth_mod.get_active_provider() == "nous"


def test_auth_spotify_status_command_reports_logged_in(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_mod,
        "get_auth_status",
        lambda provider=None: {
            "logged_in": True,
            "auth_type": "oauth_pkce",
            "client_id": "spotify-client",
            "redirect_uri": "http://127.0.0.1:43827/spotify/callback",
            "scope": "user-library-read",
        },
    )

    from hermes_cli.auth_commands import auth_status_command

    auth_status_command(SimpleNamespace(provider="spotify"))
    output = capsys.readouterr().out
    assert "spotify: logged in" in output
    assert "client_id: spotify-client" in output






# ---------------------------------------------------------------------------
# Quarantine: terminal refresh failure clears dead tokens (#28139)
# ---------------------------------------------------------------------------

_STALE_SPOTIFY_STATE = {
    "client_id": "test-client",
    "redirect_uri": "http://127.0.0.1:43827/spotify/callback",
    "api_base_url": auth_mod.DEFAULT_SPOTIFY_API_BASE_URL,
    "accounts_base_url": auth_mod.DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
    "scope": auth_mod.DEFAULT_SPOTIFY_SCOPE,
    "granted_scope": auth_mod.DEFAULT_SPOTIFY_SCOPE,
    "token_type": "Bearer",
    "access_token": "dead-access-token",
    "refresh_token": "dead-refresh-token",
    "expires_at": "2000-01-01T00:00:00+00:00",
    "expires_in": 3600,
    "obtained_at": "2000-01-01T00:00:00+00:00",
    "auth_type": "oauth_pkce",
}


def _seed_spotify_state(tmp_path, state: dict) -> None:
    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        store["active_provider"] = "nous"
        auth_mod._store_provider_state(store, "spotify", state, set_active=False)
        auth_mod._save_auth_store(store)


def test_resolve_credentials_quarantines_dead_tokens_on_terminal_refresh_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal refresh failure (relogin_required=True + refresh_token present)
    must clear access_token/refresh_token/expires_* from auth.json and write a
    last_auth_error marker so subsequent calls fail fast without a network retry.
    Mirrors Nous / xAI-OAuth / Codex-OAuth / MiniMax quarantine pattern.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_spotify_state(tmp_path, dict(_STALE_SPOTIFY_STATE))

    def _terminal_refresh(_state, **_kw):
        raise AuthError(
            "Spotify token refresh failed. Run `hermes auth spotify` again.",
            provider="spotify",
            code="spotify_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "_refresh_spotify_oauth_state", _terminal_refresh)
    monkeypatch.setattr(auth_spotify, "_refresh_spotify_oauth_state", _terminal_refresh)

    with pytest.raises(AuthError) as exc_info:
        resolve_spotify_runtime_credentials(force_refresh=True)

    assert exc_info.value.code == "spotify_refresh_failed"
    assert exc_info.value.relogin_required is True

    persisted = auth_mod.get_provider_auth_state("spotify")
    assert persisted is not None

    # Dead OAuth fields must be cleared.
    assert "access_token" not in persisted
    assert "refresh_token" not in persisted
    assert "expires_at" not in persisted
    assert "expires_in" not in persisted
    assert "obtained_at" not in persisted

    # Non-credential metadata must be preserved.
    assert persisted["client_id"] == "test-client"
    assert persisted["api_base_url"] == auth_mod.DEFAULT_SPOTIFY_API_BASE_URL
    assert persisted["accounts_base_url"] == auth_mod.DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL

    # Structured diagnostic blob must be written.
    err = persisted.get("last_auth_error")
    assert isinstance(err, dict)
    assert err["provider"] == "spotify"
    assert err["code"] == "spotify_refresh_failed"
    assert err["reason"] == "runtime_refresh_failure"
    assert err["relogin_required"] is True
    assert "at" in err

    # Active provider must be unchanged.
    assert auth_mod.get_active_provider() == "nous"


