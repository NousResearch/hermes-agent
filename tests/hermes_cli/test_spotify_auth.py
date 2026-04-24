from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import auth as auth_mod


def test_store_provider_state_can_skip_active_provider() -> None:
    auth_store = {"active_provider": "nous", "providers": {}}

    auth_mod._store_provider_state(
        auth_store,
        "spotify",
        {"access_token": "abc"},
        set_active=False,
    )

    assert auth_store["active_provider"] == "nous"
    assert auth_store["providers"]["spotify"]["access_token"] == "abc"


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
