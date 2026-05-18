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


def test_spotify_logout_does_not_reset_model_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  default: gemini-3-flash\n"
        "  provider: custom:local\n"
        "  base_url: http://localhost:11434/v1\n"
        "  api_key: ${LOCAL_API_KEY}\n",
        encoding="utf-8",
    )

    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        auth_mod._store_provider_state(
            store,
            "spotify",
            {
                "client_id": "spotify-client",
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
            set_active=False,
        )
        auth_mod._save_auth_store(store)

    auth_mod.logout_command(SimpleNamespace(provider="spotify"))

    output = capsys.readouterr().out
    assert "Logged out of Spotify." in output
    assert "Model provider configuration was unchanged." in output
    assert auth_mod.get_provider_auth_state("spotify") is None
    assert config_path.read_text(encoding="utf-8") == (
        "model:\n"
        "  default: gemini-3-flash\n"
        "  provider: custom:local\n"
        "  base_url: http://localhost:11434/v1\n"
        "  api_key: ${LOCAL_API_KEY}\n"
    )


def test_spotify_interactive_setup_persists_client_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """The wizard writes HERMES_SPOTIFY_CLIENT_ID to .env and returns the value."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": "wizard-client-123")
    # Prevent actually opening the browser during tests.
    monkeypatch.setattr(auth_mod, "webbrowser", SimpleNamespace(open=lambda *_a, **_k: False))
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)

    result = auth_mod._spotify_interactive_setup(
        redirect_uri_hint=auth_mod.DEFAULT_SPOTIFY_REDIRECT_URI,
    )
    assert result == "wizard-client-123"

    env_path = tmp_path / ".env"
    assert env_path.exists()
    env_text = env_path.read_text()
    assert "HERMES_SPOTIFY_CLIENT_ID=wizard-client-123" in env_text
    # Default redirect URI should NOT be persisted.
    assert "HERMES_SPOTIFY_REDIRECT_URI" not in env_text

    # Docs URL should appear in wizard output so users can find the guide.
    output = capsys.readouterr().out
    assert auth_mod.SPOTIFY_DOCS_URL in output


def test_spotify_interactive_setup_empty_aborts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty input aborts cleanly instead of persisting an empty client_id."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(auth_mod, "webbrowser", SimpleNamespace(open=lambda *_a, **_k: False))
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)

    with pytest.raises(SystemExit):
        auth_mod._spotify_interactive_setup(
            redirect_uri_hint=auth_mod.DEFAULT_SPOTIFY_REDIRECT_URI,
        )

    env_path = tmp_path / ".env"
    if env_path.exists():
        assert "HERMES_SPOTIFY_CLIENT_ID" not in env_path.read_text()
