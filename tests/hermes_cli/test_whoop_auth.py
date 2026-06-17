from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import auth as auth_mod


def test_whoop_default_api_base_url_uses_developer_v2() -> None:
    assert auth_mod.DEFAULT_WHOOP_API_BASE_URL == "https://api.prod.whoop.com/developer/v2"


def test_store_provider_state_whoop_does_not_overwrite_active_provider() -> None:
    auth_store = {"active_provider": "nous", "providers": {}}

    auth_mod._store_provider_state(
        auth_store,
        "whoop",
        {"access_token": "tok"},
        set_active=False,
    )

    assert auth_store["active_provider"] == "nous"
    assert auth_store["providers"]["whoop"]["access_token"] == "tok"


def test_resolve_whoop_runtime_credentials_refreshes_expiring_token(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        store["active_provider"] = "nous"
        auth_mod._store_provider_state(
            store,
            "whoop",
            {
                "client_id": "whoop-client",
                "client_secret": "whoop-secret",
                "redirect_uri": auth_mod.DEFAULT_WHOOP_REDIRECT_URI,
                "api_base_url": auth_mod.DEFAULT_WHOOP_API_BASE_URL,
                "oauth_base_url": auth_mod.DEFAULT_WHOOP_OAUTH_BASE_URL,
                "scope": auth_mod.DEFAULT_WHOOP_SCOPE,
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
        "_refresh_whoop_oauth_state",
        lambda state, timeout_seconds=20.0: {
            **{k: v for k, v in state.items() if k != "client_secret"},
            "access_token": "fresh-token",
            "expires_at": "2099-01-01T00:00:00+00:00",
        },
    )

    creds = auth_mod.resolve_whoop_runtime_credentials()

    assert creds["access_token"] == "fresh-token"
    assert "refresh_token" not in creds
    persisted = auth_mod.get_provider_auth_state("whoop")
    assert persisted is not None
    assert persisted["access_token"] == "fresh-token"
    assert "client_secret" not in persisted
    assert auth_mod.get_active_provider() == "nous"


def test_resolve_whoop_runtime_credentials_skips_refresh_if_fresh(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    refresh_calls: list = []

    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        auth_mod._store_provider_state(
            store,
            "whoop",
            {
                "client_id": "whoop-client",
                "access_token": "fresh-token",
                "refresh_token": "rtoken",
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
            set_active=False,
        )
        auth_mod._save_auth_store(store)

    monkeypatch.setattr(
        auth_mod,
        "_refresh_whoop_oauth_state",
        lambda state, **kw: refresh_calls.append(1) or state,
    )

    creds = auth_mod.resolve_whoop_runtime_credentials()
    assert creds["access_token"] == "fresh-token"
    assert "refresh_token" not in creds
    assert refresh_calls == []


def test_resolve_whoop_runtime_credentials_uses_provider_state_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    fallback_state = {
        "client_id": "global-whoop-client",
        "access_token": "global-access-token",
        "refresh_token": "global-refresh-token",
        "expires_at": "2099-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr(auth_mod, "get_provider_auth_state", lambda provider: fallback_state if provider == "whoop" else None)

    creds = auth_mod.resolve_whoop_runtime_credentials()

    assert creds["access_token"] == "global-access-token"
    assert "refresh_token" not in creds


def test_get_whoop_auth_status_logged_in(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        auth_mod._store_provider_state(
            store,
            "whoop",
            {
                "client_id": "whoop-client",
                "access_token": "tok",
                "refresh_token": "rtok",
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
            set_active=False,
        )
        auth_mod._save_auth_store(store)

    status = auth_mod.get_whoop_auth_status()
    assert status["logged_in"] is True
    assert status.get("client_id") == "whoop-client"
    assert "has_refresh_token" in status


def test_get_whoop_auth_status_logged_out(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status = auth_mod.get_whoop_auth_status()
    assert status["logged_in"] is False


def test_whoop_state_token_is_exactly_eight_chars() -> None:
    state = auth_mod._whoop_state_token()
    assert len(state) == 8


def test_whoop_interactive_setup_persists_client_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    inputs = iter(["wizard-client-123"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "wizard-secret-456")
    monkeypatch.setattr(auth_mod, "webbrowser", SimpleNamespace(open=lambda *_a, **_k: False))
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)

    result = auth_mod._whoop_interactive_setup(
        redirect_uri_hint=auth_mod.DEFAULT_WHOOP_REDIRECT_URI,
    )
    assert result == "wizard-client-123"

    env_path = tmp_path / ".env"
    assert env_path.exists()
    env_text = env_path.read_text()
    assert "HERMES_WHOOP_CLIENT_ID=wizard-client-123" in env_text
    assert "HERMES_WHOOP_CLIENT_SECRET=wizard-secret-456" in env_text

    output = capsys.readouterr().out
    assert auth_mod.WHOOP_DOCS_URL in output


def test_whoop_interactive_setup_empty_aborts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(auth_mod, "webbrowser", SimpleNamespace(open=lambda *_a, **_k: False))
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)

    with pytest.raises(SystemExit):
        auth_mod._whoop_interactive_setup(
            redirect_uri_hint=auth_mod.DEFAULT_WHOOP_REDIRECT_URI,
        )

    env_path = tmp_path / ".env"
    if env_path.exists():
        assert "HERMES_WHOOP_CLIENT_ID" not in env_path.read_text()


def test_get_auth_status_whoop_dispatches(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status = auth_mod.get_auth_status("whoop")
    assert "logged_in" in status

def test_whoop_token_state_does_not_persist_client_secret() -> None:
    state = auth_mod._whoop_token_payload_to_state(
        {"access_token": "access", "refresh_token": "refresh", "expires_in": 3600},
        client_id="whoop-client",
        client_secret="do-not-store",
        redirect_uri=auth_mod.DEFAULT_WHOOP_REDIRECT_URI,
        requested_scope=auth_mod.DEFAULT_WHOOP_SCOPE,
        oauth_base_url=auth_mod.DEFAULT_WHOOP_OAUTH_BASE_URL,
        api_base_url=auth_mod.DEFAULT_WHOOP_API_BASE_URL,
        previous_state={"client_secret": "old-secret"},
    )

    assert state["client_id"] == "whoop-client"
    assert state["access_token"] == "access"
    assert "client_secret" not in state
