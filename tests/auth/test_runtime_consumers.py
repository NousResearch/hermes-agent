"""Runtime credential consumers retain explicit profile ownership."""
from contextlib import nullcontext
from dataclasses import replace

import pytest

from auth.api_keys import resolve_api_key_provider_secret
from auth.keepalive import refresh_nous_auth_keepalive_once
from hermes_cli.config_credentials import credential_pool_environment
from tests.auth.test_pool_environment import profile_scope, multiplex_scope


def test_api_key_resolution_uses_each_profiles_file_and_rejects_retained_inputs(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / ".env").write_text("OPENAI_API_KEY=profile-a-secret\n", encoding="utf-8")
    (b / ".env").write_text("OPENAI_API_KEY=profile-b-secret\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "launch-profile-secret")
    with profile_scope(a):
        supplied = credential_pool_environment()
        config = supplied.provider_config("openai-api")
        assert resolve_api_key_provider_secret("openai-api", config, environment=supplied) == (
            "profile-a-secret", "OPENAI_API_KEY")
    with profile_scope(b):
        with pytest.raises(ValueError, match="different profile"):
            resolve_api_key_provider_secret("openai-api", config, environment=supplied)
        own = credential_pool_environment()
        assert resolve_api_key_provider_secret("openai-api", config, environment=own) == (
            "profile-b-secret", "OPENAI_API_KEY")
    with profile_scope(a):
        assert resolve_api_key_provider_secret("openai-api", config, environment=supplied)[0] == "profile-a-secret"


def test_api_key_scope_check_precedes_source_callbacks(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with profile_scope(a):
        supplied = replace(credential_pool_environment(),
                           read_secret=lambda name: pytest.fail("foreign credential source read"))
    with profile_scope(b):
        with pytest.raises(ValueError, match="different profile"):
            resolve_api_key_provider_secret("copilot", None, environment=supplied)


def test_keepalive_rejects_a_factory_for_another_profile_before_storage(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with profile_scope(a):
        supplied = credential_pool_environment()
    monkeypatch.setattr("auth.keepalive.get_provider_auth_state",
                        lambda provider: pytest.fail("foreign authentication store read"))
    with profile_scope(b):
        with pytest.raises(ValueError, match="different profile"):
            refresh_nous_auth_keepalive_once(
                environment_factory=lambda: supplied, scope_context=nullcontext)


def test_spotify_plugin_availability_follows_the_active_profile(tmp_path):
    from auth.provider_state import _save_active_provider_state
    from plugins.spotify.tools import _check_spotify_available

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with profile_scope(a):
        _save_active_provider_state("spotify", {
            "auth_type": "oauth_pkce", "access_token": "owned-access",
            "refresh_token": "owned-refresh", "expires_at": "2030-01-01T00:00:00+00:00",
        })
        assert _check_spotify_available() is True
    with profile_scope(b):
        assert _check_spotify_available() is False
        assert not (b / "auth.json").exists()
    with profile_scope(a):
        assert _check_spotify_available() is True
