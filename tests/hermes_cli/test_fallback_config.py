"""Tests for hermes_cli/fallback_config.py — fallback entry API-key resolution."""

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import resolve_entry_api_key

import pytest

from hermes_cli.auth import AuthError


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"


    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None


    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"

    def test_key_env_resolves_from_active_secret_scope_not_raw_env(self, monkeypatch):
        # Multiplexed gateway: os.environ holds another profile's key, but the
        # active per-turn secret scope holds this profile's key. The scoped
        # value must win — a raw os.getenv() would leak the other profile's
        # credential (issue #74311).
        monkeypatch.setenv("FB_KEY", "fake-other-profile-key")
        token = set_secret_scope({"FB_KEY": "fake-active-profile-key"})
        try:
            assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "fake-active-profile-key"
        finally:
            reset_secret_scope(token)

    def test_key_env_falls_back_to_env_when_no_active_scope(self, monkeypatch):
        # Non-multiplexed / single-profile behavior must be unchanged: with no
        # secret scope installed, resolution still reads os.environ.
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"


@pytest.mark.parametrize("entry", [
    {"api_key": ""}, {"api_key": None}, {"api_key": "  "},
    {"key_env": ""}, {"api_key_env": "  "}, {"key_env": "FB_MISSING"},
    {"api_key": "${FB_PRIVATE_REFERENCE}"},
    {"api_key": "prefix-${FB_PRIVATE_REFERENCE}-suffix", "key_env": "FB_KEY"},
    {"key_env": "FB_UNRESOLVED"}, {"api_key_env": "FB_BLANK"},
])
def test_strict_declared_credentials_fail_without_disclosing_values(entry, monkeypatch, caplog):
    monkeypatch.delenv("FB_MISSING", raising=False)
    monkeypatch.setenv("FB_KEY", "usable-synthetic-key")
    monkeypatch.setenv("FB_UNRESOLVED", "${FB_PRIVATE_REFERENCE}")
    monkeypatch.setenv("FB_BLANK", "  ")
    with pytest.raises(AuthError) as error:
        resolve_entry_api_key(entry, strict=True)
    assert error.value.code == "missing_api_key"
    assert str(error.value) == "Fallback entry has no usable explicit API key."
    assert "FB_" not in caplog.text
    assert "usable-synthetic-key" not in caplog.text


@pytest.mark.parametrize("entry,expected", [
    ({"api_key": " x ", "key_env": "FB_KEY"}, "x"),
    ({"api_key": "  ", "key_env": "FB_KEY"}, "scoped-key"),
    ({"api_key_env": "FB_KEY"}, "scoped-key"),
    ({"key_env": "FB_KEY", "api_key_env": "FB_OTHER"}, "scoped-key"),
    ({"provider": "lmstudio", "model": "local"}, None),
    ({}, None), (None, None),
])
def test_strict_preserves_precedence_alias_and_absent_fields(entry, expected):
    token = set_secret_scope({"FB_KEY": " scoped-key ", "FB_OTHER": "other-key"})
    try:
        assert resolve_entry_api_key(entry, strict=True) == expected
    finally:
        reset_secret_scope(token)


def test_strict_multiplex_miss_never_uses_ambient_key(monkeypatch):
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("FB_KEY", "other-profile-key")
    token = set_secret_scope({})
    try:
        with pytest.raises(AuthError, match="Fallback entry has no usable explicit API key"):
            resolve_entry_api_key({"key_env": "FB_KEY"}, strict=True)
    finally:
        reset_secret_scope(token)


@pytest.mark.parametrize("strict", [False, True])
def test_unscoped_secret_error_is_preserved(strict, monkeypatch):
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("FB_KEY", "other-profile-key")
    token = set_secret_scope(None)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            resolve_entry_api_key({"key_env": "FB_KEY"}, strict=strict)
    finally:
        reset_secret_scope(token)


def test_automatic_key_resolution_keeps_permissive_default(monkeypatch):
    monkeypatch.delenv("FB_MISSING", raising=False)
    assert resolve_entry_api_key({"key_env": "FB_MISSING"}) is None
    assert resolve_entry_api_key({"api_key": "${FB_MISSING}"}) == "${FB_MISSING}"
