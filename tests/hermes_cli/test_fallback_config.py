"""Tests for hermes_cli/fallback_config.py."""

import logging

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import (
    effective_runtime_provider,
    get_fallback_chain,
    normalize_fallback_entries,
    resolve_entry_api_key,
)


def test_fallback_chain_accepts_compact_and_json_string_entries():
    assert get_fallback_chain({
        "fallback_providers": ["openrouter:qwen/qwen3.6-plus", "nous:model:free"],
        "fallback_model": '[{"provider":"anthropic","model":"claude-sonnet-4-6"}]',
    }) == [
        {"provider": "openrouter", "model": "qwen/qwen3.6-plus"},
        {"provider": "nous", "model": "model:free"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]


def test_invalid_configured_fallback_entries_are_reported(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_cli.fallback_config"):
        assert normalize_fallback_entries(["missing-model", {"provider": "nous"}, 7]) == []

    assert len(caplog.records) == 3
    assert all("Ignoring fallback entry" in record.message for record in caplog.records)


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


class TestEffectiveRuntimeProvider:
    """Named custom fallback entries must keep their configured identity (#98739)."""

    def test_named_custom_entry_keeps_configured_id(self):
        entry = {"provider": "my-custom-provider", "model": "some-model"}
        runtime = {"provider": "custom", "requested_provider": "my-custom-provider"}
        assert effective_runtime_provider(entry, runtime) == "my-custom-provider"

    def test_requested_provider_missing_falls_back_to_entry(self):
        entry = {"provider": "my-custom-provider", "model": "some-model"}
        runtime = {"provider": "custom"}
        assert effective_runtime_provider(entry, runtime) == "my-custom-provider"

    def test_builtin_provider_untouched(self):
        entry = {"provider": "openrouter", "model": "glm"}
        runtime = {"provider": "openrouter", "requested_provider": "openrouter"}
        assert effective_runtime_provider(entry, runtime) == "openrouter"

    def test_genuinely_bare_custom_stays_custom(self):
        # Ad-hoc endpoint: user literally configured provider: custom.
        entry = {"provider": "custom", "model": "some-model"}
        runtime = {"provider": "custom", "requested_provider": "custom"}
        assert effective_runtime_provider(entry, runtime) == "custom"

    def test_none_inputs_are_safe(self):
        assert effective_runtime_provider(None, None) == ""
