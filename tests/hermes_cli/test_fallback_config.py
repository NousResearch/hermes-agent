"""Tests for hermes_cli/fallback_config.py — fallback entry API-key resolution."""

import pytest

from hermes_cli.fallback_config import (
    get_fallback_chain,
    resolve_entry_api_key,
    resolve_entry_api_mode,
)


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"

    def test_key_env_resolves_from_environment(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"

    def test_api_key_env_alias(self, monkeypatch):
        monkeypatch.setenv("FB_ALIAS_KEY", "alias-key")
        assert resolve_entry_api_key({"api_key_env": "FB_ALIAS_KEY"}) == "alias-key"

    def test_unset_env_var_returns_none(self, monkeypatch):
        monkeypatch.delenv("FB_MISSING", raising=False)
        # None (not "") lets resolve_runtime_provider fall through to the
        # provider's standard credential resolution.
        assert resolve_entry_api_key({"key_env": "FB_MISSING"}) is None

    def test_empty_env_var_returns_none(self, monkeypatch):
        monkeypatch.setenv("FB_EMPTY", "   ")
        assert resolve_entry_api_key({"key_env": "FB_EMPTY"}) is None

    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None

    def test_non_dict_returns_none(self):
        assert resolve_entry_api_key(None) is None
        assert resolve_entry_api_key("nope") is None  # type: ignore[arg-type]

    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"


class TestResolveEntryApiMode:
    def test_explicit_api_mode_is_preserved(self):
        assert resolve_entry_api_mode({"api_mode": "codex_responses"}) == "codex_responses"

    def test_provider_transport_name_is_normalized(self):
        assert resolve_entry_api_mode({"transport": "openai_chat"}) == "chat_completions"

    def test_unknown_explicit_mode_fails_closed(self):
        with pytest.raises(ValueError, match="unsupported fallback api_mode"):
            resolve_entry_api_mode({"api_mode": "made_up_transport"})

    def test_missing_mode_allows_legacy_inference(self):
        assert resolve_entry_api_mode({"provider": "openrouter"}) is None

    def test_chain_dedup_keeps_same_endpoint_with_distinct_transports(self):
        config = {
            "fallback_providers": [
                {
                    "provider": "relay",
                    "model": "shared-model",
                    "base_url": "https://relay.example/v1",
                    "api_mode": "chat_completions",
                }
            ],
            "fallback_model": {
                "provider": "relay",
                "model": "shared-model",
                "base_url": "https://relay.example/v1",
                "api_mode": "codex_responses",
            },
        }
        assert [entry["api_mode"] for entry in get_fallback_chain(config)] == [
            "chat_completions",
            "codex_responses",
        ]
