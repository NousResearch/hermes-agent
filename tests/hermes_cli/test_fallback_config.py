"""Tests for hermes_cli/fallback_config.py — fallback entry API-key resolution."""

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import (
    get_auxiliary_fallback_chain,
    resolve_entry_api_key,
)


def test_auxiliary_fallback_chain_precedes_and_deduplicates_global_chain():
    task_entry = {
        "provider": "custom",
        "model": "review-backup",
        "base_url": "https://review.example/v1/",
    }
    global_entry = {
        "provider": "openrouter",
        "model": "openai/gpt-5.5",
    }
    config = {
        "auxiliary": {
            "curator": {
                "fallback_chain": [task_entry, {**task_entry, "base_url": "https://review.example/v1"}],
            },
        },
        "fallback_providers": [global_entry],
    }

    assert get_auxiliary_fallback_chain(config, "curator") == [
        {**task_entry, "base_url": "https://review.example/v1"},
        global_entry,
    ]


def test_auxiliary_fallback_chain_ignores_empty_and_malformed_entries():
    config = {
        "auxiliary": {
            "curator": {
                "fallback_chain": [None, "bad", {}, {"provider": "custom"}],
            },
        },
        "fallback_providers": [42, {"model": "missing-provider"}],
    }

    assert get_auxiliary_fallback_chain(config, "curator") == []


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
