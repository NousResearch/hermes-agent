"""Tests for hermes_cli/fallback_config.py — API-key resolution and chain round-trip."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import resolve_entry_api_key


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


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class TestGetFallbackChainIntegration:
    """set_config_value() persists fallback_providers as a JSON string; verify
    get_fallback_chain() still returns the correct entries after reload."""

    def test_json_string_round_trip(self, isolated_home):
        """hermes config set serializes lists as JSON strings; chain must survive."""
        from hermes_cli.config import set_config_value, load_config
        from hermes_cli.fallback_config import get_fallback_chain

        entries = [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            {"provider": "nous", "model": "Hermes-4-Llama-3.1-405B"},
            {"provider": "llamacpp", "model": "local", "base_url": "http://localhost:8080/v1"},
        ]
        set_config_value("fallback_providers", json.dumps(entries))

        cfg = load_config()
        chain = get_fallback_chain(cfg)

        assert len(chain) == 3
        assert chain[0]["provider"] == "openrouter"
        assert chain[0]["model"] == "anthropic/claude-sonnet-4.6"
        assert chain[1]["provider"] == "nous"
        assert chain[1]["model"] == "Hermes-4-Llama-3.1-405B"
        assert chain[2]["provider"] == "llamacpp"
        assert chain[2]["base_url"] == "http://localhost:8080/v1"

    def test_order_preserved(self, isolated_home):
        """Entry order from the JSON string must be preserved in the chain."""
        from hermes_cli.config import set_config_value, load_config
        from hermes_cli.fallback_config import get_fallback_chain

        entries = [
            {"provider": "first", "model": "m1"},
            {"provider": "second", "model": "m2"},
            {"provider": "third", "model": "m3"},
        ]
        set_config_value("fallback_providers", json.dumps(entries))

        chain = get_fallback_chain(load_config())

        assert [e["provider"] for e in chain] == ["first", "second", "third"]

    def test_native_list_still_works(self, isolated_home):
        """YAML-native list format (written directly) must continue to work."""
        import yaml
        from hermes_cli.config import load_config
        from hermes_cli.fallback_config import get_fallback_chain

        config_path = isolated_home / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({
                "fallback_providers": [
                    {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
                ]
            }),
            encoding="utf-8",
        )

        chain = get_fallback_chain(load_config())

        assert len(chain) == 1
        assert chain[0]["provider"] == "openrouter"

    def test_empty_config_returns_empty_chain(self, isolated_home):
        from hermes_cli.config import load_config
        from hermes_cli.fallback_config import get_fallback_chain

        chain = get_fallback_chain(load_config())

        assert chain == []
