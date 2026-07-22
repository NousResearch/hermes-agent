"""Tests for hermes_cli/fallback_config.py — fallback entry API-key resolution."""

from hermes_cli.fallback_config import get_fallback_chain, resolve_entry_api_key


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


class TestJsonStringFallbackEntries:
    def test_json_list_preserves_order_and_normalization(self):
        config = {
            "fallback_providers": (
                '[{"provider":" first ","model":" model-a ",'
                '"base_url":"https://example.test/v1/"},'
                '{"provider":"second","model":"model-b"}]'
            )
        }

        assert get_fallback_chain(config) == [
            {
                "provider": "first",
                "model": "model-a",
                "base_url": "https://example.test/v1",
            },
            {"provider": "second", "model": "model-b"},
        ]

    def test_json_dict_works_for_legacy_key(self):
        config = {
            "fallback_model": '{"provider":"legacy","model":"model-a"}',
        }

        assert get_fallback_chain(config) == [
            {"provider": "legacy", "model": "model-a"},
        ]

    def test_double_encoded_json_string_is_decoded(self):
        config = {
            "fallback_providers": (
                '"[{\\"provider\\":\\"fallback\\",'
                '\\"model\\":\\"model-a\\"}]"'
            )
        }

        assert get_fallback_chain(config) == [
            {"provider": "fallback", "model": "model-a"},
        ]

    def test_invalid_json_string_returns_empty_chain(self):
        assert get_fallback_chain({"fallback_providers": "not-json"}) == []
