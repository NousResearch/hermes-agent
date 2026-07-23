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


class TestGetFallbackChainPayloadShapes:
    """``fallback_providers`` accepts multiple YAML-serialised shapes because
    older ``hermes config set fallback_providers [...]`` invocations round-tripped
    the value through a YAML serializer that wrote the list as a quoted JSON
    string. Silently dropping the chain on that path produced cron jobs that
    failed instead of falling back when their primary provider hit a hard
    quota wall — see hermes-agent #41590. These tests pin the behaviour so the
    any-shape parsing cannot regress."""

    def test_canonical_list_of_dicts(self):
        cfg = {
            "fallback_providers": [
                {"provider": "openrouter", "model": "minimax/minimax-m3"},
                {"provider": "zai", "model": "glm-4-flash", "base_url": "https://api.z.ai/api/coding/paas/v4"},
            ]
        }
        chain = get_fallback_chain(cfg)
        assert [e["provider"] for e in chain] == ["openrouter", "zai"]
        assert chain[1]["base_url"] == "https://api.z.ai/api/coding/paas/v4"

    def test_legacy_single_dict(self):
        """Backward-compat: older configs that used ``fallback_providers`` as a
        single dict (not a list) must still resolve to one entry."""
        cfg = {
            "fallback_providers": {"provider": "openrouter", "model": "minimax/minimax-m3"}
        }
        assert get_fallback_chain(cfg) == [
            {"provider": "openrouter", "model": "minimax/minimax-m3"}
        ]

    def test_json_encoded_list_string_round_trip(self):
        """Production-shaped config: ``fallback_providers`` is a single quoted
        JSON string (the shape an older ``hermes config set`` invocation
        produced). On main this returns ``[]`` — the cron job then has no
        fallback configured and hard-fails when the primary hits a quota
        wall. With this fix it parses through to the canonical entries.
        """
        cfg = {
            "fallback_providers": '[{"provider": "openrouter", "model": "minimax/minimax-m3"}]'
        }
        chain = get_fallback_chain(cfg)
        assert chain == [
            {"provider": "openrouter", "model": "minimax/minimax-m3"}
        ]

    def test_json_encoded_single_dict_string(self):
        cfg = {
            "fallback_providers": '{"provider": "openrouter", "model": "minimax/minimax-m3"}'
        }
        assert get_fallback_chain(cfg) == [
            {"provider": "openrouter", "model": "minimax/minimax-m3"}
        ]

    def test_string_payload_that_does_not_look_like_json_is_ignored(self):
        """An accidental string literal (env-var name, free-form note, etc.)
        must NOT raise — it falls through to an empty chain so the failure
        surfaces at the actual primary-provider call rather than at config
        load."""
        cfg = {"fallback_providers": "this is not a fallback chain"}
        assert get_fallback_chain(cfg) == []

    def test_malformed_json_string_is_ignored(self):
        """Bad JSON in the encoded string falls through to an empty chain.
        Same rationale as ``test_string_payload_that_does_not_look_like_json_is_ignored``."""
        cfg = {"fallback_providers": "[{unparseable"}
        assert get_fallback_chain(cfg) == []

    def test_none_and_missing_key(self):
        assert get_fallback_chain(None) == []
        assert get_fallback_chain({}) == []

    def test_legacy_fallback_model_merges_after_fallback_providers(self):
        """A non-empty ``fallback_providers`` chain takes precedence; a legacy
        ``fallback_model`` entry is appended afterwards, deduplicated by
        ``(provider, model, base_url)`` identity."""
        cfg = {
            "fallback_providers": [{"provider": "openrouter", "model": "minimax/minimax-m3"}],
            "fallback_model": {"provider": "zai", "model": "glm-4-flash"},
        }
        chain = get_fallback_chain(cfg)
        providers = [e["provider"] for e in chain]
        # ``openrouter`` came first; ``zai`` legacy is appended.
        assert providers == ["openrouter", "zai"]

    def test_duplicate_entries_across_keys_are_deduplicated(self):
        cfg = {
            "fallback_providers": [{"provider": "openrouter", "model": "minimax/minimax-m3"}],
            "fallback_model": {"provider": "openrouter", "model": "minimax/minimax-m3"},
        }
        assert len(get_fallback_chain(cfg)) == 1
