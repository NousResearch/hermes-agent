"""Tests for the effective fallback chain builder (get_fallback_chain).

Regression coverage for the silent-drop bug where a config written as

    fallback_providers:
      - deepseek            # bare string, not a {provider, model} dict

produced an EMPTY chain, so empty-response failover never fired and the
user only ever saw "model provider failed after retries".
"""

from hermes_cli.fallback_config import get_fallback_chain
from hermes_cli.models import get_default_model_for_provider


class TestBareProviderStrings:
    def test_bare_string_expands_to_default_model(self):
        chain = get_fallback_chain({"fallback_providers": ["deepseek"]})
        assert len(chain) == 1
        assert chain[0]["provider"] == "deepseek"
        # Expanded to the provider's catalog default, not silently dropped.
        assert chain[0]["model"] == get_default_model_for_provider("deepseek")
        assert chain[0]["model"]  # non-empty

    def test_top_level_bare_string(self):
        chain = get_fallback_chain({"fallback_providers": "deepseek"})
        assert len(chain) == 1
        assert chain[0]["provider"] == "deepseek"

    def test_unknown_bare_provider_is_dropped(self):
        # No default model resolvable → can't build a usable entry.
        chain = get_fallback_chain({"fallback_providers": ["totally-unknown-xyz"]})
        assert chain == []

    def test_empty_string_is_dropped(self):
        chain = get_fallback_chain({"fallback_providers": ["", "  "]})
        assert chain == []


class TestDictEntries:
    def test_full_dict_passes_through(self):
        entry = {"provider": "deepseek", "model": "deepseek-v4-pro"}
        chain = get_fallback_chain({"fallback_providers": [entry]})
        assert chain == [entry]

    def test_dict_missing_model_gets_default(self):
        chain = get_fallback_chain(
            {"fallback_providers": [{"provider": "deepseek"}]}
        )
        assert len(chain) == 1
        assert chain[0]["provider"] == "deepseek"
        assert chain[0]["model"] == get_default_model_for_provider("deepseek")

    def test_dict_missing_provider_is_dropped(self):
        chain = get_fallback_chain(
            {"fallback_providers": [{"model": "deepseek-v4-pro"}]}
        )
        assert chain == []

    def test_base_url_is_normalized_and_preserved(self):
        chain = get_fallback_chain({
            "fallback_providers": [{
                "provider": "custom",
                "model": "claude-sonnet-4-6",
                "base_url": "http://127.0.0.1:11435/v1/",
            }],
        })
        assert chain[0]["base_url"] == "http://127.0.0.1:11435/v1"


class TestChainMerge:
    def test_mixed_string_and_dict_entries(self):
        chain = get_fallback_chain({
            "fallback_providers": [
                "deepseek",
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
            ],
        })
        assert [c["provider"] for c in chain] == ["deepseek", "openrouter"]

    def test_dedup_across_fallback_model_key(self):
        chain = get_fallback_chain({
            "fallback_providers": [{"provider": "deepseek", "model": "deepseek-v4-pro"}],
            "fallback_model": {"provider": "deepseek", "model": "deepseek-v4-pro"},
        })
        # Same route from both keys collapses to one entry.
        assert len(chain) == 1

    def test_none_and_unknown_types_yield_empty(self):
        assert get_fallback_chain(None) == []
        assert get_fallback_chain({}) == []
        assert get_fallback_chain({"fallback_providers": 42}) == []
