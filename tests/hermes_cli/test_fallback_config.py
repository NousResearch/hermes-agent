"""Tests for hermes_cli.fallback_config — fallback provider chain helpers."""

from __future__ import annotations

import pytest

from hermes_cli.fallback_config import (
    _entry_identity,
    _iter_fallback_entries,
    _normalized_base_url,
    get_fallback_chain,
)


# ============================================================================
# _normalized_base_url
# ============================================================================
class TestNormalizedBaseUrl:
    def test_strips_trailing_slash(self):
        assert _normalized_base_url("https://api.example.com/") == "https://api.example.com"

    def test_strips_multiple_trailing_slashes(self):
        assert _normalized_base_url("https://api.example.com///") == "https://api.example.com"

    def test_strips_whitespace(self):
        assert _normalized_base_url("  https://api.example.com  ") == "https://api.example.com"

    def test_no_trailing_slash_unchanged(self):
        assert _normalized_base_url("https://api.example.com") == "https://api.example.com"

    def test_non_string_returns_empty(self):
        assert _normalized_base_url(None) == ""
        assert _normalized_base_url(42) == ""
        assert _normalized_base_url([]) == ""
        assert _normalized_base_url({}) == ""

    def test_empty_string(self):
        assert _normalized_base_url("") == ""

    def test_only_slashes(self):
        assert _normalized_base_url("///") == ""


# ============================================================================
# _iter_fallback_entries
# ============================================================================
class TestIterFallbackEntries:
    def test_single_dict(self):
        raw = {"provider": "openai", "model": "gpt-4"}
        result = _iter_fallback_entries(raw)
        assert len(result) == 1
        assert result[0]["provider"] == "openai"
        assert result[0]["model"] == "gpt-4"

    def test_list_of_dicts(self):
        raw = [
            {"provider": "openai", "model": "gpt-4"},
            {"provider": "anthropic", "model": "claude-3"},
        ]
        result = _iter_fallback_entries(raw)
        assert len(result) == 2
        assert result[0]["provider"] == "openai"
        assert result[1]["provider"] == "anthropic"

    def test_filters_non_dict_entries(self):
        raw = [
            {"provider": "openai", "model": "gpt-4"},
            "not a dict",
            42,
            {"provider": "anthropic", "model": "claude-3"},
        ]
        result = _iter_fallback_entries(raw)
        assert len(result) == 2

    def test_filters_missing_provider(self):
        raw = [{"model": "gpt-4"}]
        result = _iter_fallback_entries(raw)
        assert len(result) == 0

    def test_filters_missing_model(self):
        raw = [{"provider": "openai"}]
        result = _iter_fallback_entries(raw)
        assert len(result) == 0

    def test_filters_empty_provider_string(self):
        raw = [{"provider": "  ", "model": "gpt-4"}]
        result = _iter_fallback_entries(raw)
        assert len(result) == 0

    def test_filters_empty_model_string(self):
        raw = [{"provider": "openai", "model": ""}]
        result = _iter_fallback_entries(raw)
        assert len(result) == 0

    def test_non_dict_non_list_returns_empty(self):
        assert _iter_fallback_entries(None) == []
        assert _iter_fallback_entries("string") == []
        assert _iter_fallback_entries(42) == []

    def test_empty_list(self):
        assert _iter_fallback_entries([]) == []

    def test_strips_provider_and_model(self):
        raw = [{"provider": " openai ", "model": " gpt-4 "}]
        result = _iter_fallback_entries(raw)
        assert result[0]["provider"] == "openai"
        assert result[0]["model"] == "gpt-4"

    def test_includes_base_url_when_present(self):
        raw = [{"provider": "openai", "model": "gpt-4", "base_url": "https://api.example.com/"}]
        result = _iter_fallback_entries(raw)
        assert result[0]["base_url"] == "https://api.example.com"

    def test_omits_empty_base_url(self):
        """Empty base_url persists from dict(entry) copy — key stays but value is ''."""
        raw = [{"provider": "openai", "model": "gpt-4", "base_url": ""}]
        result = _iter_fallback_entries(raw)
        assert result[0]["base_url"] == ""

    def test_preserves_extra_keys(self):
        raw = [{"provider": "openai", "model": "gpt-4", "api_key": "sk-123"}]
        result = _iter_fallback_entries(raw)
        assert result[0]["api_key"] == "sk-123"


# ============================================================================
# _entry_identity
# ============================================================================
class TestEntryIdentity:
    def test_basic_identity(self):
        entry = {"provider": "OpenAI", "model": "GPT-4", "base_url": "https://api.openai.com/"}
        result = _entry_identity(entry)
        assert result == ("openai", "gpt-4", "https://api.openai.com")

    def test_missing_base_url(self):
        entry = {"provider": "openai", "model": "gpt-4"}
        result = _entry_identity(entry)
        assert result == ("openai", "gpt-4", "")

    def test_case_insensitive(self):
        a = _entry_identity({"provider": "OpenAI", "model": "GPT-4"})
        b = _entry_identity({"provider": "openai", "model": "gpt-4"})
        assert a == b

    def test_strips_whitespace(self):
        a = _entry_identity({"provider": " openai ", "model": " gpt-4 "})
        b = _entry_identity({"provider": "openai", "model": "gpt-4"})
        assert a == b


# ============================================================================
# get_fallback_chain
# ============================================================================
class TestGetFallbackChain:
    def test_empty_config(self):
        assert get_fallback_chain(None) == []
        assert get_fallback_chain({}) == []

    def test_single_fallback_provider(self):
        config = {
            "fallback_providers": [
                {"provider": "openai", "model": "gpt-4"}
            ]
        }
        result = get_fallback_chain(config)
        assert len(result) == 1
        assert result[0]["provider"] == "openai"

    def test_fallback_providers_take_priority(self):
        config = {
            "fallback_providers": [
                {"provider": "openai", "model": "gpt-4"}
            ],
            "fallback_model": [
                {"provider": "anthropic", "model": "claude-3"}
            ]
        }
        result = get_fallback_chain(config)
        assert len(result) == 2
        assert result[0]["provider"] == "openai"
        assert result[1]["provider"] == "anthropic"

    def test_legacy_fallback_model_only(self):
        config = {
            "fallback_model": [
                {"provider": "openai", "model": "gpt-3.5"}
            ]
        }
        result = get_fallback_chain(config)
        assert len(result) == 1
        assert result[0]["provider"] == "openai"

    def test_deduplicates_by_identity(self):
        config = {
            "fallback_providers": [
                {"provider": "openai", "model": "gpt-4"}
            ],
            "fallback_model": [
                {"provider": "openai", "model": "gpt-4"}  # same identity
            ]
        }
        result = get_fallback_chain(config)
        assert len(result) == 1

    def test_different_base_url_not_deduped(self):
        config = {
            "fallback_providers": [
                {"provider": "openai", "model": "gpt-4", "base_url": "https://a.com"}
            ],
            "fallback_model": [
                {"provider": "openai", "model": "gpt-4", "base_url": "https://b.com"}
            ]
        }
        result = get_fallback_chain(config)
        assert len(result) == 2

    def test_config_is_none(self):
        assert get_fallback_chain(None) == []

    def test_returns_fresh_dicts(self):
        original = {"fallback_providers": [{"provider": "x", "model": "y"}]}
        result = get_fallback_chain(original)
        # Modify returned entry — should not affect original
        result[0]["extra"] = True
        assert "extra" not in original["fallback_providers"][0]

    def test_skips_invalid_entries_in_chain(self):
        config = {
            "fallback_providers": [
                {"provider": "openai", "model": "gpt-4"},
                {"missing": "keys"},
            ]
        }
        result = get_fallback_chain(config)
        assert len(result) == 1
