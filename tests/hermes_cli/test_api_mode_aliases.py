"""API-mode spelling is owned by the canonical Phase 5.6 routing domain."""

from __future__ import annotations

import pytest

from hermes_cli.config import _normalize_custom_provider_entry
from providers.routing import canonicalize_api_mode


class TestCanonicalApiMode:
    @pytest.mark.parametrize(
        "alias, canonical",
        [
            ("openai", "chat_completions"),
            ("OpenAI", "chat_completions"),
            (" openai ", "chat_completions"),
            ("openai_chat", "chat_completions"),
            ("chat-completions", "chat_completions"),
            ("responses", "codex_responses"),
            ("openai_responses", "codex_responses"),
            ("anthropic", "anthropic_messages"),
            ("messages", "anthropic_messages"),
            ("bedrock", "bedrock_converse"),
        ],
    )
    def test_alias_maps_to_canonical(self, alias, canonical):
        assert canonicalize_api_mode(alias) == canonical

    @pytest.mark.parametrize(
        "canonical",
        [
            "chat_completions",
            "codex_responses",
            "anthropic_messages",
            "bedrock_converse",
        ],
    )
    def test_canonical_names_pass_through(self, canonical):
        assert canonicalize_api_mode(canonical) == canonical

    def test_plugin_mode_passes_through_unchanged(self):
        assert canonicalize_api_mode("vendor_native") == "vendor_native"

    def test_empty_values_normalize_to_empty(self):
        assert canonicalize_api_mode(None) == ""
        assert canonicalize_api_mode("") == ""


class TestNormalizedEntryCanonicalizes:
    def _entry(self, api_mode):
        return {
            "name": "relay",
            "api": "https://relay.example.invalid/v1",
            "api_mode": api_mode,
        }

    def test_legacy_openai_becomes_chat_completions(self):
        normalized = _normalize_custom_provider_entry(
            self._entry("openai"), provider_key="relay"
        )
        assert normalized["api_mode"] == "chat_completions"

    def test_canonical_value_unchanged(self):
        normalized = _normalize_custom_provider_entry(
            self._entry("codex_responses"), provider_key="relay"
        )
        assert normalized["api_mode"] == "codex_responses"

    def test_transport_key_also_canonicalized(self):
        entry = {
            "name": "relay",
            "api": "https://relay.example.invalid/v1",
            "transport": "openai",
        }
        normalized = _normalize_custom_provider_entry(entry, provider_key="relay")
        assert normalized["api_mode"] == "chat_completions"

    def test_plugin_mode_is_preserved_for_transport_registry(self):
        normalized = _normalize_custom_provider_entry(
            self._entry("vendor_native"), provider_key="relay"
        )
        assert normalized["api_mode"] == "vendor_native"
