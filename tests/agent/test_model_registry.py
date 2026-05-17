"""Tests for agent.model_registry — centralized model resolution.

TDD: write failing tests first, then implement.
"""
from __future__ import annotations

import pytest

from agent.model_registry import (
    ModelRef,
    ModelRegistry,
    ResolvedModel,
    split_model_id,
    model_id,
    legacy_entry_to_ref,
    resolved_to_legacy_dict,
)


# ── ModelRef ──────────────────────────────────────────────────────────

class TestModelRef:
    def test_id_construction(self):
        ref = ModelRef(provider="openrouter", model="anthropic/claude-sonnet-4")
        assert ref.id == "openrouter/anthropic/claude-sonnet-4"

    def test_custom_provider_id(self):
        ref = ModelRef(provider="custom:ollama-cloud", model="gpt-oss:20b")
        assert ref.id == "custom:ollama-cloud/gpt-oss:20b"

    def test_simple_model(self):
        ref = ModelRef(provider="gemini", model="gemini-2.5-flash")
        assert ref.id == "gemini/gemini-2.5-flash"

    def test_split_model_id(self):
        provider, model = split_model_id("openrouter/anthropic/claude-sonnet-4")
        assert provider == "openrouter"
        assert model == "anthropic/claude-sonnet-4"

    def test_split_custom_provider(self):
        provider, model = split_model_id("custom:ollama-cloud/gpt-oss:20b")
        assert provider == "custom:ollama-cloud"
        assert model == "gpt-oss:20b"

    def test_model_id(self):
        assert model_id("gemini", "gemini-2.5-flash") == "gemini/gemini-2.5-flash"


# ── parse_ref ─────────────────────────────────────────────────────────

class TestParseRef:
    def test_parse_string_id(self):
        reg = ModelRegistry({})
        ref = reg.parse_ref("openrouter/anthropic/claude-sonnet-4")
        assert ref.provider == "openrouter"
        assert ref.model == "anthropic/claude-sonnet-4"
        assert ref.id == "openrouter/anthropic/claude-sonnet-4"

    def test_parse_custom_provider(self):
        reg = ModelRegistry({})
        ref = reg.parse_ref("custom:ollama-cloud/gpt-oss:20b")
        assert ref.provider == "custom:ollama-cloud"
        assert ref.model == "gpt-oss:20b"

    def test_parse_legacy_dict(self):
        reg = ModelRegistry({})
        ref = reg.parse_ref({"provider": "gemini", "model": "gemini-2.5-flash"})
        assert ref.provider == "gemini"
        assert ref.model == "gemini-2.5-flash"

    def test_parse_legacy_dict_with_extra_fields(self):
        reg = ModelRegistry({})
        ref = reg.parse_ref({"provider": "openrouter", "model": "openai/gpt-4o-mini", "base_url": "https://x"})
        assert ref.provider == "openrouter"
        assert ref.model == "openai/gpt-4o-mini"

    def test_parse_none_uses_default_provider(self):
        reg = ModelRegistry({})
        ref = reg.parse_ref(None, default_provider="anthropic")
        assert ref.provider == "anthropic"
        assert ref.model == ""

    def test_parse_empty_string_uses_default_provider(self):
        reg = ModelRegistry({})
        ref = reg.parse_ref("", default_provider="anthropic")
        assert ref.provider == "anthropic"
        assert ref.model == ""


# ── main() ────────────────────────────────────────────────────────────

class TestMain:
    def test_resolve_main_from_legacy_model_block(self):
        cfg = {"model": {"provider": "gemini", "default": "gemini-2.5-flash"}}
        resolved = ModelRegistry(cfg).main()
        assert resolved.id == "gemini/gemini-2.5-flash"
        assert resolved.provider == "gemini"
        assert resolved.model == "gemini-2.5-flash"

    def test_resolve_main_from_model_block_with_name(self):
        cfg = {"model": {"provider": "openrouter", "name": "anthropic/claude-opus-4.7"}}
        resolved = ModelRegistry(cfg).main()
        assert resolved.id == "openrouter/anthropic/claude-opus-4.7"

    def test_resolve_main_from_model_block_with_model_key(self):
        cfg = {"model": {"provider": "openrouter", "model": "anthropic/claude-opus-4.7"}}
        resolved = ModelRegistry(cfg).main()
        assert resolved.id == "openrouter/anthropic/claude-opus-4.7"

    def test_resolve_main_with_base_url(self):
        cfg = {"model": {"provider": "openrouter", "default": "openai/gpt-4o-mini", "base_url": "https://x"}}
        resolved = ModelRegistry(cfg).main()
        assert resolved.base_url == "https://x"

    def test_main_empty_config_raises(self):
        with pytest.raises(ValueError, match="no main model configured"):
            ModelRegistry({}).main()


# ── fallback_chain() ──────────────────────────────────────────────────

class TestFallbackChain:
    def test_resolve_fallback_chain_from_legacy_list(self):
        cfg = {"fallback_providers": [
            {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ]}
        chain = ModelRegistry(cfg).fallback_chain()
        assert [m.id for m in chain] == [
            "openrouter/openai/gpt-4o-mini",
            "gemini/gemini-2.5-flash",
        ]

    def test_fallback_chain_empty(self):
        cfg = {}
        chain = ModelRegistry(cfg).fallback_chain()
        assert chain == []

    def test_legacy_fallback_model_dict(self):
        cfg = {"fallback_model": {"provider": "openrouter", "model": "openai/gpt-4o-mini"}}
        chain = ModelRegistry(cfg).fallback_chain()
        assert len(chain) == 1
        assert chain[0].id == "openrouter/openai/gpt-4o-mini"

    def test_fallback_chain_preserves_order(self):
        cfg = {"fallback_providers": [
            {"provider": "gemini", "model": "gemini-2.5-flash"},
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        ]}
        chain = ModelRegistry(cfg).fallback_chain()
        assert [m.id for m in chain] == [
            "gemini/gemini-2.5-flash",
            "openrouter/anthropic/claude-sonnet-4",
        ]

    def test_fallback_chain_with_base_url(self):
        cfg = {"fallback_providers": [
            {"provider": "openrouter", "model": "openai/gpt-4o-mini", "base_url": "https://x"},
        ]}
        chain = ModelRegistry(cfg).fallback_chain()
        assert chain[0].base_url == "https://x"


# ── auxiliary() ───────────────────────────────────────────────────────

class TestAuxiliary:
    def test_resolve_explicit_auxiliary(self):
        cfg = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
            "auxiliary": {
                "vision": {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
            }
        }
        resolved = ModelRegistry(cfg).auxiliary("vision")
        assert resolved.id == "openrouter/openai/gpt-4o-mini"

    def test_resolve_auto_auxiliary_returns_main(self):
        cfg = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
            "auxiliary": {
                "vision": {"provider": "auto"},
            }
        }
        resolved = ModelRegistry(cfg).auxiliary("vision")
        assert resolved.id == "openrouter/anthropic/claude-sonnet-4"

    def test_resolve_missing_auxiliary_returns_main(self):
        cfg = {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
        }
        resolved = ModelRegistry(cfg).auxiliary("vision")
        assert resolved.id == "openrouter/anthropic/claude-sonnet-4"

    def test_resolve_auxiliary_auto_no_main_raises(self):
        cfg = {"auxiliary": {"vision": {"provider": "auto"}}}
        with pytest.raises(ValueError, match="no main model"):
            ModelRegistry(cfg).auxiliary("vision")


# ── to_legacy_agent_kwargs ────────────────────────────────────────────

class TestToLegacyAgentKwargs:
    def test_basic_conversion(self):
        cfg = {"model": {"provider": "gemini", "default": "gemini-2.5-flash"}}
        reg = ModelRegistry(cfg)
        resolved = reg.main()
        kwargs = reg.to_legacy_agent_kwargs(resolved)
        assert kwargs["provider"] == "gemini"
        assert kwargs["model"] == "gemini-2.5-flash"

    def test_conversion_includes_base_url(self):
        cfg = {"model": {"provider": "openrouter", "default": "openai/gpt-4o-mini", "base_url": "https://x"}}
        reg = ModelRegistry(cfg)
        resolved = reg.main()
        kwargs = reg.to_legacy_agent_kwargs(resolved)
        assert kwargs["base_url"] == "https://x"


# ── legacy_entry_to_ref / resolved_to_legacy_dict ─────────────────────

class TestLegacyHelpers:
    def test_legacy_entry_to_ref(self):
        ref = legacy_entry_to_ref({"provider": "gemini", "model": "gemini-2.5-flash"})
        assert ref.provider == "gemini"
        assert ref.model == "gemini-2.5-flash"

    def test_resolved_to_legacy_dict(self):
        resolved = ResolvedModel(id="gemini/gemini-2.5-flash", provider="gemini", model="gemini-2.5-flash", base_url="https://x")
        d = resolved_to_legacy_dict(resolved)
        assert d["provider"] == "gemini"
        assert d["model"] == "gemini-2.5-flash"
        assert d["base_url"] == "https://x"
