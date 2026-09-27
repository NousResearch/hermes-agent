"""Tests for OSS provider definitions and validation."""

import pytest

from plugins.memory.mem0._oss_providers import (
    LLM_PROVIDERS,
    EMBEDDER_PROVIDERS,
    VECTOR_PROVIDERS,
    KNOWN_DIMS,
    validate_oss_config,
)


class TestProviderDefinitions:

    def test_llm_providers_have_required_keys(self):
        for pid, p in LLM_PROVIDERS.items():
            assert "label" in p
            assert "needs_key" in p
            assert "default_model" in p

    def test_embedder_providers_have_required_keys(self):
        for pid, p in EMBEDDER_PROVIDERS.items():
            assert "label" in p
            assert "needs_key" in p
            assert "default_model" in p
            assert "dims" in p


    def test_vector_providers_have_required_keys(self):
        for pid, p in VECTOR_PROVIDERS.items():
            assert "label" in p
            assert "default_config" in p


    def test_known_dims_covers_defaults(self):
        for pid, p in EMBEDDER_PROVIDERS.items():
            assert p["default_model"] in KNOWN_DIMS


class TestValidation:

    def test_valid_openai_config(self):
        cfg = {
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
            "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/test"}},
        }
        errors = validate_oss_config(cfg)
        assert errors == []

    def test_unknown_llm_provider(self):
        cfg = {
            "llm": {"provider": "gemini", "config": {}},
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {}},
        }
        errors = validate_oss_config(cfg)
        assert any("llm" in e.lower() for e in errors)


    def test_missing_llm_section(self):
        cfg = {
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {}},
        }
        errors = validate_oss_config(cfg)
        assert any("llm" in e.lower() for e in errors)

    def test_pgvector_needs_user(self):
        cfg = {
            "llm": {"provider": "openai", "config": {}},
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "pgvector", "config": {"host": "localhost"}},
        }
        errors = validate_oss_config(cfg)
        assert any("user" in e.lower() for e in errors)



class TestProviderBlockTranslation:
    """``_provider_block`` normalises one OSS section before mem0 sees it."""

    @staticmethod
    def _block(section, registry_name="embedder"):
        from plugins.memory.mem0._backend import _provider_block
        from plugins.memory.mem0._oss_providers import EMBEDDER_PROVIDERS, LLM_PROVIDERS

        registry = EMBEDDER_PROVIDERS if registry_name == "embedder" else LLM_PROVIDERS
        return _provider_block(section, registry)

    def test_known_mem0_provider_is_untouched(self):
        block = self._block({"provider": "openai", "config": {"model": "text-embedding-3-small"}})
        assert block["provider"] == "openai"
        assert block["config"] == {"model": "text-embedding-3-small"}

    def test_legacy_api_base_maps_to_the_canonical_key(self):
        block = self._block({"provider": "ollama", "config": {"api_base": "http://box:11434"}})
        assert block["config"] == {"ollama_base_url": "http://box:11434"}

    def test_mittwald_rides_mem0_openai(self, monkeypatch):
        # mem0 has no "mittwald" provider, so the id must be translated before MemoryConfig sees it.
        monkeypatch.delenv("MITTWALD_LLM_API_KEY", raising=False)
        monkeypatch.delenv("MITTWALD_AI_API_KEY", raising=False)
        from hermes_cli import config
        env_values = {"MITTWALD_AI_API_KEY": "sk-alias"}
        monkeypatch.setattr(config, "get_env_value", lambda name: env_values.get(name))
        block = self._block({"provider": "mittwald", "config": {"model": "Qwen3-Embedding-8B"}})
        assert block["provider"] == "openai"
        assert block["config"]["openai_base_url"] == "https://llm.aihosting.mittwald.de/v1"
        # mem0's OpenAI embedder would otherwise look for OPENAI_API_KEY only; this comes from .env.
        assert block["config"]["api_key"] == "sk-alias"

        env_values["MITTWALD_LLM_API_KEY"] = "sk-primary"
        assert self._block({"provider": "mittwald", "config": {}})["config"]["api_key"] == "sk-primary"

    def test_configured_values_win_over_the_defaults(self, monkeypatch):
        monkeypatch.setenv("MITTWALD_LLM_API_KEY", "sk-env")
        block = self._block({"provider": "mittwald", "config": {
            "model": "Qwen3-Embedding-8B", "openai_base_url": "https://proxy.example/v1",
            "api_key": "sk-configured"}})
        assert block["config"]["openai_base_url"] == "https://proxy.example/v1"
        assert block["config"]["api_key"] == "sk-configured"

    def test_missing_key_is_left_to_mem0(self, monkeypatch):
        monkeypatch.delenv("MITTWALD_LLM_API_KEY", raising=False)
        monkeypatch.delenv("MITTWALD_AI_API_KEY", raising=False)
        block = self._block({"provider": "mittwald", "config": {"model": "Qwen3-Embedding-8B"}})
        assert "api_key" not in block["config"]

    def test_input_section_is_not_mutated(self, monkeypatch):
        monkeypatch.setenv("MITTWALD_LLM_API_KEY", "sk-test")
        section = {"provider": "mittwald", "config": {"model": "Qwen3-Embedding-8B"}}
        self._block(section)
        assert section == {"provider": "mittwald", "config": {"model": "Qwen3-Embedding-8B"}}
