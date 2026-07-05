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

    def test_embedder_provider_ids(self):
        assert set(EMBEDDER_PROVIDERS.keys()) == {"openai", "ollama"}

    def test_vector_providers_have_required_keys(self):
        for pid, p in VECTOR_PROVIDERS.items():
            assert "label" in p
            assert "default_config" in p

    def test_vector_provider_ids(self):
        assert set(VECTOR_PROVIDERS.keys()) == {"qdrant", "pgvector"}

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

    def test_unknown_embedder_provider(self):
        cfg = {
            "llm": {"provider": "openai", "config": {}},
            "embedder": {"provider": "cohere", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {}},
        }
        errors = validate_oss_config(cfg)
        assert any("embedder" in e.lower() for e in errors)

    def test_unknown_vector_provider(self):
        cfg = {
            "llm": {"provider": "openai", "config": {}},
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "redis", "config": {}},
        }
        errors = validate_oss_config(cfg)
        assert any("vector" in e.lower() for e in errors)

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

    def test_pgvector_with_user_valid(self):
        cfg = {
            "llm": {"provider": "openai", "config": {}},
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "pgvector", "config": {"host": "localhost", "user": "pg"}},
        }
        errors = validate_oss_config(cfg)
        assert errors == []
