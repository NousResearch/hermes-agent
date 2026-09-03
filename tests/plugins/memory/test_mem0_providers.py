"""Tests for OSS provider definitions and validation."""

import pytest

from plugins.memory.mem0._oss_providers import (
    LLM_PROVIDERS,
    EMBEDDER_PROVIDERS,
    VECTOR_PROVIDERS,
    KNOWN_DIMS,
    default_qdrant_path,
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

    def test_qdrant_default_config_has_no_frozen_path(self):
        """Regression for #85830: the qdrant default must not bake a HOME-derived
        path at import time, or HERMES_HOME redirection can never reach it."""
        cfg = VECTOR_PROVIDERS["qdrant"]["default_config"]
        assert "path" not in cfg
        assert "url" not in cfg


class TestDefaultQdrantPath:
    """Regression tests for #85830: default path must follow HERMES_HOME."""

    def test_resolves_under_hermes_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert default_qdrant_path() == str(tmp_path / "mem0_qdrant")

    def test_changes_when_hermes_home_changes(self, monkeypatch, tmp_path):
        first = tmp_path / "profile-a"
        second = tmp_path / "profile-b"
        monkeypatch.setenv("HERMES_HOME", str(first))
        p1 = default_qdrant_path()
        monkeypatch.setenv("HERMES_HOME", str(second))
        p2 = default_qdrant_path()
        assert p1 == str(first / "mem0_qdrant")
        assert p2 == str(second / "mem0_qdrant")
        assert p1 != p2

    def test_not_frozen_at_import_time(self, monkeypatch, tmp_path):
        """The path must be re-evaluated on every call, not captured at import."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        before = default_qdrant_path()
        other = tmp_path.parent / "hermes-redirect"
        monkeypatch.setenv("HERMES_HOME", str(other))
        assert default_qdrant_path() == str(other / "mem0_qdrant")
        assert default_qdrant_path() != before


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

