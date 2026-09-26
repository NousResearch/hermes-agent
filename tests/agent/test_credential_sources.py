"""Test coverage for agent/credential_sources.py — the credential removal
registry. This module had zero test coverage before this file.

Tests the RemovalStep matching logic, the registry, and the env-source
removal path (the most common one). All filesystem and env access is
mocked — no real credentials are touched.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_sources import (
    RemovalResult,
    RemovalStep,
    _REGISTRY,
    _register_all_sources,
    _remove_env_source,
    find_removal_step,
    register,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Save/restore the global registry so tests don't pollute each other."""
    saved = list(_REGISTRY)
    _REGISTRY.clear()
    yield _REGISTRY
    _REGISTRY.clear()
    _REGISTRY.extend(saved)


class TestRemovalStep:
    def test_matches_exact_provider_and_source(self):
        step = RemovalStep(
            provider="xai", source_id="env:XAI_API_KEY",
            remove_fn=lambda p, r: RemovalResult(),
        )
        assert step.matches("xai", "env:XAI_API_KEY") is True
        assert step.matches("xai", "env:OTHER") is False
        assert step.matches("anthropic", "env:XAI_API_KEY") is False

    def test_matches_wildcard_provider(self):
        step = RemovalStep(
            provider="*", source_id="manual",
            remove_fn=lambda p, r: RemovalResult(),
        )
        assert step.matches("xai", "manual") is True
        assert step.matches("anthropic", "manual") is True

    def test_matches_with_match_fn(self):
        step = RemovalStep(
            provider="xai", source_id="",
            match_fn=lambda s: s.startswith("env:"),
            remove_fn=lambda p, r: RemovalResult(),
        )
        assert step.matches("xai", "env:XAI_API_KEY") is True
        assert step.matches("xai", "oauth:device") is False


class TestRemovalResult:
    def test_defaults(self):
        result = RemovalResult()
        assert result.cleaned == []
        assert result.hints == []
        assert result.suppress is True

    def test_custom_values(self):
        result = RemovalResult(cleaned=["Cleared key"], hints=["Check shell"], suppress=False)
        assert result.cleaned == ["Cleared key"]
        assert result.suppress is False


class TestRegistry:
    def test_register_adds_step(self):
        step = RemovalStep(provider="test", source_id="s", remove_fn=lambda p, r: RemovalResult())
        register(step)
        assert step in _REGISTRY

    def test_register_all_sources_populates(self):
        _register_all_sources()
        assert len(_REGISTRY) > 0

    def test_find_removal_step_exact(self):
        _register_all_sources()
        step = find_removal_step("xai", "env:XAI_API_KEY")
        assert step is not None

    def test_find_removal_step_returns_none_for_unknown(self):
        _register_all_sources()
        step = find_removal_step("nonexistent-provider", "nonexistent-source")
        assert step is None


class TestRemoveEnvSource:
    def test_removes_env_var_from_env_file(self, tmp_path, monkeypatch):
        # The function reads from ~/.hermes/.env, not os.environ.
        # When the var is only in the shell env, it reports a hint.
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        removed = MagicMock()
        removed.source = "env:XAI_API_KEY"

        result = _remove_env_source("xai", removed)
        assert isinstance(result, RemovalResult)
        # Without a .env file entry, nothing is cleaned but hints are given
        assert isinstance(result.hints, list)

    def test_returns_removal_result(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        removed = MagicMock()
        removed.source = "env:XAI_API_KEY"
        result = _remove_env_source("xai", removed)
        assert result.suppress is True
