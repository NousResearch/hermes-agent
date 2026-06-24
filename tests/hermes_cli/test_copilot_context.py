"""Tests for Copilot live /models context-window resolution."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hermes_cli.models import get_copilot_model_context, get_copilot_reasoning_efforts


# Sample catalog items mimicking the Copilot /models API response
_SAMPLE_CATALOG = [
    {
        "id": "claude-opus-4.6-1m",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 1000000, "max_output_tokens": 64000},
        },
    },
    {
        "id": "gpt-4.1",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 128000, "max_output_tokens": 32768},
        },
    },
    {
        "id": "claude-sonnet-4",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 200000, "max_output_tokens": 64000},
        },
    },
    {
        "id": "model-without-limits",
        "capabilities": {"type": "chat"},
    },
    {
        "id": "model-zero-limit",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 0},
        },
    },
]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module-level caches before each test."""
    import hermes_cli.models as mod

    mod._copilot_context_cache = {}
    mod._copilot_context_cache_time = 0.0
    mod._copilot_reasoning_catalog_cache = None
    mod._copilot_reasoning_catalog_cache_time = 0.0
    yield
    mod._copilot_context_cache = {}
    mod._copilot_context_cache_time = 0.0
    mod._copilot_reasoning_catalog_cache = None
    mod._copilot_reasoning_catalog_cache_time = 0.0


class TestGetCopilotModelContext:
    """Tests for get_copilot_model_context()."""

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_returns_max_prompt_tokens(self, mock_fetch):
        assert get_copilot_model_context("claude-opus-4.6-1m") == 1_000_000
        assert get_copilot_model_context("gpt-4.1") == 128_000

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_returns_none_for_unknown_model(self, mock_fetch):
        assert get_copilot_model_context("nonexistent-model") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_skips_models_without_limits(self, mock_fetch):
        assert get_copilot_model_context("model-without-limits") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_skips_zero_limit(self, mock_fetch):
        assert get_copilot_model_context("model-zero-limit") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_caches_results(self, mock_fetch):
        get_copilot_model_context("gpt-4.1")
        get_copilot_model_context("claude-sonnet-4")
        # Only one API call despite two lookups
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_cache_expires(self, mock_fetch):
        import hermes_cli.models as mod

        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 1

        # Expire the cache
        mod._copilot_context_cache_time = time.time() - 7200
        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 2

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_returns_none_when_catalog_unavailable(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=[])
    def test_returns_none_for_empty_catalog(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None


class TestModelMetadataCopilotIntegration:
    """Test that get_model_context_length() uses Copilot live API for copilot provider."""

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_copilot_provider_uses_live_api(self, mock_fetch):
        from agent.model_metadata import get_model_context_length

        ctx = get_model_context_length("claude-opus-4.6-1m", provider="copilot")
        assert ctx == 1_000_000

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_copilot_acp_provider_uses_live_api(self, mock_fetch):
        from agent.model_metadata import get_model_context_length

        ctx = get_model_context_length("claude-sonnet-4", provider="copilot-acp")
        assert ctx == 200_000

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_falls_through_when_catalog_unavailable(self, mock_fetch):
        from agent.model_metadata import get_model_context_length

        # Should not raise, should fall through to models.dev or defaults
        ctx = get_model_context_length("gpt-4.1", provider="copilot")
        assert isinstance(ctx, int)
        assert ctx > 0


# Catalog whose entries advertise reasoning_effort under capabilities.supports,
# the shape the live Copilot /models API returns for opus/sonnet. This is the
# data the bare github_model_reasoning_efforts(model) call never sees (it has no
# catalog), which is why Claude resolved to [] before get_copilot_reasoning_efforts.
_REASONING_CATALOG = [
    {
        "id": "claude-opus-4.8",
        "capabilities": {
            "type": "chat",
            "supports": {"reasoning_effort": ["low", "medium", "high", "max"]},
        },
    },
    {
        "id": "claude-sonnet-4.6",
        "capabilities": {
            "type": "chat",
            "supports": {"reasoning_effort": ["low", "medium", "high"]},
        },
    },
    {
        "id": "gpt-4.1",
        "capabilities": {"type": "chat", "supports": {}},
    },
]


class TestGetCopilotReasoningEfforts:
    """Tests for get_copilot_reasoning_efforts(): the cached catalog wrapper."""

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_REASONING_CATALOG)
    def test_claude_resolves_efforts_from_catalog(self, mock_fetch):
        # The bug: without the catalog, Claude returned []. With the catalog
        # supplied by this wrapper, the advertised levels come through.
        assert get_copilot_reasoning_efforts("claude-opus-4.8") == [
            "low",
            "medium",
            "high",
            "max",
        ]
        assert get_copilot_reasoning_efforts("claude-sonnet-4.6") == [
            "low",
            "medium",
            "high",
        ]

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_REASONING_CATALOG)
    def test_bare_resolver_returns_empty_for_claude_without_catalog(self, mock_fetch):
        # Regression guard documenting the underlying bug: the bare resolver with
        # no catalog falls through to the static table and yields [] for Claude.
        from hermes_cli.models import github_model_reasoning_efforts

        assert github_model_reasoning_efforts("claude-opus-4.8") == []

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_REASONING_CATALOG)
    def test_caches_catalog_across_calls(self, mock_fetch):
        get_copilot_reasoning_efforts("claude-opus-4.8")
        get_copilot_reasoning_efforts("claude-sonnet-4.6")
        get_copilot_reasoning_efforts("gpt-4.1")
        # One fetch despite three lookups, no per-turn HTTP hit.
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_REASONING_CATALOG)
    def test_catalog_cache_expires(self, mock_fetch):
        import hermes_cli.models as mod

        get_copilot_reasoning_efforts("claude-opus-4.8")
        assert mock_fetch.call_count == 1

        mod._copilot_reasoning_catalog_cache_time = time.time() - 7200
        get_copilot_reasoning_efforts("claude-opus-4.8")
        assert mock_fetch.call_count == 2

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_degrades_gracefully_without_catalog(self, mock_fetch):
        # No catalog available: must not raise. Claude has no static entry, so
        # it resolves to []; a static-table model still resolves from the table.
        assert get_copilot_reasoning_efforts("claude-opus-4.8") == []
        efforts = get_copilot_reasoning_efforts("gpt-5.4")
        assert isinstance(efforts, list)
