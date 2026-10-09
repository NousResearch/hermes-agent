"""Tests for Copilot live /models context-window resolution."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hermes_cli.models import get_copilot_model_context


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
    """Reset module-level cache before each test."""
    import hermes_cli.models as mod

    mod._copilot_context_cache = {}
    mod._copilot_context_cache_time = 0.0
    yield
    mod._copilot_context_cache = {}
    mod._copilot_context_cache_time = 0.0


class TestGetCopilotModelContext:
    """Tests for get_copilot_model_context()."""

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_returns_max_prompt_tokens(self, mock_fetch):
        assert get_copilot_model_context("claude-opus-4.6-1m") == 1_000_000
        assert get_copilot_model_context("gpt-4.1") == 128_000


    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_cache_expires(self, mock_fetch):
        import hermes_cli.models as mod

        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 1

        # Expire the cache
        mod._copilot_context_cache_time = time.time() - 7200
        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 2


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


