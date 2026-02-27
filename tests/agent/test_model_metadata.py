"""Tests for agent/model_metadata.py — token estimation and context lengths."""

import pytest
from unittest.mock import patch, MagicMock

from agent.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    estimate_tokens_rough,
    estimate_messages_tokens_rough,
    get_model_context_length,
    fetch_model_metadata,
    _MODEL_CACHE_TTL,
)


# =========================================================================
# Token estimation
# =========================================================================

class TestEstimateTokensRough:
    def test_empty_string(self):
        assert estimate_tokens_rough("") == 0

    def test_none_returns_zero(self):
        assert estimate_tokens_rough(None) == 0

    def test_known_length(self):
        # 400 chars / 4 = 100 tokens
        text = "a" * 400
        assert estimate_tokens_rough(text) == 100

    def test_short_text(self):
        # "hello" = 5 chars -> 5 // 4 = 1
        assert estimate_tokens_rough("hello") == 1

    def test_proportional(self):
        short = estimate_tokens_rough("hello world")
        long = estimate_tokens_rough("hello world " * 100)
        assert long > short


class TestEstimateMessagesTokensRough:
    def test_empty_list(self):
        assert estimate_messages_tokens_rough([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        result = estimate_messages_tokens_rough(msgs)
        assert result > 0

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there, how can I help?"},
        ]
        result = estimate_messages_tokens_rough(msgs)
        assert result > 0


# =========================================================================
# Default context lengths
# =========================================================================

class TestDefaultContextLengths:
    def test_claude_models_200k(self):
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            if "claude" in key:
                assert value == 200000, f"{key} should be 200000"

    def test_gpt4_models_128k(self):
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            if "gpt-4" in key:
                assert value == 128000, f"{key} should be 128000"

    def test_gemini_models_1m(self):
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            if "gemini" in key:
                assert value == 1048576, f"{key} should be 1048576"

    def test_all_values_positive(self):
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            assert value > 0, f"{key} has non-positive context length"


# =========================================================================
# get_model_context_length (with mocked API)
# =========================================================================

class TestGetModelContextLength:
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_known_model_from_api(self, mock_fetch):
        mock_fetch.return_value = {
            "test/model": {"context_length": 32000}
        }
        assert get_model_context_length("test/model") == 32000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_fallback_to_defaults(self, mock_fetch):
        mock_fetch.return_value = {}  # API returns nothing
        result = get_model_context_length("anthropic/claude-sonnet-4")
        assert result == 200000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_unknown_model_returns_128k(self, mock_fetch):
        mock_fetch.return_value = {}
        result = get_model_context_length("unknown/never-heard-of-this")
        assert result == 128000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_partial_match_in_defaults(self, mock_fetch):
        mock_fetch.return_value = {}
        # "gpt-4o" is a substring match for "openai/gpt-4o"
        result = get_model_context_length("openai/gpt-4o")
        assert result == 128000


# =========================================================================
# fetch_model_metadata (cache behavior)
# =========================================================================

class TestFetchModelMetadata:
    @patch("agent.model_metadata.requests.get")
    def test_caches_result(self, mock_get):
        import agent.model_metadata as mm
        # Reset cache
        mm._model_metadata_cache = {}
        mm._model_metadata_cache_time = 0

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "test/model", "context_length": 99999, "name": "Test Model"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # First call fetches
        result1 = fetch_model_metadata(force_refresh=True)
        assert "test/model" in result1
        assert mock_get.call_count == 1

        # Second call uses cache
        result2 = fetch_model_metadata()
        assert "test/model" in result2
        assert mock_get.call_count == 1  # Not called again

    @patch("agent.model_metadata.requests.get")
    def test_api_failure_returns_empty(self, mock_get):
        import agent.model_metadata as mm
        mm._model_metadata_cache = {}
        mm._model_metadata_cache_time = 0

        mock_get.side_effect = Exception("Network error")
        result = fetch_model_metadata(force_refresh=True)
        assert result == {}
