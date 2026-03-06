"""Tests for agent/model_metadata.py — token estimation and context lengths."""

import os
import tempfile

import pytest
import yaml
from unittest.mock import patch, MagicMock

from agent.model_metadata import (
    CONTEXT_PROBE_TIERS,
    DEFAULT_CONTEXT_LENGTHS,
    estimate_tokens_rough,
    estimate_messages_tokens_rough,
    get_model_context_length,
    get_next_probe_tier,
    get_cached_context_length,
    parse_context_limit_from_error,
    save_context_length,
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
    def test_unknown_model_returns_first_probe_tier(self, mock_fetch):
        mock_fetch.return_value = {}
        result = get_model_context_length("unknown/never-heard-of-this")
        assert result == CONTEXT_PROBE_TIERS[0]  # 2M — will be narrowed on context error

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


# =========================================================================
# Context probe tiers
# =========================================================================

class TestContextProbeTiers:
    def test_tiers_descending(self):
        for i in range(len(CONTEXT_PROBE_TIERS) - 1):
            assert CONTEXT_PROBE_TIERS[i] > CONTEXT_PROBE_TIERS[i + 1]

    def test_first_tier_is_2m(self):
        assert CONTEXT_PROBE_TIERS[0] == 2_000_000

    def test_last_tier_is_32k(self):
        assert CONTEXT_PROBE_TIERS[-1] == 32_000


class TestGetNextProbeTier:
    def test_from_2m(self):
        assert get_next_probe_tier(2_000_000) == 1_000_000

    def test_from_1m(self):
        assert get_next_probe_tier(1_000_000) == 512_000

    def test_from_128k(self):
        assert get_next_probe_tier(128_000) == 64_000

    def test_from_32k_returns_none(self):
        assert get_next_probe_tier(32_000) is None

    def test_from_below_min_returns_none(self):
        assert get_next_probe_tier(16_000) is None

    def test_from_arbitrary_value(self):
        # 300K is between 512K and 200K, should return 200K
        assert get_next_probe_tier(300_000) == 200_000


# =========================================================================
# Error message parsing
# =========================================================================

class TestParseContextLimitFromError:
    def test_openai_format(self):
        msg = "This model's maximum context length is 32768 tokens. However, your messages resulted in 45000 tokens."
        assert parse_context_limit_from_error(msg) == 32768

    def test_context_length_exceeded(self):
        msg = "context_length_exceeded: maximum context length is 131072"
        assert parse_context_limit_from_error(msg) == 131072

    def test_context_size_exceeded(self):
        msg = "Maximum context size 65536 exceeded"
        assert parse_context_limit_from_error(msg) == 65536

    def test_no_limit_in_message(self):
        msg = "Something went wrong with the API"
        assert parse_context_limit_from_error(msg) is None

    def test_unreasonable_number_rejected(self):
        msg = "context length is 42 tokens"  # too small
        assert parse_context_limit_from_error(msg) is None

    def test_ollama_format(self):
        msg = "Context size has been exceeded. Maximum context size is 32768"
        assert parse_context_limit_from_error(msg) == 32768


# =========================================================================
# Persistent context length cache
# =========================================================================

class TestContextLengthCache:
    def test_save_and_load(self, tmp_path):
        cache_file = tmp_path / "context_length_cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("test/model", "http://localhost:8080/v1", 32768)
            result = get_cached_context_length("test/model", "http://localhost:8080/v1")
            assert result == 32768

    def test_missing_cache_returns_none(self, tmp_path):
        cache_file = tmp_path / "nonexistent.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            assert get_cached_context_length("test/model", "http://x") is None

    def test_multiple_models_cached(self, tmp_path):
        cache_file = tmp_path / "context_length_cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("model-a", "http://a", 64000)
            save_context_length("model-b", "http://b", 128000)
            assert get_cached_context_length("model-a", "http://a") == 64000
            assert get_cached_context_length("model-b", "http://b") == 128000

    def test_same_model_different_providers(self, tmp_path):
        cache_file = tmp_path / "context_length_cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("llama-3", "http://local:8080", 32768)
            save_context_length("llama-3", "https://openrouter.ai/api/v1", 131072)
            assert get_cached_context_length("llama-3", "http://local:8080") == 32768
            assert get_cached_context_length("llama-3", "https://openrouter.ai/api/v1") == 131072

    def test_idempotent_save(self, tmp_path):
        cache_file = tmp_path / "context_length_cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("model", "http://x", 32768)
            save_context_length("model", "http://x", 32768)  # same value
            with open(cache_file) as f:
                data = yaml.safe_load(f)
            assert len(data["context_lengths"]) == 1

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_cached_value_takes_priority(self, mock_fetch, tmp_path):
        """Cached context length should be used before API or defaults."""
        mock_fetch.return_value = {}
        cache_file = tmp_path / "context_length_cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("unknown/model", "http://local", 65536)
            result = get_model_context_length("unknown/model", base_url="http://local")
            assert result == 65536
