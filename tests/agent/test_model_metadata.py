"""Tests for agent/model_metadata.py — token estimation, context lengths,
probing, caching, and error parsing.

Coverage levels:
  Token estimation       — concrete value assertions, edge cases
  Context length lookup  — resolution order, fuzzy match, cache priority
  API metadata fetch     — caching, TTL, canonical slugs, stale fallback
  Probe tiers            — descending, boundaries, extreme inputs
  Error parsing          — OpenAI, Ollama, Anthropic, edge cases
  Persistent cache       — save/load, corruption, update, provider isolation
"""

import os
import time
import tempfile

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.model_metadata import (
    CONTEXT_PROBE_TIERS,
    DEFAULT_CONTEXT_LENGTHS,
    _strip_provider_prefix,
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
        assert estimate_tokens_rough("a" * 400) == 100

    def test_short_text(self):
        assert estimate_tokens_rough("hello") == 1

    def test_proportional(self):
        short = estimate_tokens_rough("hello world")
        long = estimate_tokens_rough("hello world " * 100)
        assert long > short

    def test_unicode_multibyte(self):
        """Unicode chars are still 1 Python char each — 4 chars/token holds."""
        text = "你好世界"  # 4 CJK characters
        assert estimate_tokens_rough(text) == 1


class TestEstimateMessagesTokensRough:
    def test_empty_list(self):
        assert estimate_messages_tokens_rough([]) == 0

    def test_single_message_concrete_value(self):
        """Verify against known str(msg) length."""
        msg = {"role": "user", "content": "a" * 400}
        result = estimate_messages_tokens_rough([msg])
        expected = len(str(msg)) // 4
        assert result == expected

    def test_multiple_messages_additive(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there, how can I help?"},
        ]
        result = estimate_messages_tokens_rough(msgs)
        expected = sum(len(str(m)) for m in msgs) // 4
        assert result == expected

    def test_tool_call_message(self):
        """Tool call messages with no 'content' key still contribute tokens."""
        msg = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "1", "function": {"name": "terminal", "arguments": "{}"}}]}
        result = estimate_messages_tokens_rough([msg])
        assert result > 0
        assert result == len(str(msg)) // 4

    def test_message_with_list_content(self):
        """Vision messages with multimodal content arrays."""
        msg = {"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}
        result = estimate_messages_tokens_rough([msg])
        assert result == len(str(msg)) // 4


# =========================================================================
# Default context lengths
# =========================================================================

class TestDefaultContextLengths:
    def test_claude_models_context_lengths(self):
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            if "claude" not in key:
                continue
            # Claude 4.6 models have 1M context
            if "4.6" in key or "4-6" in key:
                assert value == 1000000, f"{key} should be 1000000"
            else:
                assert value == 200000, f"{key} should be 200000"

    def test_gpt4_models_128k_or_1m(self):
        # gpt-4.1 and gpt-4.1-mini have 1M context; other gpt-4* have 128k
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            if "gpt-4" in key and "gpt-4.1" not in key:
                assert value == 128000, f"{key} should be 128000"

    def test_gpt41_models_1m(self):
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            if "gpt-4.1" in key:
                assert value == 1047576, f"{key} should be 1047576"

    def test_gemini_models_1m(self):
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            if "gemini" in key:
                assert value == 1048576, f"{key} should be 1048576"

    def test_all_values_positive(self):
        for key, value in DEFAULT_CONTEXT_LENGTHS.items():
            assert value > 0, f"{key} has non-positive context length"

    def test_dict_is_not_empty(self):
        assert len(DEFAULT_CONTEXT_LENGTHS) >= 10


# =========================================================================
# get_model_context_length — resolution order
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
        mock_fetch.return_value = {}
        assert get_model_context_length("anthropic/claude-sonnet-4") == 200000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_unknown_model_returns_first_probe_tier(self, mock_fetch):
        mock_fetch.return_value = {}
        assert get_model_context_length("unknown/never-heard-of-this") == CONTEXT_PROBE_TIERS[0]

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_partial_match_in_defaults(self, mock_fetch):
        mock_fetch.return_value = {}
        assert get_model_context_length("openai/gpt-4o") == 128000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_api_missing_context_length_key(self, mock_fetch):
        """Model in API but without context_length → defaults to 128000."""
        mock_fetch.return_value = {"test/model": {"name": "Test"}}
        assert get_model_context_length("test/model") == 128000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_cache_takes_priority_over_api(self, mock_fetch, tmp_path):
        """Persistent cache should be checked BEFORE API metadata."""
        mock_fetch.return_value = {"my/model": {"context_length": 999999}}
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("my/model", "http://local", 32768)
            result = get_model_context_length("my/model", base_url="http://local")
            assert result == 32768  # cache wins over API's 999999

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_no_base_url_skips_cache(self, mock_fetch, tmp_path):
        """Without base_url, cache lookup is skipped."""
        mock_fetch.return_value = {}
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("custom/model", "http://local", 32768)
            # No base_url → cache skipped → falls to probe tier
            result = get_model_context_length("custom/model")
            assert result == CONTEXT_PROBE_TIERS[0]

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_custom_endpoint_metadata_beats_fuzzy_default(self, mock_endpoint_fetch, mock_fetch):
        mock_fetch.return_value = {}
        mock_endpoint_fetch.return_value = {
            "zai-org/GLM-5-TEE": {"context_length": 65536}
        }

        result = get_model_context_length(
            "zai-org/GLM-5-TEE",
            base_url="https://llm.chutes.ai/v1",
            api_key="test-key",
        )

        assert result == 65536

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_custom_endpoint_without_metadata_skips_name_based_default(self, mock_endpoint_fetch, mock_fetch):
        mock_fetch.return_value = {}
        mock_endpoint_fetch.return_value = {}

        result = get_model_context_length(
            "zai-org/GLM-5-TEE",
            base_url="https://llm.chutes.ai/v1",
            api_key="test-key",
        )

        assert result == CONTEXT_PROBE_TIERS[0]

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_custom_endpoint_single_model_fallback(self, mock_endpoint_fetch, mock_fetch):
        """Single-model servers: use the only model even if name doesn't match."""
        mock_fetch.return_value = {}
        mock_endpoint_fetch.return_value = {
            "Qwen3.5-9B-Q4_K_M.gguf": {"context_length": 131072}
        }

        result = get_model_context_length(
            "qwen3.5:9b",
            base_url="http://myserver.example.com:8080/v1",
            api_key="test-key",
        )

        assert result == 131072

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_custom_endpoint_fuzzy_substring_match(self, mock_endpoint_fetch, mock_fetch):
        """Fuzzy match: configured model name is substring of endpoint model."""
        mock_fetch.return_value = {}
        mock_endpoint_fetch.return_value = {
            "org/llama-3.3-70b-instruct-fp8": {"context_length": 131072},
            "org/qwen-2.5-72b": {"context_length": 32768},
        }

        result = get_model_context_length(
            "llama-3.3-70b-instruct",
            base_url="http://myserver.example.com:8080/v1",
            api_key="test-key",
        )

        assert result == 131072

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_config_context_length_overrides_all(self, mock_fetch):
        """Explicit config_context_length takes priority over everything."""
        mock_fetch.return_value = {
            "test/model": {"context_length": 200000}
        }

        result = get_model_context_length(
            "test/model",
            config_context_length=65536,
        )

        assert result == 65536

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_config_context_length_zero_is_ignored(self, mock_fetch):
        """config_context_length=0 should be treated as unset."""
        mock_fetch.return_value = {}

        result = get_model_context_length(
            "anthropic/claude-sonnet-4",
            config_context_length=0,
        )

        assert result == 200000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_config_context_length_none_is_ignored(self, mock_fetch):
        """config_context_length=None should be treated as unset."""
        mock_fetch.return_value = {}

        result = get_model_context_length(
            "anthropic/claude-sonnet-4",
            config_context_length=None,
        )

        assert result == 200000


# =========================================================================
# _strip_provider_prefix — Ollama model:tag vs provider:model
# =========================================================================

class TestStripProviderPrefix:
    def test_known_provider_prefix_is_stripped(self):
        assert _strip_provider_prefix("local:my-model") == "my-model"
        assert _strip_provider_prefix("openrouter:anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"
        assert _strip_provider_prefix("anthropic:claude-sonnet-4") == "claude-sonnet-4"

    def test_ollama_model_tag_preserved(self):
        """Ollama model:tag format must NOT be stripped."""
        assert _strip_provider_prefix("qwen3.5:27b") == "qwen3.5:27b"
        assert _strip_provider_prefix("llama3.3:70b") == "llama3.3:70b"
        assert _strip_provider_prefix("gemma2:9b") == "gemma2:9b"
        assert _strip_provider_prefix("codellama:13b-instruct-q4_0") == "codellama:13b-instruct-q4_0"

    def test_http_urls_preserved(self):
        assert _strip_provider_prefix("http://example.com") == "http://example.com"
        assert _strip_provider_prefix("https://example.com") == "https://example.com"

    def test_no_colon_returns_unchanged(self):
        assert _strip_provider_prefix("gpt-4o") == "gpt-4o"
        assert _strip_provider_prefix("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_ollama_model_tag_not_mangled_in_context_lookup(self, mock_fetch):
        """Ensure 'qwen3.5:27b' is NOT reduced to '27b' during context length lookup.

        We mock a custom endpoint that knows 'qwen3.5:27b' — the full name
        must reach the endpoint metadata lookup intact.
        """
        mock_fetch.return_value = {}
        with patch("agent.model_metadata.fetch_endpoint_model_metadata") as mock_ep, \
             patch("agent.model_metadata._is_custom_endpoint", return_value=True):
            mock_ep.return_value = {"qwen3.5:27b": {"context_length": 32768}}
            result = get_model_context_length(
                "qwen3.5:27b",
                base_url="http://localhost:11434/v1",
            )
        assert result == 32768


# =========================================================================
# fetch_model_metadata — caching, TTL, slugs, failures
# =========================================================================

class TestFetchModelMetadata:
    def _reset_cache(self):
        import agent.model_metadata as mm
        mm._model_metadata_cache = {}
        mm._model_metadata_cache_time = 0

    @patch("agent.model_metadata.requests.get")
    def test_caches_result(self, mock_get):
        self._reset_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "test/model", "context_length": 99999, "name": "Test"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result1 = fetch_model_metadata(force_refresh=True)
        assert "test/model" in result1
        assert mock_get.call_count == 1

        result2 = fetch_model_metadata()
        assert "test/model" in result2
        assert mock_get.call_count == 1  # cached

    @patch("agent.model_metadata.requests.get")
    def test_api_failure_returns_empty_on_cold_cache(self, mock_get):
        self._reset_cache()
        mock_get.side_effect = Exception("Network error")
        result = fetch_model_metadata(force_refresh=True)
        assert result == {}

    @patch("agent.model_metadata.requests.get")
    def test_api_failure_returns_stale_cache(self, mock_get):
        """On API failure with existing cache, stale data is returned."""
        import agent.model_metadata as mm
        mm._model_metadata_cache = {"old/model": {"context_length": 50000}}
        mm._model_metadata_cache_time = 0  # expired

        mock_get.side_effect = Exception("Network error")
        result = fetch_model_metadata(force_refresh=True)
        assert "old/model" in result
        assert result["old/model"]["context_length"] == 50000

    @patch("agent.model_metadata.requests.get")
    def test_canonical_slug_aliasing(self, mock_get):
        """Models with canonical_slug get indexed under both IDs."""
        self._reset_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{
                "id": "anthropic/claude-3.5-sonnet:beta",
                "canonical_slug": "anthropic/claude-3.5-sonnet",
                "context_length": 200000,
                "name": "Claude 3.5 Sonnet"
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_model_metadata(force_refresh=True)
        # Both the original ID and canonical slug should work
        assert "anthropic/claude-3.5-sonnet:beta" in result
        assert "anthropic/claude-3.5-sonnet" in result
        assert result["anthropic/claude-3.5-sonnet"]["context_length"] == 200000

    @patch("agent.model_metadata.requests.get")
    def test_provider_prefixed_models_get_bare_aliases(self, mock_get):
        self._reset_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{
                "id": "provider/test-model",
                "context_length": 123456,
                "name": "Provider: Test Model",
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_model_metadata(force_refresh=True)

        assert result["provider/test-model"]["context_length"] == 123456
        assert result["test-model"]["context_length"] == 123456

    @patch("agent.model_metadata.requests.get")
    def test_ttl_expiry_triggers_refetch(self, mock_get):
        """Cache expires after _MODEL_CACHE_TTL seconds."""
        import agent.model_metadata as mm
        self._reset_cache()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "m1", "context_length": 1000, "name": "M1"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        fetch_model_metadata(force_refresh=True)
        assert mock_get.call_count == 1

        # Simulate TTL expiry
        mm._model_metadata_cache_time = time.time() - _MODEL_CACHE_TTL - 1
        fetch_model_metadata()
        assert mock_get.call_count == 2  # refetched

    @patch("agent.model_metadata.requests.get")
    def test_malformed_json_no_data_key(self, mock_get):
        """API returns JSON without 'data' key — empty cache, no crash."""
        self._reset_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "something"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

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
        assert get_next_probe_tier(300_000) == 200_000

    def test_above_max_tier(self):
        """Value above 2M should return 2M."""
        assert get_next_probe_tier(5_000_000) == 2_000_000

    def test_zero_returns_none(self):
        assert get_next_probe_tier(0) is None


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
        assert parse_context_limit_from_error("Something went wrong with the API") is None

    def test_unreasonable_small_number_rejected(self):
        assert parse_context_limit_from_error("context length is 42 tokens") is None

    def test_ollama_format(self):
        msg = "Context size has been exceeded. Maximum context size is 32768"
        assert parse_context_limit_from_error(msg) == 32768

    def test_anthropic_format(self):
        msg = "prompt is too long: 250000 tokens > 200000 maximum"
        # Should extract 200000 (the limit), not 250000 (the input size)
        assert parse_context_limit_from_error(msg) == 200000

    def test_lmstudio_format(self):
        msg = "Error: context window of 4096 tokens exceeded"
        assert parse_context_limit_from_error(msg) == 4096

    def test_completely_unrelated_error(self):
        assert parse_context_limit_from_error("Invalid API key") is None

    def test_empty_string(self):
        assert parse_context_limit_from_error("") is None

    def test_number_outside_reasonable_range(self):
        """Very large number (>10M) should be rejected."""
        msg = "maximum context length is 99999999999"
        assert parse_context_limit_from_error(msg) is None


# =========================================================================
# Persistent context length cache
# =========================================================================

class TestContextLengthCache:
    def test_save_and_load(self, tmp_path):
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("test/model", "http://localhost:8080/v1", 32768)
            assert get_cached_context_length("test/model", "http://localhost:8080/v1") == 32768

    def test_missing_cache_returns_none(self, tmp_path):
        cache_file = tmp_path / "nonexistent.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            assert get_cached_context_length("test/model", "http://x") is None

    def test_multiple_models_cached(self, tmp_path):
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("model-a", "http://a", 64000)
            save_context_length("model-b", "http://b", 128000)
            assert get_cached_context_length("model-a", "http://a") == 64000
            assert get_cached_context_length("model-b", "http://b") == 128000

    def test_same_model_different_providers(self, tmp_path):
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("llama-3", "http://local:8080", 32768)
            save_context_length("llama-3", "https://openrouter.ai/api/v1", 131072)
            assert get_cached_context_length("llama-3", "http://local:8080") == 32768
            assert get_cached_context_length("llama-3", "https://openrouter.ai/api/v1") == 131072

    def test_idempotent_save(self, tmp_path):
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("model", "http://x", 32768)
            save_context_length("model", "http://x", 32768)
            with open(cache_file) as f:
                data = yaml.safe_load(f)
            assert len(data["context_lengths"]) == 1

    def test_update_existing_value(self, tmp_path):
        """Saving a different value for the same key overwrites it."""
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("model", "http://x", 128000)
            save_context_length("model", "http://x", 64000)
            assert get_cached_context_length("model", "http://x") == 64000

    def test_corrupted_yaml_returns_empty(self, tmp_path):
        """Corrupted cache file is handled gracefully."""
        cache_file = tmp_path / "cache.yaml"
        cache_file.write_text("{{{{not valid yaml: [[[")
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            assert get_cached_context_length("model", "http://x") is None

    def test_wrong_structure_returns_none(self, tmp_path):
        """YAML that loads but has wrong structure."""
        cache_file = tmp_path / "cache.yaml"
        cache_file.write_text("just_a_string\n")
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            assert get_cached_context_length("model", "http://x") is None

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_cached_value_takes_priority(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {}
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("unknown/model", "http://local", 65536)
            assert get_model_context_length("unknown/model", base_url="http://local") == 65536

    def test_special_chars_in_model_name(self, tmp_path):
        """Model names with colons, slashes, etc. don't break the cache."""
        cache_file = tmp_path / "cache.yaml"
        model = "anthropic/claude-3.5-sonnet:beta"
        url = "https://api.example.com/v1"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length(model, url, 200000)
            assert get_cached_context_length(model, url) == 200000
