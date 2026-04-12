"""Tests for agent.models_dev — models.dev registry integration."""
import json
from unittest.mock import patch, MagicMock

import pytest
from agent.models_dev import (
    PROVIDER_TO_MODELS_DEV,
    _extract_context,
    fetch_models_dev,
    get_model_capabilities,
    lookup_models_dev_context,
)


SAMPLE_REGISTRY = {
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
            "claude-opus-4-6": {
                "id": "claude-opus-4-6",
                "limit": {"context": 1000000, "output": 128000},
            },
            "claude-sonnet-4-6": {
                "id": "claude-sonnet-4-6",
                "limit": {"context": 1000000, "output": 64000},
            },
            "claude-sonnet-4-0": {
                "id": "claude-sonnet-4-0",
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
    "github-copilot": {
        "id": "github-copilot",
        "name": "GitHub Copilot",
        "models": {
            "claude-opus-4.6": {
                "id": "claude-opus-4.6",
                "limit": {"context": 128000, "output": 32000},
            },
        },
    },
    "kilo": {
        "id": "kilo",
        "name": "Kilo Gateway",
        "models": {
            "anthropic/claude-sonnet-4.6": {
                "id": "anthropic/claude-sonnet-4.6",
                "limit": {"context": 1000000, "output": 128000},
            },
        },
    },
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek",
        "models": {
            "deepseek-chat": {
                "id": "deepseek-chat",
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "audio-only": {
        "id": "audio-only",
        "models": {
            "tts-model": {
                "id": "tts-model",
                "limit": {"context": 0, "output": 0},
            },
        },
    },
}


class TestProviderMapping:
    def test_all_mapped_providers_are_strings(self):
        for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
            assert isinstance(hermes_id, str)
            assert isinstance(mdev_id, str)

    def test_known_providers_mapped(self):
        assert PROVIDER_TO_MODELS_DEV["anthropic"] == "anthropic"
        assert PROVIDER_TO_MODELS_DEV["copilot"] == "github-copilot"
        assert PROVIDER_TO_MODELS_DEV["kilocode"] == "kilo"
        assert PROVIDER_TO_MODELS_DEV["ai-gateway"] == "vercel"

    def test_unmapped_provider_not_in_dict(self):
        assert "nous" not in PROVIDER_TO_MODELS_DEV

    def test_openai_codex_mapped_to_openai(self):
        assert PROVIDER_TO_MODELS_DEV["openai"] == "openai"
        assert PROVIDER_TO_MODELS_DEV["openai-codex"] == "openai"


class TestExtractContext:
    def test_valid_entry(self):
        assert _extract_context({"limit": {"context": 128000}}) == 128000

    def test_zero_context_returns_none(self):
        assert _extract_context({"limit": {"context": 0}}) is None

    def test_missing_limit_returns_none(self):
        assert _extract_context({"id": "test"}) is None

    def test_missing_context_returns_none(self):
        assert _extract_context({"limit": {"output": 8192}}) is None

    def test_non_dict_returns_none(self):
        assert _extract_context("not a dict") is None

    def test_float_context_coerced_to_int(self):
        assert _extract_context({"limit": {"context": 131072.0}}) == 131072


class TestLookupModelsDevContext:
    @patch("agent.models_dev.fetch_models_dev")
    def test_exact_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") == 1000000

    @patch("agent.models_dev.fetch_models_dev")
    def test_case_insensitive_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "Claude-Opus-4-6") == 1000000

    @patch("agent.models_dev.fetch_models_dev")
    def test_provider_not_mapped(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("nous", "some-model") is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_model_not_found(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "nonexistent-model") is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_provider_aware_context(self, mock_fetch):
        """Same model, different context per provider."""
        mock_fetch.return_value = SAMPLE_REGISTRY
        # Anthropic direct: 1M
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") == 1000000
        # GitHub Copilot: only 128K for same model
        assert lookup_models_dev_context("copilot", "claude-opus-4.6") == 128000

    @patch("agent.models_dev.fetch_models_dev")
    def test_zero_context_filtered(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        # audio-only is not a mapped provider, but test the filtering directly
        data = SAMPLE_REGISTRY["audio-only"]["models"]["tts-model"]
        assert _extract_context(data) is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_empty_registry(self, mock_fetch):
        mock_fetch.return_value = {}
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") is None


class TestFetchModelsDev:
    @patch("agent.models_dev.requests.get")
    def test_fetch_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Clear caches
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev(force_refresh=True)

        assert "anthropic" in result
        assert len(result) == len(SAMPLE_REGISTRY)

    @patch("agent.models_dev.requests.get")
    def test_fetch_failure_returns_stale_cache(self, mock_get):
        mock_get.side_effect = Exception("network error")

        import agent.models_dev as md
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = 0  # expired

        with patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY):
            result = fetch_models_dev(force_refresh=True)

        assert "anthropic" in result

    @patch("agent.models_dev.requests.get")
    def test_in_memory_cache_used(self, mock_get):
        import agent.models_dev as md
        import time
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = time.time()  # fresh

        result = fetch_models_dev()
        mock_get.assert_not_called()
        assert result == SAMPLE_REGISTRY


# ---------------------------------------------------------------------------
# get_model_capabilities — vision via modalities.input
# ---------------------------------------------------------------------------


CAPS_REGISTRY = {
    "google": {
        "id": "google",
        "models": {
            "gemma-4-31b-it": {
                "id": "gemma-4-31b-it",
                "attachment": False,
                "tool_call": True,
                "modalities": {"input": ["text", "image"]},
                "limit": {"context": 128000, "output": 8192},
            },
            "gemma-3-1b": {
                "id": "gemma-3-1b",
                "tool_call": True,
                "limit": {"context": 32000, "output": 8192},
            },
        },
    },
    "anthropic": {
        "id": "anthropic",
        "models": {
            "claude-sonnet-4": {
                "id": "claude-sonnet-4",
                "attachment": True,
                "tool_call": True,
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
}


class TestGetModelCapabilities:
    """Tests for get_model_capabilities vision detection."""

    def test_vision_from_attachment_flag(self):
        """Models with attachment=True should report supports_vision=True."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        assert caps.supports_vision is True

    def test_vision_from_modalities_input_image(self):
        """Models with 'image' in modalities.input but attachment=False should
        still report supports_vision=True (the core fix in this PR)."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "gemma-4-31b-it")
        assert caps is not None
        assert caps.supports_vision is True

    def test_no_vision_without_attachment_or_modalities(self):
        """Models with neither attachment nor image modality should be non-vision."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "gemma-3-1b")
        assert caps is not None
        assert caps.supports_vision is False

    def test_modalities_non_dict_handled(self):
        """Non-dict modalities field should not crash."""
        registry = {
            "google": {"id": "google", "models": {
                "weird-model": {
                    "id": "weird-model",
                    "modalities": "text",  # not a dict
                    "limit": {"context": 200000, "output": 8192},
                },
            }},
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=registry):
            caps = get_model_capabilities("gemini", "weird-model")
        assert caps is not None
        assert caps.supports_vision is False

    def test_model_not_found_returns_none(self):
        """Unknown model should return None."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "nonexistent-model")
        assert caps is None
