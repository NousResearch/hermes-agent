"""Tests for agent.models_dev — models.dev registry integration."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

import pytest

from agent.models_dev import (
    PROVIDER_TO_MODELS_DEV,
    _extract_context,
    _override_context_window,
    _resolve_model_override,
    fetch_models_dev,
    get_model_capabilities,
    get_model_info,
    get_provider_info,
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
    "xai": {
        "id": "xai",
        "name": "xAI",
        "models": {
            "grok-build-0.1": {
                "id": "grok-build-0.1",
                "limit": {"context": 256000, "output": 64000},
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
        assert PROVIDER_TO_MODELS_DEV["stepfun"] == "stepfun"
        assert PROVIDER_TO_MODELS_DEV["kilocode"] == "kilo"
        assert PROVIDER_TO_MODELS_DEV["ai-gateway"] == "vercel"

    def test_xai_oauth_uses_xai_catalog(self):
        assert PROVIDER_TO_MODELS_DEV["xai"] == "xai"
        assert PROVIDER_TO_MODELS_DEV["xai-oauth"] == "xai"

    def test_unmapped_provider_not_in_dict(self):
        assert "nous" not in PROVIDER_TO_MODELS_DEV



class TestExtractContext:
    def test_valid_entry(self):
        assert _extract_context({"limit": {"context": 128000}}) == 128000




    def test_non_dict_returns_none(self):
        assert _extract_context("not a dict") is None



class TestLookupModelsDevContext:
    @patch("agent.models_dev.fetch_models_dev")
    def test_exact_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") == 1000000






    @patch("agent.models_dev.fetch_models_dev")
    def test_zero_context_filtered(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        # audio-only is not a mapped provider, but test the filtering directly
        data = SAMPLE_REGISTRY["audio-only"]["models"]["tts-model"]
        assert _extract_context(data) is None



class TestFetchModelsDev:
    @pytest.fixture(autouse=True)
    def _reset_fetch_state(self):
        import agent.models_dev as md

        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_refresh_in_flight = False
        yield
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_refresh_in_flight = False






    @patch("agent.models_dev.requests.get")
    def test_stale_disk_cache_returns_without_foreground_network(self, mock_get):
        """#35838: stale disk cache should not wait on models.dev timeout."""
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_disk_cache_age_seconds",
                          return_value=md._MODELS_DEV_CACHE_TTL + 60), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_start_background_refresh_models_dev") as mock_refresh:
            result = fetch_models_dev()

        mock_get.assert_not_called()
        mock_refresh.assert_called_once()
        assert "anthropic" in result



    @patch("agent.models_dev.requests.get")
    def test_stale_cache_failure_enters_backoff_and_suppresses_retry(self, mock_get):
        import agent.models_dev as md

        mock_get.side_effect = OSError("models.dev unreachable")
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1

        with patch.object(
            md,
            "_disk_cache_age_seconds",
            return_value=md._MODELS_DEV_CACHE_TTL + 60,
        ), patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY):
            first = fetch_models_dev()
            # Join the background refresh worker so its failure backoff is
            # observable and requests.get stays patched for its lifetime.
            for worker in threading.enumerate():
                if worker.name == "models-dev-refresh":
                    worker.join(timeout=5)
                    assert not worker.is_alive()

        assert first == SAMPLE_REGISTRY
        assert not md._models_dev_refresh_in_flight
        assert md._models_dev_retry_after > time.time()
        mock_get.assert_called_once()

        # A subsequent stale-cache hit inside the backoff window must not
        # spawn another refresh worker (in_flight is set synchronously
        # before the worker thread starts, so False proves no spawn).
        md._models_dev_cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1
        second = fetch_models_dev()
        assert second == SAMPLE_REGISTRY
        assert not md._models_dev_refresh_in_flight
        mock_get.assert_called_once()

    @patch("agent.models_dev.requests.get")
    def test_background_refresh_success_commits_registry(self, mock_get):
        """The bg worker must save disk + swap mem cache + clear backoff."""
        import agent.models_dev as md

        response = MagicMock()
        response.json.return_value = SAMPLE_REGISTRY
        mock_get.return_value = response

        md._models_dev_cache = {"stale": {}}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = time.time() - 1

        with patch.object(md, "_save_disk_cache") as mock_save:
            # Run the worker synchronously — deterministic, no thread.
            md._models_dev_refresh_in_flight = True
            md._background_refresh_models_dev()

        mock_save.assert_called_once_with(SAMPLE_REGISTRY)
        assert md._models_dev_cache == SAMPLE_REGISTRY
        assert md._models_dev_cache_time > 0
        assert md._models_dev_retry_after == 0
        assert not md._models_dev_refresh_in_flight


    @patch("agent.models_dev.requests.get")
    def test_concurrent_refreshes_share_one_network_request(self, mock_get):
        import agent.models_dev as md

        request_started = threading.Event()
        release_request = threading.Event()
        response = MagicMock()
        response.json.return_value = SAMPLE_REGISTRY

        def blocking_get(*_args, **_kwargs):
            request_started.set()
            assert release_request.wait(timeout=5)
            return response

        mock_get.side_effect = blocking_get
        with patch.object(md, "_disk_cache_age_seconds", return_value=None), patch.object(
            md, "_save_disk_cache"
        ), ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(fetch_models_dev) for _ in range(6)]
            assert request_started.wait(timeout=2)
            release_request.set()
            results = [future.result(timeout=5) for future in futures]

        assert results == [SAMPLE_REGISTRY] * 6
        mock_get.assert_called_once()

    @patch("agent.models_dev.requests.get")
    def test_force_refresh_bypasses_failure_backoff(self, mock_get):
        import agent.models_dev as md

        response = MagicMock()
        response.json.return_value = SAMPLE_REGISTRY
        mock_get.side_effect = [OSError("models.dev unreachable"), response]

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), patch.object(
            md, "_load_disk_cache", return_value={}
        ), patch.object(md, "_save_disk_cache"):
            assert fetch_models_dev() == {}
            assert fetch_models_dev(force_refresh=True) == SAMPLE_REGISTRY

        assert mock_get.call_count == 2
        assert md._models_dev_retry_after == 0

    @pytest.mark.parametrize(
        ("cache", "cache_time", "disk_data", "expected"),
        [
            (SAMPLE_REGISTRY, lambda md: time.time(), {}, SAMPLE_REGISTRY),
            (
                SAMPLE_REGISTRY,
                lambda md: time.time() - md._MODELS_DEV_CACHE_TTL - 1,
                {},
                SAMPLE_REGISTRY,
            ),
            ({}, lambda _md: 0, {}, {}),
        ],
        ids=["fresh-memory", "stale-memory", "missing"],
    )
    @patch("agent.models_dev.requests.get")
    def test_network_disabled_never_fetches(
        self, mock_get, cache, cache_time, disk_data, expected
    ):
        import agent.models_dev as md

        md._models_dev_cache = cache
        md._models_dev_cache_time = cache_time(md)
        with patch.object(md, "_load_disk_cache", return_value=disk_data):
            result = fetch_models_dev(allow_network=False)

        assert result == expected
        mock_get.assert_not_called()







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
            "text-only-with-stale-attachment": {
                "id": "text-only-with-stale-attachment",
                "attachment": True,
                "tool_call": True,
                "modalities": {"input": ["text"]},
                "limit": {"context": 128000, "output": 8192},
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
        """Models with attachment=True and no modalities should report supports_vision=True."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        assert caps.supports_vision is True




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


# ---------------------------------------------------------------------------
# Per-model metadata overrides (model_overrides config)
# ---------------------------------------------------------------------------


class TestModelOverrides:
    """Tests for the model_overrides config system."""

    def _setup_overrides(self, overrides_dict):
        """Patch _load_model_overrides to return the given dict."""
        import agent.models_dev as md
        return patch.object(md, "_load_model_overrides", return_value=overrides_dict)

    # --- _resolve_model_override ---

    def test_per_provider_model_override(self):
        """Per-provider+model override is found first."""
        overrides = {
            "upstage": {
                "solar-pro4": {"context_window": 524288},
            },
        }
        with self._setup_overrides(overrides):
            result = _resolve_model_override("upstage", "solar-pro4")
        assert result is not None
        assert result["context_window"] == 524288

    def test_per_provider_default_fallback(self):
        """Per-provider _default is used when model not found."""
        overrides = {
            "upstage": {
                "_default": {"context_window": 128000},
            },
        }
        with self._setup_overrides(overrides):
            result = _resolve_model_override("upstage", "unknown-model")
        assert result is not None
        assert result["context_window"] == 128000

    def test_global_default_fallback(self):
        """Global _default is used when provider not found."""
        overrides = {
            "_default": {"context_window": 65536},
        }
        with self._setup_overrides(overrides):
            result = _resolve_model_override("unknown-provider", "unknown-model")
        assert result is not None
        assert result["context_window"] == 65536

    def test_no_override_returns_none(self):
        """No override found returns None."""
        with self._setup_overrides({}):
            result = _resolve_model_override("anthropic", "claude-sonnet-4")
        assert result is None

    def test_per_provider_model_beats_default(self):
        """Per-provider+model wins over per-provider _default."""
        overrides = {
            "upstage": {
                "solar-pro4": {"context_window": 524288},
                "_default": {"context_window": 128000},
            },
        }
        with self._setup_overrides(overrides):
            result = _resolve_model_override("upstage", "solar-pro4")
        assert result is not None
        assert result["context_window"] == 524288

    def test_per_provider_default_beats_global(self):
        """Per-provider _default wins over global _default."""
        overrides = {
            "upstage": {
                "_default": {"context_window": 128000},
            },
            "_default": {"context_window": 65536},
        }
        with self._setup_overrides(overrides):
            result = _resolve_model_override("upstage", "unknown-model")
        assert result is not None
        assert result["context_window"] == 128000

    # --- _override_context_window ---

    def test_override_context_window_returns_value(self):
        overrides = {
            "upstage": {
                "syn-pro": {"context_window": 65536},
            },
        }
        with self._setup_overrides(overrides):
            ctx = _override_context_window("upstage", "syn-pro")
        assert ctx == 65536

    def test_override_context_window_returns_none_when_missing(self):
        with self._setup_overrides({}):
            ctx = _override_context_window("upstage", "syn-pro")
        assert ctx is None

    def test_override_context_window_rejects_zero(self):
        overrides = {
            "upstage": {
                "bad-model": {"context_window": 0},
            },
        }
        with self._setup_overrides(overrides):
            ctx = _override_context_window("upstage", "bad-model")
        assert ctx is None

    # --- get_model_capabilities with overrides ---

    def test_caps_override_unknown_model(self):
        """Override provides capabilities for a model NOT in the catalog (#8731)."""
        overrides = {
            "custom:my-vllm": {
                "my-llava-model": {
                    "context_window": 8192,
                    "supports_vision": True,
                    "supports_reasoning": False,
                    "supports_tools": True,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            caps = get_model_capabilities("custom:my-vllm", "my-llava-model")
        assert caps is not None
        assert caps.context_window == 8192
        assert caps.supports_vision is True
        assert caps.supports_reasoning is False
        assert caps.supports_tools is True

    def test_caps_override_patches_existing_catalog_entry(self):
        """Override patches specific fields on a known catalog entry (#84482)."""
        overrides = {
            "anthropic": {
                "claude-sonnet-4": {
                    "context_window": 500000,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        # Override wins
        assert caps.context_window == 500000
        # Non-overridden fields preserved from catalog
        assert caps.supports_vision is True
        assert caps.supports_tools is True

    def test_caps_no_override_no_catalog_returns_none(self):
        """No override and no catalog entry → None."""
        with self._setup_overrides({}), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            caps = get_model_capabilities("anthropic", "unknown-model")
        assert caps is None

    def test_caps_override_default_for_unknown_model(self):
        """Per-provider _default provides capabilities for unknown models."""
        overrides = {
            "custom:my-vllm": {
                "_default": {
                    "context_window": 32768,
                    "supports_tools": True,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            caps = get_model_capabilities("custom:my-vllm", "some-new-model")
        assert caps is not None
        assert caps.context_window == 32768
        assert caps.supports_tools is True

    # --- lookup_models_dev_context with overrides ---

    def test_context_lookup_override_wins_over_catalog(self):
        """Override context_window wins over models.dev catalog value."""
        overrides = {
            "anthropic": {
                "claude-opus-4-6": {"context_window": 500000},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=SAMPLE_REGISTRY):
            ctx = lookup_models_dev_context("anthropic", "claude-opus-4-6")
        assert ctx == 500000

    def test_context_lookup_override_for_unknown_provider(self):
        """Override works for providers not in PROVIDER_TO_MODELS_DEV."""
        overrides = {
            "upstage": {
                "solar-pro4": {"context_window": 524288},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            ctx = lookup_models_dev_context("upstage", "solar-pro4")
        assert ctx == 524288

    # --- get_model_info with overrides ---

    def test_model_info_override_for_unknown_model(self):
        """Override provides full metadata for a model not in the catalog."""
        overrides = {
            "custom:my-vllm": {
                "my-llava-model": {
                    "name": "My LLaVA Model",
                    "family": "llava",
                    "reasoning": False,
                    "tool_call": True,
                    "limit": {"context": 8192, "output": 4096},
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            info = get_model_info("custom:my-vllm", "my-llava-model")
        assert info is not None
        assert info.name == "My LLaVA Model"
        assert info.family == "llava"
        assert info.context_window == 8192
        assert info.max_output == 4096
        assert info.tool_call is True
        assert info.reasoning is False

    def test_model_info_override_merges_with_catalog(self):
        """Override patches specific fields on a known catalog entry."""
        overrides = {
            "anthropic": {
                "claude-sonnet-4-6": {
                    "limit": {"context": 500000, "output": 64000},
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=SAMPLE_REGISTRY):
            info = get_model_info("anthropic", "claude-sonnet-4-6")
        assert info is not None
        # Override wins
        assert info.context_window == 500000
        # Non-overridden fields preserved from catalog
        assert info.name == "claude-sonnet-4-6"

