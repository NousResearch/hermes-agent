"""Tests for agent.models_dev — models.dev registry integration."""
from unittest.mock import patch, MagicMock

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
        "name": "audio-only",
        "models": {
            "tts-model": {
                "id": "tts-model",
                "limit": {"context": 0, "output": 0},
            },
        },
    },
    "opencode-go": {
        "id": "opencode-go",
        "name": "OpenCode Go",
        "models": {
            "qwen3.7-plus": {
                "id": "qwen3.7-plus",
                "limit": {"context": 1_000_000, "output": 65_536},
            },
            "qwen3.5-plus": {
                "id": "qwen3.5-plus",
                "limit": {"context": 262_144, "output": 65_536},
            },
        },
    },
    "requesty": {
        "id": "requesty",
        "name": "Requesty",
        "models": {
            "openai/gpt-5": {
                "id": "openai/gpt-5",
                "limit": {"context": 100_000, "output": 32_000},
            },
        },
    },
}


class TestProviderMapping:
    def test_xai_oauth_uses_xai_catalog(self):
        assert PROVIDER_TO_MODELS_DEV["xai"] == "xai"
        assert PROVIDER_TO_MODELS_DEV["xai-oauth"] == "xai"

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
    def test_xai_oauth_resolves_xai_context(self, mock_fetch):
        """xAI OAuth is an auth path, not a separate model catalog."""
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("xai-oauth", "grok-build-0.1") == 256000

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

    @patch("agent.models_dev.fetch_models_dev")
    def test_prefixed_slash_model_id_strips_provider(self, mock_fetch):
        """Regression: 'opencode-go/qwen3.7-plus' must resolve to the same
        context window as 'qwen3.7-plus' with provider='opencode-go'.

        Without the slash-form prefix strip in
        agent/model_metadata.py::_strip_provider_prefix, the prefixed
        id falls through to DEFAULT_CONTEXT_LENGTHS and matches a
        generic 'qwen' entry (131_072) instead of the provider-
        specific models.dev value (1_000_000)."""
        mock_fetch.return_value = SAMPLE_REGISTRY
        from agent.model_metadata import get_model_context_length
        bare     = get_model_context_length("qwen3.7-plus", provider="opencode-go")
        prefixed = get_model_context_length("opencode-go/qwen3.7-plus")
        assert bare == prefixed == 1_000_000

    @patch("agent.models_dev.fetch_models_dev")
    def test_prefixed_slash_model_id_qwen3_5(self, mock_fetch):
        """Same regression for qwen3.5-plus (262_144 ceiling)."""
        mock_fetch.return_value = SAMPLE_REGISTRY
        from agent.model_metadata import get_model_context_length
        bare     = get_model_context_length("qwen3.5-plus", provider="opencode-go")
        prefixed = get_model_context_length("opencode-go/qwen3.5-plus")
        assert bare == prefixed == 262_144

    def test_prefixed_slash_only_strips_first_segment(self):
        """The strip is 'first separator only' — 'requesty/openai/gpt-5'
        (two slashes) must look up the key 'openai/gpt-5' under provider
        'requesty', not 'gpt-5'. models.dev has thousands of multi-slash
        ids under 'requesty/<upstream>/<model>', so 'split on first
        separator only' is the correct semantics."""
        from agent.model_metadata import _strip_provider_prefix
        # Two slashes: keep first segment as prefix, keep the rest as-is.
        assert _strip_provider_prefix("requesty/openai/gpt-5") == "openai/gpt-5"
        # Three slashes: same — only the first gets stripped.
        assert _strip_provider_prefix("requesty/openai/gpt-5/variant") == "openai/gpt-5/variant"
        # Colon form still works (regression check on the legacy path).
        assert _strip_provider_prefix("local:my-model") == "my-model"
        # Ollama-style model:tag colons are NOT stripped.
        assert _strip_provider_prefix("qwen3.5:27b") == "qwen3.5:27b"
        # Unknown provider prefix is NOT stripped (only the allowlist matches).
        assert _strip_provider_prefix("unknown-provider/model-x") == "unknown-provider/model-x"


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

    @patch("agent.models_dev.requests.get")
    def test_fresh_disk_cache_skips_network(self, mock_get):
        """When in-mem cache is empty but disk cache exists and is fresh by
        mtime (< TTL), fetch_models_dev returns disk data without ever
        making the network call.

        This is the cold-start fast path: every fresh process previously
        paid ~500 ms re-fetching a registry that was already on disk
        from an earlier run.
        """
        import agent.models_dev as md
        # Empty in-mem cache so stage 1 doesn't short-circuit.
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_disk_cache_age_seconds", return_value=60.0), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY):
            result = fetch_models_dev()

        # The whole point: no network call.
        mock_get.assert_not_called()
        assert "anthropic" in result
        # In-mem cache populated so subsequent calls within the same
        # process stay on stage 1.
        assert md._models_dev_cache == SAMPLE_REGISTRY

    @patch("agent.models_dev.requests.get")
    def test_stale_disk_cache_falls_through_to_network(self, mock_get):
        """When the disk cache is OLDER than TTL, we must hit the network
        (and only fall back to the stale disk data if network fails)."""
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Disk cache exists but is older than the TTL — must NOT short-circuit.
        with patch.object(md, "_disk_cache_age_seconds",
                          return_value=md._MODELS_DEV_CACHE_TTL + 60), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev()

        mock_get.assert_called_once()
        assert "anthropic" in result

    @patch("agent.models_dev.requests.get")
    def test_force_refresh_skips_disk_cache(self, mock_get):
        """force_refresh=True bypasses BOTH the in-mem cache AND the
        disk-cache fast path. Used by ``hermes config refresh`` and
        anywhere else the user explicitly asked for fresh data.
        """
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Disk cache is fresh, but force_refresh must override it.
        with patch.object(md, "_disk_cache_age_seconds", return_value=60.0), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev(force_refresh=True)

        mock_get.assert_called_once()
        assert "anthropic" in result

    @patch("agent.models_dev.requests.get")
    def test_missing_disk_cache_falls_through_to_network(self, mock_get):
        """If the disk cache file doesn't exist (first-ever run, or it
        was deleted), fall through cleanly to network."""
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev()

        mock_get.assert_called_once()
        assert "anthropic" in result


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

    def test_vision_from_modalities_input_image(self):
        """Models with 'image' in modalities.input but attachment=False should
        still report supports_vision=True (the core fix in this PR)."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "gemma-4-31b-it")
        assert caps is not None
        assert caps.supports_vision is True

    def test_text_only_modalities_override_stale_attachment_flag(self):
        """Text-only modalities must win over stale attachment=True metadata."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "text-only-with-stale-attachment")
        assert caps is not None
        assert caps.supports_vision is False

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
