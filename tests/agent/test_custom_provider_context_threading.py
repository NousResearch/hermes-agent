"""Tests for custom_providers context_length threading across all call sites.

Regression tests ensuring that ``custom_providers[].models.<id>.context_length``
overrides are honored not just at agent startup (agent_init) and /model switch
(model_switch), but also at every deferred resolution point:

  * ContextCompressor._resolve_context_length  (deferred first-access probe)
  * auxiliary_client._candidate_context_window (fallback chain screening)
  * moa_loop._trim_messages_for_reference      (MoA reference model trimming)
  * web_server.get_model_info                  (WebUI model info endpoint)

See #15779 for the original /model switch fix; this extends the same contract
to sibling call paths that were missed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.config import get_custom_provider_context_length

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CUSTOM_PROVIDERS = [
    {
        "name": "test-provider",
        "base_url": "https://custom.example.com/v1",
        "models": {
            "my-model": {"context_length": 1_000_000},
        },
    },
]

BASE_URL = "https://custom.example.com/v1"
MODEL = "my-model"
EXPECTED_CTX = 1_000_000


def _mock_all_probes():
    """Disable every downstream resolution step so only the
    custom_providers override (step 0b) can produce a result."""
    from agent import model_metadata as _mm
    return [
        patch.object(_mm, "get_cached_context_length", return_value=None),
        patch.object(_mm, "fetch_endpoint_model_metadata", return_value={}),
        patch.object(_mm, "fetch_model_metadata", return_value={}),
        patch.object(_mm, "is_local_endpoint", return_value=False),
        patch.object(_mm, "_is_known_provider_base_url", return_value=False),
    ]


# ---------------------------------------------------------------------------
# 1. ContextCompressor._resolve_context_length
# ---------------------------------------------------------------------------

class TestContextCompressorCustomProviders:
    """ContextCompressor must honor custom_providers per-model overrides
    when it lazily resolves context_length on first property access."""

    def test_resolve_context_length_uses_custom_providers(self):
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            model=MODEL,
            base_url=BASE_URL,
            provider="custom",
            custom_providers=CUSTOM_PROVIDERS,
        )
        patches = _mock_all_probes()
        for p in patches:
            p.start()
        try:
            ctx = compressor._resolve_context_length()
        finally:
            for p in patches:
                p.stop()

        assert ctx == EXPECTED_CTX, (
            f"Expected {EXPECTED_CTX} from custom_providers override, got {ctx}"
        )

    def test_resolve_context_length_without_custom_providers_falls_through(self):
        """Without custom_providers, resolver falls through to default."""
        from agent.context_compressor import ContextCompressor
        from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT

        compressor = ContextCompressor(
            model="unknown-model",
            base_url=BASE_URL,
            provider="custom",
            custom_providers=None,
        )
        patches = _mock_all_probes()
        for p in patches:
            p.start()
        try:
            ctx = compressor._resolve_context_length()
        finally:
            for p in patches:
                p.stop()

        assert ctx == DEFAULT_FALLBACK_CONTEXT

    def test_config_context_length_still_wins_over_custom_providers(self):
        """Explicit config_context_length (step 0) outranks custom_providers (step 0b)."""
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            model=MODEL,
            base_url=BASE_URL,
            provider="custom",
            config_context_length=500_000,
            custom_providers=CUSTOM_PROVIDERS,
        )
        ctx = compressor._resolve_context_length()
        assert ctx == 500_000

    def test_custom_providers_stored_on_instance(self):
        """The custom_providers list is stored for deferred resolution."""
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            model=MODEL,
            base_url=BASE_URL,
            provider="custom",
            custom_providers=CUSTOM_PROVIDERS,
        )
        assert compressor._custom_providers is CUSTOM_PROVIDERS

    def test_default_custom_providers_is_none(self):
        """Omitting custom_providers defaults to None (backward compat)."""
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            model=MODEL,
            base_url=BASE_URL,
            provider="custom",
        )
        assert compressor._custom_providers is None


# ---------------------------------------------------------------------------
# 2. auxiliary_client._candidate_context_window
# ---------------------------------------------------------------------------

class TestCandidateContextWindowCustomProviders:
    """_candidate_context_window must load custom_providers from config
    and pass them to get_model_context_length."""

    def test_honors_custom_providers_override(self):
        from agent.auxiliary_client import _candidate_context_window

        mock_config = {"custom_providers": CUSTOM_PROVIDERS}
        with (
            patch(
                "hermes_cli.config.load_config_readonly",
                return_value=mock_config,
            ),
            patch(
                "hermes_cli.config.get_compatible_custom_providers",
                return_value=CUSTOM_PROVIDERS,
            ),
        ):
            patches = _mock_all_probes()
            for p in patches:
                p.start()
            try:
                ctx = _candidate_context_window(
                    "custom", MODEL, base_url=BASE_URL,
                )
            finally:
                for p in patches:
                    p.stop()

        assert ctx == EXPECTED_CTX

    def test_config_load_failure_falls_through_gracefully(self):
        """If config loading fails, resolver still works (returns default)."""
        from agent.auxiliary_client import _candidate_context_window
        from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT

        with patch(
            "hermes_cli.config.load_config_readonly",
            side_effect=RuntimeError("config unavailable"),
        ):
            patches = _mock_all_probes()
            for p in patches:
                p.start()
            try:
                ctx = _candidate_context_window(
                    "custom", "unknown-model", base_url=BASE_URL,
                )
            finally:
                for p in patches:
                    p.stop()

        assert ctx == DEFAULT_FALLBACK_CONTEXT

    def test_empty_model_returns_none(self):
        from agent.auxiliary_client import _candidate_context_window

        assert _candidate_context_window("custom", "", base_url=BASE_URL) is None


# ---------------------------------------------------------------------------
# 3. moa_loop._load_custom_providers + _trim_messages_for_reference
# ---------------------------------------------------------------------------

class TestMoACustomProviders:
    """MoA reference trimming must honor custom_providers overrides."""

    def test_load_custom_providers_returns_list(self):
        from agent.moa_loop import _load_custom_providers

        mock_config = {"custom_providers": CUSTOM_PROVIDERS}
        with (
            patch(
                "hermes_cli.config.load_config_readonly",
                return_value=mock_config,
            ),
            patch(
                "hermes_cli.config.get_compatible_custom_providers",
                return_value=CUSTOM_PROVIDERS,
            ),
        ):
            result = _load_custom_providers()

        assert result == CUSTOM_PROVIDERS

    def test_load_custom_providers_returns_none_on_failure(self):
        from agent.moa_loop import _load_custom_providers

        with patch(
            "hermes_cli.config.load_config_readonly",
            side_effect=RuntimeError("no config"),
        ):
            result = _load_custom_providers()

        assert result is None

    def test_trim_messages_uses_custom_providers_context(self):
        """_trim_messages_for_reference resolves context via custom_providers."""
        from agent.moa_loop import _trim_messages_for_reference

        slot = {"model": MODEL, "provider": "custom"}
        runtime = {"base_url": BASE_URL, "api_key": "test-key", "provider": "custom"}

        # Build messages that would fit in 1M but not in 256K
        # ~300K tokens worth of text (chars/4 heuristic)
        big_content = "x" * 1_200_000  # ~300K tokens
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": big_content},
            {"role": "assistant", "content": "OK"},
        ]

        mock_config = {"custom_providers": CUSTOM_PROVIDERS}
        with (
            patch(
                "hermes_cli.config.load_config_readonly",
                return_value=mock_config,
            ),
            patch(
                "hermes_cli.config.get_compatible_custom_providers",
                return_value=CUSTOM_PROVIDERS,
            ),
        ):
            patches = _mock_all_probes()
            for p in patches:
                p.start()
            try:
                result = _trim_messages_for_reference(
                    messages, slot, runtime,
                )
            finally:
                for p in patches:
                    p.stop()

        # With 1M context, messages should NOT be trimmed (they fit)
        assert len(result) == len(messages), (
            f"Messages should not be trimmed with 1M context, "
            f"got {len(result)} of {len(messages)}"
        )


# ---------------------------------------------------------------------------
# 4. web_server.get_model_info
# ---------------------------------------------------------------------------

class TestWebServerModelInfoCustomProviders:
    """WebUI /api/model/info must pass custom_providers to the resolver."""

    def test_get_model_info_passes_custom_providers(self):
        """Verify get_model_context_length receives custom_providers kwarg."""
        from hermes_cli.web_server import get_model_info

        mock_config = {
            "model": {
                "default": MODEL,
                "provider": "custom",
                "base_url": BASE_URL,
            },
            "custom_providers": CUSTOM_PROVIDERS,
        }

        captured_kwargs = {}

        def _capture_get_model_context_length(**kwargs):
            captured_kwargs.update(kwargs)
            return 256_000  # default

        with (
            patch("hermes_cli.web_server.load_config", return_value=mock_config),
            patch(
                "agent.model_metadata.get_model_context_length",
                side_effect=_capture_get_model_context_length,
            ),
            patch(
                "hermes_cli.config.get_compatible_custom_providers",
                return_value=CUSTOM_PROVIDERS,
            ),
        ):
            try:
                get_model_info()
            except Exception:
                pass  # endpoint may need app context; we only care about kwargs

        assert "custom_providers" in captured_kwargs, (
            "get_model_context_length was not called with custom_providers"
        )
        assert captured_kwargs["custom_providers"] == CUSTOM_PROVIDERS


# ---------------------------------------------------------------------------
# 5. get_custom_provider_context_length (existing helper — extended coverage)
# ---------------------------------------------------------------------------

class TestGetCustomProviderContextLengthExtended:
    """Extended coverage for the lookup helper used by all call sites."""

    def test_model_not_in_entry_returns_none(self):
        assert (
            get_custom_provider_context_length(
                "other-model", BASE_URL, CUSTOM_PROVIDERS,
            )
            is None
        )

    def test_base_url_mismatch_returns_none(self):
        assert (
            get_custom_provider_context_length(
                MODEL, "https://wrong.example.com/v1", CUSTOM_PROVIDERS,
            )
            is None
        )

    def test_models_as_list_returns_none(self):
        """List-format models (no per-model config) must return None."""
        providers = [
            {
                "base_url": BASE_URL,
                "models": [MODEL, "other-model"],
            }
        ]
        assert (
            get_custom_provider_context_length(MODEL, BASE_URL, providers)
            is None
        )

    def test_zero_context_length_returns_none(self):
        providers = [
            {
                "base_url": BASE_URL,
                "models": {MODEL: {"context_length": 0}},
            }
        ]
        assert (
            get_custom_provider_context_length(MODEL, BASE_URL, providers)
            is None
        )

    def test_negative_context_length_returns_none(self):
        providers = [
            {
                "base_url": BASE_URL,
                "models": {MODEL: {"context_length": -100}},
            }
        ]
        assert (
            get_custom_provider_context_length(MODEL, BASE_URL, providers)
            is None
        )

    def test_string_context_length_coerced(self):
        """String integers are coerced (config YAML may parse as str)."""
        providers = [
            {
                "base_url": BASE_URL,
                "models": {MODEL: {"context_length": "1000000"}},
            }
        ]
        assert (
            get_custom_provider_context_length(MODEL, BASE_URL, providers)
            == 1_000_000
        )

    def test_multiple_entries_first_match_wins(self):
        providers = [
            {
                "base_url": BASE_URL,
                "models": {MODEL: {"context_length": 500_000}},
            },
            {
                "base_url": BASE_URL,
                "models": {MODEL: {"context_length": 1_000_000}},
            },
        ]
        assert (
            get_custom_provider_context_length(MODEL, BASE_URL, providers)
            == 500_000
        )

    def test_model_cfg_not_dict_returns_none(self):
        """models.<id> must be a dict; a bare string is invalid."""
        providers = [
            {
                "base_url": BASE_URL,
                "models": {MODEL: "1000000"},
            }
        ]
        assert (
            get_custom_provider_context_length(MODEL, BASE_URL, providers)
            is None
        )
