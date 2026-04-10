"""Tests for MiniMax provider hardening — context lengths, thinking guard, catalog, beta headers."""

from unittest.mock import patch


class TestMinimaxContextLengths:
    """Verify per-model context length entries for MiniMax models."""

    def test_m1_variants_have_1m_context(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        # Keys are lowercase because the lookup lowercases model names
        for model in ("minimax-m1", "minimax-m1-40k", "minimax-m1-80k",
                       "minimax-m1-128k", "minimax-m1-256k"):
            assert model in DEFAULT_CONTEXT_LENGTHS, f"{model} missing from context lengths"
            assert DEFAULT_CONTEXT_LENGTHS[model] == 1_000_000, f"{model} expected 1M"

    def test_m2_variants_have_1m_context(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        # Keys are lowercase because the lookup lowercases model names
        for model in ("minimax-m2.5", "minimax-m2.7"):
            assert model in DEFAULT_CONTEXT_LENGTHS, f"{model} missing from context lengths"
            assert DEFAULT_CONTEXT_LENGTHS[model] == 1_048_576, f"{model} expected 1048576"

    def test_minimax_prefix_fallback(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        # The generic "minimax" prefix entry should be 1M for unknown models
        assert DEFAULT_CONTEXT_LENGTHS["minimax"] == 1_048_576



class TestMinimaxThinkingGuard:
    """Verify that build_anthropic_kwargs does NOT add thinking params for MiniMax models."""

    def test_no_thinking_for_minimax_m27(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "thinking" not in kwargs
        assert "output_config" not in kwargs

    def test_no_thinking_for_minimax_m1(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="MiniMax-M1-128k",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert "thinking" not in kwargs

    def test_thinking_still_works_for_claude(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "thinking" in kwargs


class TestMinimaxAuxModel:
    """Verify auxiliary model is standard (not highspeed)."""

    def test_minimax_aux_is_standard(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert _API_KEY_PROVIDER_AUX_MODELS["minimax"] == "MiniMax-M2.7"
        assert _API_KEY_PROVIDER_AUX_MODELS["minimax-cn"] == "MiniMax-M2.7"

    def test_minimax_aux_not_highspeed(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert "highspeed" not in _API_KEY_PROVIDER_AUX_MODELS["minimax"]
        assert "highspeed" not in _API_KEY_PROVIDER_AUX_MODELS["minimax-cn"]


class TestMinimaxModelCatalog:
    """Verify the model catalog includes M1 family and excludes deprecated models."""

    def test_catalog_includes_m1_family(self):
        from hermes_cli.models import _PROVIDER_MODELS
        for provider in ("minimax", "minimax-cn"):
            models = _PROVIDER_MODELS[provider]
            assert "MiniMax-M1" in models
            assert "MiniMax-M1-40k" in models
            assert "MiniMax-M1-80k" in models
            assert "MiniMax-M1-128k" in models
            assert "MiniMax-M1-256k" in models

    def test_catalog_excludes_deprecated(self):
        from hermes_cli.models import _PROVIDER_MODELS
        for provider in ("minimax", "minimax-cn"):
            models = _PROVIDER_MODELS[provider]
            assert "MiniMax-M2.1" not in models

    def test_catalog_excludes_highspeed(self):
        from hermes_cli.models import _PROVIDER_MODELS
        for provider in ("minimax", "minimax-cn"):
            models = _PROVIDER_MODELS[provider]
            assert "MiniMax-M2.7-highspeed" not in models
            assert "MiniMax-M2.5-highspeed" not in models


class TestMinimaxBetaHeaders:
    """MiniMax Anthropic-compat endpoints reject fine-grained-tool-streaming beta.

    Verify that build_anthropic_client omits the tool-streaming beta for MiniMax
    (both global and China domains) while keeping it for native Anthropic and
    other third-party endpoints.  Covers the fix for #6510 / #6555.
    """

    _TOOL_BETA = "fine-grained-tool-streaming-2025-05-14"
    _THINKING_BETA = "interleaved-thinking-2025-05-14"

    # -- helper ----------------------------------------------------------

    def _build_and_get_betas(self, api_key, base_url=None):
        """Build client, return the anthropic-beta header string."""
        from agent.anthropic_adapter import build_anthropic_client
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(api_key, base_url=base_url)
            kwargs = mock_sdk.Anthropic.call_args[1]
            headers = kwargs.get("default_headers", {})
            return headers.get("anthropic-beta", "")

    # -- MiniMax global --------------------------------------------------

    def test_minimax_global_omits_tool_streaming(self):
        betas = self._build_and_get_betas(
            "mm-key-123", base_url="https://api.minimax.io/anthropic"
        )
        assert self._TOOL_BETA not in betas
        assert self._THINKING_BETA in betas

    def test_minimax_global_trailing_slash(self):
        betas = self._build_and_get_betas(
            "mm-key-123", base_url="https://api.minimax.io/anthropic/"
        )
        assert self._TOOL_BETA not in betas

    # -- MiniMax China ---------------------------------------------------

    def test_minimax_cn_omits_tool_streaming(self):
        betas = self._build_and_get_betas(
            "mm-cn-key-456", base_url="https://api.minimaxi.com/anthropic"
        )
        assert self._TOOL_BETA not in betas
        assert self._THINKING_BETA in betas

    def test_minimax_cn_trailing_slash(self):
        betas = self._build_and_get_betas(
            "mm-cn-key-456", base_url="https://api.minimaxi.com/anthropic/"
        )
        assert self._TOOL_BETA not in betas

    # -- Non-MiniMax keeps full betas ------------------------------------

    def test_native_anthropic_keeps_tool_streaming(self):
        betas = self._build_and_get_betas("sk-ant-api03-real-key-here")
        assert self._TOOL_BETA in betas
        assert self._THINKING_BETA in betas

    def test_third_party_proxy_keeps_tool_streaming(self):
        betas = self._build_and_get_betas(
            "custom-key", base_url="https://my-proxy.example.com/anthropic"
        )
        assert self._TOOL_BETA in betas

    def test_custom_base_url_keeps_tool_streaming(self):
        betas = self._build_and_get_betas(
            "custom-key", base_url="https://custom.api.com"
        )
        assert self._TOOL_BETA in betas

    # -- _common_betas_for_base_url unit tests ---------------------------

    def test_common_betas_none_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _COMMON_BETAS
        assert _common_betas_for_base_url(None) == _COMMON_BETAS

    def test_common_betas_empty_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _COMMON_BETAS
        assert _common_betas_for_base_url("") == _COMMON_BETAS

    def test_common_betas_minimax_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _TOOL_STREAMING_BETA
        betas = _common_betas_for_base_url("https://api.minimax.io/anthropic")
        assert _TOOL_STREAMING_BETA not in betas
        assert len(betas) > 0  # still has other betas

    def test_common_betas_minimax_cn_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _TOOL_STREAMING_BETA
        betas = _common_betas_for_base_url("https://api.minimaxi.com/anthropic")
        assert _TOOL_STREAMING_BETA not in betas

    def test_common_betas_regular_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _COMMON_BETAS
        assert _common_betas_for_base_url("https://api.anthropic.com") == _COMMON_BETAS
