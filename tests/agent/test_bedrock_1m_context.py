"""Tests for the 1M-context beta header on AWS Bedrock Claude models.

Claude Opus 4.6/4.7 and Sonnet 4.6 support a 1M context window, but on AWS
Bedrock (and Azure AI Foundry) that window is still gated behind the
``context-1m-2025-08-07`` beta header as of 2026-04. Without it, Bedrock
caps these models at 200K even though ``model_metadata.py`` advertises 1M.

These tests guard the invariant that the header is always emitted on the
Bedrock client path, and that it survives the MiniMax bearer-auth strip.
"""

from unittest.mock import MagicMock, patch


class TestBedrockContext1MBeta:
    """``context-1m-2025-08-07`` must reach Bedrock Claude requests."""

    def test_common_betas_omit_1m(self):
        from agent.anthropic_adapter import _COMMON_BETAS, _CONTEXT_1M_BETA

        assert _CONTEXT_1M_BETA == "context-1m-2025-08-07"
        assert _CONTEXT_1M_BETA not in _COMMON_BETAS

    def test_common_betas_for_native_anthropic_omit_1m(self):
        """Native Anthropic omits the gated 1M beta by default."""
        from agent.anthropic_adapter import (
            _common_betas_for_base_url,
            _CONTEXT_1M_BETA,
        )

        assert _CONTEXT_1M_BETA not in _common_betas_for_base_url(None)
        assert _CONTEXT_1M_BETA not in _common_betas_for_base_url("")
        assert _CONTEXT_1M_BETA not in _common_betas_for_base_url(
            "https://api.anthropic.com"
        )

    def test_common_betas_for_azure_include_1m(self):
        """Azure AI Foundry still gates 1M context behind the beta header."""
        from agent.anthropic_adapter import (
            _common_betas_for_base_url,
            _CONTEXT_1M_BETA,
        )

        assert _CONTEXT_1M_BETA in _common_betas_for_base_url(
            "https://example.services.ai.azure.com/models/anthropic"
        )

    def test_common_betas_strips_1m_for_minimax(self):
        """MiniMax bearer-auth endpoints host their own models — strip 1M beta."""
        from agent.anthropic_adapter import (
            _common_betas_for_base_url,
            _CONTEXT_1M_BETA,
        )

        for url in (
            "https://api.minimax.io/anthropic",
            "https://api.minimaxi.com/anthropic",
        ):
            betas = _common_betas_for_base_url(url)
            assert _CONTEXT_1M_BETA not in betas, (
                f"1M beta must be stripped for MiniMax bearer endpoint {url}"
            )
            # Other betas still present
            assert "interleaved-thinking-2025-05-14" in betas

    def test_build_anthropic_bedrock_client_sends_1m_beta(self):
        """AnthropicBedrock client must carry the 1M beta in default_headers.

        This is the load-bearing assertion for the reported bug:
        without this header Bedrock serves Opus 4.6/4.7 with a 200K cap.
        """
        import agent.anthropic_adapter as adapter

        fake_sdk = MagicMock()
        fake_sdk.AnthropicBedrock = MagicMock()

        with patch.object(adapter, "_anthropic_sdk", fake_sdk):
            adapter.build_anthropic_bedrock_client(region="us-west-2")

        call_kwargs = fake_sdk.AnthropicBedrock.call_args.kwargs
        assert call_kwargs["aws_region"] == "us-west-2"

        default_headers = call_kwargs.get("default_headers") or {}
        beta_header = default_headers.get("anthropic-beta", "")
        assert "context-1m-2025-08-07" in beta_header, (
            "Bedrock client must send context-1m-2025-08-07 or Opus 4.6/4.7 "
            "silently caps at 200K context"
        )
        # Other common betas still present — no regression.
        assert "interleaved-thinking-2025-05-14" in beta_header
        assert "fine-grained-tool-streaming-2025-05-14" in beta_header

    def test_build_anthropic_client_includes_1m_for_azure(self):
        """Azure AI Foundry clients must still opt into the 1M beta."""
        import agent.anthropic_adapter as adapter

        fake_sdk = MagicMock()
        fake_sdk.Anthropic = MagicMock()

        with patch.object(adapter, "_anthropic_sdk", fake_sdk):
            adapter.build_anthropic_client(
                api_key="azure-key",
                base_url="https://example.services.ai.azure.com/models/anthropic",
            )

        kwargs = fake_sdk.Anthropic.call_args.kwargs
        beta_header = kwargs.get("default_headers", {}).get("anthropic-beta", "")
        assert "context-1m-2025-08-07" in beta_header, (
            "Azure client must send context-1m-2025-08-07 or 1M context "
            "can be capped behind the provider-side beta gate"
        )
