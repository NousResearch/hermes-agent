"""Tests for the 1M-context beta header on AWS Bedrock Claude models.

Claude Opus 4.6/4.7 and Sonnet 4.6 support a 1M context window, but on AWS
Bedrock (and Microsoft Foundry) that window is still gated behind the
``context-1m-2025-08-07`` beta header as of 2026-04. Without it, Bedrock
caps these models at 200K even though ``model_metadata.py`` advertises 1M.

These tests guard the invariant that the header is always emitted on the
Bedrock client path, and that it survives the MiniMax bearer-auth strip.
"""

from unittest.mock import MagicMock, patch


class TestBedrockContext1MBeta:
    """``context-1m-2025-08-07`` must reach Bedrock Claude requests."""



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


class TestBedrockConverse1MContextOptIn:
    """Native boto3 Converse path (agent/bedrock_adapter.py) — issue #31277.

    Opus 4.8 and Sonnet 5 (and the earlier 4.6/4.7 families) need the
    ``context-1m-2025-08-07`` beta forwarded via ``additionalModelRequestFields``
    on the Converse API. Off by default — only active when the user opts in
    via ``HERMES_BEDROCK_1M_CONTEXT``, since the entitlement is gated
    per-account and an unentitled account gets a hard rejection otherwise.
    """

    def test_capability_matrix(self):
        from agent.bedrock_adapter import is_bedrock_1m_context_capable

        capable = [
            "anthropic.claude-opus-4-8-v1:0",
            "us.anthropic.claude-opus-4-8-v1:0",
            "global.anthropic.claude-sonnet-5-20260201-v1:0",
            "anthropic.claude-sonnet-4-6-v1:0",
            "anthropic.claude-opus-4-6-v1:0",
            "anthropic.claude-opus-4-7-v1:0",
        ]
        not_capable = [
            "anthropic.claude-sonnet-4-5-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "amazon.nova-pro-v1:0",
        ]
        for model_id in capable:
            assert is_bedrock_1m_context_capable(model_id), model_id
        for model_id in not_capable:
            assert not is_bedrock_1m_context_capable(model_id), model_id

    def test_opt_in_env_var(self, monkeypatch):
        from agent.bedrock_adapter import bedrock_1m_context_enabled

        monkeypatch.delenv("HERMES_BEDROCK_1M_CONTEXT", raising=False)
        assert bedrock_1m_context_enabled() is False

        monkeypatch.setenv("HERMES_BEDROCK_1M_CONTEXT", "true")
        assert bedrock_1m_context_enabled() is True

        monkeypatch.setenv("HERMES_BEDROCK_1M_CONTEXT", "0")
        assert bedrock_1m_context_enabled() is False

    def test_build_converse_kwargs_adds_beta_when_opted_in(self, monkeypatch):
        from agent.bedrock_adapter import build_converse_kwargs

        monkeypatch.setenv("HERMES_BEDROCK_1M_CONTEXT", "true")
        kwargs = build_converse_kwargs(
            model="anthropic.claude-opus-4-8-v1:0",
            messages=[{"role": "user", "content": "hi"}],
        )
        betas = kwargs["additionalModelRequestFields"]["anthropic_beta"]
        assert "context-1m-2025-08-07" in betas

    def test_build_converse_kwargs_no_beta_when_not_opted_in(self, monkeypatch):
        from agent.bedrock_adapter import build_converse_kwargs

        monkeypatch.delenv("HERMES_BEDROCK_1M_CONTEXT", raising=False)
        kwargs = build_converse_kwargs(
            model="anthropic.claude-opus-4-8-v1:0",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "additionalModelRequestFields" not in kwargs

    def test_build_converse_kwargs_no_beta_for_incapable_model(self, monkeypatch):
        from agent.bedrock_adapter import build_converse_kwargs

        monkeypatch.setenv("HERMES_BEDROCK_1M_CONTEXT", "true")
        kwargs = build_converse_kwargs(
            model="anthropic.claude-sonnet-4-5-v1:0",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "additionalModelRequestFields" not in kwargs

    def test_get_bedrock_context_length_opt_in(self, monkeypatch):
        from agent.bedrock_adapter import get_bedrock_context_length

        monkeypatch.delenv("HERMES_BEDROCK_1M_CONTEXT", raising=False)
        assert get_bedrock_context_length("anthropic.claude-opus-4-8-v1:0") == 200_000

        monkeypatch.setenv("HERMES_BEDROCK_1M_CONTEXT", "true")
        assert get_bedrock_context_length("anthropic.claude-opus-4-8-v1:0") == 1_000_000
        # Non-capable model is unaffected by opt-in.
        assert get_bedrock_context_length("amazon.nova-pro-v1:0") == 300_000
