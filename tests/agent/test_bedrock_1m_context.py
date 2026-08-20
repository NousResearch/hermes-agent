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


class TestBedrockContextTableParity:
    """``BEDROCK_CONTEXT_LENGTHS`` must agree with ``DEFAULT_CONTEXT_LENGTHS``.

    The two tables are maintained by hand and the Bedrock one carries a
    comment requiring the 1M entries to match. Nothing enforced it, so
    ``claude-opus-5`` was added to ``DEFAULT_CONTEXT_LENGTHS`` while the
    Bedrock table kept falling through to the generic
    ``anthropic.claude-opus-4`` (200K) entry — the agent then compressed
    at 200K on a model that serves 1M.

    This is a parity invariant, not a snapshot: it derives the expected
    value from the source table, so a future model only has to be added
    in one place for the mismatch to be caught rather than hardcoded here.
    """

    def _bedrock_lookup(self, bedrock_table, model_id):
        """Resolve a Bedrock model id the way the adapter does: longest key."""
        matches = [(len(k), v) for k, v in bedrock_table.items() if k in model_id]
        return max(matches)[1] if matches else None

    def test_1m_claude_models_match_across_tables(self):
        from agent.bedrock_adapter import BEDROCK_CONTEXT_LENGTHS
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS

        mismatches = []
        for model, expected in DEFAULT_CONTEXT_LENGTHS.items():
            # Only bare Claude ids with a 1M window; skip dotted aliases,
            # which are spelling variants of an entry already covered.
            if not model.startswith("claude-") or "." in model:
                continue
            if expected != 1_000_000:
                continue
            actual = self._bedrock_lookup(
                BEDROCK_CONTEXT_LENGTHS, f"anthropic.{model}"
            )
            if actual != expected:
                mismatches.append(
                    f"anthropic.{model}: bedrock={actual} metadata={expected}"
                )

        assert not mismatches, (
            "BEDROCK_CONTEXT_LENGTHS is out of sync with "
            "DEFAULT_CONTEXT_LENGTHS for 1M-context Claude models. The "
            "agent will compress prematurely on Bedrock for:\n  "
            + "\n  ".join(mismatches)
        )

    def test_non_1m_claude_models_not_widened(self):
        """Guard the other direction: 200K models must stay 200K."""
        from agent.bedrock_adapter import BEDROCK_CONTEXT_LENGTHS

        for model_id in (
            "anthropic.claude-haiku-4-5",
            "anthropic.claude-sonnet-4-5",
            "anthropic.claude-opus-4",
        ):
            assert self._bedrock_lookup(BEDROCK_CONTEXT_LENGTHS, model_id) == 200_000, (
                f"{model_id} must remain 200K — a longest-key regression here "
                "would over-advertise context and cause hard API errors"
            )
