"""Tests for the 1M-context beta header on AWS Bedrock Claude models.

Claude Opus 4.6/4.7/4.8 and Sonnet 4.6 support a 1M context window, but on AWS
Bedrock (and Microsoft Foundry) that window has been gated behind the
``context-1m-2025-08-07`` beta header on some routes. Without it, Bedrock
can cap these models at 200K even though ``model_metadata.py`` advertises 1M.

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
            "Bedrock client must send context-1m-2025-08-07 or Opus 4.6/4.7/4.8 "
            "silently caps at 200K context"
        )
        # Other common betas still present — no regression.
        assert "interleaved-thinking-2025-05-14" in beta_header
        assert "fine-grained-tool-streaming-2025-05-14" in beta_header

    def test_build_converse_kwargs_sends_1m_beta_for_long_context_claude(self):
        """Bedrock Converse needs anthropic_beta in additionalModelRequestFields."""
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="au.anthropic.claude-opus-4-8",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert kwargs["additionalModelRequestFields"]["anthropic_beta"] == [
            "context-1m-2025-08-07"
        ]

    def test_build_converse_kwargs_skips_1m_beta_for_standard_context_claude(self):
        """Do not send the 1M beta for Claude IDs still capped at 200K."""
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="us.anthropic.claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert "additionalModelRequestFields" not in kwargs


class TestBedrockClientTimeout:
    """The AnthropicBedrock read timeout must default to 1800s and be overridable.

    The httpx read timeout caps how long a single Bedrock streaming response
    may run. A high-reasoning Opus review can stream past the historical
    900s ceiling and have its API call killed. The Anthropic idle-gap
    stream watchdog (see ``agent.chat_completion_helpers``) now owns staleness
    detection on a per-event basis (12–180s idle scaled by context, 120s
    TTFB for small contexts), so the SDK ``read`` timeout is a high
    defense-in-depth backstop (default 1800s) that must NOT preempt the
    watchdog. ``build_anthropic_bedrock_client`` takes an optional
    ``timeout`` so ``providers.bedrock.request_timeout_seconds``
    (resolved via ``get_provider_request_timeout``) can override that
    ceiling from config without a code change — but a value below the
    watchdog floor (``idle_max * attempts + ttfb`` = 660s with defaults) is
    raised to the floor so misconfig can't reintroduce the old bug.
    """

    def _timeout_obj(self, fake_sdk):
        return fake_sdk.AnthropicBedrock.call_args.kwargs["timeout"]

    def test_default_timeout_is_1800(self):
        """No timeout arg → 1800s read backstop (raised from the old 900s)."""
        import agent.anthropic_adapter as adapter

        fake_sdk = MagicMock()
        fake_sdk.AnthropicBedrock = MagicMock()

        with patch.object(adapter, "_anthropic_sdk", fake_sdk):
            adapter.build_anthropic_bedrock_client(region="us-west-2")

        timeout_obj = self._timeout_obj(fake_sdk)
        assert timeout_obj.read == 1800.0
        assert timeout_obj.connect == 10.0

    def test_explicit_timeout_overrides_default(self):
        """A positive timeout above the watchdog floor overrides the default."""
        import agent.anthropic_adapter as adapter

        fake_sdk = MagicMock()
        fake_sdk.AnthropicBedrock = MagicMock()

        # 1200s is above the 660s watchdog floor and below 1800s — a clean
        # override (no flooring, no clamping). Connect stays at 10s.
        with patch.object(adapter, "_anthropic_sdk", fake_sdk):
            adapter.build_anthropic_bedrock_client(region="us-west-2", timeout=1200)

        timeout_obj = self._timeout_obj(fake_sdk)
        assert timeout_obj.read == 1200.0
        assert timeout_obj.connect == 10.0

    def test_too_low_timeout_is_floored_to_watchdog_budget(self):
        """A misconfigured tiny timeout is RAISED to the watchdog floor (660s).

        The SDK ``read`` timeout is a backstop; if it fires before the
        per-event watchdog, the idle/TTFB detector can never run. Floor at
        ``idle_max * attempts + ttfb`` (180 * 3 + 120 = 660 with defaults).
        """
        import agent.anthropic_adapter as adapter

        fake_sdk = MagicMock()
        fake_sdk.AnthropicBedrock = MagicMock()

        with patch.object(adapter, "_anthropic_sdk", fake_sdk):
            adapter.build_anthropic_bedrock_client(region="us-west-2", timeout=120)

        timeout_obj = self._timeout_obj(fake_sdk)
        # 120 < 660 → floored up to 660
        assert timeout_obj.read == 660.0
        assert timeout_obj.connect == 10.0

    def test_none_and_nonpositive_timeout_fall_back_to_1800(self):
        """None / 0 / negative are ignored — the 1800s default holds."""
        import agent.anthropic_adapter as adapter

        for bad in (None, 0, -5):
            fake_sdk = MagicMock()
            fake_sdk.AnthropicBedrock = MagicMock()
            with patch.object(adapter, "_anthropic_sdk", fake_sdk):
                adapter.build_anthropic_bedrock_client(region="us-west-2", timeout=bad)
            timeout_obj = self._timeout_obj(fake_sdk)
            assert timeout_obj.read == 1800.0, f"timeout={bad!r} should fall back to 1800"
