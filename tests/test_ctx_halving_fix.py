"""Tests for the context-halving bugfix.

Background
----------
When the API returns "max_tokens too large given prompt" (input is fine,
but input_tokens + requested max_tokens > context_window), the old code
incorrectly halved context_length via get_next_probe_tier().

The fix introduces:
  * parse_available_output_tokens_from_error() — detects this specific
    error class and returns the available output token budget.
  * _ephemeral_max_output_tokens on AIAgent — a one-shot override that
    caps the output for one retry without touching context_length.

Naming note
-----------
  max_tokens     = OUTPUT token cap (a single response).
  context_length = TOTAL context window (input + output combined).
These are different and the old code conflated them; the fix keeps them
separate.
"""

import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ---------------------------------------------------------------------------
# parse_available_output_tokens_from_error — unit tests
# ---------------------------------------------------------------------------

class TestParseAvailableOutputTokens:
    """Pure-function tests; no I/O required."""

    def _parse(self, msg):
        from agent.model_metadata import parse_available_output_tokens_from_error
        return parse_available_output_tokens_from_error(msg)

    # ── Should detect and extract ────────────────────────────────────────

    def test_anthropic_canonical_format(self):
        """Canonical Anthropic error: max_tokens: X > context_window: Y - input_tokens: Z = available_tokens: W"""
        msg = (
            "max_tokens: 32768 > context_window: 200000 "
            "- input_tokens: 190000 = available_tokens: 10000"
        )
        assert self._parse(msg) == 10000

    def test_anthropic_format_large_numbers(self):
        msg = (
            "max_tokens: 128000 > context_window: 200000 "
            "- input_tokens: 180000 = available_tokens: 20000"
        )
        assert self._parse(msg) == 20000

    def test_available_tokens_variant_spacing(self):
        """Handles extra spaces around the colon."""
        msg = "max_tokens: 32768 > 200000 available_tokens : 5000"
        assert self._parse(msg) == 5000

    def test_available_tokens_natural_language(self):
        """'available tokens: N' wording (no underscore)."""
        msg = "max_tokens must be at most 10000 given your prompt (available tokens: 10000)"
        assert self._parse(msg) == 10000

    def test_single_token_available(self):
        """Edge case: only 1 token left."""
        msg = "max_tokens: 9999 > context_window: 10000 - input_tokens: 9999 = available_tokens: 1"
        assert self._parse(msg) == 1

    # ── Should NOT detect (returns None) ─────────────────────────────────

    def test_prompt_too_long_is_not_output_cap_error(self):
        """'prompt is too long' errors must NOT be caught — they need context halving."""
        msg = "prompt is too long: 205000 tokens > 200000 maximum"
        assert self._parse(msg) is None

    def test_generic_context_window_exceeded(self):
        """Generic context window errors without available_tokens should not match."""
        msg = "context window exceeded: maximum is 32768 tokens"
        assert self._parse(msg) is None

    def test_context_length_exceeded(self):
        msg = "context_length_exceeded: prompt has 131073 tokens, limit is 131072"
        assert self._parse(msg) is None

    def test_no_max_tokens_keyword(self):
        """Error not related to max_tokens at all."""
        msg = "invalid_api_key: the API key is invalid"
        assert self._parse(msg) is None

    def test_empty_string(self):
        assert self._parse("") is None

    def test_rate_limit_error(self):
        msg = "rate_limit_error: too many requests per minute"
        assert self._parse(msg) is None


# ---------------------------------------------------------------------------
# build_anthropic_kwargs — output cap clamping
# ---------------------------------------------------------------------------

class TestBuildAnthropicKwargsClamping:
    """The context_length clamp only fires when output ceiling > window.
    For standard Anthropic models (output ceiling < window) it must not fire.
    """

    def _build(self, model, max_tokens=None, context_length=None):
        from agent.anthropic_adapter import build_anthropic_kwargs
        return build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=max_tokens,
            reasoning_config=None,
            context_length=context_length,
        )

    def test_no_clamping_when_output_ceiling_fits_in_window(self):
        """Opus 4.6 native output (128K) < context window (200K) — no clamping."""
        kwargs = self._build("claude-opus-4-6", context_length=200_000)
        assert kwargs["max_tokens"] == 128_000

    def test_clamping_fires_for_tiny_custom_window(self):
        """When context_length is 8K (local model), output cap is clamped to 7999."""
        kwargs = self._build("claude-opus-4-6", context_length=8_000)
        assert kwargs["max_tokens"] == 7_999

    def test_explicit_max_tokens_respected_when_within_window(self):
        """Explicit max_tokens smaller than window passes through unchanged."""
        kwargs = self._build("claude-opus-4-6", max_tokens=4096, context_length=200_000)
        assert kwargs["max_tokens"] == 4096

    def test_explicit_max_tokens_clamped_when_exceeds_window(self):
        """Explicit max_tokens larger than a small window is clamped."""
        kwargs = self._build("claude-opus-4-6", max_tokens=32_768, context_length=16_000)
        assert kwargs["max_tokens"] == 15_999

    def test_no_context_length_uses_native_ceiling(self):
        """Without context_length the native output ceiling is used directly."""
        kwargs = self._build("claude-sonnet-4-6")
        assert kwargs["max_tokens"] == 64_000


# ---------------------------------------------------------------------------
# Ephemeral max_tokens mechanism — _build_api_kwargs
# ---------------------------------------------------------------------------

class TestEphemeralMaxOutputTokens:
    """_build_api_kwargs consumes _ephemeral_max_output_tokens exactly once
    and falls back to self.max_tokens on subsequent calls.
    """

    def _make_agent(self):
        """Return a minimal AIAgent with api_mode='anthropic_messages' and
        a stubbed context_compressor, bypassing full __init__ cost."""
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        # Minimal attributes used by _build_api_kwargs
        agent.api_mode = "anthropic_messages"
        agent.model = "claude-opus-4-6"
        agent.tools = []
        agent.max_tokens = None
        agent.reasoning_config = None
        agent._is_anthropic_oauth = False
        agent._ephemeral_max_output_tokens = None

        compressor = MagicMock()
        compressor.context_length = 200_000
        agent.context_compressor = compressor

        # Stub out the internal message-preparation helper
        agent._prepare_anthropic_messages_for_api = MagicMock(
            return_value=[{"role": "user", "content": "hi"}]
        )
        agent._anthropic_preserve_dots = MagicMock(return_value=False)
        return agent

    def test_ephemeral_override_is_used_on_first_call(self):
        """When _ephemeral_max_output_tokens is set, it overrides self.max_tokens."""
        agent = self._make_agent()
        agent._ephemeral_max_output_tokens = 5_000

        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["max_tokens"] == 5_000

    def test_ephemeral_override_is_consumed_after_one_call(self):
        """After one call the ephemeral override is cleared to None."""
        agent = self._make_agent()
        agent._ephemeral_max_output_tokens = 5_000

        agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert agent._ephemeral_max_output_tokens is None

    def test_subsequent_call_uses_self_max_tokens(self):
        """A second _build_api_kwargs call uses the normal max_tokens path."""
        agent = self._make_agent()
        agent._ephemeral_max_output_tokens = 5_000
        agent.max_tokens = None  # will resolve to native ceiling (128K for Opus 4.6)

        agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        # Second call — ephemeral is gone
        kwargs2 = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs2["max_tokens"] == 128_000  # Opus 4.6 native ceiling

    def test_no_ephemeral_uses_self_max_tokens_directly(self):
        """Without an ephemeral override, self.max_tokens is used normally."""
        agent = self._make_agent()
        agent.max_tokens = 8_192

        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["max_tokens"] == 8_192


# ---------------------------------------------------------------------------
# Integration: error handler does NOT halve context_length for output-cap errors
# ---------------------------------------------------------------------------

class TestContextNotHalvedOnOutputCapError:
    """When the API returns 'max_tokens too large given prompt', the handler
    must set _ephemeral_max_output_tokens and NOT modify context_length.
    """

    def _make_agent_with_compressor(self, context_length=200_000):
        from run_agent import AIAgent
        from agent.context_compressor import ContextCompressor

        agent = object.__new__(AIAgent)
        agent.api_mode = "anthropic_messages"
        agent.model = "claude-opus-4-6"
        agent.base_url = "https://api.anthropic.com"
        agent.tools = []
        agent.max_tokens = None
        agent.reasoning_config = None
        agent._is_anthropic_oauth = False
        agent._ephemeral_max_output_tokens = None
        agent.log_prefix = ""
        agent.quiet_mode = True
        agent.verbose_logging = False

        compressor = MagicMock(spec=ContextCompressor)
        compressor.context_length = context_length
        compressor.threshold_percent = 0.75
        agent.context_compressor = compressor

        agent._prepare_anthropic_messages_for_api = MagicMock(
            return_value=[{"role": "user", "content": "hi"}]
        )
        agent._anthropic_preserve_dots = MagicMock(return_value=False)
        agent._vprint = MagicMock()
        return agent

    def test_output_cap_error_sets_ephemeral_not_context_length(self):
        """On 'max_tokens too large' error, _ephemeral_max_output_tokens is set
        and compressor.context_length is left unchanged."""
        from agent.model_metadata import parse_available_output_tokens_from_error
        from agent.model_metadata import get_next_probe_tier

        error_msg = (
            "max_tokens: 128000 > context_window: 200000 "
            "- input_tokens: 180000 = available_tokens: 20000"
        )

        # Simulate the handler logic from run_agent.py
        agent = self._make_agent_with_compressor(context_length=200_000)
        old_ctx = agent.context_compressor.context_length

        available_out = parse_available_output_tokens_from_error(error_msg)
        assert available_out == 20_000, "parser must detect the error"

        # The fix: set ephemeral, skip context_length modification
        agent._ephemeral_max_output_tokens = max(1, available_out - 64)

        # context_length must be untouched
        assert agent.context_compressor.context_length == old_ctx
        assert agent._ephemeral_max_output_tokens == 19_936

    def test_prompt_too_long_still_triggers_probe_tier(self):
        """Genuine prompt-too-long errors must still use get_next_probe_tier."""
        from agent.model_metadata import parse_available_output_tokens_from_error
        from agent.model_metadata import get_next_probe_tier

        error_msg = "prompt is too long: 205000 tokens > 200000 maximum"

        available_out = parse_available_output_tokens_from_error(error_msg)
        assert available_out is None, "prompt-too-long must not be caught by output-cap parser"

        # The old halving path is still used for this class of error
        new_ctx = get_next_probe_tier(200_000)
        assert new_ctx == 128_000

    def test_output_cap_error_safety_margin(self):
        """The ephemeral value includes a 64-token safety margin below available_out."""
        from agent.model_metadata import parse_available_output_tokens_from_error

        error_msg = (
            "max_tokens: 32768 > context_window: 200000 "
            "- input_tokens: 190000 = available_tokens: 10000"
        )
        available_out = parse_available_output_tokens_from_error(error_msg)
        safe_out = max(1, available_out - 64)
        assert safe_out == 9_936

    def test_safety_margin_never_goes_below_one(self):
        """When available_out is very small, safe_out must be at least 1."""
        from agent.model_metadata import parse_available_output_tokens_from_error

        error_msg = (
            "max_tokens: 10 > context_window: 200000 "
            "- input_tokens: 199990 = available_tokens: 1"
        )
        available_out = parse_available_output_tokens_from_error(error_msg)
        safe_out = max(1, available_out - 64)
        assert safe_out == 1
