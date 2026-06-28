"""Tests for the rate-limit retry backoff caps in conversation_loop.

Backstory: NVIDIA NIM 429 cool-down windows are often 90-300s.  The
default cap of 120s on the Retry-After header and 60s on jittered
backoff meant we always retried before the bucket refilled and saw
the same 429 again, burning the 3-retry budget on the same hung quota.

This file pins the new caps (600s Retry-After / 120s jittered) so they
don't regress, and confirms the agent-level knob (api_max_retries)
gives users room to extend retries when dealing with a known 429-prone
provider.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestJitteredBackoffCap:
    """Pure-function tests on the helper, no full agent loop required."""

    def test_jittered_backoff_respects_new_120s_max(self):
        """With max_delay=120, deeper attempts hit the ceiling at ~120+jitter."""
        from agent.retry_utils import jittered_backoff

        # At deep attempts, the helper must cap at the supplied max.  With
        # the production cap moved from 60 → 120, a value like 180 should
        # be reachable for old hops that fed max_delay=120.
        seen = 0
        for n in range(1, 15):
            d = jittered_backoff(n, base_delay=2.0, max_delay=120.0, jitter_ratio=0.5)
            seen = max(seen, d)
        # Max possible with jitter_ratio=0.5 is 120 * 1.5 = 180.
        assert 100 <= seen <= 180, (
            f"jittered_backoff max_delay=120 should hit ~120s ceiling; "
            f"max observed was {seen:.1f}s — likely regressed to 60s cap?"
        )

    def test_jittered_backoff_old_60s_max_was_too_low(self):
        """Regression guard: 60s cap was insufficient for NVIDIA 429 windows.

        At attempt 4 (2^3 = 8 → 16s base), the jitter can swing the result up
        to base * 1.5 either way, so the cap doesn't yet dominate.  Use a
        deeper attempt where the cap is the binding constraint so the test
        is deterministic instead of jitter-only.
        """
        from agent.retry_utils import jittered_backoff

        # attempt=8 → 2^7 = 128, saturation point of both caps.
        d_120 = jittered_backoff(8, base_delay=2.0, max_delay=120.0)
        d_60 = jittered_backoff(8, base_delay=2.0, max_delay=60.0)
        # Both must be capped (since 128 > both 60 and 120), and 120s cap
        # produces a strictly larger value (capped value + jitter).
        assert d_120 >= 120.0, f"120s cap should be saturated; got {d_120}"
        assert d_60 >= 60.0, f"60s cap should be saturated; got {d_60}"
        assert d_120 > d_60, (
            f"120s cap should produce larger backoff than 60s at saturation; "
            f"got {d_120} vs {d_60}"
        )


class TestRateLimitRetrySurface:
    """Direct unit test of the cap constants we changed.

    These pin the exact literal values in the conversation_loop patch so
    a future "let's tighten that back" refactor can't silently undo the
    NVIDIA-friendly defaults.
    """

    def test_retry_after_cap_is_now_600s_not_120s(self):
        """The literal `min(float(_ra_raw), 600)` must still be in the code."""
        from agent import conversation_loop as cl

        with open(cl.__file__) as f:
            src = f.read()

        # The OLD cap (120) must still appear as a comment for context,
        # but the ACTIVE cap must be 600.
        assert "min(float(_ra_raw), 600)" in src, (
            "conversation_loop.py: Retry-After cap regression-detector "
            "expected the literal `min(float(_ra_raw), 600)`, didn't find it. "
            "Did someone lower the cap back to 120?"
        )
        assert "max_delay=120.0" in src, (
            "conversation_loop.py: jittered_backoff cap regression-detector "
            "expected `max_delay=120.0`, didn't find it.  Did someone revert "
            "the 60s → 120s cap bump for unknown-header 429s?"
        )

    def test_old_60s_cap_commented_out_not_active(self):
        """The literal `max_delay=60.0` near rate-limit context must be gone."""
        from agent import conversation_loop as cl

        with open(cl.__file__) as f:
            src = f.read()

        # Find the rate-limit backoff block and assert no 60s cap remains.
        # If the new code is the only place referencing 60 there, we're clean.
        # Simple regex over the window works because the cap is single-use.
        import re
        rate_limit_section = re.search(
            r"# For rate limits.*?wait_time = _retry_after if _retry_after",
            src,
            re.DOTALL,
        )
        assert rate_limit_section, "could not locate the rate-limit retry block"
        section = rate_limit_section.group(0)
        assert "max_delay=60.0" not in section, (
            "rate-limit retry block still uses max_delay=60.0 — NVIDIA 429s "
            "without Retry-After header won't survive long enough to retry."
        )


class TestApiMaxRetriesKnob:
    """The agent.api_max_retries config knob already exists; ensure it covers
    the user's need to bump retries when living with a flaky provider."""

    def test_default_api_max_retries_is_three(self):
        """Default 3 retries preserved for users who didn't ask for more."""
        from run_agent import AIAgent

        cfg = {"agent": {}}
        with patch("run_agent.OpenAI"), \
             patch("hermes_cli.config.load_config", return_value=cfg):
            agent = AIAgent(
                api_key="k",
                base_url="https://integrate.api.nvidia.com/v1",
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert agent._api_max_retries == 3

    def test_user_can_bump_retries_to_five(self):
        """Setting agent.api_max_retries=5 propagates to AIAgent._api_max_retries."""
        from run_agent import AIAgent

        cfg = {"agent": {"api_max_retries": 5}}
        with patch("run_agent.OpenAI"), \
             patch("hermes_cli.config.load_config", return_value=cfg):
            agent = AIAgent(
                api_key="k",
                base_url="https://integrate.api.nvidia.com/v1",
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert agent._api_max_retries == 5
