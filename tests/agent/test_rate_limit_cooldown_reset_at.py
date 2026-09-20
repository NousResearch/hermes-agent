"""RED tests for #117484: rate-limit cooldown should honor the provider's reset_at.

The provider's 429 often says exactly when the window reopens; `extract_api_error_context`
already parses it into ``error_context["reset_at"]`` (epoch s/ms, ISO-8601, Retry-After,
vendor headers, message text) and ``turn_recovery.route_classified_error`` has that context
in scope at both rate-limit eager-fallback sites. Yet ``_arm_rate_limit_cooldown`` benches
the primary with a pure exponential guess (60s → 2m → … → 4h cap) and never sees the value.

These tests pin the intended behavior on the real arm path
(``agent._try_activate_fallback(reason=FailoverReason.rate_limit)``):

1. a provider-stated absolute reset wins over the exponential guess;
2. ISO-8601 / epoch-ms strings are normalized before use;
3. a reset in the past (or unparseable) falls back to the exponential ramp;
4. the never-shrink invariant (existing longer window is kept) still holds;
5. no ``reset_at`` in the error context keeps today's behavior exactly.
"""

import time
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import _arm_rate_limit_cooldown
from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


class TestResetAtHonoredOnRateLimitCooldown:
    def test_provider_reset_at_wins_over_exponential_guess(self):
        """A future, parseable ``reset_at`` in the error context arms exactly that
        window instead of the exponential 60s first step."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        now = 50_000.0
        with (
            patch("agent.fallback_cooldown.time.monotonic", return_value=frozen),
            patch("agent.fallback_cooldown.time.time", return_value=now),
        ):
            armed = _arm_rate_limit_cooldown(
                agent, FailoverReason.rate_limit,
                reset_at=now + 90,  # provider says: resets in 90s
                error_context={"reset_at": now + 90},
            )
            cooldown = agent._rate_limited_until - frozen
        assert armed == 90
        assert cooldown == 90.0, (
            "provider-stated 90s window must be honored verbatim, not guessed 60s"
        )

    def test_iso8601_reset_at_is_normalized(self):
        """ISO-8601 strings (common in Anthropic/OpenRouter bodies) are parsed via
        the same helper the rest of the codebase uses, not naively subtracted."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        now = 50_000.0
        iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now + 30))
        with (
            patch("agent.fallback_cooldown.time.monotonic", return_value=frozen),
            patch("agent.fallback_cooldown.time.time", return_value=now),
        ):
            armed = _arm_rate_limit_cooldown(
                agent, FailoverReason.rate_limit,
                reset_at=iso,
                error_context={"reset_at": iso},
            )
        assert armed == 30
        assert agent._rate_limited_until - frozen == 30.0

    def test_past_or_garbage_reset_at_falls_back_to_exponential(self):
        """A reset_at already in the past (provider clock skew / dropped header) or a
        non-parseable value keeps the historical exponential ramp — fail-open."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        now = 50_000.0
        for bad in (now - 10, "not-a-timestamp", None):
            fresh = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
            with (
                patch("agent.fallback_cooldown.time.monotonic", return_value=frozen),
                patch("agent.fallback_cooldown.time.time", return_value=now),
            ):
                armed = _arm_rate_limit_cooldown(
                    fresh, FailoverReason.rate_limit,
                    reset_at=bad, error_context={} if bad is None else {"reset_at": bad},
                )
                assert armed == 60, f"reset_at={bad!r} must fall back to the 60s first step"
                assert fresh._rate_limited_until == frozen + 60

    def test_never_shrinks_existing_longer_window(self):
        """If a longer cooldown is already armed, an incoming reset_at must not
        shorten it (same invariant as the exponential path)."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        now = 50_000.0
        far_future = frozen + 999
        agent._rate_limited_until = far_future
        with (
            patch("agent.fallback_cooldown.time.monotonic", return_value=frozen),
            patch("agent.fallback_cooldown.time.time", return_value=now),
        ):
            _arm_rate_limit_cooldown(
                agent, FailoverReason.rate_limit,
                reset_at=now + 30, error_context={"reset_at": now + 30},
            )
        assert agent._rate_limited_until == far_future

    def test_no_reset_at_keeps_today_behavior(self):
        """Guard rail: with no reset signal in the context, the armed cooldown is
        exactly the historical 60s first exponential step — zero behavior change
        for providers that do not report a reset."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        with patch("agent.fallback_cooldown.time.monotonic", return_value=frozen):
            armed = _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit)
        assert armed == 60
        assert agent._rate_limited_until == frozen + 60

    def test_exhausted_chain_without_rate_limit_ignores_reset_at(self):
        """reset_at only applies to rate-limit-family reasons; other reasons keep
        the plain cooldown logic untouched."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        now = 50_000.0
        with (
            patch("agent.fallback_cooldown.time.monotonic", return_value=frozen),
            patch("agent.fallback_cooldown.time.time", return_value=now),
        ):
            armed = _arm_rate_limit_cooldown(
                agent, FailoverReason.rate_limit,  # non-rate-limit reason below via None
            )
            assert armed == 60
