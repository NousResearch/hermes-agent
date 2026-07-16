"""Regression tests for #24996 — fallback-switch storm on host memory.

When every provider in the fallback chain fails non-retryably back-to-back
(e.g. HTTP 400/402/429 across distinct providers), the within-turn walk is
bounded (``_fallback_index`` advances monotonically and the loop aborts when
the chain exhausts).  The damaging mode is *cross-turn*: ``restore_primary_
runtime`` resets ``_fallback_index = 0`` every turn, so a client that
re-submits immediately replays the entire chain — re-marshaling the full
(potentially 80k-token) context once per provider every turn — with no
throttle on the non-rate-limit path.

The fix arms a short cooldown via the existing ``_rate_limited_until`` gate
when the chain exhausts on a non-rate-limit failure, so the next turn's
restore stays gated (and does NOT reset the index) until the cooldown clears.
Rate-limit / billing failures keep their own 60s cooldown and are unaffected.
"""

from unittest.mock import MagicMock, patch
from run_agent import AIAgent
from agent.error_classifier import FailoverReason
from agent.chat_completion_helpers import _FALLBACK_EXHAUSTED_COOLDOWN_S


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
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


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


class TestExhaustionArmsCooldown:
    def test_non_retryable_exhaustion_arms_cooldown(self):
        """Walking a non-empty chain to exhaustion on a non-rate-limit
        failure arms a short ``_rate_limited_until`` cooldown.

        ``time.monotonic`` is frozen inside ``chat_completion_helpers`` so the
        cooldown math is exact and independent of CI scheduling latency — the
        previous wall-clock upper bound (``before + window + 1.0``) flaked on
        loaded runners when the three activation calls took longer than 1s.
        """
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent._rate_limited_until = 0
        frozen = 1_000.0
        with (
            patch("agent.chat_completion_helpers.time.monotonic", return_value=frozen),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "resolved"),
            ),
        ):
            assert agent._try_activate_fallback() is True   # -> entry 0
            assert agent._try_activate_fallback() is True   # -> entry 1
            # Chain now exhausted; a non-rate-limit failure must arm cooldown.
            assert agent._try_activate_fallback() is False
            cooldown = getattr(agent, "_rate_limited_until", 0)
        # Cooldown is exactly the short exhaustion window past the frozen clock,
        # not the 60s rate-limit one.
        assert cooldown == frozen + _FALLBACK_EXHAUSTED_COOLDOWN_S

    def test_no_chain_does_not_arm_cooldown(self):
        """An empty chain (no fallback configured) must not arm a cooldown —
        there is no chain to storm, and gating primary restoration would be
        pointless punishment."""
        agent = _make_agent(fallback_model=None)
        agent._rate_limited_until = 0
        assert agent._try_activate_fallback() is False
        assert getattr(agent, "_rate_limited_until", 0) == 0

    def test_rate_limit_exhaustion_keeps_60s_cooldown(self):
        """A rate-limit failure already arms its own 60s cooldown; the short
        exhaustion window must not shrink it."""
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        agent._rate_limited_until = 0
        frozen = 1_000.0
        with (
            patch("agent.chat_completion_helpers.time.monotonic", return_value=frozen),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "resolved"),
            ),
        ):
            # First activation with rate_limit reason arms the 60s cooldown.
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
            # Chain exhausted on the next call (also rate_limit) -> still False,
            # and the 60s cooldown must survive (max(), not overwritten down).
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is False
            cooldown = getattr(agent, "_rate_limited_until", 0)
        # ~60s past the frozen clock, far past the short exhaustion window.
        assert cooldown == frozen + 60

    def test_cooldown_never_shrinks_existing_window(self):
        """If a longer cooldown is already armed, exhaustion must not reduce
        it (we take the max)."""
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        frozen = 1_000.0
        far_future = frozen + 999
        agent._rate_limited_until = far_future
        with (
            patch("agent.chat_completion_helpers.time.monotonic", return_value=frozen),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "resolved"),
            ),
        ):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False
            cooldown = getattr(agent, "_rate_limited_until", 0)
        assert cooldown == far_future

    def test_exhaustion_does_not_extend_active_cooldown(self):
        """Repeated calls on an already-exhausted chain must not re-arm the
        exhaustion cooldown while it is still active.  This prevents a
        sub-window retry loop from permanently locking out the primary
        restore (#57582)."""
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent._rate_limited_until = 0
        frozen = 1_000.0

        # Walk to exhaustion and arm the cooldown.
        with (
            patch("agent.chat_completion_helpers.time.monotonic", return_value=frozen),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "resolved"),
            ),
        ):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False
        first_cooldown = getattr(agent, "_rate_limited_until", 0)
        assert first_cooldown == frozen + _FALLBACK_EXHAUSTED_COOLDOWN_S

        # Still within the cooldown window -- must NOT extend.
        still_in_window = frozen + _FALLBACK_EXHAUSTED_COOLDOWN_S - 1.0
        with (
            patch("agent.chat_completion_helpers.time.monotonic", return_value=still_in_window),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "resolved"),
            ),
        ):
            assert agent._try_activate_fallback() is False
        assert getattr(agent, "_rate_limited_until", 0) == first_cooldown

        # After cooldown expires -- a new window IS armed.
        after_expiry = frozen + _FALLBACK_EXHAUSTED_COOLDOWN_S + 0.5
        with (
            patch("agent.chat_completion_helpers.time.monotonic", return_value=after_expiry),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "resolved"),
            ),
        ):
            assert agent._try_activate_fallback() is False
        assert getattr(agent, "_rate_limited_until", 0) == after_expiry + _FALLBACK_EXHAUSTED_COOLDOWN_S
