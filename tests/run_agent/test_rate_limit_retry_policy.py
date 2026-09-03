"""Tests for agent.rate_limit_retry policy surface.

Covers: config resolution (agent_init wiring), the RateLimitRetryPolicy
dataclass semantics, and the policy backoff schedule used by the turn loop
to hold the primary provider on long exponential backoff for rate-limit
429s instead of eagerly failing over.
"""
from unittest.mock import patch

import pytest

from agent.rate_limit_policy import (
    RateLimitRetryPolicy,
    policy_backoff_wait,
    resolve_rate_limit_retry_policy,
)


# ── resolve_rate_limit_retry_policy ─────────────────────────────────────


class TestResolvePolicy:
    def test_absent_section_returns_none(self):
        """No agent.rate_limit_retry → policy disabled (legacy behavior)."""
        assert resolve_rate_limit_retry_policy(None) is None

    def test_malformed_section_returns_none(self):
        assert resolve_rate_limit_retry_policy("yes") is None
        assert resolve_rate_limit_retry_policy([1, 2]) is None

    def test_zero_max_retries_returns_none(self):
        """max_retries: 0 is the explicit off switch."""
        assert resolve_rate_limit_retry_policy({"max_retries": 0}) is None

    def test_empty_section_returns_none(self):
        """Present but empty → default max_retries 0 → disabled."""
        assert resolve_rate_limit_retry_policy({}) is None

    def test_negative_enables_unlimited(self):
        policy = resolve_rate_limit_retry_policy({"max_retries": -1})
        assert policy is not None
        assert policy.max_retries == -1
        assert policy.enabled

    def test_finite_budget_enabled(self):
        policy = resolve_rate_limit_retry_policy({"max_retries": 20})
        assert policy is not None
        assert policy.max_retries == 20

    def test_default_backoff_values(self):
        policy = resolve_rate_limit_retry_policy({"max_retries": 5})
        assert policy.backoff_base == 10.0
        assert policy.backoff_max == 600.0

    def test_custom_backoff_values(self):
        policy = resolve_rate_limit_retry_policy(
            {"max_retries": -1, "backoff_base": 15, "backoff_max": 300}
        )
        assert policy.backoff_base == 15.0
        assert policy.backoff_max == 300.0

    def test_garbage_max_retries_disables_policy(self):
        """Unparseable max_retries → default 0 → disabled (legacy path).

        Malformed values never crash init and never enable aggressive
        retrying by accident.
        """
        policy = resolve_rate_limit_retry_policy(
            {"max_retries": "forever", "backoff_base": "lots", "backoff_max": None}
        )
        assert policy is None

    def test_garbage_backoff_falls_back_to_defaults(self):
        policy = resolve_rate_limit_retry_policy(
            {"max_retries": 5, "backoff_base": "lots", "backoff_max": None}
        )
        assert policy is not None
        assert policy.backoff_base == 10.0
        assert policy.backoff_max == 600.0

    def test_backoff_max_clamped_to_base(self):
        """max < base would shrink the schedule; clamp to base."""
        policy = resolve_rate_limit_retry_policy(
            {"max_retries": 3, "backoff_base": 60, "backoff_max": 30}
        )
        assert policy.backoff_max == 60.0

    def test_extreme_values_clamped_to_sane_bounds(self):
        policy = resolve_rate_limit_retry_policy(
            {"max_retries": 10**9, "backoff_base": 10**6, "backoff_max": 10**7}
        )
        assert policy.max_retries == 100000
        assert policy.backoff_base == 3600.0
        assert policy.backoff_max == 86400.0


# ── policy_backoff_wait ──────────────────────────────────────────────────


class TestPolicyBackoffWait:
    def test_first_wait_near_base(self):
        policy = RateLimitRetryPolicy(max_retries=-1, backoff_base=10.0, backoff_max=600.0)
        wait = policy_backoff_wait(policy, 1)
        assert 9.0 <= wait <= 11.0  # ±10% jitter

    def test_exponential_growth(self):
        policy = RateLimitRetryPolicy(max_retries=-1, backoff_base=10.0, backoff_max=100000.0)
        wait_1 = policy_backoff_wait(policy, 1)
        wait_2 = policy_backoff_wait(policy, 2)
        wait_3 = policy_backoff_wait(policy, 3)
        # Jitter is ±10%; growth is 2x per step, far outside the band.
        assert wait_2 > wait_1 * 1.6
        assert wait_3 > wait_2 * 1.6

    def test_caps_at_backoff_max(self):
        policy = RateLimitRetryPolicy(max_retries=-1, backoff_base=10.0, backoff_max=300.0)
        for index in (6, 7, 20, 50):
            assert policy_backoff_wait(policy, index) <= 300.0

    def test_long_plateau_sits_at_cap(self):
        """After enough doublings, every wait is at (or a hair under) the cap."""
        policy = RateLimitRetryPolicy(max_retries=-1, backoff_base=10.0, backoff_max=600.0)
        for index in (10, 11, 12):
            assert policy_backoff_wait(policy, index) > 540.0


# ── agent_init wiring ────────────────────────────────────────────────────


def _make_agent(agent_section):
    from run_agent import AIAgent

    cfg = {"agent": dict(agent_section)}
    with patch("run_agent.OpenAI"), \
         patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("hermes_cli.config.load_config", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            base_url="https://api.z.ai/api/paas/v4",
            model="glm-5.3-flash",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


class TestAgentInitWiring:
    def test_no_policy_by_default(self):
        agent = _make_agent({})
        assert agent._rate_limit_retry_policy is None

    def test_policy_from_config(self):
        agent = _make_agent({"rate_limit_retry": {"max_retries": -1, "backoff_base": 15}})
        policy = agent._rate_limit_retry_policy
        assert policy is not None
        assert policy.max_retries == -1
        assert policy.backoff_base == 15.0

    def test_malformed_policy_does_not_break_init(self):
        agent = _make_agent({"rate_limit_retry": "unlimited please"})
        assert agent._rate_limit_retry_policy is None
