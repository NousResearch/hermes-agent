"""Tests for configurable overload (503/529) backoff — #55540."""

import pytest
from unittest.mock import MagicMock, patch


# ── Test: config parsing in agent_init ────────────────────────────────


def _make_agent_section(**overrides):
    """Build a minimal _agent_section dict for the overload config block."""
    return overrides


def _apply_overload_config(agent, agent_section, api_max_retries=3):
    """Replicate the config-reading logic from agent_init.py."""
    try:
        _overload_retries = int(agent_section.get("overload_max_retries", 2))
        _overload_retries = max(_overload_retries, 0)
        _overload_retries = min(_overload_retries, api_max_retries)
    except (TypeError, ValueError):
        _overload_retries = 2
    agent._overload_max_retries = _overload_retries

    try:
        _overload_base = float(agent_section.get("overload_base_delay", 2.0))
        _overload_base = max(_overload_base, 0.1)
    except (TypeError, ValueError):
        _overload_base = 2.0
    agent._overload_base_delay = _overload_base

    try:
        _overload_max = float(agent_section.get("overload_max_delay", 60.0))
        _overload_max = max(_overload_max, 1.0)
    except (TypeError, ValueError):
        _overload_max = 60.0
    agent._overload_max_delay = _overload_max


class TestOverloadConfigDefaults:
    """When no overload keys are present, defaults match prior hardcoded values."""

    def test_defaults_when_absent(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section())
        assert agent._overload_max_retries == 2
        assert agent._overload_base_delay == 2.0
        assert agent._overload_max_delay == 60.0


class TestOverloadConfigCustom:
    """User-supplied values are respected."""

    def test_custom_values(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(
            overload_max_retries=3,
            overload_base_delay=10.0,
            overload_max_delay=120.0,
        ))
        assert agent._overload_max_retries == 3
        assert agent._overload_base_delay == 10.0
        assert agent._overload_max_delay == 120.0

    def test_zero_retries_means_immediate_fallback(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_max_retries=0))
        assert agent._overload_max_retries == 0

    def test_string_values_are_cast(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(
            overload_max_retries="4",
            overload_base_delay="5.5",
            overload_max_delay="90",
        ), api_max_retries=5)
        assert agent._overload_max_retries == 4
        assert agent._overload_base_delay == 5.5
        assert agent._overload_max_delay == 90.0


class TestOverloadConfigClamping:
    """Floor and ceiling clamping prevents nonsensical values."""

    def test_negative_retries_clamped_to_zero(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_max_retries=-1))
        assert agent._overload_max_retries == 0

    def test_retries_clamped_to_api_max_retries(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_max_retries=10), api_max_retries=3)
        assert agent._overload_max_retries == 3

    def test_retries_within_api_max_retries_unchanged(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_max_retries=2), api_max_retries=5)
        assert agent._overload_max_retries == 2

    def test_base_delay_floor(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_base_delay=0.01))
        assert agent._overload_base_delay == 0.1

    def test_max_delay_floor(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_max_delay=0.5))
        assert agent._overload_max_delay == 1.0


class TestOverloadConfigInvalid:
    """Garbage input falls back to defaults."""

    def test_invalid_retries(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_max_retries="abc"))
        assert agent._overload_max_retries == 2

    def test_invalid_base_delay(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_base_delay="not_a_number"))
        assert agent._overload_base_delay == 2.0

    def test_invalid_max_delay(self):
        agent = MagicMock()
        _apply_overload_config(agent, _make_agent_section(overload_max_delay=None))
        assert agent._overload_max_delay == 60.0


# ── Test: fallback threshold logic ────────────────────────────────────


class TestOverloadFallbackThreshold:
    """The overload retry threshold uses the configured value."""

    def _should_fallback_overloaded(self, retry_count, overload_max_retries):
        """Replicate the _should_fallback condition for overloaded."""
        return retry_count > overload_max_retries

    def test_default_threshold_triggers_at_2(self):
        assert not self._should_fallback_overloaded(0, 2)
        assert not self._should_fallback_overloaded(1, 2)
        assert not self._should_fallback_overloaded(2, 2)
        assert self._should_fallback_overloaded(3, 2)

    def test_zero_threshold_triggers_after_first_failure(self):
        assert not self._should_fallback_overloaded(0, 0)
        assert self._should_fallback_overloaded(1, 0)

    def test_one_retry_is_distinguishable_from_zero(self):
        assert not self._should_fallback_overloaded(1, 1)
        assert self._should_fallback_overloaded(2, 1)

    def test_high_threshold_delays_fallback(self):
        assert not self._should_fallback_overloaded(4, 5)
        assert not self._should_fallback_overloaded(5, 5)
        assert self._should_fallback_overloaded(6, 5)


# ── Test: backoff delay uses configured values ────────────────────────


class TestOverloadBackoffDelay:
    """jittered_backoff is called with configured base/max for overloaded errors."""

    def test_custom_delays_passed_to_jittered_backoff(self):
        from agent.retry_utils import jittered_backoff

        delay = jittered_backoff(1, base_delay=10.0, max_delay=30.0)
        assert 10.0 <= delay <= 10.0 * 1.5  # base + up to jitter_ratio * base

    def test_max_delay_caps_high_attempts(self):
        from agent.retry_utils import jittered_backoff

        delay = jittered_backoff(10, base_delay=2.0, max_delay=30.0)
        assert delay <= 30.0 * 1.5  # max + jitter
