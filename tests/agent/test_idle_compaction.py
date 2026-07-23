"""Tests for the opt-in idle-triggered compaction policy.

Covers ``agent.turn_context._should_idle_compact`` — the pure predicate that
decides whether a session resuming after an idle gap should compact up front.
The predicate is intentionally side-effect-free so the policy can be verified
without constructing a live agent or DB.
"""

from agent.turn_context import _should_idle_compact


def _decide(**overrides):
    """Call the predicate with sensible defaults (idle + large context => fire)."""
    kwargs = dict(
        enabled=True,
        idle_after_seconds=1800,
        idle_gap_seconds=3600.0,
        tokens=100_000,
        floor_tokens=40_000,
        cooldown_active=False,
    )
    kwargs.update(overrides)
    return _should_idle_compact(**kwargs)


class TestShouldIdleCompact:
    def test_fires_when_idle_long_enough_and_context_large(self):
        assert _decide() is True

    def test_disabled_when_idle_after_zero(self):
        # 0 is the documented "off" value — must never fire regardless of gap.
        assert _decide(idle_after_seconds=0, idle_gap_seconds=10_000.0) is False

    def test_disabled_when_idle_after_negative(self):
        assert _decide(idle_after_seconds=-1) is False

    def test_disabled_when_compression_off(self):
        assert _decide(enabled=False) is False

    def test_skips_when_gap_below_threshold(self):
        assert _decide(idle_gap_seconds=600.0) is False

    def test_gap_exactly_at_threshold_fires(self):
        assert _decide(idle_after_seconds=1800, idle_gap_seconds=1800.0) is True

    def test_skips_when_context_at_or_below_floor(self):
        # At/below the post-compression target there is nothing worth saving.
        assert _decide(tokens=40_000, floor_tokens=40_000) is False
        assert _decide(tokens=39_999, floor_tokens=40_000) is False

    def test_fires_just_above_floor(self):
        assert _decide(tokens=40_001, floor_tokens=40_000) is True

    def test_defers_to_active_compression_cooldown(self):
        assert _decide(cooldown_active=True) is False
