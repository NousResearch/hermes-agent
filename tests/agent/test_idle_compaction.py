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

    def test_disabled_when_idle_after_zero(self):
        # 0 is the documented "off" value — must never fire regardless of gap.
        assert _decide(idle_after_seconds=0, idle_gap_seconds=10_000.0) is False


    def test_disabled_when_compression_off(self):
        assert _decide(enabled=False) is False




    def test_fires_just_above_floor(self):
        assert _decide(tokens=40_001, floor_tokens=40_000) is True

