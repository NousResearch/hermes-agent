"""Unit tests for ``safety_redline.redline``.

These tests use a fake monotonic clock so the cooldown behaviour can be
exercised deterministically. They never touch the network or any IO. Pure
unit tests, suitable for any Python 3.13+ environment with pytest 9.x.
"""

from __future__ import annotations

import pytest

from safety_redline import (
    REDLINE_VERSION,
    SafetyConfig,
    SafetyRedline,
    SafetyState,
)


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def test_initial_state_is_healthy():
    redline = SafetyRedline()
    assert redline.state is SafetyState.HEALTHY
    assert redline.snapshot()["version"] == REDLINE_VERSION
    assert redline.is_traffic_allowed() is True


def test_single_failure_keeps_healthy():
    clock = _Clock()
    redline = SafetyRedline()
    event = redline.record_failure(reason="api timeout", now=clock())
    assert event.state is SafetyState.HEALTHY
    assert redline.is_traffic_allowed()


def test_two_failures_warn_but_still_pass_traffic():
    clock = _Clock()
    redline = SafetyRedline()
    redline.record_failure(now=clock())
    redline.advance = clock.advance  # not used, kept for symmetry
    event = redline.record_failure(now=clock())
    assert event.state is SafetyState.WARN
    assert redline.is_traffic_allowed()


def test_three_failures_paused_and_blocks_traffic():
    clock = _Clock()
    redline = SafetyRedline()
    for _ in range(3):
        redline.record_failure(now=clock())
    assert redline.state is SafetyState.PAUSED
    # Test clock is at 1000.0; cooldown is 300s; traffic still blocked.
    assert not redline.is_traffic_allowed(now=clock())


def test_four_failures_hard_pause():
    clock = _Clock()
    redline = SafetyRedline()
    for _ in range(4):
        redline.record_failure(now=clock())
    assert redline.state is SafetyState.HARD_PAUSE
    assert not redline.is_traffic_allowed()


def test_hard_pause_persists_even_after_cooldown_until_reset():
    clock = _Clock()
    redline = SafetyRedline()
    for _ in range(4):
        redline.record_failure(now=clock())
    clock.advance(10_000.0)
    # Even with cooldown elapsed, HARD_PAUSE is sticky: only ``reset()``
    # returns the redline to HEALTHY.
    redline._maybe_clear_pause(clock())
    assert redline.state is SafetyState.HARD_PAUSE
    redline.reset()
    assert redline.state is SafetyState.HEALTHY


def test_success_resets_failure_streak_from_healthy_or_warn():
    clock = _Clock()
    redline = SafetyRedline()
    redline.record_failure(now=clock())
    redline.record_failure(now=clock())
    assert redline.state is SafetyState.WARN
    redline.record_success(now=clock())
    assert redline.state is SafetyState.HEALTHY
    assert redline._failure_streak == 0


def test_paused_clears_after_cooldown_once_event_observed():
    clock = _Clock()
    redline = SafetyRedline()
    for _ in range(3):
        redline.record_failure(now=clock())
    assert redline.state is SafetyState.PAUSED
    clock.advance(400.0)  # > default cooldown
    # Any event triggers the cooldown evaluation.
    redline.record_success(now=clock())
    assert redline.state is SafetyState.HEALTHY


def test_notifier_called_on_pause_only_once_until_reset():
    seen = []
    config = SafetyConfig(notifier=lambda level, msg, snap: seen.append((level, msg)))
    clock = _Clock()
    redline = SafetyRedline(config=config)
    for _ in range(3):
        redline.record_failure(now=clock())
    assert [level for level, _ in seen] == ["paused"]
    # A 4th failure escalates to hard_pause -- another notification.
    redline.record_failure(now=clock())
    assert [level for level, _ in seen] == ["paused", "hard_pause"]


def test_notifier_swallowed_on_error():
    def broken_notifier(level, msg, snap):
        raise RuntimeError("notifier offline")

    config = SafetyConfig(notifier=broken_notifier)
    clock = _Clock()
    redline = SafetyRedline(config=config)
    # Must not raise; notification failure must never disable safety.
    for _ in range(4):
        redline.record_failure(now=clock())
    assert redline.state is SafetyState.HARD_PAUSE


def test_history_bounded_to_64():
    clock = _Clock()
    redline = SafetyRedline()
    for i in range(200):
        redline.record_failure(now=clock() + i * 0.001)
    assert len(redline.history()) == 64


def test_snapshot_shape():
    redline = SafetyRedline()
    snap = redline.snapshot()
    assert set(snap) == {
        "state",
        "failure_streak",
        "last_failure_at",
        "pause_started_at",
        "version",
    }
    assert snap["state"] == "healthy"


def test_config_is_immutable():
    config = SafetyConfig()
    with pytest.raises(Exception):
        config.pause_threshold = 99  # frozen dataclass


def test_window_old_failures_counted_as_stale_only_after_threshold():
    """Older failures inside the streak window still escalate, but a single
    success resets the streak regardless of how recent the failures are."""
    clock = _Clock()
    redline = SafetyRedline()
    redline.record_failure(now=clock())
    clock.advance(60.0)
    redline.record_failure(now=clock())
    clock.advance(60.0)
    # Two recent failures: still WARN.
    assert redline.state is SafetyState.WARN
    # A success clears back to HEALTHY without crossing the cooldown.
    redline.record_success(now=clock())
    assert redline.state is SafetyState.HEALTHY
