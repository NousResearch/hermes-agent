"""Regression tests for cooldown copy in issue #140737."""

from agent.turn_failure_copy import _NEXT_STEPS_RETRY, exhausted_copy


def _copy(seconds=None, *, attempts=3):
    return exhausted_copy(
        reason="rate_limit", label="Acme", attempts=attempts,
        summary="429: slow down", reset_seconds=seconds,
    )


def test_known_one_to_two_minute_cooldown_does_not_say_wait_a_minute():
    for seconds in (60, 61, 90, 119.9):
        out = _copy(seconds)
        assert "its limit resets soon" in out
        assert "reset time below" in out
        assert _NEXT_STEPS_RETRY not in out
        assert "switch models with /model" in out


def test_cooldowns_below_one_minute_keep_existing_copy():
    for seconds in (0, 1, 59.9):
        assert _copy(seconds) == _copy(None)


def test_two_minutes_and_above_keep_existing_named_window():
    for seconds in (120, 121, 300):
        out = _copy(seconds)
        assert "its usage limit resets in" in out
        assert "Send /retry after that" in out
        assert "reset time below" not in out


def test_first_attempt_copy_keeps_existing_named_window():
    out = _copy(61, attempts=1)
    assert "its usage limit resets in" in out
    assert "Send /retry after that" in out
    assert "reset time below" not in out
