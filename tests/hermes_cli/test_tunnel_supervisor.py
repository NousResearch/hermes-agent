import pytest
from hermes_cli.tunnel_supervisor import reset_idle_on, should_close_now


class TestPolicy:
    def test_reset_when_counter_increases(self):
        assert reset_idle_on(10, 11) is True

    def test_no_reset_when_counter_unchanged(self):
        assert reset_idle_on(10, 10) is False

    def test_no_reset_when_counter_decreases(self):
        # a poll hiccup / counter reset should NOT count as activity
        assert reset_idle_on(11, 10) is False

    def test_close_after_idle_timeout(self):
        state = {"now": 1000.0, "last_activity": 100.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": None}
        assert should_close_now(state) is True

    def test_open_before_idle_timeout(self):
        state = {"now": 1000.0, "last_activity": 999.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": None}
        assert should_close_now(state) is False

    def test_hold_active_keeps_open_past_idle(self):
        state = {"now": 1000.0, "last_activity": 0.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 2000.0}
        assert should_close_now(state) is False

    def test_hold_expired_falls_back_to_idle_not_hard_kill(self):
        # hold_until in the past: fall back to idle rule.
        # last_activity recent -> still open (no hard kill on approval expiry).
        state = {"now": 1000.0, "last_activity": 999.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 500.0}
        assert should_close_now(state) is False

    def test_hold_expired_and_idle_closes(self):
        state = {"now": 1000.0, "last_activity": 0.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 500.0}
        assert should_close_now(state) is True