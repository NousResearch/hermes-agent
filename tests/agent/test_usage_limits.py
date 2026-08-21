"""Wall-clock / token usage belts (agent/usage_limits.py)."""

from agent.usage_limits import TurnUsageTracker, UsageLimitsConfig


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now


class TestConfig:
    def test_disabled_by_default(self):
        cfg = UsageLimitsConfig.from_mapping({})
        assert not cfg.enabled
        assert UsageLimitsConfig.from_mapping(None).enabled is False

    def test_junk_values_disable_instead_of_crashing(self):
        cfg = UsageLimitsConfig.from_mapping(
            {
                "turn_wall_clock_seconds": "abc",
                "turn_total_tokens": -5,
                "session_total_tokens": None,
            }
        )
        assert not cfg.enabled

    def test_nested_section_parses(self):
        cfg = UsageLimitsConfig.from_mapping(
            {"turn_wall_clock_seconds": 1800, "session_total_tokens": "200000"}
        )
        assert cfg.enabled
        assert cfg.turn_wall_clock_seconds == 1800
        assert cfg.session_total_tokens == 200000


class TestTracker:
    def test_disabled_never_breaches(self):
        clock = FakeClock()
        t = TurnUsageTracker(UsageLimitsConfig(), 0, clock=clock)
        clock.now += 10**9
        assert t.check(10**12) is None

    def test_wall_clock_breach(self):
        clock = FakeClock()
        t = TurnUsageTracker(
            UsageLimitsConfig(turn_wall_clock_seconds=1800), 0, clock=clock
        )
        clock.now += 1799
        assert t.check(0) is None
        clock.now += 1
        breach = t.check(0)
        assert breach is not None
        assert breach.code == "usage_limit_wall_clock"

    def test_turn_tokens_measured_from_watermark(self):
        t = TurnUsageTracker(
            UsageLimitsConfig(turn_total_tokens=1000),
            50_000,
            clock=FakeClock(),
        )
        assert t.check(50_999) is None
        breach = t.check(51_000)
        assert breach is not None
        assert breach.code == "usage_limit_turn_tokens"

    def test_session_tokens_absolute(self):
        t = TurnUsageTracker(
            UsageLimitsConfig(session_total_tokens=200_000),
            199_500,
            clock=FakeClock(),
        )
        assert t.check(199_999) is None
        breach = t.check(200_000)
        assert breach is not None
        assert breach.code == "usage_limit_session_tokens"

    def test_wall_clock_wins_over_tokens(self):
        clock = FakeClock()
        t = TurnUsageTracker(
            UsageLimitsConfig(turn_wall_clock_seconds=60, turn_total_tokens=10),
            0,
            clock=clock,
        )
        clock.now += 61
        assert t.check(10**6).code == "usage_limit_wall_clock"

    def test_none_config_is_disabled(self):
        t = TurnUsageTracker(None, 0, clock=FakeClock())
        assert t.check(10**9) is None
