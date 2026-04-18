"""Tests for agent.rate_limiter – per-model stepped cooldown."""

from __future__ import annotations

import threading
import time
from unittest import mock

import pytest

from agent.rate_limiter import RateLimiter, _COOLDOWN_STEPS, _RESET_WINDOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_limiter(steps: tuple[int, ...] = _COOLDOWN_STEPS, reset_window: float = _RESET_WINDOW) -> RateLimiter:
    """Create a fresh RateLimiter (not the module-level singleton)."""
    return RateLimiter(cooldown_steps=steps, reset_window=reset_window)


# ---------------------------------------------------------------------------
# Test stepped cooldown escalation
# ---------------------------------------------------------------------------

class TestSteppedCooldown:
    """The cooldown should escalate through the ladder: 30s → 60s → 300s."""

    def test_first_hit_returns_30s(self) -> None:
        rl = _make_limiter()
        assert rl.record_rate_limit("gpt-4") == 30

    def test_second_hit_returns_60s(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("gpt-4")
        assert rl.record_rate_limit("gpt-4") == 60

    def test_third_hit_returns_300s(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("gpt-4")
        rl.record_rate_limit("gpt-4")
        assert rl.record_rate_limit("gpt-4") == 300

    def test_fourth_hit_stays_at_max(self) -> None:
        rl = _make_limiter()
        for _ in range(3):
            rl.record_rate_limit("gpt-4")
        # 4th hit should stay clamped at step 3 (300s)
        assert rl.record_rate_limit("gpt-4") == 300

    def test_step_number_increments(self) -> None:
        rl = _make_limiter()
        assert rl.get_step("gpt-4") == 0
        rl.record_rate_limit("gpt-4")
        assert rl.get_step("gpt-4") == 1
        rl.record_rate_limit("gpt-4")
        assert rl.get_step("gpt-4") == 2
        rl.record_rate_limit("gpt-4")
        assert rl.get_step("gpt-4") == 3
        # Stays clamped
        rl.record_rate_limit("gpt-4")
        assert rl.get_step("gpt-4") == 3

    def test_custom_steps(self) -> None:
        rl = _make_limiter(steps=(5, 10))
        assert rl.record_rate_limit("m") == 5
        assert rl.record_rate_limit("m") == 10
        assert rl.record_rate_limit("m") == 10  # clamped


# ---------------------------------------------------------------------------
# Test cooldown reset after no hits
# ---------------------------------------------------------------------------

class TestCooldownReset:
    """Step counter should reset after reset_window seconds of no hits."""

    def test_reset_after_window(self) -> None:
        rl = _make_limiter(reset_window=10.0)

        # Bump to step 2
        rl.record_rate_limit("gpt-4")
        rl.record_rate_limit("gpt-4")
        assert rl.get_step("gpt-4") == 2

        # Simulate 10+ seconds passing by manipulating last_hit
        with rl._lock:
            state = rl._models["gpt-4"]
            state.last_hit = time.monotonic() - 11.0
            state.cooldown_until = 0  # clear active cooldown too

        # Next recording should start from step 1 again (reset happened)
        assert rl.record_rate_limit("gpt-4") == 30
        assert rl.get_step("gpt-4") == 1

    def test_no_reset_within_window(self) -> None:
        rl = _make_limiter(reset_window=600.0)

        rl.record_rate_limit("gpt-4")
        rl.record_rate_limit("gpt-4")
        assert rl.get_step("gpt-4") == 2

        # No time manipulation → still within window
        assert rl.record_rate_limit("gpt-4") == 300
        assert rl.get_step("gpt-4") == 3

    def test_get_step_resets_when_window_elapsed(self) -> None:
        rl = _make_limiter(reset_window=5.0)
        rl.record_rate_limit("x")
        assert rl.get_step("x") == 1

        with rl._lock:
            rl._models["x"].last_hit = time.monotonic() - 6.0
        assert rl.get_step("x") == 0


# ---------------------------------------------------------------------------
# Test per-model isolation
# ---------------------------------------------------------------------------

class TestPerModelIsolation:
    """Each model should have its own independent cooldown state."""

    def test_different_models_are_independent(self) -> None:
        rl = _make_limiter()

        rl.record_rate_limit("gpt-4")
        rl.record_rate_limit("gpt-4")

        # Claude has not been hit yet → should start at step 1
        assert rl.record_rate_limit("claude-3") == 30
        assert rl.get_step("claude-3") == 1

        # GPT-4 should still be at step 2 (plus the third hit now)
        assert rl.record_rate_limit("gpt-4") == 300
        assert rl.get_step("gpt-4") == 3

    def test_reset_single_model(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("a")
        rl.record_rate_limit("b")

        rl.reset("a")
        assert rl.get_step("a") == 0
        assert rl.get_step("b") == 1

    def test_reset_all(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("a")
        rl.record_rate_limit("b")
        rl.reset()
        assert rl.get_step("a") == 0
        assert rl.get_step("b") == 0


# ---------------------------------------------------------------------------
# Test check_rate_limit returns correct remaining time
# ---------------------------------------------------------------------------

class TestCheckRateLimit:
    """check_rate_limit should return remaining cooldown or 0."""

    def test_no_cooldown_initially(self) -> None:
        rl = _make_limiter()
        assert rl.check_rate_limit("gpt-4") == 0.0

    def test_remaining_time_after_hit(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("gpt-4")  # 30s cooldown

        remaining = rl.check_rate_limit("gpt-4")
        # Should be very close to 30 (within a small tolerance)
        assert 28.0 < remaining <= 30.0

    def test_remaining_decreases_over_time(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("gpt-4")

        # Simulate 10 seconds passing by adjusting cooldown_until
        with rl._lock:
            rl._models["gpt-4"].cooldown_until = time.monotonic() + 20.0

        remaining = rl.check_rate_limit("gpt-4")
        assert 18.0 < remaining <= 20.0

    def test_returns_zero_after_cooldown_expires(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("gpt-4")

        # Expire the cooldown
        with rl._lock:
            rl._models["gpt-4"].cooldown_until = time.monotonic() - 1.0

        assert rl.check_rate_limit("gpt-4") == 0.0


# ---------------------------------------------------------------------------
# Test get_cooldown_status
# ---------------------------------------------------------------------------

class TestGetCooldownStatus:
    """get_cooldown_status should report all models with active cooldowns."""

    def test_empty_when_no_hits(self) -> None:
        rl = _make_limiter()
        assert rl.get_cooldown_status() == {}

    def test_shows_active_cooldowns(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("gpt-4")
        rl.record_rate_limit("claude-3")

        status = rl.get_cooldown_status()
        assert "gpt-4" in status
        assert "claude-3" in status
        assert status["gpt-4"]["step"] == 1
        assert status["gpt-4"]["remaining"] > 0

    def test_omits_expired_cooldowns(self) -> None:
        rl = _make_limiter()
        rl.record_rate_limit("old")
        rl.record_rate_limit("new")

        # Expire "old"
        with rl._lock:
            rl._models["old"].cooldown_until = time.monotonic() - 1.0

        status = rl.get_cooldown_status()
        assert "old" not in status
        assert "new" in status


# ---------------------------------------------------------------------------
# Test thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Concurrent access should not corrupt state."""

    def test_concurrent_record(self) -> None:
        rl = _make_limiter()
        errors: list[Exception] = []

        def _hit(model: str, n: int) -> None:
            try:
                for _ in range(n):
                    rl.record_rate_limit(model)
            except Exception as exc:
                errors.append(exc)

        threads = []
        for i in range(10):
            t = threading.Thread(target=_hit, args=(f"model-{i % 3}", 50))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"

        # Each of the 3 models should have a valid step (clamped to max)
        for i in range(3):
            step = rl.get_step(f"model-{i}")
            assert 1 <= step <= len(rl._cooldown_steps)

    def test_concurrent_check_and_record(self) -> None:
        rl = _make_limiter()
        errors: list[Exception] = []

        def _checker(model: str) -> None:
            try:
                for _ in range(100):
                    remaining = rl.check_rate_limit(model)
                    assert remaining >= 0
            except Exception as exc:
                errors.append(exc)

        def _recorder(model: str) -> None:
            try:
                for _ in range(50):
                    cd = rl.record_rate_limit(model)
                    assert cd > 0
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_checker, args=("m",)),
            threading.Thread(target=_checker, args=("m",)),
            threading.Thread(target=_recorder, args=("m",)),
            threading.Thread(target=_recorder, args=("m",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# Test module-level singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    """The module-level ``rate_limiter`` should be usable directly."""

    def test_singleton_import(self) -> None:
        from agent.rate_limiter import rate_limiter
        assert isinstance(rate_limiter, RateLimiter)

    def test_singleton_records(self) -> None:
        from agent.rate_limiter import rate_limiter
        # Reset to avoid pollution from other tests
        rate_limiter.reset()
        cd = rate_limiter.record_rate_limit("test-singleton-model")
        assert cd == 30
        rate_limiter.reset("test-singleton-model")
