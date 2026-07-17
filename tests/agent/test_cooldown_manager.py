"""Tests for cooldown tracking."""

from __future__ import annotations

import threading
import time

import pytest

from agent.cooldown_manager import CooldownManager, get_cooldown_manager, set_cooldown_manager


def _make_mgr(**kwargs) -> CooldownManager:
    defaults = dict(
        base_seconds=10.0,
        multiplier=2.0,
        max_seconds=100.0,
        billing_base_hours=1.0,
        billing_max_hours=8.0,
    )
    defaults.update(kwargs)
    return CooldownManager(**defaults)


class TestRateLimitCooldown:
    def test_fresh_manager_not_cooling(self):
        mgr = _make_mgr()
        assert mgr.is_cooling("openrouter") is False

    def test_mark_failure_returns_positive_seconds(self):
        mgr = _make_mgr(base_seconds=30.0, multiplier=2.0)
        secs = mgr.mark_failure("openrouter", "rate_limit")
        assert secs > 0

    def test_is_cooling_true_immediately_after_mark(self):
        mgr = _make_mgr(base_seconds=60.0, multiplier=2.0)
        mgr.mark_failure("openrouter", "rate_limit")
        assert mgr.is_cooling("openrouter") is True

    def test_is_cooling_false_after_expiry(self, monkeypatch):
        """Mock time.monotonic so the clock jumps past the cooldown."""
        mgr = _make_mgr(base_seconds=10.0, multiplier=2.0)

        base_time = time.monotonic()
        call_count = [0]

        def fake_monotonic():
            call_count[0] += 1
            if call_count[0] <= 1:
                return base_time
            return base_time + 15.0  # past the 10s cooldown

        monkeypatch.setattr("agent.cooldown_manager.time.monotonic", fake_monotonic)

        mgr.mark_failure("openrouter", "rate_limit")
        assert mgr.is_cooling("openrouter") is False

    def test_first_failure_uses_base_seconds(self):
        mgr = _make_mgr(base_seconds=30.0, multiplier=5.0, max_seconds=9999.0)
        secs = mgr.mark_failure("provider_a", "rate_limit")
        # count=1: 30 * 5^0 = 30
        assert secs == pytest.approx(30.0, abs=0.01)

    def test_second_failure_multiplied(self):
        mgr = _make_mgr(base_seconds=30.0, multiplier=5.0, max_seconds=9999.0)
        mgr.mark_failure("provider_a", "rate_limit")           # count=1 → 30s
        secs2 = mgr.mark_failure("provider_a", "rate_limit")  # count=2 → 150s
        assert secs2 == pytest.approx(150.0, abs=0.01)

    def test_rate_limit_capped_at_max_seconds(self):
        mgr = _make_mgr(base_seconds=60.0, multiplier=5.0, max_seconds=200.0)
        mgr.mark_failure("p", "rate_limit")  # 60s
        mgr.mark_failure("p", "rate_limit")  # 300s → capped to 200s
        secs = mgr.mark_failure("p", "rate_limit")
        assert secs == pytest.approx(200.0, abs=0.01)


class TestBillingCooldown:
    def test_billing_marks_is_cooling(self):
        mgr = _make_mgr(billing_base_hours=1.0, billing_max_hours=8.0)
        mgr.mark_failure("anthropic", "billing")
        assert mgr.is_cooling("anthropic") is True

    def test_billing_first_failure_base_hours(self):
        mgr = _make_mgr(billing_base_hours=2.0, billing_max_hours=48.0)
        secs = mgr.mark_failure("anthropic", "billing")
        # count=1: 2h * 2^0 = 2h = 7200s
        assert secs == pytest.approx(7200.0, abs=1.0)

    def test_billing_second_failure_doubles(self):
        mgr = _make_mgr(billing_base_hours=2.0, billing_max_hours=48.0)
        mgr.mark_failure("anthropic", "billing")            # 2h
        secs = mgr.mark_failure("anthropic", "billing")    # 4h
        assert secs == pytest.approx(4 * 3600.0, abs=1.0)

    def test_billing_capped_at_max_hours(self):
        mgr = _make_mgr(billing_base_hours=5.0, billing_max_hours=24.0)
        # 5h, 10h, 20h, 40h->capped at 24h
        secs = 0.0
        for _ in range(4):
            secs = mgr.mark_failure("billing_provider", "billing")
        assert secs == pytest.approx(24 * 3600.0, abs=1.0)

    def test_billing_longer_than_rate_limit(self):
        mgr = _make_mgr(
            base_seconds=60.0, multiplier=5.0, max_seconds=3600.0,
            billing_base_hours=5.0, billing_max_hours=24.0,
        )
        rate_secs = mgr.mark_failure("p1", "rate_limit")
        billing_secs = mgr.mark_failure("p2", "billing")
        # First billing failure (5h=18000s) >> first rate_limit (60s)
        assert billing_secs > rate_secs


class TestExponentialProgression:
    def test_three_rate_limit_failures_increase(self):
        mgr = _make_mgr(base_seconds=10.0, multiplier=3.0, max_seconds=9999.0)
        s1 = mgr.mark_failure("p", "rate_limit")   # 10s
        s2 = mgr.mark_failure("p", "rate_limit")   # 30s
        s3 = mgr.mark_failure("p", "rate_limit")   # 90s
        assert s1 < s2
        assert s2 < s3

    def test_progression_matches_formula(self):
        """Verify exact formula: base * multiplier^(count-1)."""
        base, mult = 60.0, 5.0
        mgr = _make_mgr(base_seconds=base, multiplier=mult, max_seconds=9999.0)
        for i in range(1, 5):
            secs = mgr.mark_failure("p", "rate_limit")
            expected = base * (mult ** (i - 1))
            assert secs == pytest.approx(expected, abs=0.01), \
                f"count={i}: got {secs}, expected {expected}"

    def test_billing_progression_doubles_each_time(self):
        mgr = _make_mgr(billing_base_hours=1.0, billing_max_hours=9999.0)
        prev = None
        for _ in range(4):
            secs = mgr.mark_failure("b", "billing")
            if prev is not None:
                assert abs(secs / prev - 2.0) < 0.01
            prev = secs


class TestClear:
    def test_clear_removes_cooling_state(self):
        mgr = _make_mgr(base_seconds=3600.0)
        mgr.mark_failure("openai", "rate_limit")
        assert mgr.is_cooling("openai") is True
        mgr.clear("openai")
        assert mgr.is_cooling("openai") is False

    def test_clear_resets_count(self):
        """After clear, the next failure should start from count=1 again."""
        mgr = _make_mgr(base_seconds=10.0, multiplier=2.0, max_seconds=9999.0)
        mgr.mark_failure("p", "rate_limit")
        mgr.mark_failure("p", "rate_limit")  # count=2 → 20s
        mgr.clear("p")
        secs = mgr.mark_failure("p", "rate_limit")  # count resets to 1 → 10s
        assert secs == pytest.approx(10.0, abs=0.01)

    def test_clear_nonexistent_key_is_noop(self):
        mgr = _make_mgr()
        mgr.clear("ghost_provider")  # should not raise

    def test_clear_only_affects_target_key(self):
        mgr = _make_mgr(base_seconds=3600.0)
        mgr.mark_failure("a", "rate_limit")
        mgr.mark_failure("b", "rate_limit")
        mgr.clear("a")
        assert mgr.is_cooling("a") is False
        assert mgr.is_cooling("b") is True


class TestStateInspection:
    def test_get_all_states_empty(self):
        mgr = _make_mgr()
        assert mgr.get_all_states() == {}

    def test_get_all_states_reflects_cooling(self):
        mgr = _make_mgr(base_seconds=3600.0)
        mgr.mark_failure("openrouter", "rate_limit")
        states = mgr.get_all_states()
        assert "openrouter" in states
        info = states["openrouter"]
        assert info["count"] == 1
        assert info["cooling"] is True
        assert info["remaining_seconds"] > 0

    def test_get_cooldown_status_structure(self):
        mgr = _make_mgr(base_seconds=3600.0)
        mgr.mark_failure("x", "rate_limit")
        status = mgr.get_cooldown_status()
        assert "total_tracked" in status
        assert "cooling" in status
        assert "expired" in status
        assert "details" in status
        assert status["total_tracked"] == 1
        assert "x" in status["cooling"]


class TestThreadSafety:
    def test_concurrent_mark_failure_no_corruption(self):
        """Multiple threads hammering mark_failure concurrently must not corrupt state."""
        mgr = _make_mgr(base_seconds=1.0, multiplier=2.0, max_seconds=9999.0)
        errors: list[Exception] = []
        results: list[float] = []
        lock = threading.Lock()

        def worker(key: str):
            try:
                for _ in range(20):
                    secs = mgr.mark_failure(key, "rate_limit")
                    with lock:
                        results.append(secs)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"provider_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised: {errors}"
        assert all(s > 0 for s in results)

    def test_concurrent_mark_and_clear(self):
        """Concurrent mark + clear on the same key must not raise."""
        mgr = _make_mgr(base_seconds=0.01, multiplier=2.0, max_seconds=9999.0)
        errors: list[Exception] = []
        stop = threading.Event()

        def marker():
            for _ in range(50):
                try:
                    mgr.mark_failure("shared", "rate_limit")
                except Exception as e:
                    errors.append(e)

        def clearer():
            for _ in range(50):
                try:
                    mgr.clear("shared")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=marker),
            threading.Thread(target=clearer),
            threading.Thread(target=marker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised: {errors}"

    def test_concurrent_is_cooling_reads(self):
        """Read-only is_cooling from many threads is safe."""
        mgr = _make_mgr(base_seconds=3600.0)
        mgr.mark_failure("p", "rate_limit")
        errors: list[Exception] = []

        def reader():
            try:
                for _ in range(100):
                    mgr.is_cooling("p")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


class TestSingleton:
    def test_get_cooldown_manager_returns_same_instance(self):
        m1 = get_cooldown_manager()
        m2 = get_cooldown_manager()
        assert m1 is m2

    def test_set_cooldown_manager_replaces_singleton(self):
        original = get_cooldown_manager()
        replacement = CooldownManager()
        try:
            set_cooldown_manager(replacement)
            assert get_cooldown_manager() is replacement
        finally:
            set_cooldown_manager(original)

    def test_singleton_is_cooldown_manager_instance(self):
        mgr = get_cooldown_manager()
        assert isinstance(mgr, CooldownManager)


class TestCooldownKeySafety:
    def test_distinct_keys_with_a_shared_prefix_have_distinct_fingerprints(self):
        from agent.cooldown_manager import build_cooldown_key

        first = build_cooldown_key("openai", "sk-shared-abcdefgh", "rate_limit")
        second = build_cooldown_key("openai", "sk-shared-ijklmnop", "rate_limit")

        assert first != second
        assert "sk-shared" not in first
        assert "sk-shared" not in second

    def test_fixed_duration_override_does_not_advance_rate_limit_backoff(self):
        manager = _make_mgr(base_seconds=60.0, multiplier=5.0, max_seconds=999.0)

        assert manager.mark_failure("openai:key", "rate_limit", cooldown_seconds=5.0) == 5.0
        assert manager.mark_failure("openai:key", "rate_limit") == 60.0

    def test_load_discards_legacy_raw_key_prefixes(self, tmp_path):
        import json

        storage_path = tmp_path / "cooldowns.json"
        storage_path.write_text(
            json.dumps(
                {
                    "openai:sk-shared": {
                        "count": 1,
                        "reason": "rate_limit",
                        "until_wall": time.time() + 60.0,
                    }
                }
            ),
            encoding="utf-8",
        )

        manager = _make_mgr(storage_path=storage_path)

        assert manager.get_all_states() == {}
        assert json.loads(storage_path.read_text(encoding="utf-8")) == {}
