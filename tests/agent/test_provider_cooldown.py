"""Tests for the opt-in lenient key-cooldown policy (agent/provider_cooldown).

Covers the user-facing rules:
1. strict is the DEFAULT and inert (classify/lenient never fires).
2. lenient: a single failure only PARKS a key (rotate away, retry soon).
3. lenient: crossing the window threshold BENCHES, with a growing ladder.
4. Billing ladder grows x3, x1.5, x1.25 ... saturating at x1.01 (uncapped).
5. A success clears the rolling state (ladder restarts).
6. Non-rolled classes (auth/401) fall through to the upstream TTL.
7. Hostile config values never crash and fall back to safe defaults.
"""
import pytest

from agent import provider_cooldown as pc


@pytest.fixture(autouse=True)
def _reset_cache():
    pc._reset_settings_cache_for_tests()
    yield
    pc._reset_settings_cache_for_tests()


def _set_config(monkeypatch, tmp_path, block):
    import yaml
    from agent import provider_cooldown as pco

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"provider_cooldown": block}), encoding="utf-8"
    )
    pc._reset_settings_cache_for_tests()


class TestStrictDefault:
    def test_default_mode_is_strict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert pc.get_settings(force_reload=True)["mode"] == pc.MODE_STRICT
        assert pc.is_lenient() is False

    def test_strict_ignores_rolling(self):
        # Even if called, classify still maps the class, but the CALLER (pool)
        # only consults this when is_lenient(). We assert the guard exists.
        assert pc.is_lenient() is False

    def test_classify_maps_429_and_402(self):
        assert pc.classify(429, "rate_limit") == pc.CLASS_RATE_LIMIT
        assert pc.classify(402, "billing") == pc.CLASS_BILLING
        assert pc.classify(403, "billing") == pc.CLASS_BILLING
        # Not rolled -> upstream TTL stays in charge.
        assert pc.classify(401, "auth") is None
        assert pc.classify(500, None) is None
        assert pc.classify(None, None) is None


class TestLenientParking:
    def _enable(self, tmp_path, monkeypatch, **overrides):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        rl = {"window_seconds": 1800, "fail_threshold": 5, "park_seconds": 30,
              "base_cooldown_seconds": 300, "backoff_multipliers": [1.0]}
        rl.update(overrides.get("rate_limit", {}))
        _set_config(monkeypatch, tmp_path, {"mode": "lenient", "rate_limit": rl})

    def test_single_failure_parks_not_benches(self, tmp_path, monkeypatch):
        self._enable(tmp_path, monkeypatch)
        extra = {}
        now = 1000.0
        ttl, benched = pc.record_lenient_failure(extra, status_code=429, now=now)
        assert benched is False           # only a park
        assert ttl == 30.0
        # A mere park neither starts a blackout nor consumes a ladder step.
        state = pc._read_state(extra)
        assert state["blackout_until"] == 0.0
        assert state["step"] == 0

    def test_threshold_benches(self, tmp_path, monkeypatch):
        self._enable(tmp_path, monkeypatch)
        extra = {}
        now = 5000.0
        # 4 failures park; the 5th crosses the threshold -> bench.
        for i in range(4):
            ttl, benched = pc.record_lenient_failure(extra, status_code=429, now=now + i)
            assert benched is False
        ttl, benched = pc.record_lenient_failure(extra, status_code=429, now=now + 4)
        assert benched is True
        assert ttl == 300.0  # flat ladder for 429


class TestBillingLadder:
    def _enable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {
            "mode": "lenient",
            "billing": {"window_seconds": 3600, "fail_threshold": 1,
                        "park_seconds": 30, "base_cooldown_seconds": 300,
                        "backoff_multipliers": [3.0, 1.5, 1.25, 1.25, 1.2, 1.15, 1.11, 1.1, 1.01],
                        # The blackout IS the growable value, so the ladder is
                        # observable directly (no separate probe cap anymore).
                        "probe_requests": 1},
        })

    def test_ladder_grows_per_consecutive_trigger(self, tmp_path, monkeypatch):
        self._enable(tmp_path, monkeypatch)
        extra = {}
        now = 0.0
        # threshold=1 -> every 402 benches; each bench grows the ladder.
        expected = [900.0, 1350.0, 1687.5, 2109.375]
        for i, exp in enumerate(expected):
            ttl, benched = pc.record_lenient_failure(extra, status_code=402, now=now + i)
            assert benched is True
            assert ttl == pytest.approx(exp, rel=1e-6)

    def test_ladder_saturates_to_slow_growth_no_cap(self, tmp_path, monkeypatch):
        self._enable(tmp_path, monkeypatch)
        extra = {}
        now = 0.0
        last = None
        for i in range(60):
            ttl, _ = pc.record_lenient_failure(extra, status_code=402, now=now + i)
            last = ttl
        # After saturation the growth per step is x1.01 — slow, but uncapped.
        ttl_next, _ = pc.record_lenient_failure(extra, status_code=402, now=now + 100)
        assert ttl_next == pytest.approx(last * 1.01, rel=1e-6)
        assert ttl_next > 3600  # well past an hour, no ceiling

    def test_probe_requires_x_selections_after_blackout(self, tmp_path, monkeypatch):
        """After the bench elapses the key is still held for X selections, then
        a single probe is allowed. The blackout itself is uncapped."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {
            "mode": "lenient",
            "billing": {"window_seconds": 3600, "fail_threshold": 1,
                        "park_seconds": 30, "base_cooldown_seconds": 300,
                        "backoff_multipliers": [3.0],
                        "probe_requests": 2},
        })
        extra = {}
        now = 0.0
        ttl, benched = pc.record_lenient_failure(extra, status_code=402, now=now)
        assert benched is True
        assert ttl == 900.0  # blackout: base 300 x first ladder factor 3
        # Inside the blackout: no number of selections opens the gate.
        assert pc.probe_after_blackout(extra, now=now + 100) is False
        # Blackout over: the 1st selection is consumed but not released...
        after = now + 900 + 1
        assert pc.probe_after_blackout(extra, now=after) is False
        # ...the 2nd selection (== X) opens the gate.
        assert pc.probe_after_blackout(extra, now=after) is True
        # A held probe does not consume a ladder step.
        assert pc._read_state(extra)["step"] == 1

    def test_billing_first_failure_benches_immediately(self, tmp_path, monkeypatch):
        """First 402 enters the special state at once (no window wait)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {
            "mode": "lenient",
            "billing": {"window_seconds": 3600, "fail_threshold": 1,
                        "park_seconds": 30, "base_cooldown_seconds": 300,
                        "backoff_multipliers": [3.0, 1.5],
                        "probe_requests": 1},
        })
        extra = {}
        ttl, benched = pc.record_lenient_failure(extra, status_code=402, now=1000.0)
        assert benched is True          # immediate — not parked
        assert ttl == 900.0             # base 300 x first ladder factor 3
        assert pc._read_state(extra)["step"] == 1

    def test_billing_backoff_grows_when_uncapped(self, tmp_path, monkeypatch):
        """With N<=0 the pure exponential backoff governs the retry interval."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {
            "mode": "lenient",
            "billing": {"window_seconds": 3600, "fail_threshold": 1,
                        "park_seconds": 30, "base_cooldown_seconds": 300,
                        "backoff_multipliers": [3.0, 1.5, 1.25],
                        "probe_window_seconds": 0, "probe_attempts": 1},
        })
        extra = {}
        now = 0.0
        out = [pc.record_lenient_failure(extra, status_code=402, now=now + i)[0] for i in range(3)]
        assert out == [pytest.approx(900.0), pytest.approx(1350.0), pytest.approx(1687.5)]


class TestRecovery:
    def _enable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {"mode": "lenient"})

    def test_success_clears_state(self, tmp_path, monkeypatch):
        self._enable(tmp_path, monkeypatch)
        extra = {}
        pc.record_lenient_failure(extra, status_code=429, now=100.0)
        assert pc.STATE_KEY in extra
        assert pc.record_success(extra) is True
        assert pc.STATE_KEY not in extra

    def test_failed_bench_continues_ladder(self, tmp_path, monkeypatch):
        # A failed bench must NOT clear state — the ladder keeps growing.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {
            "mode": "lenient",
            "rate_limit": {"window_seconds": 1800, "fail_threshold": 1,
                           "park_seconds": 30, "base_cooldown_seconds": 300,
                           "backoff_multipliers": [1.0]},
        })
        extra = {}
        # threshold=1 -> every 429 benches; step grows per bench.
        pc.record_lenient_failure(extra, status_code=429, now=100.0)
        step_after_first = pc._read_state(extra)["step"]
        assert step_after_first == 1
        pc.record_lenient_failure(extra, status_code=429, now=200.0)
        assert pc._read_state(extra)["step"] == step_after_first + 1


class TestNonRolledFallsThrough:
    def _enable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {"mode": "lenient"})

    def test_auth_returns_none(self, tmp_path, monkeypatch):
        self._enable(tmp_path, monkeypatch)
        assert pc.record_lenient_failure({}, status_code=401, failure_reason="auth") is None

    def test_unknown_returns_none(self, tmp_path, monkeypatch):
        self._enable(tmp_path, monkeypatch)
        assert pc.record_lenient_failure({}, status_code=500) is None


class TestHostileConfig:
    def test_bad_values_fall_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {
            "mode": "lenient",
            "rate_limit": {"window_seconds": "abc", "fail_threshold": 0,
                           "park_seconds": float("nan"), "base_cooldown_seconds": float("inf"),
                           "backoff_multipliers": ["x", None, -1]},
        })
        s = pc.get_settings(force_reload=True)
        assert s["mode"] == pc.MODE_LENIENT
        assert s["rate_limit"]["window_seconds"] == 1800.0
        assert s["rate_limit"]["fail_threshold"] == 5
        assert s["rate_limit"]["park_seconds"] == 30.0
        assert s["rate_limit"]["base_cooldown_seconds"] == 300.0
        assert s["rate_limit"]["backoff_multipliers"] == [1.0]

    def test_never_raises_with_broken_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("provider_cooldown: [broken", encoding="utf-8")
        pc._reset_settings_cache_for_tests()
        s = pc.get_settings(force_reload=True)
        assert s["mode"] == pc.MODE_STRICT  # safe default

    def test_returned_settings_do_not_alias_defaults(self, tmp_path, monkeypatch):
        """Regression: _merge_settings must deep-copy, so mutating the returned
        settings cannot corrupt the module-level defaults (shared list alias)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pc._reset_settings_cache_for_tests()
        s = pc.get_settings(force_reload=True)
        s["rate_limit"]["backoff_multipliers"].append(999.0)
        assert 999.0 not in pc._DEFAULT_SETTINGS["rate_limit"]["backoff_multipliers"]


class TestCurveAndCap:
    """User-selectable curve (auto/custom) and a cooldown ceiling."""

    def _set(self, tmp_path, monkeypatch, block):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_config(monkeypatch, tmp_path, {"mode": "lenient", "billing": block})
        return pc.get_settings(force_reload=True)["billing"]

    def test_default_is_auto_curve(self, tmp_path, monkeypatch):
        s = self._set(tmp_path, monkeypatch, {})
        assert s["curve"] == "auto"
        assert s["backoff_multipliers"] == pc.AUTO_CURVES["billing"]

    def test_custom_curve_is_used(self, tmp_path, monkeypatch):
        s = self._set(tmp_path, monkeypatch, {
            "curve": "custom", "backoff_multipliers": [2.0, 1.5],
        })
        assert s["curve"] == "custom"
        assert s["backoff_multipliers"] == [2.0, 1.5]

    def test_custom_curve_without_list_falls_back_to_auto(self, tmp_path, monkeypatch):
        # curve=custom but no/empty multipliers -> the built-in curve is used.
        s = self._set(tmp_path, monkeypatch, {"curve": "custom", "backoff_multipliers": []})
        assert s["backoff_multipliers"] == pc.AUTO_CURVES["billing"]

    def test_auto_ignores_stale_multipliers(self, tmp_path, monkeypatch):
        # curve=auto must not read the multipliers a user left behind.
        s = self._set(tmp_path, monkeypatch, {
            "curve": "auto", "backoff_multipliers": [9.0, 9.0],
        })
        assert s["backoff_multipliers"] == pc.AUTO_CURVES["billing"]

    def test_max_cooldown_caps_the_ladder(self, tmp_path, monkeypatch):
        self._set(tmp_path, monkeypatch, {
            "curve": "custom", "backoff_multipliers": [3.0, 1.5],
            "base_cooldown_seconds": 300, "max_cooldown_seconds": 1000,
            "probe_window_seconds": 0,
        })
        extra = {}
        now = 0.0
        # ladder: 900 (< cap) then 1350 -> clamped to the 1000s ceiling.
        t1, _ = pc.record_lenient_failure(extra, status_code=402, now=now)
        t2, _ = pc.record_lenient_failure(extra, status_code=402, now=now + 1)
        assert t1 == 900.0
        assert t2 == 1000.0  # capped

    def test_zero_cap_means_uncapped(self, tmp_path, monkeypatch):
        self._set(tmp_path, monkeypatch, {
            "curve": "custom", "backoff_multipliers": [3.0, 1.5],
            "base_cooldown_seconds": 300, "max_cooldown_seconds": 0,
            "probe_window_seconds": 0,
        })
        extra = {}
        now = 0.0
        pc.record_lenient_failure(extra, status_code=402, now=now)
        t2, _ = pc.record_lenient_failure(extra, status_code=402, now=now + 1)
        assert t2 == 1350.0  # not clamped