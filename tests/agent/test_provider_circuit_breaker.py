"""Tests for the provider-level rolling-window circuit breaker.

Covers the behaviour the fallback chain relies on:
1. Failures accumulate per backend and clear when the rolling window slides past.
2. Reaching the threshold quiesces the backend; below it does not.
3. A success clears both the window and any quiesce (automatic recovery).
4. Quiesce expires by wall-clock (state survives a restart / fresh read).
5. Provider/base_url identity keeps distinct backends separate.
6. Config overrides are honored and validated; disabling turns it off.
7. State never persists credentials — only provider/base_url/timestamps/counters.
"""
import json
import time

import pytest

from agent import provider_circuit_breaker as cb


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a tmp dir and reset module caches per test."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cb.reset_all_for_tests()
    yield
    cb.reset_all_for_tests()


def _write_config(tmp_path, block):
    """Write a config.yaml with the breaker block.

    Accepts either the bare breaker dict or a dict already wrapped as
    ``{"fallback_circuit_breaker": {...}}`` (all call sites use the wrapped
    form), so the on-disk yaml is never double-nested.

    The breaker auto-follows the project's API pool (fallback_providers chain),
    so to exercise the real circuit-breaker behaviour tests must configure a
    pool. We inject one automatically unless the test explicitly turns the
    feature off as a hard kill-switch (``enabled: false``) — that case stays
    chain-less and inert.
    """
    import yaml

    if isinstance(block, dict) and "fallback_circuit_breaker" in block:
        block = block["fallback_circuit_breaker"]
    cfg = {"fallback_circuit_breaker": block}
    explicit_off = isinstance(block.get("enabled"), bool) and not block["enabled"]
    if not explicit_off and "fallback_providers" not in block and "fallback_model" not in block:
        cfg["fallback_providers"] = [
            {"provider": "openrouter", "model": "gpt-fallback", "base_url": "https://openrouter.ai/api/v1"}
        ]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cb.get_settings(force_reload=True)


# ---------------------------------------------------------------------------
# 1. Rolling window
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_failures_accumulate_and_prune(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 100, "fail_threshold": 5, "cooldown_seconds": 100}})
        now = 1000.0
        for offset in (0.0, 10.0, 20.0):
            cb.record_failure("openrouter", "", now=now + offset)
        # All three are inside the window.
        assert cb.status(now=now + 30).count("fails=3/5") == 1

        # Jump past the window: the old failures are pruned, none remain.
        cb.record_failure("openrouter", "", now=now + 500)
        assert "fails=1/5" in cb.status(now=now + 500)

    def test_threshold_trips_breaker(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 3600, "fail_threshold": 3, "cooldown_seconds": 3600}})
        now = 2000.0
        assert cb.record_failure("openrouter", "", now=now) is False
        assert cb.record_failure("openrouter", "", now=now + 1) is False
        assert cb.record_failure("openrouter", "", now=now + 2) is True
        assert cb.is_quiesced("openrouter", "", now=now + 3) is True

    def test_below_threshold_does_not_quiesce(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 3600, "fail_threshold": 3, "cooldown_seconds": 3600}})
        cb.record_failure("openrouter", "")
        cb.record_failure("openrouter", "")
        assert cb.is_quiesced("openrouter") is False


# ---------------------------------------------------------------------------
# 2. Recovery
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_success_clears_window_and_quiesce(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 3600, "fail_threshold": 2, "cooldown_seconds": 3600}})
        now = 3000.0
        cb.record_failure("openrouter", "", now=now)
        assert cb.record_failure("openrouter", "", now=now + 1) is True
        assert cb.is_quiesced("openrouter", "", now=now + 2) is True

        cb.record_success("openrouter", "")
        assert cb.is_quiesced("openrouter", "", now=now + 3) is False
        assert cb.remaining_cooldown("openrouter", "", now=now + 3) == 0.0
        assert "fails=0/2" in cb.status(now=now + 3)

    def test_quiesce_expires_by_clock(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 3600, "fail_threshold": 1, "cooldown_seconds": 60}})
        now = 5000.0
        assert cb.record_failure("openrouter", "", now=now) is True
        assert cb.is_quiesced("openrouter", "", now=now + 30) is True
        # Cooldown elapsed: retried again.
        assert cb.is_quiesced("openrouter", "", now=now + 61) is False

    def test_remaining_cooldown_counts_down(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 3600, "fail_threshold": 1, "cooldown_seconds": 600}})
        now = 8000.0
        cb.record_failure("openrouter", "", now=now)
        assert cb.remaining_cooldown("openrouter", "", now=now + 100) == pytest.approx(500.0, abs=1.0)


# ---------------------------------------------------------------------------
# 3. Identity
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_distinct_providers_are_independent(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 3600, "fail_threshold": 2, "cooldown_seconds": 3600}})
        cb.record_failure("openrouter", "")
        cb.record_failure("openrouter", "")
        assert cb.is_quiesced("openrouter", "") is True
        assert cb.is_quiesced("anthropic", "") is False

    def test_distinct_base_urls_are_independent(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 3600, "fail_threshold": 1, "cooldown_seconds": 3600}})
        cb.record_failure("custom", "https://a.example/v1/")
        # Trailing slash and case are normalized away: same backend.
        assert cb.is_quiesced("custom", "https://A.example/v1") is True
        # A different endpoint is a different backend.
        assert cb.is_quiesced("custom", "https://b.example/v1") is False

    def test_empty_provider_is_ignored(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {"enabled": True, "fail_threshold": 1}})
        assert cb.record_failure("", "") is False
        assert cb.is_quiesced("", "") is False


# ---------------------------------------------------------------------------
# 4. Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults_when_unconfigured(self, tmp_path):
        # No config at all => no fallback pool configured => the breaker is inert.
        # With a single provider, quenching it on repeated failures would make it
        # unusable, so it must never fire until a pool exists.
        settings = cb.get_settings(force_reload=True)
        assert settings["enabled"] is False
        assert settings["window_seconds"] == 600
        assert settings["fail_threshold"] == 5
        assert settings["cooldown_seconds"] == 120

    def test_auto_enables_when_fallback_chain_configured(self, tmp_path):
        # The core requirement: the breaker rides on the project's API pool. Merely
        # configuring a fallback_providers chain is enough to enable it — no manual
        # `enabled: true` needed.
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "fail_threshold": 1, "cooldown_seconds": 3600}})
        settings = cb.get_settings(force_reload=True)
        assert settings["enabled"] is True

    def test_explicit_false_disables_even_with_pool(self, tmp_path):
        # A pool exists but the user hard-disables: conservative kill-switch wins.
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": False, "fail_threshold": 1, "cooldown_seconds": 3600}})
        assert cb.record_failure("openrouter", "https://openrouter.ai/api/v1") is False
        assert cb.is_quiesced("openrouter", "https://openrouter.ai/api/v1") is False

    def test_disabled_short_circuits(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": False, "fail_threshold": 1, "cooldown_seconds": 3600}})
        assert cb.record_failure("openrouter", "") is False
        assert cb.is_quiesced("openrouter", "") is False

    def test_overrides_are_honored(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": 120, "fail_threshold": 1, "cooldown_seconds": 30}})
        settings = cb.get_settings(force_reload=True)
        assert settings["window_seconds"] == 120
        assert settings["fail_threshold"] == 1
        assert settings["cooldown_seconds"] == 30

    def test_invalid_values_fall_back_to_defaults(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "window_seconds": "not-a-number", "fail_threshold": 0, "cooldown_seconds": -5}})
        settings = cb.get_settings(force_reload=True)
        assert settings["window_seconds"] == 600
        assert settings["fail_threshold"] == 5
        assert settings["cooldown_seconds"] == 120


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_state_round_trips_across_reads(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": True, "window_seconds": 3600, "fail_threshold": 1, "cooldown_seconds": 3600}})
        now = time.time()
        cb.record_failure("openrouter", "https://openrouter.ai/api/v1", now=now)

        # Simulate a restart: drop in-memory state, re-read from disk.
        cb.reset_all_for_tests()
        cb.get_settings(force_reload=True)
        assert cb.is_quiesced("openrouter", "https://openrouter.ai/api/v1", now=now + 1) is True

    def test_persisted_state_contains_no_secrets(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {"enabled": True, "fail_threshold": 1}})
        cb.record_failure("openrouter", "https://openrouter.ai/api/v1")
        path = tmp_path / "state" / "fallback_circuit_breaker.json"
        payload = json.loads(path.read_text())
        blob = json.dumps(payload).lower()
        for forbidden in ("api_key", "apikey", "sk-", "token", "secret", "password"):
            assert forbidden not in blob
        # The shape is exactly what the module documents.
        entry = next(iter(payload["backends"].values()))
        assert set(entry) == {
            "provider", "base_url", "failures", "quiesced_until",
            "total_failures", "total_successes",
        }


# ---------------------------------------------------------------------------
# 6. Robustness: hostile state/config must never crash the failover path
# ---------------------------------------------------------------------------
# The breaker is called from the hot failover path (is_quiesced/record_failure
# during every provider retry). These tests pin down the guarantee that a
# corrupted state file, pathological config values, or concurrent access can
# never raise out of the public API.

class TestRobustness:
    def test_bad_state_file_never_crashes(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": True, "fail_threshold": 1, "cooldown_seconds": 3600}})
        key = ("openrouter", "https://openrouter.ai/api/v1")
        state_dir = tmp_path / "state"
        state_dir.mkdir(exist_ok=True)
        state_file = state_dir / "fallback_circuit_breaker.json"

        hostile_blobs = [
            "{ not valid json !!",              # garbage
            '"a plain string"',                 # non-dict root
            '{"backends": [1,2,3]}',            # backends not a dict
            '{"backends": {"k": "not-dict"}}',  # entry not a dict
            '{"backends": {"k": {"failures": "oops", "quiesced_until": "abc", '
            '"total_failures": "x", "total_successes": "y"}}}',  # bad field types
            "",                                  # empty file
        ]
        for i, blob in enumerate(hostile_blobs):
            state_file.write_text(blob, encoding="utf-8")
            # Public calls must degrade to "not quiesced", never raise.
            assert cb.is_quiesced(*key) is False
            assert cb.record_failure(*key) in (False, True)
            assert cb.record_success(*key) is None
            assert cb.remaining_cooldown(*key) == 0.0
            cb.status()  # must not raise either

    def test_hostile_config_values_never_crash(self, tmp_path):
        # _coerce_* must survive every pathological raw value.
        for raw in (None, "abc", -5, 0, 1.5, 999, [], {}, True, float("nan"), float("inf")):
            assert cb._coerce_positive(raw, 3600.0, 1.0) >= 1.0
        for raw in (None, "abc", 0, -3, 1, 7, "42", [], {}, True, float("nan")):
            assert cb._coerce_threshold(raw, 3) >= 1

    def test_bad_config_yaml_falls_back_to_defaults(self, tmp_path):
        # A broken config.yaml must not raise out of get_settings().
        (tmp_path / "config.yaml").write_text("fallback_circuit_breaker: [unbalanced", encoding="utf-8")
        settings = cb.get_settings(force_reload=True)
        assert settings["enabled"] in (True, False)
        assert settings["window_seconds"] == 600
        assert settings["fail_threshold"] == 5
        assert settings["cooldown_seconds"] == 120

    def test_concurrent_access_no_deadlock_no_crash(self, tmp_path):
        # 16 threads hammering mixed ops on distinct backends: the module lock
        # is non-reentrant but never self-recurses, so this must complete
        # without deadlock or exception.
        import threading

        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": True, "fail_threshold": 3, "cooldown_seconds": 3600}})
        errors = []
        barrier = threading.Barrier(16)

        def worker(pidx):
            try:
                barrier.wait()
                provider = f"p{pidx}"
                for i in range(300):
                    base = f"https://api{pidx}.example/v{i % 3}"
                    if i % 3 == 0:
                        cb.record_failure(provider, base)
                    elif i % 3 == 1:
                        cb.is_quiesced(provider, base)
                    else:
                        cb.remaining_cooldown(provider, base)
                        cb.status()
                    cb.record_success(provider, base)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# 7. Half-open auto-probe (opt-in self-healing)
# ---------------------------------------------------------------------------

class TestAutoProbe:
    def test_probe_is_off_by_default(self, tmp_path):
        # Conservative default: a quiesced backend stays skipped until cooldown
        # elapses — no extra probe traffic.
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": True, "fail_threshold": 1, "cooldown_seconds": 3600}})
        settings = cb.get_settings(force_reload=True)
        assert settings["auto_probe"] is False

    def test_probe_passes_one_attempt_after_interval(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": True, "fail_threshold": 1, "cooldown_seconds": 100,
            "auto_probe": True, "probe_interval_seconds": 40}})
        now = 9000.0
        cb.record_failure("openrouter", "", now=now)  # trip, quiesced_until=now+100
        # Well before the probe interval -> still quiesced.
        assert cb.is_quiesced("openrouter", "", now=now + 10) is True
        # Probe window reached (now >= quiesced_until - 40) -> one attempt passes.
        assert cb.is_quiesced("openrouter", "", now=now + 61) is False
        # A second concurrent check while the probe is in flight stays gated.
        assert cb.is_quiesced("openrouter", "", now=now + 61) is True

    def test_failed_probe_rearms_cooldown(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": True, "fail_threshold": 1, "cooldown_seconds": 100,
            "auto_probe": True, "probe_interval_seconds": 40}})
        now = 10000.0
        cb.record_failure("openrouter", "", now=now)
        # Probe passes through at now+61, then the attempt FAILS -> re-arm.
        assert cb.is_quiesced("openrouter", "", now=now + 61) is False
        cb.record_failure("openrouter", "", now=now + 61)
        # Now firmly quiesced again (fresh cooldown from now+61).
        assert cb.is_quiesced("openrouter", "", now=now + 80) is True

    def test_successful_probe_recovers_immediately(self, tmp_path):
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": True, "fail_threshold": 1, "cooldown_seconds": 100,
            "auto_probe": True, "probe_interval_seconds": 40}})
        now = 11000.0
        cb.record_failure("openrouter", "", now=now)
        assert cb.is_quiesced("openrouter", "", now=now + 61) is False  # probe passes
        cb.record_success("openrouter", "")  # probe answered
        # Immediately recovered, not quiesced, no cooldown remaining.
        assert cb.is_quiesced("openrouter", "", now=now + 62) is False
        assert cb.remaining_cooldown("openrouter", "", now=now + 62) == 0.0

    def test_unsettled_probe_does_not_leak_slot(self, tmp_path):
        """Regression: a probe that passes but never settles must not wedge the
        backend's probe slot forever. When the cooldown elapses the marker is
        dropped, so the next cooldown can probe again."""
        _write_config(tmp_path, {"fallback_circuit_breaker": {
            "enabled": True, "fail_threshold": 1, "cooldown_seconds": 100,
            "auto_probe": True, "probe_interval_seconds": 40}})
        now = 12000.0
        cb.record_failure("openrouter", "", now=now)  # quiesced until now+100
        # Probe passes at now+61 but is NOT settled (candidate skipped elsewhere).
        assert cb.is_quiesced("openrouter", "", now=now + 61) is False
        assert "openrouter|" in cb._probe_inflight
        # Cooldown elapses; the stale marker must be cleared.
        assert cb.is_quiesced("openrouter", "", now=now + 101) is False
        assert cb._probe_inflight == set()
        # A fresh cooldown can still probe (slot not leaked).
        cb.record_failure("openrouter", "", now=now + 200)  # quiesced until now+300
        assert cb.is_quiesced("openrouter", "", now=now + 261) is False