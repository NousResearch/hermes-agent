"""Integration tests for the opt-in lenient key-cooldown wired into credential_pool.

These exercise the wiring, not just the policy module:
1. strict (default) keeps the upstream single-failure bench, byte-for-byte.
2. lenient: a single 429 only PARKS the key (short cooldown), not an hour.
3. lenient: crossing the window threshold BENCHES with the growing ladder.
4. mark_success clears the rolling state (probe must succeed to reset).
5. Non-rolled failures (auth) keep the upstream TTL even in lenient mode.
"""
from __future__ import annotations

import json
import time

import pytest

from agent import provider_cooldown as pc


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _entry(entry_id, provider="openrouter", api_key="key-1"):
    return {"id": entry_id, "provider": provider, "api_key": api_key,
            "auth_type": "api_key", "source": "manual", "priority": 0}


def _load(tmp_path, monkeypatch, entries: list[dict]):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {"openrouter": entries}})
    from agent.credential_pool import load_pool

    return load_pool("openrouter")


def _write_cooldown_config(tmp_path, monkeypatch, block):
    import yaml

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"provider_cooldown": block}), encoding="utf-8"
    )
    pc._reset_settings_cache_for_tests()


@pytest.fixture(autouse=True)
def _reset_policy_cache():
    pc._reset_settings_cache_for_tests()
    yield
    pc._reset_settings_cache_for_tests()


class TestStrictUnchanged:
    def test_single_429_benches_upstream_way(self, tmp_path, monkeypatch):
        # No provider_cooldown config -> strict. A single 429 benches the key
        # for the upstream 1h TTL (window logic must NOT engage).
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        entry = pool.entries()[0]
        pool._mark_exhausted(entry, 429, failure_reason="rate_limit")
        again = pool.entries()[0]
        assert again.last_status == "exhausted"
        from agent.credential_pool import _exhausted_until

        until = _exhausted_until(again, sole_credential=True)
        assert until is not None
        # Upstream behaviour must be intact AND the lenient window logic must NOT
        # have engaged: no rolling state, and the cooldown is the upstream
        # sole-credential 429 TTL (60s), not the lenient 30s park.
        assert pc.STATE_KEY not in again.extra
        assert 31 < until - time.time() <= 61

    def test_mark_success_is_noop_in_strict(self, tmp_path, monkeypatch):
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        # strict -> mark_success must not touch anything.
        assert pool.mark_success("cred-1") is False


class TestLenientParking:
    def test_single_429_only_parks(self, tmp_path, monkeypatch):
        _write_cooldown_config(tmp_path, monkeypatch, {
            "mode": "lenient",
            "rate_limit": {"window_seconds": 1800, "fail_threshold": 5,
                           "park_seconds": 30, "base_cooldown_seconds": 300,
                           "backoff_multipliers": [1.0]},
        })
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        entry = pool.entries()[0]
        pool._mark_exhausted(entry, 429, failure_reason="rate_limit")
        again = pool.entries()[0]
        from agent.credential_pool import _exhausted_until

        until = _exhausted_until(again, sole_credential=True)
        assert until is not None
        # Parked only ~30s, not the upstream hour.
        assert 0 < until - time.time() <= 31

    def test_threshold_benches_with_ladder(self, tmp_path, monkeypatch):
        _write_cooldown_config(tmp_path, monkeypatch, {
            "mode": "lenient",
            "billing": {"window_seconds": 3600, "fail_threshold": 3,
                        "park_seconds": 30, "base_cooldown_seconds": 300,
                        "backoff_multipliers": [3.0, 1.5]},
        })
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        from agent.credential_pool import _exhausted_until

        # Failure 1 & 2 -> park (~30s). Failure 3 crosses threshold -> bench (900s).
        for _ in range(2):
            pool._mark_exhausted(pool.entries()[0], 402, failure_reason="billing")
            until = _exhausted_until(pool.entries()[0], sole_credential=True)
            assert until - time.time() <= 31
        pool._mark_exhausted(pool.entries()[0], 402, failure_reason="billing")
        until = _exhausted_until(pool.entries()[0], sole_credential=True)
        assert until - time.time() > 800  # climbed the ladder to ~900s


class TestLenientRecovery:
    def test_mark_success_clears_state(self, tmp_path, monkeypatch):
        _write_cooldown_config(tmp_path, monkeypatch, {"mode": "lenient"})
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        pool._mark_exhausted(pool.entries()[0], 429, failure_reason="rate_limit")
        assert pc.STATE_KEY in pool.entries()[0].extra
        assert pool.mark_success("cred-1") is True
        assert pc.STATE_KEY not in pool.entries()[0].extra

    def test_failed_probe_keeps_state(self, tmp_path, monkeypatch):
        # A failed bench must NOT clear state — the ladder keeps counting.
        _write_cooldown_config(tmp_path, monkeypatch, {
            "mode": "lenient",
            "rate_limit": {"window_seconds": 1800, "fail_threshold": 5,
                           "park_seconds": 30, "base_cooldown_seconds": 300,
                           "backoff_multipliers": [1.0]},
        })
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        # First 5 windowed failures PARK (step stays 0); the 5th benches (step 1).
        for _ in range(5):
            pool._mark_exhausted(pool.entries()[0], 429, failure_reason="rate_limit")
        assert pc._read_state(pool.entries()[0].extra)["step"] == 1
        # A 6th failure within the window benches again -> step grows.
        pool._mark_exhausted(pool.entries()[0], 429, failure_reason="rate_limit")
        assert pc._read_state(pool.entries()[0].extra)["step"] == 2

    def test_billing_recovery_after_recharge(self, tmp_path, monkeypatch):
        """The ONLY 402 clearing path: an external fix (recharge) makes the next
        real call succeed, and that success clears the rolling state."""
        _write_cooldown_config(tmp_path, monkeypatch, {
            "mode": "lenient",
            "billing": {"window_seconds": 3600, "fail_threshold": 1,
                        "park_seconds": 30, "base_cooldown_seconds": 300,
                        "backoff_multipliers": [3.0, 1.5]},
        })
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        pool._mark_exhausted(pool.entries()[0], 402, failure_reason="billing")
        assert pc.STATE_KEY in pool.entries()[0].extra
        # User recharges; the next real call succeeds -> clears everything.
        assert pool.mark_success("cred-1") is True
        assert pc.STATE_KEY not in pool.entries()[0].extra


class TestNonRolledUnaffected:
    def test_auth_failure_keeps_upstream_ttl_in_lenient(self, tmp_path, monkeypatch):
        _write_cooldown_config(tmp_path, monkeypatch, {"mode": "lenient"})
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        pool._mark_exhausted(pool.entries()[0], 401, failure_reason="auth")
        from agent.credential_pool import _exhausted_until

        until = _exhausted_until(pool.entries()[0], sole_credential=False)
        assert until is not None
        # 401 keeps its 5-minute upstream TTL, not a 30s park.
        assert until - time.time() > 200
        assert pc.STATE_KEY not in pool.entries()[0].extra


class TestPersistence:
    def test_rolling_state_survives_pool_reload(self, tmp_path, monkeypatch):
        """Regression: the rolling state must round-trip through auth.json.

        from_dict() rebuilds ``extra`` from the _EXTRA_KEYS whitelist, so a
        state key not listed there is dropped on every reload — which resets the
        rolling count per request in direct integrations and silently defeats
        the lenient policy. This drives the real load_pool() reload path.
        """
        _write_cooldown_config(tmp_path, monkeypatch, {"mode": "lenient"})
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        pool._mark_exhausted(pool.entries()[0], 429, failure_reason="rate_limit")
        assert pc.STATE_KEY in pool.entries()[0].extra

        # Reload from disk (fresh pool instance, same HERMES_HOME).
        from agent.credential_pool import load_pool

        reloaded = load_pool("openrouter")
        state = reloaded.entries()[0].extra.get(pc.STATE_KEY)
        assert state is not None, "rolling state was dropped on reload"
        # A single failure only parks (no ladder step), but the failure timestamp
        # must survive so the window keeps counting across reloads.
        assert state["step"] == 0
        assert len(state["failures"]) == 1


class TestProbeGate:
    """The post-blackout "X selections then one probe" hold (pool wiring)."""

    def _bench_cred1(self, tmp_path, monkeypatch, *, probe_requests):
        _write_cooldown_config(tmp_path, monkeypatch, {
            "mode": "lenient",
            "billing": {"window_seconds": 3600, "fail_threshold": 1,
                        "park_seconds": 30, "base_cooldown_seconds": 300,
                        "backoff_multipliers": [3.0],
                        "probe_requests": probe_requests},
        })
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1"), _entry("cred-2", api_key="key-2")])
        pool._mark_exhausted(pool.entries()[0], 402, failure_reason="billing")
        return pool

    def test_hold_until_x_selections_then_release(self, tmp_path, monkeypatch):
        from agent.credential_pool import _lenient_holds_probe

        pool = self._bench_cred1(tmp_path, monkeypatch, probe_requests=2)
        entry = pool.entries()[0]
        # Force the blackout into the past so we exercise the selection gate.
        state = dict(entry.extra[pc.STATE_KEY])
        state["blackout_until"] = time.time() - 1
        entry.extra[pc.STATE_KEY] = state

        now = time.time()
        # Without advancing (advisory callers) the key is reported as held.
        assert _lenient_holds_probe(entry, advance=False, now=now) is True
        # 1st real selection is consumed but does not open the gate...
        assert _lenient_holds_probe(entry, advance=True, now=now) is True
        # ...the 2nd (== X) opens it, and the next selection is allowed.
        assert _lenient_holds_probe(entry, advance=True, now=now) is False
        assert _lenient_holds_probe(entry, advance=True, now=now) is False

    def test_still_held_inside_blackout(self, tmp_path, monkeypatch):
        from agent.credential_pool import _lenient_holds_probe

        pool = self._bench_cred1(tmp_path, monkeypatch, probe_requests=1)
        entry = pool.entries()[0]
        # The bench is 900s out; no selection may open the gate yet.
        now = time.time()
        assert _lenient_holds_probe(entry, advance=True, now=now) is True
        assert _lenient_holds_probe(entry, advance=True, now=now) is True

    def test_strict_never_holds(self, tmp_path, monkeypatch):
        from agent.credential_pool import _lenient_holds_probe

        # No provider_cooldown config -> strict: even with stale rolling state
        # on the entry, the hold must be a no-op (upstream behaviour intact).
        _write_cooldown_config(tmp_path, monkeypatch, {})
        pool = _load(tmp_path, monkeypatch, [_entry("cred-1")])
        entry = pool.entries()[0]
        entry.extra[pc.STATE_KEY] = {
            "failures": [], "step": 1, "blackout_until": time.time() - 1,
            "probe_seen": 0, "probe_requests": 5,
        }
        assert _lenient_holds_probe(entry, advance=True, now=time.time()) is False