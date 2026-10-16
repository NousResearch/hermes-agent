"""Per-credential circuit breaker + per-IP storm backstop for POST
/auth/native/refresh (Refs #98338, Defect 3).

A dead token fails fast (401); a *struggling* upstream (429/5xx/transport →
ProviderError → 503) invited unbounded retries with zero state change. The
breaker trips after a few consecutive transient failures for one credential
hash and refuses fast (503 + Retry-After) through a cooldown, then lets a
single half-open probe through. Permanent rejections (401 all-rejected) never
count — only transient failures do. Token rotation (fresh garbage per attempt)
evades any per-credential budget, so a coarse per-IP transient-failure count
refuses fast once a single IP storms past its own budget.

Keying is hash-only (never raw tokens); refusal reasons are audited so the
storm stays visible in dashboard-auth.log.

Run: scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_native_refresh_breaker.py
"""

from __future__ import annotations

import re
import time

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth import native_flow
from hermes_cli.dashboard_auth import routes as routes_mod
from hermes_cli.dashboard_auth import refresh_singleflight
from hermes_cli.dashboard_auth.base import ProviderError
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


class _FlakyProvider(StubAuthProvider):
    """Counts refresh calls; raises transient ProviderError until released."""

    name = "flaky"

    def __init__(self):
        super().__init__()
        self.refresh_calls = 0
        self.fail = True

    def refresh_session(self, *, refresh_token: str):
        self.refresh_calls += 1
        if self.fail:
            raise ProviderError("upstream timeout")
        return super().refresh_session(refresh_token=refresh_token)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    native_flow._reset_for_tests()
    refresh_singleflight._reset_for_tests()
    routes_mod._reset_native_refresh_breaker()
    # Deterministic windows: no sleeping in tests.
    monkeypatch.setattr(routes_mod, "_BREAKER_FAIL_THRESHOLD", 3)
    monkeypatch.setattr(routes_mod, "_BREAKER_WINDOW_SEC", 60.0)
    monkeypatch.setattr(routes_mod, "_BREAKER_COOLDOWN_SEC", 3600.0)
    monkeypatch.setattr(routes_mod, "_IP_STORM_MAX", 1_000_000)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    yield
    native_flow._reset_for_tests()
    refresh_singleflight._reset_for_tests()
    routes_mod._reset_native_refresh_breaker()
    clear_providers()
    web_server.app.state.auth_required = prev_required
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port


@pytest.fixture
def breaker_client():
    provider = _FlakyProvider()
    clear_providers()
    register_provider(provider)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(
        web_server.app, base_url="https://fly-app.fly.dev", follow_redirects=False
    )
    yield client, provider
    clear_providers()


def _refresh(client, token="rt-looping", provider="flaky"):
    # Breaker tests replay the same token repeatedly; the single-flight cache
    # would serve a stale verdict and mask the breaker decision we're testing.
    refresh_singleflight._reset_for_tests()
    return client.post(
        "/auth/native/refresh",
        json={"refresh_token": token, "provider": provider},
    )


class TestCredentialBreaker:
    def test_transient_storm_trips_breaker_and_stops_fanout(self, breaker_client):
        client, provider = breaker_client
        statuses = [_refresh(client).status_code for _ in range(6)]
        # 3 transient 503s, then the breaker refuses fast without fan-out.
        assert statuses[:3] == [503, 503, 503]
        assert statuses[3] == 503
        assert provider.refresh_calls == 3
        last = _refresh(client)
        assert last.status_code == 503
        assert last.json()["error"] == "breaker_open"
        assert int(last.headers["retry-after"]) >= 0
        assert provider.refresh_calls == 3

    def test_half_open_probe_after_cooldown(self, breaker_client, monkeypatch):
        client, provider = breaker_client
        for _ in range(3):
            assert _refresh(client).status_code == 503
        assert _refresh(client).json()["error"] == "breaker_open"
        calls_at_open = provider.refresh_calls
        assert _refresh(client).json()["error"] == "breaker_open"
        assert provider.refresh_calls == calls_at_open
        # Cooldown elapsed → one probe goes through. The stub rejects the
        # garbage RT on its merits (401): a provider verdict proves the
        # upstream is reachable again, so the breaker closes.
        monkeypatch.setattr(routes_mod, "_BREAKER_COOLDOWN_SEC", 0.0)
        provider.fail = False
        # Same credential that tripped the breaker. Probing with a *different* token
        # never enters the open-bucket branch, so the assertion would hold even if the
        # half-open admit path were deleted entirely.
        assert _refresh(client).status_code == 401
        assert provider.refresh_calls > calls_at_open
        # Counter reset by the verdict: one fresh transient is a plain 503,
        # not a refusal.
        provider.fail = True
        r = _refresh(client)
        assert r.status_code == 503
        assert r.json().get("error") != "breaker_open"

    def test_failed_probe_rearms_full_cooldown(self, breaker_client, monkeypatch):
        client, provider = breaker_client
        for _ in range(3):
            assert _refresh(client).status_code == 503
        assert _refresh(client).json()["error"] == "breaker_open"
        calls_at_open = provider.refresh_calls
        # Elapsed cooldown → one probe goes through and fails transiently.
        monkeypatch.setattr(routes_mod, "_BREAKER_COOLDOWN_SEC", 0.0)
        assert _refresh(client).status_code == 503
        assert provider.refresh_calls == calls_at_open + 1
        # Cooldown re-armed by the failed probe. The cooldown must be back at its
        # REAL value here: leaving it at 0.0 would admit a fresh probe, and leaving
        # it long for both the trip and the re-arm makes the two indistinguishable
        # (the assertion would hold even with the re-arm removed).
        monkeypatch.setattr(routes_mod, "_BREAKER_COOLDOWN_SEC", 3600.0)
        assert _refresh(client).json()["error"] == "breaker_open"
        assert provider.refresh_calls == calls_at_open + 1

    def test_permanent_rejections_never_trip_breaker(self, breaker_client):
        client, provider = breaker_client
        provider.fail = False
        for _ in range(10):
            r = _refresh(client, token="dead-token-x", provider="flaky")
            assert r.status_code == 401
        # No refusal: dead tokens fail on their merits every time.
        assert provider.refresh_calls == 10

    def test_distinct_credential_unaffected(self, breaker_client):
        client, provider = breaker_client
        for _ in range(3):
            assert _refresh(client, token="rt-a").status_code == 503
        assert _refresh(client, token="rt-a").json()["error"] == "breaker_open"
        r = _refresh(client, token="rt-b")
        assert r.status_code == 503
        assert r.json().get("error") != "breaker_open"


class TestIpStormBackstop:
    def test_rotating_tokens_from_one_ip_trips_backstop(
        self, breaker_client, monkeypatch
    ):
        client, provider = breaker_client
        monkeypatch.setattr(routes_mod, "_IP_STORM_MAX", 5)
        last = None
        for i in range(10):
            last = _refresh(client, token=f"rotated-garbage-{i}")
        assert last is not None
        assert last.status_code == 503
        assert last.json()["error"] == "storm_backstop"
        # Bounded fan-out despite a fresh credential every attempt.
        assert provider.refresh_calls <= 5

    def test_admitting_healthy_traffic_never_allocates_ip_buckets(self):
        """The admission path must not create per-IP buckets.

        `_breaker_check` reads the storm table for every admitted request. If that read
        creates an entry, a fleet with no failures at all grows the table without bound,
        and the cap/prune — which only runs from the transient-failure path in
        `_breaker_record` — never gets a chance to fire. Rotating X-Forwarded-For reaches
        it with no valid token: unbounded memory on a public endpoint, no rate limit.
        """
        for i in range(routes_mod._IP_TABLE_MAX + 500):
            routes_mod._breaker_check(f"tok-{i}", f"10.{i // 250}.{i % 250}")
        assert not routes_mod._ip_transients, (
            "the admission path allocated %d storm buckets for traffic that never failed"
            % len(routes_mod._ip_transients)
        )

    def test_ip_table_stays_bounded(self):
        for i in range(routes_mod._IP_TABLE_MAX + 500):
            routes_mod._breaker_record(
                f"tok-{i}", f"10.{i // 250}.{i % 250}", "transient"
            )
        assert len(routes_mod._ip_transients) <= routes_mod._IP_TABLE_MAX + 1


class TestStaleProbeExpiry:
    """A half-open probe whose record never lands (crash between check and
    record) must not wedge the credential: it expires after a full cooldown
    and exactly one fresh probe is admitted."""

    def _tripped_state(self, client, provider, token="rt-looping"):
        for _ in range(3):
            assert _refresh(client, token=token).status_code == 503
        assert _refresh(client, token=token).json()["error"] == "breaker_open"
        key = routes_mod._breaker_bucket(token)
        st = routes_mod._breaker_state[key]
        # Let the cooldown elapse without sleeping.
        st["open_at"] -= routes_mod._BREAKER_COOLDOWN_SEC + 1.0
        return st

    def test_stale_probe_expires_and_admits_fresh_probe(self, breaker_client):
        client, provider = breaker_client
        st = self._tripped_state(client, provider)
        # Simulate the lost record: probe admitted long ago, never resolved.
        st["probing"] = True
        st["probe_at"] = time.monotonic() - routes_mod._BREAKER_COOLDOWN_SEC - 1.0
        calls = provider.refresh_calls
        r = _refresh(client)
        assert provider.refresh_calls == calls + 1
        assert r.json().get("error") != "breaker_open"

    def test_fresh_probe_still_refuses_concurrent_probers(self, breaker_client):
        client, provider = breaker_client
        st = self._tripped_state(client, provider)
        # A live probe still guards: concurrent probers are refused.
        st["probing"] = True
        st["probe_at"] = time.monotonic()
        calls = provider.refresh_calls
        r = _refresh(client)
        assert r.json()["error"] == "breaker_open"
        assert provider.refresh_calls == calls


class TestBreakerContractGaps:
    """Pins the claims the module docstring and the route make about the breaker
    that the other tests exercise only as a side effect."""

    def test_success_closes_the_breaker_after_it_opened(self, breaker_client, monkeypatch):
        client, provider = breaker_client
        for _ in range(3):
            assert _refresh(client).status_code == 503
        assert _refresh(client).json()["error"] == "breaker_open"
        calls_at_open = provider.refresh_calls
        # Upstream recovers; the very next attempt is admitted.
        monkeypatch.setattr(routes_mod, "_BREAKER_COOLDOWN_SEC", 0.0)
        provider.fail = False
        # The stub rejects this RT shape on its merits (401) — a provider verdict, which
        # is what closes the breaker ("permanent": upstream answered).
        assert _refresh(client).status_code == 401
        assert provider.refresh_calls > calls_at_open
        # Closed: the credential's breaker state is gone, and a long cooldown no longer
        # refuses it — the re-arm that a failed probe installs is gone with it.
        assert not routes_mod._breaker_state
        monkeypatch.setattr(routes_mod, "_BREAKER_COOLDOWN_SEC", 3600.0)
        r = _refresh(client)
        assert r.json().get("error") != "breaker_open", (
            "a successful verdict did not close the breaker"
        )

    def test_refusal_is_audited_with_its_reason(self, breaker_client, monkeypatch):
        """The docstring promises refusals are audited so the storm stays visible in
        dashboard-auth.log. Without this the refusals are invisible — the exact failure
        #98338 reported, where a runaway client left no audit trace."""
        from hermes_cli.dashboard_auth import audit as audit_mod

        events = []
        monkeypatch.setattr(routes_mod, "_audit",
                            lambda request, event, **kw: events.append((event, kw)))
        client, provider = breaker_client
        for _ in range(3):
            assert _refresh(client).status_code == 503
        assert _refresh(client).json()["error"] == "breaker_open"
        refusals = [kw.get("reason") for event, kw in events
                    if getattr(event, "value", event) == "refresh_failure"]
        assert "breaker_open" in refusals, (
            "a breaker refusal was never audited: %r" % (events,)
        )

    def test_retry_after_reflects_the_remaining_cooldown(self, breaker_client, monkeypatch):
        client, provider = breaker_client
        for _ in range(3):
            assert _refresh(client).status_code == 503
        r = _refresh(client)
        assert r.json()["error"] == "breaker_open"
        # 3600s cooldown, tripped moments ago → the header must carry roughly that,
        # not a constant. A hardcoded "1" satisfies a presence check but lies to the client.
        retry_after = int(r.headers["Retry-After"])
        assert retry_after > 60, (
            "Retry-After=%d does not reflect a 3600s cooldown" % retry_after
        )

    def test_a_failed_probe_does_not_wedge_the_credential_forever(
        self, breaker_client, monkeypatch
    ):
        """Two cooldowns, not one.

        A half-open probe that fails re-arms the cooldown. If that path did not clear
        `probing`, the stale-probe expiry would never be reached again and the credential
        would be refused forever — the exact wedge this mechanism exists to prevent. The
        existing re-arm test stops at "still refused" (with a long cooldown), which is also
        what a wedge looks like, so only a SECOND elapsed cooldown tells them apart.
        """
        client, provider = breaker_client
        for _ in range(3):
            assert _refresh(client).status_code == 503
        assert _refresh(client).json()["error"] == "breaker_open"
        key = next(iter(routes_mod._breaker_state))
        # Elapse the cooldown by age, not by shortening it: the stale-probe guard only
        # engages while the cooldown is still long.
        routes_mod._breaker_state[key]["open_at"] -= routes_mod._BREAKER_COOLDOWN_SEC + 1.0
        assert _refresh(client).status_code == 503, "probe was not admitted"
        # The probe resolved (transient failure), which re-arms the cooldown.
        routes_mod._breaker_state[key]["open_at"] -= routes_mod._BREAKER_COOLDOWN_SEC + 1.0
        r = _refresh(client)
        assert r.json().get("error") != "breaker_open", (
            "credential wedged: a failed probe left `probing` set, so the stale-probe "
            "expiry never fires again"
        )

    def test_successful_refresh_from_a_storming_ip_is_admitted(
        self, breaker_client, monkeypatch
    ):
        """Only TRANSIENT failures earn a storm entry.

        A healthy upstream seeing more than _IP_STORM_MAX refreshes a minute is exactly
        the NAT household this backstop must not punish: each of those attempts closes
        its breaker and must leave the storm table untouched.
        """
        client, provider = breaker_client
        monkeypatch.setattr(routes_mod, "_IP_STORM_MAX", 5)
        provider.fail = False
        for i in range(20):
            assert _refresh(client, token=f"healthy-{i}").status_code == 401
        assert not routes_mod._ip_transients, (
            "%d storm entries earned by healthy traffic" % len(routes_mod._ip_transients)
        )

    def test_bucket_keys_never_contain_the_refresh_token(self):
        """Both tables key on a SHA-256 hex digest, never the raw token: the state is
        process-local and the audit log is append-only, so a leaked key would put a live
        refresh token into a structure that can be logged."""
        routes_mod._breaker_record("super-secret-rt-value", "10.0.0.1", "transient")
        for key in routes_mod._breaker_state:
            assert "super-secret" not in key, "bucket key leaks the raw token: %r" % key
            assert re.fullmatch(r"[0-9a-f]{64}", key), "bucket key is not a hex digest: %r" % key

    def test_idle_closed_credentials_are_evicted_before_open_ones(self, monkeypatch):
        """The cap must prefer evicting an idle credential over an OPEN one.

        Insertion order and age point opposite ways here: the open breaker is the older
        entry, the idle credential is newer but its last failure is long past the window.
        Plain FIFO keeps the dead weight and drops the credential that is actively
        failing. Ordering the eviction by age is the whole point of the sweep.
        """
        monkeypatch.setattr(routes_mod, "_BREAKER_MAX_BUCKETS", 4)
        # Older entry, actively OPEN: must survive.
        for _ in range(3):
            routes_mod._breaker_record("open-live", "10.0.0.1", "transient")
        open_key = next(k for k, v in routes_mod._breaker_state.items() if v["open_at"])
        # Newer entries that have gone quiet.
        for name in ("idle-a", "idle-b", "idle-c", "idle-d"):
            routes_mod._breaker_record(name, "10.0.0.1", "transient")
        far_past = time.monotonic() - routes_mod._BREAKER_WINDOW_SEC - 60.0
        for key in routes_mod._breaker_state:
            if key != open_key:
                routes_mod._breaker_state[key]["fails"][0] = far_past
        routes_mod._breaker_check("probe-token", "10.0.0.1")  # admission path runs the prune
        assert len(routes_mod._breaker_state) == 4
        assert open_key in routes_mod._breaker_state, (
            "the cap evicted an OPEN credential and kept idle ones"
        )

    def test_credential_table_stays_bounded(self):
        for i in range(routes_mod._BREAKER_MAX_BUCKETS + 500):
            routes_mod._breaker_check(f"tok-{i}", "10.0.0.1")
            routes_mod._breaker_record(f"tok-{i}", "10.0.0.1", "transient")
        # The record that trips the cap is itself inserted after the prune, so the table
        # settles at cap+1; what matters is that it does not grow with the load.
        assert len(routes_mod._breaker_state) <= routes_mod._BREAKER_MAX_BUCKETS + 1
