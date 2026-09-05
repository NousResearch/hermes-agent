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

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth import native_flow
from hermes_cli.dashboard_auth import routes as routes_mod
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
        assert _refresh(client, token="rt-valid-shape").status_code == 401
        assert provider.refresh_calls > calls_at_open
        # Counter reset by the verdict: one fresh transient is a plain 503,
        # not a refusal.
        provider.fail = True
        r = _refresh(client, token="rt-valid-shape")
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
        # Cooldown re-armed by the failed probe: with a long cooldown the
        # very next request is refused WITHOUT touching the provider.
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

    def test_ip_table_stays_bounded(self):
        for i in range(routes_mod._IP_TABLE_MAX + 500):
            routes_mod._breaker_record(
                f"tok-{i}", f"10.{i // 250}.{i % 250}", "transient"
            )
        assert len(routes_mod._ip_transients) <= routes_mod._IP_TABLE_MAX + 1
