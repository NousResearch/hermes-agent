"""Native HTTP replay boundaries and deterministic provider-level concurrency."""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import refresh_singleflight as replay
from hermes_cli.dashboard_auth.base import ProviderError, RefreshExpiredError, Session
from hermes_cli.dashboard_auth.routes import router
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


class Provider(StubAuthProvider):
    def __init__(self, name, outcome="success"):
        super().__init__()
        self.name = name
        self.outcome = outcome
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def refresh_session(self, *, refresh_token):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(5), "test provider timed out"
        if self.outcome == "expired":
            raise RefreshExpiredError("expired")
        if self.outcome == "outage":
            raise ProviderError("temporarily unavailable")
        return Session(user_id=self.name, email="test@example.test", display_name=self.name,
                       org_id="test", provider=self.name, expires_at=int(time.time()) + 3600,
                       access_token=f"access-{self.name}", refresh_token=f"rotated-{self.name}")


@pytest.fixture(autouse=True)
def isolated_registry():
    clear_providers()
    with replay._guard:
        replay._cache.clear()
        replay._flights.clear()
    yield
    clear_providers()
    with replay._guard:
        assert not replay._flights
        replay._cache.clear()


@pytest.mark.parametrize("case", ["hint-fallback", "negative", "outage", "replacement",
                                      "ttl", "capacity", "independent", "network-hop"])
def test_native_http_refresh_boundaries(case, monkeypatch):
    owner = Provider("owner", "expired" if case == "negative" else "outage" if case == "outage" else "success")
    other = Provider("other", "success" if case == "independent" else "expired")
    register_provider(owner)
    register_provider(other)
    app = FastAPI()
    app.include_router(router)
    now = [100.0]
    monkeypatch.setattr(replay.time, "monotonic", lambda: now[0])
    with TestClient(app) as client:
        def request(hint="owner", token="opaque-old-token", **kwargs):
            return client.post("/auth/native/refresh", json={"refresh_token": token, "provider": hint}, **kwargs)

        first = request()
        assert first.status_code == (401 if case == "negative" else 503 if case == "outage" else 200)
        if case in {"hint-fallback", "negative", "outage"}:
            for hint in ("other", "", "unknown-a", "unknown-b", "owner"):
                response = request(hint)
                assert response.status_code == first.status_code
                if first.status_code == 200:
                    assert response.json() == first.json()
            assert owner.calls == (6 if case == "outage" else 1)
            assert other.calls == 1
        elif case == "replacement":
            clear_providers()
            replacement = Provider("owner")
            register_provider(replacement)
            assert request().status_code == 200
            assert replacement.calls == 1
        elif case == "ttl":
            assert request().json() == first.json()
            now[0] += replay._SUCCESS_TTL
            assert request().status_code == 200
            assert owner.calls == 2
        elif case == "capacity":
            monkeypatch.setattr(replay, "_MAX_ENTRIES", 2)
            for token in ("new-token-1", "new-token-2", "new-token-3"):
                assert request(token=token).status_code == 200
            assert len(replay._cache) == 2
            assert all(isinstance(key[1], bytes) and len(key[1]) == 32 for key in replay._cache)
        elif case == "independent":
            response = request("other")
            assert response.status_code == 200
            assert response.json()["provider"] == "other"
            assert first.json()["provider"] == "owner"
            assert owner.calls == other.calls == 1
        else:
            # A burst that straddles a network change (laptop wakes on another Wi-Fi) still
            # coalesces: the RT identifies the session, the peer address does not.
            with TestClient(app, client=("192.0.2.12", 2345)) as another_client:
                assert another_client.post("/auth/native/refresh", json={"refresh_token": "opaque-old-token"}).json() == first.json()
            assert owner.calls == 1


@pytest.mark.parametrize("outcome, independent", [("success", False), ("expired", False),
                                                  ("outage", False), ("success", True)])
def test_concurrent_refresh_uses_concrete_provider_identity(outcome, independent):
    def coalesced(token, hint):
        return replay.refresh_session_coalesced(
            token, hint, phase="test", log=logging.getLogger(__name__))

    owner = Provider("owner", outcome)
    other = Provider("other", "success" if independent else "expired")
    owner.release.clear()
    if independent:
        other.release.clear()
    register_provider(owner)
    register_provider(other)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(coalesced, "same-token", "owner")
        assert owner.entered.wait(3)
        second = pool.submit(coalesced, "same-token", "other")
        try:
            if independent:
                assert other.entered.wait(3), "unrelated providers must not share a lock"
            else:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    with replay._guard:
                        if any(key[0] == id(owner) and flight.users == 2 for key, flight in replay._flights.items()):
                            break
                    time.sleep(0.005)
                else:
                    pytest.fail("different hints did not converge on the concrete issuer lock")
                assert owner.calls == 1
        finally:
            owner.release.set()
            other.release.set()
        if outcome == "outage":
            for future in (first, second):
                with pytest.raises(ProviderError):
                    future.result(timeout=3)
            assert owner.calls == 2
        else:
            results = [first.result(timeout=3), second.result(timeout=3)]
            if outcome == "expired":
                assert results == [None, None]
            else:
                assert [result[1] for result in results] == ["owner", "other" if independent else "owner"]
            assert owner.calls == 1
        assert not replay._flights


class _RotatingReuseDetectingProvider(Provider):
    """A rotating-RT IdP with reuse detection: replaying a rotated RT kills the session."""

    def __init__(self):
        super().__init__("stub")
        self.rotated: set[str] = set()

    def verify_session(self, *, access_token):
        return None  # every AT presented is expired -> the gate must refresh

    def refresh_session(self, *, refresh_token):
        self.calls += 1
        if refresh_token in self.rotated:
            raise RefreshExpiredError("refresh token reuse detected")
        self.rotated.add(refresh_token)
        self.entered.set()
        assert self.release.wait(5), "test provider timed out"
        return Session(user_id="u", email="u@example.test", display_name="u", org_id="o",
                       provider=self.name, expires_at=int(time.time()) + 900,
                       access_token="fresh-at", refresh_token=f"rt-{self.calls}")


@pytest.fixture
def gated_web_app():
    from hermes_cli import web_server

    prev = {k: getattr(web_server.app.state, k, None) for k in ("bound_host", "bound_port", "auth_required")}
    web_server.app.state.bound_host = "gw.example.test"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    yield web_server.app
    for k, v in prev.items():
        setattr(web_server.app.state, k, v)


def test_cookie_gate_burst_with_stale_rt_rotates_once(gated_web_app):
    """#55712: a browser burst after AT expiry carries one stale RT in N requests; exactly one
    reaches the provider and every sibling is served under the rotated session."""
    provider = _RotatingReuseDetectingProvider()
    provider.release.clear()
    register_provider(provider)
    cookies = {"hermes_session_at": "expired-at", "hermes_session_rt": "stale-rt",
               "hermes_session_provider": "stub"}

    def call():
        # One TestClient per request: a shared jar would hand later requests the rotated RT.
        with TestClient(gated_web_app, base_url="http://gw.example.test") as client:
            return client.get("/api/auth/me", cookies=cookies)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(call) for _ in range(4)]
        assert provider.entered.wait(3)
        provider.release.set()
        statuses = sorted(f.result(timeout=10).status_code for f in futures)
    assert statuses == [200, 200, 200, 200]
    assert provider.calls == 1
