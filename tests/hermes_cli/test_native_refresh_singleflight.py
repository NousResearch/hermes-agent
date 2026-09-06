"""Native HTTP replay boundaries and deterministic provider-level concurrency."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import native_refresh as replay
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


@pytest.mark.parametrize("case", ["hint-fallback", "negative", "outage", "replacement", "client",
                                      "ttl", "capacity", "independent", "xff"])
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
        elif case == "client":
            with TestClient(app, client=("192.0.2.12", 2345)) as another_client:
                assert another_client.post("/auth/native/refresh", json={"refresh_token": "opaque-old-token"}).status_code == 200
            assert owner.calls == 2
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
            for prefix in ("192.0.2.1", "192.0.2.2"):
                assert request(headers={"x-forwarded-for": f"{prefix}, 192.0.2.100"}).status_code == 200
            # Only the ASGI peer (validated by Uvicorn), never an arbitrary header, scopes replay.
            assert owner.calls == 1


@pytest.mark.parametrize("outcome, independent", [("success", False), ("expired", False),
                                                  ("outage", False), ("success", True)])
def test_concurrent_refresh_uses_concrete_provider_identity(outcome, independent):
    owner = Provider("owner", outcome)
    other = Provider("other", "success" if independent else "expired")
    owner.release.clear()
    if independent:
        other.release.clear()
    register_provider(owner)
    register_provider(other)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(replay.refresh_native_session, "same-token", "owner", "client")
        assert owner.entered.wait(3)
        second = pool.submit(replay.refresh_native_session, "same-token", "other", "client")
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
                assert [result.provider for result in results] == ["owner", "other" if independent else "owner"]
            assert owner.calls == 1
        assert not replay._flights
