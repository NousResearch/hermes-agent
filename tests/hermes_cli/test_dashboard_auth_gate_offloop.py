"""Regression guard for the event-loop starvation reported in #54523.

``_verify_access_token`` is synchronous: it walks the provider stack and, for
JWKS-backed providers, may fetch and parse signing keys. When
``gated_auth_middleware`` called it inline on the cookie and bearer paths, a
slow IdP stalled the serve event loop for the whole verify — the exact
starvation behind #54523's multi-second ``event loop stalled`` telemetry, during
which WS handshakes and API requests time out while a plain ``curl`` to the
same server returns instantly. The refresh path already runs its provider work
via ``run_in_threadpool``; these tests pin the same property for the two verify
paths by asserting the provider's ``verify_session`` never executes on the
event-loop thread.
"""

import threading

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.base import DashboardAuthProvider, LoginStart, Session
from hermes_cli.dashboard_auth.middleware import gated_auth_middleware


class _ThreadRecordingProvider(DashboardAuthProvider):
    """Fake IdP whose ``verify_session`` records the thread it ran on."""

    name = "recorder"
    display_name = "Recorder IdP (test only)"

    def __init__(self):
        self.verify_threads: set[int] = set()

    def start_login(self, *, redirect_uri: str) -> LoginStart:  # pragma: no cover - unused
        raise NotImplementedError

    def complete_login(self, *, code, state, code_verifier, redirect_uri):  # pragma: no cover
        raise NotImplementedError

    def verify_session(self, *, access_token: str):
        self.verify_threads.add(threading.get_ident())
        return Session(
            user_id="u1", email="u1@example.com", display_name="U1", org_id="",
            provider=self.name, expires_at=4_102_444_800,
            access_token=access_token, refresh_token="")

    def refresh_session(self, *, refresh_token: str):  # pragma: no cover - unused
        raise NotImplementedError

    def revoke_session(self, *, refresh_token: str):  # pragma: no cover - unused
        raise NotImplementedError


@pytest.fixture
def gate_client():
    """Gated app whose only route reports the event-loop thread it served on."""
    provider = _ThreadRecordingProvider()
    clear_providers()
    register_provider(provider)
    app = FastAPI()
    app.state.auth_required = True

    @app.middleware("http")
    async def _gate(request: Request, call_next):
        return await gated_auth_middleware(request, call_next)

    @app.get("/api/loop-thread")
    async def _loop_thread():
        return {"loop_thread": threading.get_ident()}

    with TestClient(app, base_url="http://testserver") as client:
        yield client, provider
    clear_providers()


class TestGateVerifiesOffTheEventLoop:
    """Provider ``verify_session`` must run on a worker thread, never the loop."""

    def test_bearer_verify_runs_off_loop(self, gate_client):
        client, provider = gate_client
        r = client.get("/api/loop-thread", headers={"Authorization": "Bearer tok"})
        assert r.status_code == 200
        assert provider.verify_threads, "the gate never reached verify_session"
        assert r.json()["loop_thread"] not in provider.verify_threads

    def test_cookie_verify_runs_off_loop(self, gate_client):
        client, provider = gate_client
        r = client.get("/api/loop-thread", cookies={"hermes_session_at": "tok"})
        assert r.status_code == 200
        assert provider.verify_threads, "the gate never reached verify_session"
        assert r.json()["loop_thread"] not in provider.verify_threads
