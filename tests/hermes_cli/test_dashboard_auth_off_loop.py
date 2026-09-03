"""Every dashboard-auth provider call must run off the event loop.

``hermes_cli/dashboard_auth`` gates the dashboard, and the
:class:`~hermes_cli.dashboard_auth.base.DashboardAuthProvider` protocol is
entirely synchronous: ``start_login`` / ``complete_login`` /
``complete_password_login`` / ``verify_session`` / ``refresh_session`` /
``revoke_session`` are all plain ``def``. The shipped providers implement them
with blocking network I/O — ``httpx`` token-endpoint round trips at
``_TOKEN_ENDPOINT_TIMEOUT_SEC = 10.0``, and a ``PyJWKClient`` whose
``fetch_data`` is a blocking ``urllib.request.urlopen`` with a 30s default
ceiling and a 300s cache lifespan.

Called straight from an ``async def`` handler, each of those blocks the single
uvicorn event loop, freezing the entire dashboard — every other request, the
live feed, the terminal stream — for the duration.

These tests pin the invariant directly rather than by timing: a recording
provider asks ``asyncio.get_running_loop()`` at call time. A ``RuntimeError``
means it is on a worker thread (correct); a loop means it was invoked on the
event loop (the bug). One assertion per production call site.

Two traps this file is written around:

* ``TestClient(app)`` **without** the context-manager form spins up a fresh
  event loop per request, so a concurrency test written that way can never
  observe contention and passes even with the fix reverted. Every client here
  is entered as a context manager, which pins one anyio portal — and therefore
  one event loop — for the whole test.
* The recording provider must not be the stub the other dashboard-auth suites
  register, so this file registers its own and restores provider state itself.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.base import (
    InvalidCredentialsError,
    ProviderError,
    Session,
)
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE, SESSION_RT_COOKIE
from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider, _sign

_BASE_URL = "https://fly-app.fly.dev"
# Any gated (non-allowlisted) route works; /api/auth/me is the cheapest.
_GATED_PATH = "/api/auth/me"
# Public, provider-free, and in the shared allowlist — the probe used to show
# that a slow provider no longer holds the loop.
_PUBLIC_PATH = "/api/health"


# ---------------------------------------------------------------------------
# Recording providers
# ---------------------------------------------------------------------------


class LoopRecordingProvider(StubAuthProvider):
    """``StubAuthProvider`` that records where each provider call executed.

    ``ran_on_loop[method]`` is ``True`` when the method was entered with a
    running asyncio event loop — i.e. it was invoked directly from the
    dashboard's ``async def`` handler and is blocking it. It is ``False`` when
    the method ran on a worker thread, which is what the production code must
    arrange.
    """

    name = "stub"
    display_name = "Loop-recording stub IdP (test only)"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ran_on_loop: dict[str, bool] = {}
        # method name -> seconds to block inside the call.
        self.delays: dict[str, float] = {}
        # Set as soon as a delayed call has started blocking.
        self.entered = threading.Event()
        # method name -> exception instance to raise instead of running.
        self.raises: dict[str, BaseException] = {}

    def _record(self, method: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.ran_on_loop[method] = False
        else:
            self.ran_on_loop[method] = True
        delay = self.delays.get(method, 0.0)
        if delay:
            self.entered.set()
            # Deliberately ``time.sleep`` and not ``asyncio.sleep``: this
            # stands in for the blocking socket read a real provider performs.
            time.sleep(delay)
        exc = self.raises.get(method)
        if exc is not None:
            raise exc

    def start_login(self, **kwargs):
        self._record("start_login")
        return super().start_login(**kwargs)

    def complete_login(self, **kwargs):
        self._record("complete_login")
        return super().complete_login(**kwargs)

    def verify_session(self, **kwargs):
        self._record("verify_session")
        return super().verify_session(**kwargs)

    def refresh_session(self, **kwargs):
        self._record("refresh_session")
        return super().refresh_session(**kwargs)

    def revoke_session(self, **kwargs):
        self._record("revoke_session")
        return super().revoke_session(**kwargs)


class LoopRecordingPasswordProvider(LoopRecordingProvider):
    """Password variant, for the ``/auth/password-login`` site.

    Kept separate because ``supports_password`` diverts ``/auth/login`` to the
    credential form (so ``start_login`` is never reached) and makes
    ``/auth/native/authorize`` reject the provider outright. A single provider
    cannot cover both the OAuth and the password call sites.
    """

    name = "stubpw"
    display_name = "Loop-recording password IdP (test only)"
    supports_password = True

    def complete_password_login(self, *, username: str, password: str) -> Session:
        self._record("complete_password_login")
        if (username, password) != ("admin", "hunter2"):
            raise InvalidCredentialsError("bad credentials")
        now = int(time.time())
        exp = now + 3600
        return Session(
            user_id="stub-user-1",
            email="stub@example.test",
            display_name="Stub User",
            org_id="stub-org-1",
            provider=self.name,
            expires_at=exp,
            access_token=_sign({
                "sub": "stub-user-1",
                "email": "stub@example.test",
                "name": "Stub User",
                "org_id": "stub-org-1",
                "exp": exp,
            }),
            refresh_token=_sign({
                "sub": "stub-user-1",
                "kind": "refresh",
                "exp": now + 30 * 86400,
            }),
        )


# ---------------------------------------------------------------------------
# Token helpers — mint tokens the stub's own verify/refresh accept.
# ---------------------------------------------------------------------------


def _access_token(ttl: int = 3600) -> str:
    return _sign({
        "sub": "stub-user-1",
        "email": "stub@example.test",
        "name": "Stub User",
        "org_id": "stub-org-1",
        "exp": int(time.time()) + ttl,
    })


def _refresh_token(ttl: int = 30 * 86400) -> str:
    return _sign({
        "sub": "stub-user-1",
        "kind": "refresh",
        "exp": int(time.time()) + ttl,
    })


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def gated():
    """Yield ``make(provider) -> TestClient`` with the auth gate engaged.

    The client is entered as a context manager on purpose. ``TestClient``
    outside a ``with`` block builds a throwaway anyio portal per request, which
    means a fresh event loop per request and no possible contention between
    them; the concurrency test below would then pass against unpatched code.
    Entering it pins one portal, and therefore one loop, exactly as uvicorn
    serves the real dashboard.
    """
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    stack = contextlib.ExitStack()

    def _make(provider) -> TestClient:
        clear_providers()
        register_provider(provider)
        web_server.app.state.bound_host = "fly-app.fly.dev"
        web_server.app.state.bound_port = 443
        web_server.app.state.auth_required = True
        return stack.enter_context(
            TestClient(
                web_server.app,
                base_url=_BASE_URL,
                follow_redirects=False,
            )
        )

    try:
        yield _make
    finally:
        stack.close()
        clear_providers()
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required
        _reset_password_rate_limit()


def _assert_off_loop(provider: LoopRecordingProvider, method: str) -> None:
    assert method in provider.ran_on_loop, (
        f"{method} was never called — the test did not reach the production "
        f"site it is meant to cover (recorded: {sorted(provider.ran_on_loop)})"
    )
    assert provider.ran_on_loop[method] is False, (
        f"{method} ran ON the dashboard event loop. Provider methods perform "
        f"blocking network I/O and must be dispatched via asyncio.to_thread."
    )


# ---------------------------------------------------------------------------
# middleware.py — the auth gate
# ---------------------------------------------------------------------------


def test_bearer_verify_runs_off_the_loop(gated):
    """``middleware.py`` bearer path: ``_verify_bearer`` -> ``verify_session``."""
    provider = LoopRecordingProvider()
    client = gated(provider)

    r = client.get(
        _GATED_PATH,
        headers={"Authorization": f"Bearer {_access_token()}"},
    )

    assert r.status_code == 200, r.text
    _assert_off_loop(provider, "verify_session")


def test_cookie_verify_runs_off_the_loop(gated):
    """``middleware.py`` cookie path: the inline ``verify_session`` loop."""
    provider = LoopRecordingProvider()
    client = gated(provider)
    client.cookies.set(SESSION_AT_COOKIE, _access_token())

    r = client.get(_GATED_PATH)

    assert r.status_code == 200, r.text
    _assert_off_loop(provider, "verify_session")


def test_middleware_refresh_runs_off_the_loop(gated):
    """``middleware.py`` refresh chain: ``_attempt_refresh`` -> ``refresh_session``.

    Sends only the refresh-token cookie. That is the ordinary expiry shape, not
    an edge case: the access-token cookie's Max-Age tracks the token lifetime,
    so the browser evicts it on expiry while the RT cookie lives for 30 days.
    """
    provider = LoopRecordingProvider()
    client = gated(provider)
    client.cookies.set(SESSION_RT_COOKIE, _refresh_token())

    r = client.get(_GATED_PATH)

    assert r.status_code == 200, r.text
    _assert_off_loop(provider, "refresh_session")


# ---------------------------------------------------------------------------
# routes.py — the OAuth round trip
# ---------------------------------------------------------------------------


def test_auth_login_start_runs_off_the_loop(gated):
    """``routes.py`` ``auth_login`` -> ``start_login``."""
    provider = LoopRecordingProvider()
    client = gated(provider)

    r = client.get("/auth/login", params={"provider": "stub"})

    assert r.status_code == 302, r.text
    _assert_off_loop(provider, "start_login")


def test_native_authorize_start_runs_off_the_loop(gated):
    """``routes.py`` ``auth_native_authorize`` -> ``start_login``."""
    provider = LoopRecordingProvider()
    client = gated(provider)

    r = client.get(
        "/auth/native/authorize",
        params={
            "provider": "stub",
            "code_challenge": "x" * 43,
            "code_challenge_method": "S256",
            "redirect_uri": "http://127.0.0.1:8765/callback",
            "state": "cli-state",
        },
    )

    assert r.status_code == 302, r.text
    _assert_off_loop(provider, "start_login")


def test_auth_callback_complete_runs_off_the_loop(gated):
    """``routes.py`` ``auth_callback`` -> ``complete_login``.

    Walks the real round trip so the PKCE cookie and the state/verifier pair
    are the ones the route actually issued.
    """
    provider = LoopRecordingProvider()
    client = gated(provider)

    start = client.get("/auth/login", params={"provider": "stub"})
    assert start.status_code == 302, start.text
    cb = parse_qs(urlparse(start.headers["location"]).query)

    r = client.get(
        "/auth/callback",
        params={"code": cb["code"][0], "state": cb["state"][0]},
    )

    assert r.status_code == 302, r.text
    _assert_off_loop(provider, "complete_login")


def test_password_login_runs_off_the_loop(gated):
    """``routes.py`` ``auth_password_login`` -> ``complete_password_login``."""
    provider = LoopRecordingPasswordProvider()
    client = gated(provider)

    r = client.post(
        "/auth/password-login",
        json={"provider": "stubpw", "username": "admin", "password": "hunter2"},
    )

    assert r.status_code == 200, r.text
    _assert_off_loop(provider, "complete_password_login")


def test_logout_revoke_runs_off_the_loop(gated):
    """``routes.py`` ``auth_logout`` -> ``revoke_session``.

    The revoke is documented as best-effort and its failures are swallowed, so
    on the event loop an unreachable IDP turns an ignorable call into a
    dashboard-wide stall for the full provider timeout.
    """
    provider = LoopRecordingProvider()
    client = gated(provider)
    client.cookies.set(SESSION_RT_COOKIE, _refresh_token())

    r = client.post("/auth/logout")

    assert r.status_code == 302, r.text
    _assert_off_loop(provider, "revoke_session")


def test_native_refresh_runs_off_the_loop(gated):
    """``routes.py`` ``auth_native_refresh`` -> ``refresh_session``."""
    provider = LoopRecordingProvider()
    client = gated(provider)

    r = client.post(
        "/auth/native/refresh",
        json={"refresh_token": _refresh_token(), "provider": "stub"},
    )

    assert r.status_code == 200, r.text
    _assert_off_loop(provider, "refresh_session")


# ---------------------------------------------------------------------------
# The symptom itself
# ---------------------------------------------------------------------------


def test_a_slow_provider_does_not_stall_a_concurrent_request(gated):
    """A slow IDP must not freeze the rest of the dashboard.

    One request enters the gate against a provider that blocks for
    ``_SLOW`` seconds; a second, public, provider-free request is issued from
    another thread while it is in flight. Both share one event loop (the
    fixture pins a single portal), so if the provider call runs on the loop the
    probe cannot be served until it finishes.
    """
    _SLOW = 2.0
    _BUDGET = 1.0

    provider = LoopRecordingProvider()
    provider.delays["verify_session"] = _SLOW
    client = gated(provider)

    slow_done = threading.Event()

    def _slow_request() -> None:
        try:
            client.get(
                _GATED_PATH,
                headers={"Authorization": f"Bearer {_access_token()}"},
            )
        finally:
            slow_done.set()

    worker = threading.Thread(target=_slow_request, daemon=True)
    worker.start()
    try:
        assert provider.entered.wait(timeout=10.0), (
            "the slow provider call never started"
        )
        started = time.monotonic()
        probe = client.get(_PUBLIC_PATH)
        elapsed = time.monotonic() - started
    finally:
        worker.join(timeout=15.0)

    assert probe.status_code == 200, probe.text
    assert slow_done.is_set(), "the slow request never completed"
    assert elapsed < _BUDGET, (
        f"a public request waited {elapsed:.2f}s behind a {_SLOW:.1f}s provider "
        f"call — the provider call is blocking the event loop"
    )


def test_provider_errors_still_surface_unchanged(gated):
    """Offloading must not move an exception out of its ``except`` arm.

    ``asyncio.to_thread`` re-raises the worker's exception in the awaiting
    frame, so ``ProviderError`` still reaches the handler that turns it into a
    503. This is a behaviour-preservation guard: it passes both before and
    after the production change, and is here so a future refactor that swaps
    the offload for a fire-and-forget cannot land quietly.
    """
    provider = LoopRecordingProvider()
    provider.raises["verify_session"] = ProviderError("stub")
    client = gated(provider)

    r = client.get(
        _GATED_PATH,
        headers={"Authorization": f"Bearer {_access_token()}"},
    )

    assert r.status_code == 503, r.text
    assert "unreachable" in r.json()["detail"]
