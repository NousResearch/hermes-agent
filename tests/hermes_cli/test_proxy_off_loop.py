"""`hermes proxy` must resolve upstream credentials off the event loop.

``UpstreamAdapter`` is a synchronous contract (``adapters/base.py`` — every
method is a plain ``def``), and both shipped adapters implement it with
blocking I/O:

  * ``NousPortalAdapter.get_credential`` takes ``_auth_store_lock()``, a
    *cross-process* advisory lock with ``AUTH_LOCK_TIMEOUT_SECONDS = 15.0``
    (``hermes_cli/auth.py:110``), reads ``auth.json`` from disk, and may issue a
    token-refresh POST. Its terminal-error path takes that lock a second time to
    persist the quarantined state.
  * ``NousPortalAdapter.get_retry_credential`` routes to that same
    ``_get_credential`` with ``force_refresh=True``, so the refresh POST it only
    *may* perform above is unconditional here.
  * ``XAIGrokAdapter`` reads its key pool off disk under a ``threading.Lock``.
    Its ``get_retry_credential`` loads the pool and calls
    ``try_refresh_current`` / ``mark_exhausted_and_rotate`` under that lock.

``create_app`` registers two ``async def`` handlers, so calling those methods
directly from a handler freezes the proxy's single event loop — and with it
every other in-flight streaming completion — for the whole duration.

The primary assertions here are **thread identity**, not latency. A latency
assertion measured with an HTTP client on the blocked loop is vacuous: the
client's own timer cannot advance until the block ends, so it reports a fast
response on code that was provably frozen. Thread identity has no such failure
mode and no timing sensitivity.

The harness mirrors ``tests/hermes_cli/test_proxy.py``: the proxy and a fake
upstream run as real aiohttp servers on ephemeral ports, driven by
``asyncio.run``. That keeps everything on exactly one event loop, which is what
makes the loop-starvation observations meaningful, and it avoids taking a
pytest-aiohttp dependency for one test file.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional

import pytest

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from hermes_cli.proxy.server import create_app  # noqa: E402


# How long the fake adapter blocks. Long enough that a starved loop records
# zero heartbeats, short enough to keep the suite fast. The thread-identity
# assertions do not depend on this value at all.
_STALL_SECONDS = 0.5

# Heartbeat cadence. A healthy loop fires ~50 ticks across the stall above; a
# blocked one fires exactly 0, so the threshold has three orders of magnitude
# of headroom on a loaded runner.
_HEARTBEAT_INTERVAL = 0.01
_MIN_TICKS_ACROSS_STALL = 3


class _RecordingAdapter(UpstreamAdapter):
    """Adapter that records the thread each blocking call ran on.

    ``get_credential`` and ``is_authenticated`` are plain synchronous methods
    that sleep, standing in for the auth-store lock and token refresh under the
    real adapters. Each also samples the loop-heartbeat counter on entry and
    exit, so ``ticks_across_*`` is the number of loop iterations that got to run
    *while the adapter was blocking*.
    """

    def __init__(
        self,
        base_url: str,
        *,
        stall: float = 0.0,
        ticks: Optional[List[int]] = None,
        raise_on_credential: bool = False,
        retry_bearer: Optional[str] = None,
        raise_on_retry: bool = False,
    ) -> None:
        self._base_url = base_url
        self._stall = stall
        self._ticks = ticks if ticks is not None else [0]
        self._raise_on_credential = raise_on_credential
        self._retry_bearer = retry_bearer
        self._raise_on_retry = raise_on_retry
        self.credential_thread: Optional[int] = None
        self.authenticated_thread: Optional[int] = None
        self.retry_thread: Optional[int] = None
        self.retry_status_code: Optional[int] = None
        self.ticks_across_credential: Optional[int] = None
        self.ticks_across_is_authenticated: Optional[int] = None
        self.ticks_across_retry: Optional[int] = None

    @property
    def name(self) -> str:
        return "recording"

    @property
    def display_name(self) -> str:
        return "Recording Provider"

    @property
    def allowed_paths(self):
        return frozenset({"/chat/completions"})

    def is_authenticated(self) -> bool:
        self.authenticated_thread = threading.get_ident()
        before = self._ticks[0]
        if self._stall:
            time.sleep(self._stall)
        self.ticks_across_is_authenticated = self._ticks[0] - before
        return True

    def get_credential(self) -> UpstreamCredential:
        self.credential_thread = threading.get_ident()
        before = self._ticks[0]
        if self._stall:
            time.sleep(self._stall)
        self.ticks_across_credential = self._ticks[0] - before
        if self._raise_on_credential:
            raise RuntimeError("simulated auth failure")
        return UpstreamCredential(
            bearer="test-bearer",
            base_url=self._base_url,
            expires_at="2099-01-01T00:00:00Z",
        )

    def get_retry_credential(self, *, failed_credential, status_code):
        _ = failed_credential
        self.retry_thread = threading.get_ident()
        self.retry_status_code = status_code
        before = self._ticks[0]
        if self._stall:
            time.sleep(self._stall)
        self.ticks_across_retry = self._ticks[0] - before
        if self._raise_on_retry:
            raise RuntimeError("simulated retry-credential failure")
        if self._retry_bearer is None:
            return None
        return UpstreamCredential(
            bearer=self._retry_bearer,
            base_url=self._base_url,
            expires_at="2099-01-01T00:00:00Z",
        )


async def _start_runner(app: "web.Application"):
    """Spin up an aiohttp app on an ephemeral localhost port. Returns (runner, base_url)."""
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    sockets = list(site._server.sockets)  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _build_fake_upstream(captured: Dict[str, Any]) -> "web.Application":
    async def echo(request):
        body = await request.read()
        captured["requests"].append(
            {"path": request.path, "auth": request.headers.get("Authorization")}
        )
        return web.json_response({"echoed": True, "body": body.decode("utf-8") if body else ""})

    app = web.Application()
    app.router.add_route("*", "/v1/chat/completions", echo)
    return app


def _build_rejecting_upstream(
    captured: Dict[str, Any], *, reject_status: int, accept_bearer: str
) -> "web.Application":
    """Upstream that rejects every bearer except ``accept_bearer``.

    Drives ``handle_proxy``'s ``status in {401, 429}`` branch: the first forward
    carries the initial credential and comes back rejected, the retry carries the
    rotated one and succeeds.
    """

    async def gated(request):
        auth = request.headers.get("Authorization")
        captured["requests"].append({"path": request.path, "auth": auth})
        if auth != f"Bearer {accept_bearer}":
            return web.json_response(
                {"error": {"message": "rejected"}}, status=reject_status
            )
        return web.json_response({"echoed": True})

    app = web.Application()
    app.router.add_route("*", "/v1/chat/completions", gated)
    return app


async def _heartbeat(ticks: List[int], running: List[bool]) -> None:
    """Tick a counter on the event loop until told to stop."""
    while running[0]:
        ticks[0] += 1
        await asyncio.sleep(_HEARTBEAT_INTERVAL)


# ---------------------------------------------------------------------------
# handle_proxy -> get_credential
# ---------------------------------------------------------------------------


def test_get_credential_runs_off_the_event_loop():
    """The blocking credential resolution must not execute on the loop thread.

    On the unfixed handler ``adapter.get_credential()`` is called inline, so the
    recorded thread is the loop's own and this assertion fails.
    """
    async def run():
        loop_thread = threading.get_ident()
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = _RecordingAdapter(f"{upstream_base}/v1")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions", json={}
                ) as resp:
                    assert resp.status == 200
                    await resp.read()

            assert adapter.credential_thread is not None, "get_credential was never called"
            assert adapter.credential_thread != loop_thread, (
                "get_credential ran on the event-loop thread "
                f"({adapter.credential_thread}); it blocks on a cross-process "
                "auth-store lock and must be offloaded"
            )
            # The forward itself still worked, with our bearer attached.
            assert captured["requests"][0]["auth"] == "Bearer test-bearer"
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_event_loop_keeps_running_while_credentials_resolve():
    """A stalled credential resolution must not starve the rest of the loop.

    Measured from a heartbeat task *on the loop*, sampled by the adapter itself
    on entry and exit — not from an HTTP client, whose clock cannot advance
    while the loop is blocked and which would therefore report a false pass.
    """
    async def run():
        ticks = [0]
        running = [True]
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = _RecordingAdapter(
            f"{upstream_base}/v1", stall=_STALL_SECONDS, ticks=ticks
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        beat = asyncio.create_task(_heartbeat(ticks, running))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions", json={}
                ) as resp:
                    await resp.read()

            assert adapter.ticks_across_credential is not None
            assert adapter.ticks_across_credential >= _MIN_TICKS_ACROSS_STALL, (
                f"only {adapter.ticks_across_credential} loop iterations ran during a "
                f"{_STALL_SECONDS}s credential resolution — the event loop was frozen"
            )
        finally:
            running[0] = False
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_credential_failure_still_maps_to_401():
    """Offloading must not change the error contract.

    ``asyncio.to_thread`` re-raises the worker's exception in the awaiting
    frame, so the handler's existing ``except Exception`` still produces the
    ``upstream_auth_failed`` 401. This one is deliberately *not* in the
    red-before set — it guards the behaviour the fix must leave alone.
    """
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = _RecordingAdapter(f"{upstream_base}/v1", raise_on_credential=True)
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions", json={}
                ) as resp:
                    assert resp.status == 401
                    payload = await resp.json()

            assert payload["error"]["code"] == "upstream_auth_failed"
            assert "simulated auth failure" in payload["error"]["message"]
            # The request never reached the upstream.
            assert captured["requests"] == []
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# handle_proxy -> get_retry_credential (the 401/429 rotation path)
# ---------------------------------------------------------------------------


def test_get_retry_credential_runs_off_the_event_loop():
    """The 401/429 rotation must not resolve its credential on the loop thread.

    ``get_retry_credential`` is the third and last blocking method on the
    ``UpstreamAdapter`` contract, and it is the most expensive of them:
    ``NousPortalAdapter`` routes it into ``_get_credential(force_refresh=True)``,
    so the token-refresh POST that ``get_credential`` performs only near expiry
    is unconditional here, and it happens under the same 15s cross-process
    ``_auth_store_lock()``. ``XAIGrokAdapter`` loads its key pool off disk and
    rotates it under ``self._lock``.

    Called inline, that whole rotation runs on the loop thread and this
    assertion fails.
    """
    async def run():
        loop_thread = threading.get_ident()
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(
            _build_rejecting_upstream(
                captured, reject_status=401, accept_bearer="rotated-bearer"
            )
        )
        adapter = _RecordingAdapter(
            f"{upstream_base}/v1", retry_bearer="rotated-bearer"
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions", json={}
                ) as resp:
                    assert resp.status == 200
                    await resp.read()

            assert adapter.retry_thread is not None, "get_retry_credential was never called"
            assert adapter.retry_thread != loop_thread, (
                "get_retry_credential ran on the event-loop thread "
                f"({adapter.retry_thread}); it force-refreshes the upstream token "
                "under a cross-process auth-store lock and must be offloaded"
            )
            # The rotation itself still worked: rejected bearer, then ours.
            assert adapter.retry_status_code == 401
            assert [r["auth"] for r in captured["requests"]] == [
                "Bearer test-bearer",
                "Bearer rotated-bearer",
            ]
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_event_loop_keeps_running_while_the_retry_credential_resolves():
    """A stalled 429 rotation must not starve the rest of the loop.

    Same loop-side heartbeat as the credential test, sampled by the adapter on
    entry and exit. A 429 rotation is precisely when the proxy is busiest, so
    this is the worst moment to freeze every other in-flight completion.
    """
    async def run():
        ticks = [0]
        running = [True]
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(
            _build_rejecting_upstream(
                captured, reject_status=429, accept_bearer="rotated-bearer"
            )
        )
        adapter = _RecordingAdapter(
            f"{upstream_base}/v1",
            stall=_STALL_SECONDS,
            ticks=ticks,
            retry_bearer="rotated-bearer",
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        beat = asyncio.create_task(_heartbeat(ticks, running))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions", json={}
                ) as resp:
                    await resp.read()

            assert adapter.ticks_across_retry is not None
            assert adapter.ticks_across_retry >= _MIN_TICKS_ACROSS_STALL, (
                f"only {adapter.ticks_across_retry} loop iterations ran during a "
                f"{_STALL_SECONDS}s 429 credential rotation — the event loop was frozen"
            )
        finally:
            running[0] = False
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_retry_credential_failure_still_returns_the_upstream_rejection():
    """Offloading must not change the rotation's error contract.

    ``asyncio.to_thread`` re-raises the worker's exception in the awaiting
    frame, so the handler's ``except Exception -> retry_cred = None`` still
    swallows it and streams the upstream's own 401 back. Deliberately *not* in
    the red-before set — it guards behaviour the fix must leave alone.
    """
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(
            _build_rejecting_upstream(
                captured, reject_status=401, accept_bearer="never-offered"
            )
        )
        adapter = _RecordingAdapter(f"{upstream_base}/v1", raise_on_retry=True)
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions", json={}
                ) as resp:
                    assert resp.status == 401
                    await resp.read()

            # One forward only — the failed rotation must not be retried.
            assert len(captured["requests"]) == 1
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# handle_health -> is_authenticated
# ---------------------------------------------------------------------------


def test_is_authenticated_runs_off_the_event_loop():
    """`/health` must not resolve auth state on the loop thread.

    ``adapters/base.py`` documents ``is_authenticated`` as "cheap — no network
    calls", but ``NousPortalAdapter`` implements it via ``_read_state()``, which
    takes the same 15s cross-process ``_auth_store_lock()``. ``/health`` is what
    a supervisor, systemd unit, container healthcheck or load balancer polls, so
    it is the endpoint least able to afford a lock wait.
    """
    async def run():
        loop_thread = threading.get_ident()
        adapter = _RecordingAdapter("http://127.0.0.1:1/v1")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{proxy_base}/health") as resp:
                    assert resp.status == 200
                    payload = await resp.json()

            assert payload["authenticated"] is True
            assert adapter.authenticated_thread is not None, "is_authenticated was never called"
            assert adapter.authenticated_thread != loop_thread, (
                "is_authenticated ran on the event-loop thread "
                f"({adapter.authenticated_thread}); it takes the cross-process "
                "auth-store lock and must be offloaded"
            )
        finally:
            await proxy_runner.cleanup()

    asyncio.run(run())


def test_event_loop_keeps_running_while_health_resolves_auth_state():
    """A contended auth store must not freeze the loop behind `/health`.

    Same loop-side heartbeat measurement as the credential test — a client on
    the blocked loop cannot observe its own starvation.
    """
    async def run():
        ticks = [0]
        running = [True]
        adapter = _RecordingAdapter(
            "http://127.0.0.1:1/v1", stall=_STALL_SECONDS, ticks=ticks
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        beat = asyncio.create_task(_heartbeat(ticks, running))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{proxy_base}/health") as resp:
                    await resp.read()

            assert adapter.ticks_across_is_authenticated is not None
            assert adapter.ticks_across_is_authenticated >= _MIN_TICKS_ACROSS_STALL, (
                f"only {adapter.ticks_across_is_authenticated} loop iterations ran during "
                f"a {_STALL_SECONDS}s /health auth check — the event loop was frozen"
            )
        finally:
            running[0] = False
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)
            await proxy_runner.cleanup()

    asyncio.run(run())
