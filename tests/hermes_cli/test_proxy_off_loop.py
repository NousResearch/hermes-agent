"""`hermes proxy` must resolve upstream credentials off the event loop.

``UpstreamAdapter`` is a synchronous contract (``adapters/base.py`` — every
method is a plain ``def``), and both shipped adapters implement it with
blocking I/O:

  * ``NousPortalAdapter.get_credential`` takes ``_auth_store_lock()``, a
    *cross-process* advisory lock with ``AUTH_LOCK_TIMEOUT_SECONDS = 15.0``
    (``hermes_cli/auth.py:110``), reads ``auth.json`` from disk, and may issue a
    token-refresh POST. Its terminal-error path takes that lock a second time to
    persist the quarantined state.
  * ``XAIGrokAdapter`` reads its key pool off disk under a ``threading.Lock``.

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
    ) -> None:
        self._base_url = base_url
        self._stall = stall
        self._ticks = ticks if ticks is not None else [0]
        self._raise_on_credential = raise_on_credential
        self.credential_thread: Optional[int] = None
        self.authenticated_thread: Optional[int] = None
        self.ticks_across_credential: Optional[int] = None
        self.ticks_across_is_authenticated: Optional[int] = None

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
        _ = failed_credential, status_code
        return None


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
