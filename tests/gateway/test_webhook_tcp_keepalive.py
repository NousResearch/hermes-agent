"""Webhook listener must survive kernels that reject SO_KEEPALIVE (#123327).

On one macOS host every accepted webhook connection died: the pinned
aiohttp's ``tcp_keepalive`` calls ``setsockopt(SO_KEEPALIVE)`` with no
``OSError`` guard (unlike ``tcp_nodelay`` in the same file), so a kernel
that answers EINVAL tears the connection down and the client sees EOF.
The in-repo mitigation lives in ``gateway.platforms.tcp_site``: an
idempotent guard that makes the per-accept keepalive tuning best-effort.
It is installed once per process at runner startup (the gateway's adapter connect,
``hermes proxy start`` in its own process) plus from ``start_tcp_site`` for callers
outside those two.
"""

import asyncio
import errno
import socket
import time
from contextlib import contextmanager
from unittest.mock import patch

import aiohttp
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


@contextmanager
def _reject_so_keepalive():
    """Simulate the affected macOS kernel: SO_KEEPALIVE always fails EINVAL."""
    orig = socket.socket.setsockopt

    def flaky(self, level, opt, value):
        if level == socket.SOL_SOCKET and opt == socket.SO_KEEPALIVE:
            raise OSError(errno.EINVAL, "Invalid argument")
        return orig(self, level, opt, value)

    socket.socket.setsockopt = flaky  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.setsockopt = orig


@pytest.fixture(autouse=True)
def _clean_keepalive_guard():
    """Undo the process-global guard around every test.

    The guard patches ``aiohttp.tcp_helpers``/``web_protocol`` for the whole process, so a test
    that installs it hands every later test a working keepalive path — which hides a missing
    production call site instead of asserting on it.
    """
    from aiohttp import tcp_helpers, web_protocol

    from gateway.platforms import tcp_site

    helpers_orig = tcp_helpers.tcp_keepalive
    protocol_orig = web_protocol.tcp_keepalive
    tcp_site._keepalive_guard_installed = False
    try:
        yield
    finally:
        tcp_helpers.tcp_keepalive = helpers_orig
        web_protocol.tcp_keepalive = protocol_orig
        tcp_site._keepalive_guard_installed = False


class _EinvalSocket:
    def setsockopt(self, level, opt, value):
        raise OSError(errno.EINVAL, "Invalid argument")


class _EinvalTransport:
    def get_extra_info(self, name, default=None):
        if name == "socket":
            return _EinvalSocket()
        return default


class _StubRunner:
    """Host for the lifecycle mixin: the guard install must not need anything else from it."""

    def _platform_connect_timeout_secs(self, platform, *, initial: bool = False) -> float:
        return 0


class _SharedIngressAdapter:
    """Adapter that binds through ``shared_ingress.bind_listener`` (never ``start_tcp_site``)."""

    _shared_listener_profile = None
    _host = "127.0.0.1"
    _port = 0

    def __init__(self):
        self.runner = None
        self.port = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        from aiohttp import web

        from gateway.platforms.shared_ingress import bind_listener

        app = web.Application()

        async def health(request):
            return web.Response(text="ok")

        app.router.add_get("/health", health)
        self.runner = await bind_listener(self, app, "127.0.0.1", 0, "/health")
        self.port = sorted(self.runner.addresses)[0][1]
        return True

    async def disconnect(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()


class _StubUpstream:
    """Proxy upstream stub: ``/health`` is answered locally, so nothing forwards."""

    display_name = "stub-upstream"

    def is_authenticated(self) -> bool:
        return True


def _free_loopback_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_keepalive_guard_tolerates_einval():
    """The shared guard makes both aiohttp keepalive entry points best-effort."""
    from gateway.platforms import tcp_site

    tcp_site.ensure_tcp_keepalive_guard()

    from aiohttp import tcp_helpers, web_protocol

    # Must not raise even though every setsockopt(SO_KEEPALIVE) fails EINVAL.
    tcp_helpers.tcp_keepalive(_EinvalTransport())  # type: ignore[arg-type]
    web_protocol.tcp_keepalive(_EinvalTransport())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_webhook_serves_post_when_keepalive_rejected():
    """Real repro: a full webhook listener + POST while the kernel rejects SO_KEEPALIVE.

    The route filters on events so the POST is answered 200 without spawning
    an agent run — the assertion is purely that the accepted connection
    survives the keepalive tuning. Unfixed, the client never gets a response.
    The guard has to arrive through ``start_tcp_site`` itself: the process-global
    state is reset per test, so deleting that call site fails this test.
    """
    from gateway.platforms import tcp_site

    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "port": 0,
            "routes": {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x", "events": ["push"]}},
        },
    )
    adapter = WebhookAdapter(config)
    assert tcp_site._keepalive_guard_installed is False
    with _reject_so_keepalive():
        try:
            with patch.object(adapter, "_reload_dynamic_routes"):
                assert await adapter.connect() is True
            assert tcp_site._keepalive_guard_installed is True
            port = list(adapter._runner.addresses)[0][1]  # type: ignore[union-attr]
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/webhooks/r1", data=b"{}"
                ) as resp:
                    assert resp.status == 200
                    body = await resp.json()
                    assert body["status"] == "ignored"
        finally:
            await adapter.disconnect()


@pytest.mark.asyncio
async def test_runner_startup_covers_shared_ingress_listener():
    """Sibling bind path: a runner-connected adapter binding via ``bind_listener`` survives EINVAL.

    ``bind_listener`` builds its own ``web.TCPSite`` — the default port path for the multiplex
    SMS/LINE/Teams/BlueBubbles/Graph/WhatsApp/WeCom/Feishu adapters — and never calls
    ``start_tcp_site``; the guard reaches it because every adapter connect funnels through the
    lifecycle mixin.
    """
    from gateway.platforms import tcp_site
    from gateway.run_adapters import GatewayAdapterLifecycleMixin

    adapter = _SharedIngressAdapter()
    assert tcp_site._keepalive_guard_installed is False
    with _reject_so_keepalive():
        try:
            connected = await GatewayAdapterLifecycleMixin._connect_adapter_with_timeout(
                _StubRunner(), adapter, "webhook"
            )
            assert connected is True
            assert tcp_site._keepalive_guard_installed is True
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"http://127.0.0.1:{adapter.port}/health") as resp:
                    assert resp.status == 200
                    assert await resp.text() == "ok"
        finally:
            await adapter.disconnect()


@pytest.mark.asyncio
async def test_proxy_listener_survives_rejected_keepalive():
    """``hermes proxy start`` runs in its own process: its startup must install the guard too."""
    from gateway.platforms import tcp_site
    from hermes_cli.proxy import server as proxy_server

    port = _free_loopback_port()
    stop = asyncio.Event()
    assert tcp_site._keepalive_guard_installed is False
    with _reject_so_keepalive():
        task = asyncio.create_task(
            proxy_server.run_server(
                _StubUpstream(), host="127.0.0.1", port=port, shutdown_event=stop
            )
        )
        try:
            deadline = time.monotonic() + 10
            while True:  # readiness via a raw connect: unaffected by the accept-path guard
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        break
                except ConnectionRefusedError:
                    assert time.monotonic() < deadline, "proxy listener never bound"
                    await asyncio.sleep(0.05)
                except OSError:
                    break  # accepted then dropped: this is the regression we are here for
            timeout = aiohttp.ClientTimeout(total=5)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                        assert resp.status == 200
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                pytest.fail(f"proxy answered no request while SO_KEEPALIVE failed EINVAL: {exc!r}")
            assert tcp_site._keepalive_guard_installed is True
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=15)
