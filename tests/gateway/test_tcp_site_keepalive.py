"""An OS that rejects SO_KEEPALIVE must not cost us the connection (#123327).

aiohttp calls ``tcp_keepalive`` unguarded for every accepted connection
(``RequestHandler.connection_made``), and on macOS an external-interface bind made that call
raise ``setsockopt SO_KEEPALIVE: invalid argument`` (errno 22). The exception escaped the
accept path, so the server closed every connection and webhook deliveries failed 100%.

These tests drive a real aiohttp site with the failure injected at the same call site, because
the whole bug is in that sequencing — a unit test of the shim alone would pass whether or not
it actually kept the connection alive.
"""

from __future__ import annotations

import asyncio
import errno
import socket
import sys

import pytest

aiohttp = pytest.importorskip("aiohttp", reason="aiohttp not installed")
from aiohttp import web  # noqa: E402

from gateway.platforms import tcp_site  # noqa: E402


def _reject_keepalive(monkeypatch) -> None:
    """Make SO_KEEPALIVE raise EINVAL, the way the affected macOS host does."""
    real = socket.socket.setsockopt

    def _setsockopt(self, level, opt, value):
        if opt == socket.SO_KEEPALIVE:
            raise OSError(errno.EINVAL, "setsockopt SO_KEEPALIVE: invalid argument")
        return real(self, level, opt, value)

    monkeypatch.setattr(socket.socket, "setsockopt", _setsockopt)


async def _get(port: int, path: str = "/health") -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://127.0.0.1:{port}{path}", timeout=aiohttp.ClientTimeout(total=5)
        ) as response:
            return await response.text()


def _serve(tmp_path, monkeypatch, *, install: bool):
    """Serve one route on a real aiohttp site and return the body it answered with."""

    async def _run() -> str:
        from aiohttp import web_protocol

        # Start from stock aiohttp for every test, then apply the shim under test.
        monkeypatch.setattr(web_protocol, "tcp_keepalive", _STOCK_KEEPALIVE, raising=False)
        if install:
            assert tcp_site.install_tolerant_tcp_keepalive() is True
        app = web.Application()
        app.router.add_get("/health", lambda _r: web.Response(text="OK"))
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await _get(port)
        finally:
            await runner.cleanup()

    return asyncio.run(_run())


def _stock_keepalive(transport) -> None:
    """aiohttp's own implementation: no OSError guard (pinned 3.14.3)."""
    sock = transport.get_extra_info("socket")
    if sock is not None:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


_STOCK_KEEPALIVE = _stock_keepalive


class TestKeepaliveRejectionDoesNotCloseConnections:
    def test_stock_aiohttp_loses_the_connection(self, tmp_path, monkeypatch):
        """Pins the premise: unguarded, the rejection takes the request down.

        Without this, a pass in the next test would not prove the shim is what fixed it.
        """
        from aiohttp import web_protocol

        monkeypatch.setattr(web_protocol, "tcp_keepalive", _STOCK_KEEPALIVE, raising=False)
        _reject_keepalive(monkeypatch)

        async def _run() -> None:
            app = web.Application()
            app.router.add_get("/health", lambda _r: web.Response(text="OK"))
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                with pytest.raises(Exception):
                    await _get(port)
            finally:
                await runner.cleanup()

        asyncio.run(_run())

    def test_the_shim_keeps_the_connection_alive(self, tmp_path, monkeypatch):
        """The fix: the request is served even though SO_KEEPALIVE was rejected."""
        _reject_keepalive(monkeypatch)
        assert _serve(tmp_path, monkeypatch, install=True) == "OK"

    def test_a_healthy_platform_is_unaffected(self, tmp_path, monkeypatch):
        """Where SO_KEEPALIVE works, the request is served either way."""
        assert _serve(tmp_path, monkeypatch, install=True) == "OK"


class TestInstallIsSafe:
    def test_install_is_idempotent(self):
        from aiohttp import web_protocol

        original = web_protocol.tcp_keepalive
        try:
            assert tcp_site.install_tolerant_tcp_keepalive() is True
            first = web_protocol.tcp_keepalive
            assert tcp_site.install_tolerant_tcp_keepalive() is True
            assert web_protocol.tcp_keepalive is first
        finally:
            web_protocol.tcp_keepalive = original

    def test_missing_aiohttp_internals_degrade_instead_of_raising(self, monkeypatch):
        """If aiohttp moves the symbol, this must be a no-op — never a crash at bind time."""
        from aiohttp import web_protocol

        monkeypatch.delattr(web_protocol, "tcp_keepalive", raising=False)
        assert tcp_site.install_tolerant_tcp_keepalive() is False

    def test_the_shim_swallows_only_oserror(self):
        """A transport with no socket, and a non-OSError, are both handled."""
        transport = type("T", (), {"get_extra_info": lambda self, _k: None})()
        tcp_site._tolerant_tcp_keepalive(transport)  # no socket: returns

        class _BadSocket:
            def setsockopt(self, *_a):
                raise RuntimeError("not an OSError")

        bad = type("T", (), {"get_extra_info": lambda self, _k: _BadSocket()})()
        with pytest.raises(RuntimeError):
            tcp_site._tolerant_tcp_keepalive(bad)
