"""Regression tests for the api_server restart-handoff bind path.

Context (observed 2026-08-09 .. 2026-08-21, 14 occurrences in gateway.log):
restarting a *healthy* gateway killed the API server every single time.

    09:27:43  [Api_Server] API server stopped      <- outgoing listener closed
    09:28:01  Could not bind 127.0.0.1:8642: [Errno 48]
    09:28:14  api_server failed to connect         <- fatal, non-retryable
    09:28:14  Gateway running with 2 platform(s)   <- telegram/slack fine,
                                                      dashboard dead until the
                                                      operator restarted twice

Mechanism: the outgoing api_server's *server-side* connections linger in
TIME_WAIT on the port for 2*MSL (30s on macOS, net.inet.tcp.msl=15000), and
SO_REUSEADDR was disabled on darwin, so the replacement — which launchd starts
~11-18s later — could never bind.

``TestBindMechanics.test_immediate_rebind_after_disconnect`` did not catch this
because it binds and unbinds without ever accepting a client connection: with
no connections there is no TIME_WAIT, so the rebind succeeded even while the
real restart path was failing. These tests create the TIME_WAIT state
explicitly.
"""

import asyncio
import socket
import time

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _port_has_live_listener,
)


_KEY = "sk-test-strong-key-0123456789"


def _make_adapter(port: int) -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": port, "key": _KEY},
        )
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _leave_time_wait_on(port: int) -> None:
    """Close a served connection so the *server* side enters TIME_WAIT."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)
    cli = socket.socket()
    cli.connect(("127.0.0.1", port))
    conn, _ = srv.accept()
    conn.close()  # server closes first -> TIME_WAIT belongs to local port
    srv.close()
    cli.close()


@pytest.mark.asyncio
async def test_binds_despite_previous_listeners_time_wait():
    """The 08-21 failure: TIME_WAIT on the port must not block the handoff."""
    port = _free_port()
    _leave_time_wait_on(port)

    adapter = _make_adapter(port)
    try:
        assert await adapter.connect() is True
        assert adapter.has_fatal_error is False
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_live_listener_still_fails_fast_and_non_retryable():
    """A genuine conflict must not be softened into a 20s retry loop.

    #52132's protection depends on EADDRINUSE staying non-retryable, and the
    startup connect budget in gateway.run is 30s — a real conflict has to be
    reported well inside it.
    """
    port = _free_port()
    holder = _make_adapter(port)
    assert await holder.connect() is True
    try:
        loser = _make_adapter(port)
        started = time.monotonic()
        assert await loser.connect() is False
        elapsed = time.monotonic() - started

        assert elapsed < 5.0, f"real conflict took {elapsed:.1f}s to report"
        assert loser.has_fatal_error is True
        assert loser.fatal_error_code == "api_server_port_in_use"
        assert loser.fatal_error_retryable is False
    finally:
        await holder.disconnect()


@pytest.mark.asyncio
async def test_live_listener_probe_distinguishes_the_two_causes():
    port = _free_port()

    # Nothing bound at all.
    assert await _port_has_live_listener("127.0.0.1", port) is False

    # Only TIME_WAIT — no listener, but bind() would still say EADDRINUSE.
    _leave_time_wait_on(port)
    assert await _port_has_live_listener("127.0.0.1", port) is False

    # A real listener.
    adapter = _make_adapter(port)
    assert await adapter.connect() is True
    try:
        assert await _port_has_live_listener("127.0.0.1", port) is True
    finally:
        await adapter.disconnect()
