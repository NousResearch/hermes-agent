"""Windows live E2E probes for the CLOSE-WAIT getUpdates reconnect fix (#87057).

These probes run ONLY on a real Windows runner (the on-demand
``windows-venv-e2e.yml`` lane, fired by pushes to ``wine2e/**`` branches).
They exercise the REAL PTB ``HTTPXRequest`` transport against a live local
HTTP server whose sockets are genuinely half-closed by the server side, so
the client's pooled connection sits in CLOSE-WAIT exactly the way a dropped
``api.telegram.org`` long-poll does on Windows (selector/proactor overlapped
I/O never surfaces the peer close).

Scenario pinned by #87057: after a transient network error the reconnect
ladder calls ``_drain_polling_connections()``. On Windows the close of the
CLOSE-WAIT socket can wedge; PTB's ``HTTPXRequest.initialize()`` only builds
a fresh client when ``client.is_closed`` is true, so an abandoned close left
``start_polling()`` on the same dead socket and the gateway went silently
deaf. The fix swaps in a fresh HTTP client when the drain times out.

Probes:
1. ``test_drain_recovers_after_server_half_close_live`` — a real pooled
   keep-alive connection is half-closed by the server (CLOSE-WAIT on the
   client). The drain must complete within its bound and the next real
   request must succeed on a NEW TCP connection.
2. ``test_drain_bounded_and_functional_when_close_wedges_live`` — the real
   request's ``shutdown()`` is replaced with one that hangs forever
   (deterministic stand-in for the observed proactor CLOSE-WAIT close hang).
   The drain must return within the wall-clock bound, replace the wedged
   client, and the replacement must complete a real HTTP round-trip.
"""

import asyncio
import json
import sys
import time

import pytest

# The gateway conftest installs a MagicMock ``telegram`` package when the
# real library has not been imported yet. This probe exercises the REAL PTB
# HTTPXRequest against a live socket server, so evict any mock before the
# real import. The lane installs the messaging extra, so real PTB is present.
# Gated to win32: on other platforms these tests are skipped and evicting the
# shared mock here would poison later test modules in the same session.
if sys.platform == "win32":
    _tg = sys.modules.get("telegram")
    if _tg is not None and not hasattr(_tg, "__file__"):
        for _name in [
            m for m in list(sys.modules) if m == "telegram" or m.startswith("telegram.")
        ]:
            del sys.modules[_name]
        # The adapter module may have bound mock names at import time — reload
        # it against the real library.
        for _name in [
            m for m in list(sys.modules) if m.startswith("plugins.platforms.telegram")
        ]:
            del sys.modules[_name]

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-only live probe: CLOSE-WAIT reconnect behavior (#87057)",
    ),
]


class _LiveBotApiServer:
    """Minimal live HTTP/1.1 server that can half-close its connections.

    Speaks just enough HTTP for PTB's ``HTTPXRequest.do_request`` POSTs.
    Every accepted TCP connection is tracked so probes can assert whether a
    request arrived on a fresh connection or reused a pooled one, and the
    server can actively half-close (FIN) all live connections to park the
    client side in CLOSE-WAIT.
    """

    def __init__(self):
        self.server = None
        self.port = None
        self.connections_accepted = 0
        self.requests_served = 0
        self._writers = []

    async def start(self):
        self.server = await asyncio.start_server(
            self._handle, host="127.0.0.1", port=0
        )
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self):
        for w in self._writers:
            try:
                w.close()
            except Exception:
                pass
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/botTEST/getUpdates"

    async def half_close_all(self):
        """Send FIN on every live connection -> client side goes CLOSE-WAIT."""
        for w in self._writers:
            try:
                w.write_eof()
            except Exception:
                pass
        # Give the client's TCP stack a moment to process the FIN.
        await asyncio.sleep(0.2)

    async def _handle(self, reader, writer):
        self.connections_accepted += 1
        self._writers.append(writer)
        try:
            while True:
                # Read request head.
                head = await reader.readuntil(b"\r\n\r\n")
                headers = head.decode("latin1").lower()
                length = 0
                for line in headers.split("\r\n"):
                    if line.startswith("content-length:"):
                        length = int(line.split(":", 1)[1].strip())
                if length:
                    await reader.readexactly(length)
                self.requests_served += 1
                body = json.dumps({"ok": True, "result": []}).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                    b"Connection: keep-alive\r\n"
                    b"\r\n" + body
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass


def _make_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return TelegramAdapter(PlatformConfig(enabled=True, token="123456:TEST"))


def _diag(server, label, extra=""):
    print(
        f"[closewait-probe] {label}: connections={server.connections_accepted} "
        f"requests={server.requests_served} platform={sys.platform} {extra}",
        flush=True,
    )


async def test_drain_recovers_after_server_half_close_live(monkeypatch):
    """Drain must retire a real CLOSE-WAIT pooled connection within bound."""
    from telegram.request import HTTPXRequest
    from unittest.mock import MagicMock

    import plugins.platforms.telegram.adapter as tg_adapter

    server = _LiveBotApiServer()
    await server.start()
    try:
        polling_req = HTTPXRequest(
            connection_pool_size=1,
            read_timeout=5.0,
            connect_timeout=5.0,
            pool_timeout=5.0,
        )
        await polling_req.initialize()

        # Real round-trip 1: connection enters the keep-alive pool.
        code, payload = await polling_req.do_request(server.url, "POST")
        assert code == 200 and b'"ok"' in payload
        assert server.connections_accepted == 1
        _diag(server, "after first round-trip")

        # Server half-closes: the pooled client connection is now CLOSE-WAIT.
        await server.half_close_all()
        _diag(server, "after server half-close (client socket CLOSE-WAIT)")

        adapter = _make_adapter()
        mock_app = MagicMock()
        mock_app.bot._request = (polling_req, MagicMock())
        adapter._app = mock_app

        monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 5.0)
        started = time.monotonic()
        await adapter._drain_polling_connections()
        elapsed = time.monotonic() - started
        _diag(server, "after drain", f"elapsed={elapsed:.2f}s")
        assert elapsed < 12.0, (
            f"drain must be bounded even with a CLOSE-WAIT socket, took {elapsed:.2f}s"
        )

        # The reconnect path must be LIVE: a new real request succeeds on a
        # fresh TCP connection, not the dead pooled one.
        before = server.connections_accepted
        code, payload = await polling_req.do_request(server.url, "POST")
        assert code == 200 and b'"ok"' in payload
        assert server.connections_accepted > before, (
            "post-drain getUpdates must use a NEW connection, not the "
            "CLOSE-WAIT one"
        )
        _diag(server, "after post-drain round-trip")
        await polling_req.shutdown()
    finally:
        await server.stop()


async def test_drain_bounded_and_functional_when_close_wedges_live(monkeypatch):
    """A wedged close must not hang the drain; the swapped client must work.

    Deterministic stand-in for the Windows proactor hang: the real
    HTTPXRequest's shutdown() is replaced with a coroutine that never
    returns (what a CLOSE-WAIT close did in #87057). The drain must
    (a) return within the wall-clock bound, (b) swap in a fresh client
    because initialize() would otherwise no-op on is_closed=False, and
    (c) leave the polling request able to complete a REAL round-trip.
    """
    from telegram.request import HTTPXRequest
    from unittest.mock import MagicMock

    import plugins.platforms.telegram.adapter as tg_adapter

    server = _LiveBotApiServer()
    await server.start()
    try:
        polling_req = HTTPXRequest(
            connection_pool_size=1,
            read_timeout=5.0,
            connect_timeout=5.0,
            pool_timeout=5.0,
        )
        await polling_req.initialize()
        code, _ = await polling_req.do_request(server.url, "POST")
        assert code == 200
        old_client = polling_req._client  # noqa: SLF001
        _diag(server, "wedge-probe: after first round-trip")

        async def _wedged_shutdown():
            await asyncio.Event().wait()

        monkeypatch.setattr(polling_req, "shutdown", _wedged_shutdown)
        monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 1.0)

        adapter = _make_adapter()
        mock_app = MagicMock()
        mock_app.bot._request = (polling_req, MagicMock())
        adapter._app = mock_app

        started = time.monotonic()
        await asyncio.wait_for(adapter._drain_polling_connections(), timeout=30.0)
        elapsed = time.monotonic() - started
        _diag(server, "wedge-probe: after drain", f"elapsed={elapsed:.2f}s")
        assert elapsed < 10.0, (
            f"drain with a wedged shutdown must stay bounded, took {elapsed:.2f}s"
        )

        new_client = polling_req._client  # noqa: SLF001
        assert new_client is not old_client, (
            "drain must swap in a fresh HTTP client when shutdown wedges "
            "(initialize() no-ops while is_closed is False)"
        )

        # The replacement client must be genuinely functional: real request,
        # real socket, live server.
        before = server.connections_accepted
        code, payload = await polling_req.do_request(server.url, "POST")
        assert code == 200 and b'"ok"' in payload
        assert server.connections_accepted > before
        _diag(server, "wedge-probe: after post-swap round-trip")

        # Bounded cleanup of the orphaned client must not linger forever.
        deadline = time.monotonic() + 10.0
        while adapter._background_tasks and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        assert not adapter._background_tasks, (
            "orphaned-client cleanup task must complete/abandon within bound"
        )
    finally:
        await server.stop()
