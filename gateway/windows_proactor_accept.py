"""Keep a Windows gateway listener open when one AcceptEx peer is dropped.

CPython's proactor ``_start_serving`` closes the listening socket on any
``OSError`` from ``accept()`` (``Lib/asyncio/proactor_events.py``).
``getpeername()`` raises WinError 10014 when the peer resets during AcceptEx,
so one dropped handshake becomes "Accept failed on a socket", the port stops
listening, and Desktop reports a WebSocket error before open while the gateway
process is still up.

``hermes serve`` avoids that loop with a selector event loop (#120164). The
gateway process cannot: subprocesses on Windows need the proactor. This guard
replaces only the accept callback, and only re-arms for a dropped peer.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types

logger = logging.getLogger(__name__)

# WSAEFAULT, WSAECONNABORTED, WSAECONNRESET. A reset during AcceptEx, not a
# dead listening socket. Other OSErrors still close the listener.
_TRANSIENT_WINERRORS = frozenset({10014, 10053, 10054})


def transient_proactor_accept_error(exc: BaseException) -> bool:
    """True when this accept failure is one dropped peer, not a dead listener."""
    winerror = getattr(exc, "winerror", None)
    return isinstance(winerror, int) and winerror in _TRANSIENT_WINERRORS


def _is_proactor_loop(loop: object) -> bool:
    try:
        from asyncio.proactor_events import BaseProactorEventLoop
    except ImportError:
        return False
    return isinstance(loop, BaseProactorEventLoop)


def _guarded_start_serving(
    self,
    protocol_factory,
    sock,
    sslcontext=None,
    server=None,
    backlog=100,
    ssl_handshake_timeout=None,
    ssl_shutdown_timeout=None,
):
    """``BaseProactorEventLoop._start_serving``, except a dropped peer re-arms.

    ``backlog`` is part of the loop method's signature. The socket is already
    listening when uvicorn hands it over.
    """
    del backlog

    def _arm():
        future = self._proactor.accept(sock)
        self._accept_futures[sock.fileno()] = future
        future.add_done_callback(_on_done)
        return future

    def _handoff(conn, addr):
        protocol = protocol_factory()
        extra = {"peername": addr}
        if sslcontext is not None:
            self._make_ssl_transport(
                conn,
                protocol,
                sslcontext,
                server_side=True,
                extra=extra,
                server=server,
                ssl_handshake_timeout=ssl_handshake_timeout,
                ssl_shutdown_timeout=ssl_shutdown_timeout,
            )
            return
        self._make_socket_transport(conn, protocol, extra=extra, server=server)

    def _fail(exc: BaseException) -> None:
        if sock.fileno() == -1:
            return
        if transient_proactor_accept_error(exc):
            logger.warning(
                "Windows accept dropped a peer (%s); listening socket stays open",
                exc,
            )
            try:
                _arm()
            except OSError as retry_exc:
                exc = retry_exc
            else:
                return
            if sock.fileno() == -1:
                return
        self.call_exception_handler(
            {
                "message": "Accept failed on a socket",
                "exception": exc,
                "socket": sock,
            }
        )
        sock.close()

    def _on_done(future=None):
        if future is not None:
            try:
                conn, addr = future.result()
            except OSError as exc:
                _fail(exc)
                return
            except asyncio.CancelledError:
                sock.close()
                return
            _handoff(conn, addr)
        if self.is_closed():
            return
        try:
            _arm()
        except OSError as exc:
            _fail(exc)

    self.call_soon(_on_done)


_guarded_start_serving._hermes_accept_guard = True  # type: ignore[attr-defined]


def install_proactor_accept_guard(loop, *, enabled: bool | None = None) -> bool:
    """Install the guard on a Windows proactor loop. No-op elsewhere.

    Returns True only when this call installed it. A second call is a no-op
    so two listeners in one process do not wrap twice.
    """
    if enabled is None:
        enabled = sys.platform == "win32"
    if not enabled or not _is_proactor_loop(loop):
        return False
    current = getattr(loop, "_start_serving", None)
    if getattr(current, "_hermes_accept_guard", False):
        return False
    loop._start_serving = types.MethodType(_guarded_start_serving, loop)
    return True
