"""Bind an aiohttp ``TCPSite`` for the HTTP-serving adapters (webhook, api_server): exclusive on macOS,
yet able to rebind over a lingering TIME_WAIT socket right after a gateway restart."""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aiohttp import web

from gateway.platforms.shared_ingress import is_wildcard_host

logger = logging.getLogger(__name__)


def _tolerant_tcp_keepalive(transport) -> None:
    """``SO_KEEPALIVE`` on an accepted socket, tolerating the OS saying no.

    aiohttp calls this unguarded for every accepted connection
    (``RequestHandler.connection_made``), so a platform that rejects the option takes the whole
    connection down with it: on macOS an external-interface bind raised
    ``setsockopt SO_KEEPALIVE: invalid argument`` (errno 22) and the server closed every
    connection, so webhook deliveries failed 100% (#123327). The neighbouring ``tcp_nodelay``
    wraps the identical call in ``suppress(OSError)``; this restores that symmetry.

    Keepalive is an optimisation — a dead peer is still detected by the request timeout — so
    losing it is strictly better than refusing the connection.
    """
    sock = transport.get_extra_info("socket")
    if sock is None:
        return
    with suppress(OSError):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


def install_tolerant_tcp_keepalive() -> bool:
    """Make aiohttp's per-connection keepalive non-fatal; True when the shim was installed.

    Applied where these adapters bind rather than at import: the behaviour only matters once a
    site is being served, and aiohttp offers no supported way to disable it (there is no
    ``TCPSite(keepalive_socket=...)`` in 3.14). Idempotent, and a no-op if aiohttp's internals
    move — the symptom is a connection reset, not a crash, so degrading to stock aiohttp is safe.
    """
    try:
        from aiohttp import web_protocol
    except Exception:
        return False
    if getattr(web_protocol, "tcp_keepalive", None) is _tolerant_tcp_keepalive:
        return True
    if not hasattr(web_protocol, "tcp_keepalive"):
        return False
    with suppress(Exception):
        web_protocol.tcp_keepalive = _tolerant_tcp_keepalive
    return getattr(web_protocol, "tcp_keepalive", None) is _tolerant_tcp_keepalive


def has_live_listener(host: str, port: int) -> bool:
    """Blocking probe: True when something accepts connections on ``host:port``. Refused = nobody listens;
    any other failure (timeout, unroutable) is treated as live so the caller stays exclusive."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except ConnectionRefusedError:
        return False
    except OSError:
        return True


async def start_tcp_site(runner: web.BaseRunner, host: Optional[str], port: int, *, log_tag: str) -> web.TCPSite:
    """Bind ``host:port`` on ``runner`` and return the started site; raises OSError when unavailable.

    SO_REUSEADDR: on macOS (BSD) two wildcard/specific sockets can silently split traffic while
    both report success → disable. On Linux it only permits rebinding past TIME_WAIT (a quick
    restart would otherwise fail to bind for ~60s) → keep the default.

    The macOS exclusive bind also refuses the port while a server-side TIME_WAIT socket lingers
    (2*MSL = 30s after the previous gateway closed a connection first — its shutdown, or any
    ``Connection: close`` request), so a ``/restart`` re-binding within seconds failed with
    EADDRINUSE although nobody was listening. Preventing the TIME_WAIT at close time (SO_LINGER 0)
    would reset in-flight senders and misses the per-request case, hence the bind-side retry: for an
    explicit host, ``has_live_listener`` refused proves the address is free (the kernel still
    rejects an exact duplicate even with SO_REUSEADDR, and a foreign wildcard listener answers the
    probe), so one retry with reuse_address=True is safe. A wildcard host keeps the strict path: a
    foreign listener on a non-loopback interface could not be probed, so it must keep winning."""
    from aiohttp import web

    # Applied here, at bind time: every accepted connection goes through aiohttp's unguarded
    # keepalive, so an OS that rejects the option would otherwise close each one (#123327).
    install_tolerant_tcp_keepalive()

    exclusive = sys.platform == "darwin"
    site = web.TCPSite(runner, host, port, reuse_address=False if exclusive else None)
    try:
        await site.start()
    except OSError as exc:
        if not exclusive or exc.errno != errno.EADDRINUSE or is_wildcard_host(host):
            raise
        if await asyncio.to_thread(has_live_listener, host, port):
            raise
        await site.stop()  # aiohttp registers a site before binding: drop the dead one from the runner
        logger.info("[%s] %s:%d busy without a live listener (TIME_WAIT from the previous gateway); "
                    "rebinding with SO_REUSEADDR", log_tag, host, port)
        site = web.TCPSite(runner, host, port, reuse_address=True)
        await site.start()
    return site
