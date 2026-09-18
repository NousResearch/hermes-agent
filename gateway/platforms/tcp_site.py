"""Bind an aiohttp ``TCPSite`` for the HTTP-serving adapters (webhook, api_server): exclusive on macOS,
yet able to rebind over a lingering TIME_WAIT socket right after a gateway restart."""

import asyncio
import errno
import logging
import socket
import sys
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})


def is_wildcard_host(host: Optional[str]) -> bool:
    """True for the dual-stack default (None/"") and the per-family wildcards."""
    return not host or host.strip() in _WILDCARD_HOSTS


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
