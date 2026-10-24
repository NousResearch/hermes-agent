"""Host identity for Kanban claim locks (``<host>:<pid>``).

Every host-local decision in the dispatcher (extend a live worker's expired claim, signal a
reclaimed worker, reap a dead one, enforce max runtime) asks "was this lock issued here?".
Answering with a bare ``socket.gethostname()`` comparison breaks on macOS: with no ``HostName``
set, ``gethostname()`` follows the network (``Mac``, ``Mac.localdomain``, ``<name>.local``), so
after a flip a live local worker's lock looks remote. It is neither extended nor signalled, its
claim is reclaimed, and a duplicate worker spawns beside it (#73277).

On macOS new locks carry the Bonjour name ``<LocalHostName>.local``: unique on the LAN, changed
only by a manual rename, and equal to what ``gethostname()`` returns on an unconfigured Mac with a
healthy network, so locks from processes that predate this module still match. A lock whose host
equals the live ``gethostname()`` also counts as local, which covers those older locks and Macs
with an explicit ``HostName``. Other platforms keep ``gethostname()``.
"""

from __future__ import annotations

import os
import socket

from hermes_platform.host.facts import bonjour_hostname


def _live_hostname() -> str:
    try:
        return socket.gethostname() or "unknown"
    except OSError:
        return "unknown"


def stable_host_id() -> str:
    """The host part this process writes into new claim locks."""
    return bonjour_hostname() or _live_hostname()


def claimer_id() -> str:
    """``<stable host id>:<pid>`` for this process."""
    return f"{stable_host_id()}:{os.getpid()}"


def is_local_lock(lock: object) -> bool:
    """True when ``lock`` (``<host>:<pid>``) was issued on this machine.

    Same contract as the old ``lock.startswith(gethostname() + ":")``, but any of this machine's
    names match, case-insensitively (host names are)."""
    if not lock:
        return False
    host, sep, _pid = str(lock).partition(":")
    if not sep:
        return False
    host = host.casefold()
    return host in (stable_host_id().casefold(), _live_hostname().casefold())
