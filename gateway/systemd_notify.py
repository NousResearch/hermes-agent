"""systemd sd_notify protocol — pure-Python, no external deps.

Implements the subset of the systemd notification protocol that lets a
``Type=notify`` service tell systemd it's ready (``READY=1``), send
periodic watchdog pings (``WATCHDOG=1``), and announce shutdown
(``STOPPING=1``).

References:
* sd_notify(3) — https://www.freedesktop.org/software/systemd/man/sd_notify.html
* systemd.service(5) — ``Type=notify`` + ``WatchdogSec=`` semantics

Why no dependency
-----------------
The protocol is "write a newline-separated string to the AF_UNIX
datagram socket at ``$NOTIFY_SOCKET``." That's <40 lines of stdlib
sockets — adding ``sdnotify`` or ``systemd-python`` to ``pyproject.toml``
for that footprint isn't worth the dependency churn.

Opt-in by design
----------------
Every public helper here is a no-op unless ``$NOTIFY_SOCKET`` is set in
the environment — which only happens when systemd starts the process
under ``Type=notify``. Existing ``Type=simple`` deployments call into
this module and see nothing change.

To enable hang detection on a deployment:

1. Switch the unit to ``Type=notify`` + ``WatchdogSec=60s`` +
   ``Restart=on-watchdog`` (the latter only fires when the heartbeat
   stops, complementing ``Restart=on-failure`` which only catches
   non-zero exits).
2. The gateway calls :func:`notify_ready` after startup-complete and
   launches :func:`watchdog_heartbeat_task` if systemd advertised a
   ``$WATCHDOG_USEC`` interval. Heartbeat fires at half-interval per
   the man-page recommendation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Optional

logger = logging.getLogger(__name__)


# systemd + AF_UNIX is Linux-only. On Windows / non-POSIX platforms
# ``socket.AF_UNIX`` is absent entirely. Guard at module load so every
# helper below is a safe no-op on unsupported platforms without raising.
_HAS_AF_UNIX = hasattr(socket, "AF_UNIX")


def _notify_socket_path() -> Optional[str]:
    """Return the ``$NOTIFY_SOCKET`` path, or None if unset / unsupported.

    systemd advertises the notify socket via the ``NOTIFY_SOCKET`` env
    var. Two forms are documented:

    * Filesystem path, e.g. ``/run/systemd/notify``.
    * Linux-only abstract namespace, prefixed with ``@``; the kernel
      represents this with a leading NUL byte instead of the ``@``.

    Returns the path in the form expected by ``socket.connect()`` —
    abstract paths are converted from ``@/path`` to ``\\0/path``. Returns
    None when the env var is unset OR the platform lacks ``AF_UNIX``.
    """
    if not _HAS_AF_UNIX:
        return None
    path = os.environ.get("NOTIFY_SOCKET")
    if not path:
        return None
    if path.startswith("@"):
        return "\0" + path[1:]
    return path


def is_available() -> bool:
    """Return True when the process is running under systemd ``Type=notify``.

    Cheap, side-effect-free probe — checks only the env var, doesn't
    connect to the socket. Use this to decide whether to spawn the
    watchdog heartbeat task.
    """
    return _notify_socket_path() is not None


def notify(state: str) -> bool:
    """Send a single state line to systemd.

    Parameters
    ----------
    state : str
        One of the documented sd_notify keys, e.g. ``"READY=1"``,
        ``"WATCHDOG=1"``, ``"STOPPING=1"``, ``"STATUS=...whatever..."``.
        Multiple keys can be joined with newlines if desired.

    Returns
    -------
    bool
        True if the datagram was written to the notify socket. False
        when ``$NOTIFY_SOCKET`` is unset OR the send raised ``OSError``
        (socket gone, permissions, etc.). All failures are best-effort
        and never raise — this is a hot-path heartbeat function.
    """
    path = _notify_socket_path()
    if not path:
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(path)
            sock.sendall(state.encode("utf-8"))
        return True
    except OSError as exc:
        # systemd-notify failures should never crash the gateway. Log
        # at debug — operators get the journal entries from systemd
        # itself if the protocol actually breaks.
        logger.debug("sd_notify(%r) failed: %s", state, exc)
        return False


def notify_ready() -> bool:
    """Tell systemd the service is ready (``Type=notify`` startup complete)."""
    return notify("READY=1")


def notify_watchdog() -> bool:
    """Send a single watchdog keep-alive ping (``WATCHDOG=1``)."""
    return notify("WATCHDOG=1")


def notify_stopping() -> bool:
    """Tell systemd the service is starting orderly shutdown (``STOPPING=1``).

    Allows systemd to extend the shutdown timeout / understand the
    state transition rather than treating the process exit as a crash.
    """
    return notify("STOPPING=1")


def watchdog_usec() -> Optional[int]:
    """Return the watchdog interval in microseconds, or None.

    systemd exports ``$WATCHDOG_USEC`` when the unit has ``WatchdogSec=``
    set AND it's running under ``Type=notify``. The man page recommends
    pinging at half this interval to leave headroom for scheduling
    jitter.
    """
    raw = os.environ.get("WATCHDOG_USEC")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def watchdog_heartbeat_task() -> None:
    """Periodic ``WATCHDOG=1`` heartbeat at half the systemd interval.

    Runs forever until cancelled. No-op + immediate return when systemd
    didn't advertise a watchdog interval, so callers can spawn this
    unconditionally without checking ``$WATCHDOG_USEC`` first.

    Best-effort: socket-send errors are logged at debug and the loop
    continues. The loss of one ping is recoverable (next ping arrives
    within the half-interval); the loss of ALL pings means the process
    is hung, which is exactly what we want systemd to detect.
    """
    usec = watchdog_usec()
    if usec is None:
        return
    interval_s = max(1.0, usec / 1_000_000 / 2.0)
    logger.info(
        "systemd-notify: watchdog heartbeat starting (interval=%.1fs, half of WATCHDOG_USEC=%d)",
        interval_s,
        usec,
    )
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            logger.info("systemd-notify: watchdog heartbeat cancelled")
            raise
        notify_watchdog()
