"""systemd sd_notify watchdog gated on Slack socket health.

When the gateway runs as a systemd --user service with ``WatchdogSec`` set,
systemd expects the process to call ``sd_notify(WATCHDOG=1)`` periodically.
This module implements that heartbeat, but **only pets the watchdog when
the Slack adapter reports a recent successful event** — if the Slack socket
has been dead for longer than the healthy window, we stop petting and
systemd force-restarts the unit after ``WatchdogSec`` elapses.

This catches the class of failure where the process is alive but blind to
Slack (the June 30–July 2 outage) — the in-process reconnect logic cannot
help if the event loop itself is stalled or the adapter is in a degraded
state that still looks "alive" to its own internal probes.

The feature is entirely opt-in via the ``NOTIFY_SOCKET`` env var: when not
running under systemd, the module no-ops silently.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default: consider Slack healthy if the last event was within 60s.
# WatchdogSec in the service file is 300s (5 min), so the watchdog gets
# ~5 min of missed heartbeats before systemd force-restarts — plenty of
# time for a transient Slack hiccup to resolve without a restart.
_DEFAULT_HEALTHY_WINDOW_S: float = 60.0
_HEARTBEAT_INTERVAL_S: float = 30.0


def _send_sd_notify(message: str) -> bool:
    """Send a raw sd_notify message via the NOTIFY_SOCKET.

    Returns True if sent, False if NOTIFY_SOCKET is not set or the send
    failed. Uses a datagram socket (the sd_notify protocol) — each message
    is one datagram.
    """
    notify_socket = os.environ.get("NOTIFY_SOCKET", "")
    if not notify_socket:
        return False

    # systemd may give us an abstract socket (leading @) or a path.
    # For abstract sockets, replace the leading @ with a null byte.
    if notify_socket.startswith("@"):
        sock_addr = "\0" + notify_socket[1:]
    else:
        sock_addr = notify_socket

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(sock_addr)
            sock.sendall(message.encode("utf-8"))
        return True
    except (OSError, socket.error) as exc:
        logger.debug("sd_notify send failed: %s", exc, exc_info=True)
        return False


def is_watchdog_active() -> bool:
    """Return True if we're running under systemd with NOTIFY_SOCKET set."""
    return bool(os.environ.get("NOTIFY_SOCKET", ""))


def _read_last_event_ts(heartbeat_path: Path) -> Optional[float]:
    """Read the last Slack event timestamp from the heartbeat file.

    Returns None if the file doesn't exist or can't be parsed.
    """
    try:
        import json

        raw = heartbeat_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        data = json.loads(raw)
        ts = data.get("last_event_ts")
        if ts is not None:
            return float(ts)
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
        pass
    try:
        return os.path.getmtime(str(heartbeat_path))
    except OSError:
        return None


def is_slack_healthy(
    heartbeat_path: Path,
    healthy_window_s: float = _DEFAULT_HEALTHY_WINDOW_S,
) -> bool:
    """Check whether the Slack adapter has received a recent event.

    Reads the heartbeat file written by the Slack adapter on every incoming
    event. If the last event was within ``healthy_window_s``, the socket
    is considered healthy.
    """
    ts = _read_last_event_ts(heartbeat_path)
    if ts is None:
        # No heartbeat file yet — could be during startup before the first
        # Slack event. We treat this as healthy to avoid a false restart
        # during the boot grace period. The external health-check timer
        # (with its 5-min OnBootSec) is the backstop for this case.
        return True
    age = time.time() - ts
    return age <= healthy_window_s


class WatchdogHeartbeat:
    """Periodic sd_notify heartbeat gated on Slack socket health.

    Usage (in ``start_gateway`` or ``GatewayRunner.start``)::

        from gateway.watchdog import WatchdogHeartbeat
        watchdog = WatchdogHeartbeat(heartbeat_path)
        await watchdog.start()  # spawns the async heartbeat task
        # ...
        await watchdog.stop()   # cancels the task on shutdown
    """

    def __init__(
        self,
        heartbeat_path: Path,
        *,
        interval_s: float = _HEARTBEAT_INTERVAL_S,
        healthy_window_s: float = _DEFAULT_HEALTHY_WINDOW_S,
    ) -> None:
        self._heartbeat_path = heartbeat_path
        self._interval_s = interval_s
        self._healthy_window_s = healthy_window_s
        self._task: Optional[asyncio.Task[None]] = None
        self._stopped = False

    def start(self) -> None:
        """Start the periodic heartbeat task (if NOTIFY_SOCKET is set)."""
        if not is_watchdog_active():
            logger.debug("Watchdog not active (NOTIFY_SOCKET not set) — skipping")
            return
        if self._task is not None and not self._task.done():
            return
        logger.info(
            "systemd watchdog heartbeat started (interval=%.0fs, healthy_window=%.0fs)",
            self._interval_s,
            self._healthy_window_s,
        )
        self._task = asyncio.create_task(self._loop(), name="systemd-watchdog")

    async def stop(self) -> None:
        """Cancel the heartbeat task."""
        self._stopped = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        """Heartbeat loop: pet the watchdog only when Slack is healthy."""
        while not self._stopped:
            try:
                await asyncio.sleep(self._interval_s)
                if self._stopped:
                    break
                if is_slack_healthy(self._heartbeat_path, self._healthy_window_s):
                    _send_sd_notify("WATCHDOG=1")
                else:
                    logger.warning(
                        "Slack socket unhealthy (last event > %.0fs ago) — "
                        "not petting systemd watchdog; systemd will restart "
                        "after WatchdogSec elapses",
                        self._healthy_window_s,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover — defensive
                logger.debug(
                    "Watchdog heartbeat iteration failed; continuing",
                    exc_info=True,
                )
