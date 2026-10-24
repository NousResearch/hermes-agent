"""Minimal, optional systemd ``sd_notify`` support for the gateway."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import socket


def _notify_socket() -> str:
    return os.environ.get("NOTIFY_SOCKET", "").strip() if hasattr(socket, "AF_UNIX") else ""


def notify(message: str) -> bool:
    """Send an sd_notify datagram if systemd configured it; failures never block gateway startup."""
    if not (address := _notify_socket()) or not isinstance(message, str) or not message:
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.setblocking(False)  # a full receiver buffer must not stall the event loop
            # systemd's ``@abstract`` notation -> Python's leading-NUL address form
            sender.connect("\0" + address[1:] if address.startswith("@") else address)
            sender.send(message.encode("utf-8"))
        return True
    except (OSError, UnicodeError, ValueError):
        return False


def watchdog_interval_seconds() -> float | None:
    try:
        interval = float(os.environ.get("WATCHDOG_USEC", "") if _notify_socket() else "") / 1e6
    except (TypeError, ValueError):
        return None
    return interval if math.isfinite(interval) and interval > 0 else None


class SystemdWatchdog:
    """Feed systemd while the asyncio event loop continues to make progress.

    A SINGLE late tick must not condemn the gateway. The loop legitimately stalls for
    tens of seconds during MCP discovery, long cron turns, and adapter reconnects;
    killing the service on one late tick turns a transient stall into a restart loop
    (#t_b3e3f2cb). So the watchdog now requires SUSTAINED lag: consecutive late ticks
    must add up to ``max_lag_seconds`` of cumulative lateness before the watchdog stops
    feeding systemd and lets systemd reclaim the unit.

    The wedge this is meant to catch is persistent, not momentary: on 2026-10-23 the
    gateway's event loop froze for 27 minutes inside the inbound image/vision path and
    the in-process loop-liveness watchdog never fired, because a GIL-holding C call
    starves that watchdog's probe thread. systemd's timer lives in the systemd manager
    process and never touches the gateway's GIL, so it is the one backstop that still
    works under that failure mode.
    """

    def __init__(self, *, config_enabled: bool = True, lag_tolerance_seconds: float | None = None,
                 max_lag_seconds: float | None = None):
        self._config_enabled = bool(config_enabled)
        self.interval_seconds = watchdog_interval_seconds()
        self._lag_tolerance_seconds = lag_tolerance_seconds
        self._max_lag_seconds = max_lag_seconds
        self._task: asyncio.Task[None] | None = None
        self._unhealthy = self._stopping = self._stopping_notified = False
        self._cumulative_lag = 0.0

    enabled = property(lambda self: self._config_enabled and self.interval_seconds is not None)
    unhealthy = property(lambda self: self._unhealthy)
    task = property(lambda self: self._task)

    def _lag_tolerance(self) -> float:
        default = max(0.1, (self.interval_seconds or 0.0) * 0.25)
        with contextlib.suppress(TypeError, ValueError):
            value = float(self._lag_tolerance_seconds)
            return max(0.0, value) if math.isfinite(value) else default
        return default

    def _max_lag(self) -> float:
        """Cumulative lateness that condemns the gateway.

        Defaults to the systemd watchdog interval itself: a loop that stays behind by
        more than a whole watchdog period, cumulatively, is wedged rather than busy.
        """
        default = max(1.0, float(self.interval_seconds or 0.0))
        value = self._max_lag_seconds
        if value is None:
            return default
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric if math.isfinite(numeric) and numeric > 0 else default

    def start(self) -> bool:
        if not self.enabled:
            return False
        if self._task is not None and not self._task.done():
            return True
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._stopping = self._unhealthy = self._stopping_notified = False
        self._task = asyncio.create_task(self._run(), name="hermes-systemd-watchdog")
        return True

    def ready(self, status: str = "Gateway running") -> bool:
        safe_status = str(status or "Gateway running").replace("\n", " ")
        return self.enabled and notify(f"READY=1\nSTATUS={safe_status}")

    def record_tick(self, *, scheduled_at: float, now: float) -> bool:
        """Feed systemd only when the event loop woke within its lag budget.

        A late tick ACCUMULATES lateness instead of condemning the gateway on the
        first one: the loop gets a grace budget of ``_max_lag()`` seconds of
        cumulative lateness before the watchdog gives up on it. Feeding systemd on a
        late tick keeps the unit alive during a transient stall (MCP discovery, a long
        cron turn, an adapter reconnect); only a persistently late loop — the actual
        wedge — stops the heartbeat and lets systemd reclaim the unit.
        """
        if not self.enabled or self._stopping or self._unhealthy:
            return False
        try:
            lag = float(now) - float(scheduled_at)
        except (TypeError, ValueError):
            lag = float("inf")
        if not math.isfinite(lag) or lag < 0:
            # A clock that went backwards (or a bogus value) is not evidence of a wedge.
            lag = 0.0
        if lag > self._lag_tolerance():
            self._cumulative_lag += lag
            if self._cumulative_lag >= self._max_lag():
                self._unhealthy = True
                notify("STATUS=watchdog unhealthy: event loop progress is late")
                return False
            # Still late, but within the grace budget: keep systemd fed.
            notify("WATCHDOG=1")
            return True
        self._cumulative_lag = 0.0
        notify("WATCHDOG=1")
        return True

    async def _run(self) -> None:
        if self.interval_seconds is None:
            return
        cadence = max(0.01, self.interval_seconds / 2.0)
        loop = asyncio.get_running_loop()
        scheduled_at = loop.time() + cadence
        with contextlib.suppress(asyncio.CancelledError):
            while not self._stopping and not self._unhealthy:
                await asyncio.sleep(max(0.0, scheduled_at - loop.time()))
                now = loop.time()
                if not self.record_tick(scheduled_at=scheduled_at, now=now):
                    return
                scheduled_at += cadence
                if scheduled_at < now:
                    scheduled_at = now + cadence

    async def stop(self) -> None:
        """Stop feeding systemd and emit ``STOPPING=1`` at most once."""
        self._stopping = True
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()  # no-op on a finished task
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._task = None
        if self.enabled and not self._stopping_notified:
            notify("STOPPING=1")
            self._stopping_notified = True
