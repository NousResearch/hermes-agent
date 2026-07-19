"""Minimal, optional systemd ``sd_notify`` support for the gateway."""

from __future__ import annotations

import asyncio
import math
import os
import socket
from typing import Optional


def _notify_address(raw: str) -> str:
    """Translate systemd's ``@abstract`` notation to Python's address form."""
    return "\0" + raw[1:] if raw.startswith("@") else raw


def notify(message: str) -> bool:
    """Send one nonblocking sd_notify datagram when systemd configured it.

    Notification failures are deliberately non-fatal: a missing socket or an
    older platform must never prevent the gateway from starting.
    """
    address = os.environ.get("NOTIFY_SOCKET", "").strip()
    if not address:
        return False
    if not isinstance(message, str) or not message:
        return False
    if not hasattr(socket, "AF_UNIX"):
        return False
    try:
        payload = message.encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            # A full receiver buffer must not stall the gateway event loop.
            sender.setblocking(False)
            sender.connect(_notify_address(address))
            sender.send(payload)
        return True
    except (OSError, UnicodeError, ValueError):
        return False


def watchdog_interval_seconds() -> Optional[float]:
    """Return systemd's configured watchdog interval in seconds."""
    if not os.environ.get("NOTIFY_SOCKET", "").strip():
        return None
    if not hasattr(socket, "AF_UNIX"):
        return None
    raw = os.environ.get("WATCHDOG_USEC", "").strip()
    if not raw:
        return None
    try:
        interval = float(raw) / 1_000_000.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(interval) or interval <= 0:
        return None
    return interval


class SystemdWatchdog:
    """Feed systemd while the asyncio event loop continues to make progress."""

    def __init__(
        self,
        *,
        config_enabled: bool = True,
        lag_tolerance_seconds: Optional[float] = None,
    ) -> None:
        self._config_enabled = bool(config_enabled)
        self.interval_seconds = watchdog_interval_seconds()
        self._lag_tolerance_seconds = lag_tolerance_seconds
        self._task: Optional[asyncio.Task[None]] = None
        self._unhealthy = False
        self._stopping = False
        self._stopping_notified = False

    @property
    def enabled(self) -> bool:
        return self._config_enabled and self.interval_seconds is not None

    @property
    def unhealthy(self) -> bool:
        return self._unhealthy

    @property
    def task(self) -> Optional[asyncio.Task[None]]:
        return self._task

    def _lag_tolerance(self) -> float:
        interval = self.interval_seconds or 0.0
        configured = self._lag_tolerance_seconds
        if configured is None:
            return max(0.1, interval * 0.25)
        try:
            value = float(configured)
        except (TypeError, ValueError):
            return max(0.1, interval * 0.25)
        if not math.isfinite(value):
            return max(0.1, interval * 0.25)
        return max(0.0, value)

    def start(self) -> bool:
        """Start the loop-progress sampler when systemd watchdog is enabled."""
        if not self.enabled:
            return False
        if self._task is not None and not self._task.done():
            return True
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._stopping = False
        self._unhealthy = False
        self._stopping_notified = False
        self._task = asyncio.create_task(self._run(), name="hermes-systemd-watchdog")
        return True

    def ready(self, status: str = "Gateway running") -> bool:
        """Tell systemd that startup completed and the gateway is ready."""
        if not self.enabled:
            return False
        safe_status = str(status or "Gateway running").replace("\n", " ")
        return notify(f"READY=1\nSTATUS={safe_status}")

    def record_tick(self, *, scheduled_at: float, now: float) -> bool:
        """Feed systemd only when the event loop woke within its lag budget."""
        if not self.enabled or self._stopping or self._unhealthy:
            return False
        try:
            lag = float(now) - float(scheduled_at)
        except (TypeError, ValueError):
            lag = float("inf")
        if not math.isfinite(lag) or lag > self._lag_tolerance():
            self._unhealthy = True
            notify("STATUS=watchdog unhealthy: event loop progress is late")
            return False
        notify("WATCHDOG=1")
        return True

    async def _run(self) -> None:
        interval = self.interval_seconds
        if interval is None:
            return
        cadence = max(0.01, interval / 2.0)
        loop = asyncio.get_running_loop()
        scheduled_at = loop.time() + cadence
        try:
            while not self._stopping and not self._unhealthy:
                await asyncio.sleep(max(0.0, scheduled_at - loop.time()))
                now = loop.time()
                if not self.record_tick(scheduled_at=scheduled_at, now=now):
                    return
                scheduled_at += cadence
                if scheduled_at < now:
                    scheduled_at = now + cadence
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        """Stop feeding systemd and emit ``STOPPING=1`` at most once."""
        self._stopping = True
        task = self._task
        current = asyncio.current_task()
        if task is not None and task is not current:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._task = None
        if self.enabled and not self._stopping_notified:
            notify("STOPPING=1")
            self._stopping_notified = True
