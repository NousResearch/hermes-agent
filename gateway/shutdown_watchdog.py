"""Thread-based shutdown watchdog for gateway drain freeze recovery.

Problem: when the asyncio event loop freezes during a graceful drain
(e.g. GIL hostage by a C call, blocked I/O, pathological callback),
the asyncio-based drain timeout cannot fire because it runs on the
same frozen loop.  launchd KeepAlive / systemd Restart=only see a
still-alive process and never revive it.  Result: a zombie gateway
sits unresponsive for hours or days.

Solution (two complementary mechanisms):

1. **Thread-based shutdown watchdog** — armed at ``stop()`` start; a
   plain OS thread waits ``drain_timeout + headroom`` seconds and, if
   the drain hasn't completed, dumps all-thread tracebacks via
   faulthandler, writes a forensic snapshot, and calls ``os._exit(1)``
   so the service manager revives the process.

2. **Event-loop liveness heartbeat** — a ``asyncio.Task`` rewrites a
   small JSON file every 30 s while the gateway is running.  External
   supervision (launchd watchdog, a cron job, or a monitoring script)
   can distinguish "alive" from "loop frozen" by checking whether the
   heartbeat file is updated.  This supplements ``gateway_state.json``
   which only changes on transitions/turns.

Both are opt-in and guarded by a config flag
``gateway.shutdown_watchdog.enabled`` (default: ``true``).
"""

from __future__ import annotations

import asyncio
import faulthandler
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_DEFAULT_HEADROOM_SECONDS = 60  # drain_timeout + 60 s before watchdog fires
_HEARTBEAT_INTERVAL_SECONDS = 30
_DEFAULT_WATCHDOG_ENABLED = True


def _load_watchdog_config(cfg: Dict[str, Any]) -> dict:
    """Read the gateway.shutdown_watchdog section from the parsed config."""
    gw_cfg = cfg.get("gateway", {})
    sw_cfg = gw_cfg.get("shutdown_watchdog", {})
    enabled = sw_cfg.get("enabled", _DEFAULT_WATCHDOG_ENABLED)
    headroom = float(sw_cfg.get("headroom_seconds", _DEFAULT_HEADROOM_SECONDS))
    heartbeat_interval = float(
        sw_cfg.get("heartbeat_interval_seconds", _HEARTBEAT_INTERVAL_SECONDS)
    )
    return {
        "enabled": bool(enabled),
        "headroom_seconds": headroom,
        "heartbeat_interval_seconds": heartbeat_interval,
    }


# ---------------------------------------------------------------------------
# Thread-based shutdown watchdog
# ---------------------------------------------------------------------------

class ShutdownWatchdog:
    """Armed during stop()/drain; kills a frozen process after timeout.

    Usage::

        wd = ShutdownWatchdog(
            runner=runner_instance,
            drain_timeout=180.0,
            hermes_home=pathlib.Path("/Users/x/.hermes"),
        )
        wd.start()          # must be called BEFORE await self.stop()
        try:
            await runner.stop()
        finally:
            wd.cancel()       # cancels the watchdog thread
    """

    def __init__(
        self,
        *,
        drain_timeout: float,
        hermes_home: Path,
        headroom_seconds: float = _DEFAULT_HEADROOM_SECONDS,
        logger: Any = None,
    ) -> None:
        self._drain_deadline = (
            time.monotonic() + drain_timeout + headroom_seconds
        )
        self._hermes_home = hermes_home
        self._logger = logger
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_path = hermes_home / "state" / "gateway_shutdown_watchdog.json"

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Start the watchdog thread.  No-op if already started."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="hermes-shutdown-watchdog",
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        """Signal the watchdog to stop (drain completed successfully)."""
        self._cancel_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- internal ----------------------------------------------------------

    def _watchdog_loop(self) -> None:
        poll_interval = 1.0  # check every second
        while not self._cancel_event.is_set():
            remaining = self._drain_deadline - time.monotonic()
            if remaining <= 0:
                self._trigger()
                return
            # Sleep in small increments so cancel() is responsive.
            sleep_time = min(poll_interval, remaining)
            if self._cancel_event.wait(timeout=sleep_time):
                break

    def _trigger(self) -> None:
        """Write forensic data and exit the process."""
        msg = (
            "⚠️ Shutdown watchdog fired: drain froze for >%.0fs. "
            "Dumping traceback and exiting."
        ) % (
            time.monotonic() - self._drain_deadline + _DEFAULT_HEADROOM_SECONDS
        )
        if self._logger:
            self._logger.error(msg)
        else:
            print(msg, file=sys.stderr, flush=True)

        # --- write forensic snapshot -----------------------------------
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot: Dict[str, Any] = {
                "event": "shutdown_watchdog_fired",
                "fired_at": time.time(),
                "drain_deadline": self._drain_deadline,
                "pid": os.getpid(),
                "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            }
            # Try to grab some state info without blocking
            try:
                from gateway.status import get_running_pid
                snapshot["current_gateway_pid"] = get_running_pid()
            except Exception:
                pass
            try:
                with open(self._state_path, "w") as f:
                    json.dump(snapshot, f, indent=2)
            except Exception:
                pass
        except Exception:
            pass  # best-effort

        # --- dump all-thread traceback ---------------------------------
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:
            pass

        # --- exit so service manager revives us ------------------------
        os._exit(1)


# ---------------------------------------------------------------------------
# Event-loop liveness heartbeat
# ---------------------------------------------------------------------------

class LoopHeartbeat:
    """Writes a small JSON heartbeat file every N seconds.

    External supervision can check mtime of this file to confirm the
    asyncio loop is actually processing tasks (not just the process being
    alive).

    Usage::

        hb = LoopHeartbeat(hermes_home, interval=30.0, logger=logger)
        hb.start()        # creates an asyncio.Task
        # ... gateway runs ...
        await hb.stop()   # cancels the task
    """

    def __init__(
        self,
        hermes_home: Path,
        interval: float = _HEARTBEAT_INTERVAL_SECONDS,
        logger: Any = None,
    ) -> None:
        self._hermes_home = hermes_home
        self._interval = interval
        self._logger = logger
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_path = (
            hermes_home / "state" / "gateway.heartbeat"
        )

    async def start(self) -> None:
        """Start writing heartbeats.  No-op if already running."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Cancel the heartbeat task."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                self._write_heartbeat()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            # Final heartbeat on stop so mtime reflects last known good
            try:
                self._write_heartbeat()
            except Exception:
                pass
        except Exception:
            if self._logger:
                self._logger.debug("LoopHeartbeat error: %s", sys.exc_info()[1])

    def _write_heartbeat(self) -> None:
        """Atomically update the heartbeat file."""
        payload = {
            "pid": os.getpid(),
            "updated_at": time.time(),
        }
        tmp = str(self._heartbeat_path) + ".tmp"
        try:
            self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, str(self._heartbeat_path))
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience: integrate with GatewayRunner
# ---------------------------------------------------------------------------

def arming_shutdown_watchdog(
    drain_timeout: float,
    hermes_home: Path,
    logger: Any = None,
) -> ShutdownWatchdog:
    """Create, arm, and return a ShutdownWatchdog.

    Call ``watchdog.cancel()`` when the drain completes successfully
    (typically in a ``finally`` block around ``await runner.stop()``).
    """
    wd = ShutdownWatchdog(
        drain_timeout=drain_timeout,
        hermes_home=hermes_home,
        headroom_seconds=_DEFAULT_HEADROOM_SECONDS,
        logger=logger,
    )
    wd.start()
    return wd
