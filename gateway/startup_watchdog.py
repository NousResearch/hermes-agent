"""Startup-liveness watchdog — respawn a gateway that wedges before its loop runs (OOF-298).

The existing liveness backstops all assume startup succeeded:

* the loop-liveness watchdog (:mod:`gateway.shutdown_watchdog`) is armed by
  ``GatewayRunner._start_loop_liveness_guards`` — *inside* the running event
  loop's startup path;
* the shutdown watchdog is armed at ``stop()``;
* the loop heartbeat file is written by an asyncio task.

None of them can fire if the process deadlocks **before the event loop comes
alive**. That failure mode is real: OOF-298 documents a hosted gateway whose
process sat for ~30 hours with every thread parked in ``futex_wait_queue``,
zero log lines written, ``/health`` unreachable — while s6 saw a live PID and
therefore never respawned it, and a stale ``gateway_state.json`` from the
*previous* life told every status surface the gateway was "draining".

This module closes that gap with the simplest thing that works: a plain
daemon OS thread armed at process entry, disarmed the moment the event loop
is confirmed live (the point where the existing loop-liveness watchdog takes
over). If startup neither reaches that milestone nor exits within the
deadline, the watchdog dumps all-thread stacks via ``faulthandler``, records
the exit in the lifecycle ledger (NS-608) so the next boot classifies it
correctly, and ``os._exit``\\ s with the service-restart code so s6/systemd
revive the process instead of babysitting a zombie.

Deadline rationale: a healthy startup reaches the disarm point in seconds.
The slowest legitimate pre-loop work is MCP tool discovery (bounded 120s
internal wait), so the 300s default leaves comfortable headroom. Platform
adapter connects — which can genuinely take minutes (WhatsApp pairing, npm
cold installs) — happen *after* the disarm point and are never covered by
this watchdog.

Config surface is deliberately env-only (``HERMES_STARTUP_WATCHDOG=0`` to
disable, ``HERMES_STARTUP_WATCHDOG_TIMEOUT_S`` to tune): the watchdog must be
armed before config.yaml is loaded — a wedge during config parsing is exactly
in scope — so it cannot depend on config for its own enablement.

Everything here is best-effort: a watchdog failure must never affect the
startup it is observing.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE

logger = logging.getLogger(__name__)

DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S = 300.0
_MIN_TIMEOUT_S = 30.0

ENV_STARTUP_WATCHDOG = "HERMES_STARTUP_WATCHDOG"
ENV_STARTUP_WATCHDOG_TIMEOUT_S = "HERMES_STARTUP_WATCHDOG_TIMEOUT_S"

_DUMP_RELATIVE = ("logs", "gateway-startup-watchdog.log")

_FALSEY = frozenset({"0", "false", "no", "off"})

# Module-level singleton: the arm sites (gateway.run.main and the
# hermes_cli.gateway CLI wrapper) and the disarm site
# (GatewayRunner._start_loop_liveness_guards) have no shared object to hand a
# handle through, and only one gateway startup ever runs per process.
_handle_lock = threading.Lock()
_handle: Optional["StartupWatchdogHandle"] = None


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level diagnostic files (ignore task overrides)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def get_startup_watchdog_dump_path(home: Optional[Path] = None) -> Path:
    """Return ``<HERMES_HOME>/logs/gateway-startup-watchdog.log``."""
    base = home if home is not None else _process_hermes_home()
    return base.joinpath(*_DUMP_RELATIVE)


def startup_watchdog_disabled() -> bool:
    """True when ``HERMES_STARTUP_WATCHDOG`` opts out explicitly."""
    raw = os.environ.get(ENV_STARTUP_WATCHDOG, "").strip().lower()
    return raw in _FALSEY


def resolve_startup_watchdog_timeout() -> float:
    """Deadline in seconds; env override, floor-clamped, default on garbage."""
    raw = os.environ.get(ENV_STARTUP_WATCHDOG_TIMEOUT_S, "").strip()
    if not raw:
        return DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring non-numeric %s=%r; using default %.0fs",
            ENV_STARTUP_WATCHDOG_TIMEOUT_S,
            raw,
            DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S,
        )
        return DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S
    if value <= 0:
        return DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S
    return max(value, _MIN_TIMEOUT_S)


def _write_dump_record(record: Dict[str, Any]) -> None:
    """Append a one-line JSON metadata record beside the faulthandler dump."""
    try:
        path = get_startup_watchdog_dump_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write startup watchdog dump record", exc_info=True)


class StartupWatchdogHandle:
    """Disarm/inspect handle for the armed startup watchdog thread."""

    def __init__(self, timeout_s: float, exit_code: int):
        self.timeout_s = timeout_s
        self.exit_code = exit_code
        self.armed_at = time.monotonic()
        self._disarmed = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def disarm(self) -> None:
        """Startup reached a live event loop — stand down. Idempotent."""
        self._disarmed.set()

    @property
    def disarmed(self) -> bool:
        return self._disarmed.is_set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── internals ────────────────────────────────────────────────────────

    def _fire(self) -> None:
        elapsed = time.monotonic() - self.armed_at
        try:
            logger.critical(
                "Gateway startup did not reach a live event loop within %.0fs "
                "(elapsed %.0fs); dumping all thread stacks and exiting with "
                "code %d so the service supervisor can restart it (OOF-298).",
                self.timeout_s,
                elapsed,
                self.exit_code,
            )
        except Exception:
            pass
        _write_dump_record(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tag": "startup_watchdog.fired",
                "pid": os.getpid(),
                "timeout_s": self.timeout_s,
                "elapsed_s": round(elapsed, 3),
                "exit_code": self.exit_code,
            }
        )
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:
            logger.debug("Startup watchdog faulthandler dump failed", exc_info=True)
        # Record the exit in the lifecycle sentinel so the next boot reports
        # "startup watchdog hard-exit" instead of misclassifying this as an
        # unclean SIGKILL/OOM death (NS-608).
        try:
            from gateway.lifecycle_ledger import mark_exited

            mark_exited(self.exit_code, reason="startup_liveness_watchdog")
        except Exception:
            pass
        self._exit(self.exit_code)

    @staticmethod
    def _exit(code: int) -> None:
        """Seam for tests; production is a bare ``os._exit``."""
        os._exit(code)

    def _run(self) -> None:
        if self._disarmed.wait(timeout=self.timeout_s):
            return
        if self._disarmed.is_set():
            return
        self._fire()

    def _start(self) -> bool:
        thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="gateway-startup-watchdog",
        )
        try:
            thread.start()
        except Exception:
            logger.debug("Failed to start gateway startup watchdog", exc_info=True)
            return False
        self._thread = thread
        return True


def arm_startup_watchdog(
    timeout_s: Optional[float] = None,
    *,
    exit_code: int = GATEWAY_SERVICE_RESTART_EXIT_CODE,
) -> Optional[StartupWatchdogHandle]:
    """Arm the process-wide startup watchdog. Idempotent; never raises.

    Returns the (possibly pre-existing) handle, or ``None`` when disabled via
    ``HERMES_STARTUP_WATCHDOG=0`` or when the thread could not be started.
    """
    global _handle
    try:
        if startup_watchdog_disabled():
            return None
        with _handle_lock:
            if _handle is not None and _handle.is_alive():
                return _handle
            resolved = (
                float(timeout_s)
                if timeout_s is not None and float(timeout_s) > 0
                else resolve_startup_watchdog_timeout()
            )
            handle = StartupWatchdogHandle(resolved, exit_code)
            if not handle._start():
                return None
            _handle = handle
            return handle
    except Exception:
        logger.debug("Failed to arm gateway startup watchdog", exc_info=True)
        return None


def disarm_startup_watchdog() -> None:
    """Disarm the process-wide startup watchdog, if armed. Never raises."""
    global _handle
    try:
        with _handle_lock:
            handle = _handle
            _handle = None
        if handle is not None:
            handle.disarm()
    except Exception:
        logger.debug("Failed to disarm gateway startup watchdog", exc_info=True)


def _reset_for_tests() -> None:
    """Drop the module singleton (test isolation only)."""
    global _handle
    with _handle_lock:
        handle = _handle
        _handle = None
    if handle is not None:
        handle.disarm()
