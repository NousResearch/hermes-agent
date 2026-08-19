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

This module closes that gap with a plain daemon OS thread armed at process
entry, disarmed the moment the event loop is confirmed live (the point where
the existing loop-liveness watchdog takes over). If startup neither reaches
that milestone nor exits within the deadline, the watchdog dumps all-thread
stacks via ``faulthandler``, records the exit in the lifecycle ledger
(NS-608) so the next boot classifies it correctly, and ``os._exit``\\ s with
the service-restart code so s6/systemd revive the process instead of
babysitting a zombie.

Slow-but-alive startups are NOT killed. Before firing, the watchdog checks
whether the process consumed meaningful CPU time during the expired window
(``time.process_time()`` is process-wide). Long-but-legitimate synchronous
startup work — most importantly large ``state.db`` schema migrations, which
run inside ``SessionDB.__init__`` well before the loop starts and can
genuinely exceed any fixed deadline on multi-GB databases — burns CPU
continuously, so the deadline is extended (with a log line each time) for as
long as progress continues. The OOF-298 deadlock class parks every thread in
futex waits and accrues ~zero CPU, so it still fires on schedule. Known
limitation, documented deliberately: a *spinning* (busy-wait) startup
deadlock reads as CPU progress and will not fire; the observed incident class
is parked-thread deadlocks, which this catches.

Waits that are idle-by-design get explicit handling instead:

* the respawn-storm breaker's intentional backoff sleep (up to 300s) calls
  :func:`kick_startup_watchdog` with the sleep budget before sleeping;
* MCP tool discovery's internal wait is bounded at 120s, comfortably inside
  the 300s default deadline.

IMPORT-LIGHTNESS IS A CORRECTNESS PROPERTY of this module, not a style
preference. It lives at the repository top level (not inside the ``gateway``
package) and imports **only stdlib** because:

1. ``gateway/__init__`` eagerly imports the config/session/delivery graph —
   hundreds of modules, DB-adjacent code included. Arming must happen
   *before* that graph is imported, or an import-time deadlock (a plausible
   shape of "wedged before the loop, no logs") sits outside the watchdog's
   coverage.
2. At fire time the main thread may be wedged **holding the import lock**;
   any import attempted on the watchdog thread could then block forever.
   The fire path therefore performs no imports at all on its own thread —
   the lifecycle-ledger write (which does import) runs on a short-lived
   helper thread joined with a timeout, and ``os._exit`` happens regardless.

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
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S = 300.0
_MIN_TIMEOUT_S = 30.0

# Mirrors gateway.restart.GATEWAY_SERVICE_RESTART_EXIT_CODE. Duplicated here
# (with a parity test in tests/gateway/test_startup_watchdog.py) because this
# module must not import the gateway package — see module docstring.
SERVICE_RESTART_EXIT_CODE = 75

ENV_STARTUP_WATCHDOG = "HERMES_STARTUP_WATCHDOG"
ENV_STARTUP_WATCHDOG_TIMEOUT_S = "HERMES_STARTUP_WATCHDOG_TIMEOUT_S"

_DUMP_RELATIVE = ("logs", "gateway-startup-watchdog.log")

_FALSEY = frozenset({"0", "false", "no", "off"})

# The waiter re-reads its deadline at most this often, so kick_/deadline
# extensions take effect promptly without busy-waiting.
_POLL_SLICE_S = 5.0

# Minimum process CPU-time delta (seconds) within one expired deadline window
# for startup to count as "making progress" and earn an extension. A parked
# futex deadlock accrues microseconds; a schema migration accrues orders of
# magnitude more than this per window even on slow disks.
_CPU_PROGRESS_MIN_S = 1.0

# How long the fire path waits for the lifecycle-ledger helper thread before
# exiting anyway (the import lock may be held by the wedged main thread).
_LEDGER_JOIN_TIMEOUT_S = 5.0

# Handle lifecycle states. Transitions are guarded by the handle's state
# lock so a disarm and a fire can never both "win" (P2 race, PR #89750
# review): armed -> disarmed (startup reached a live loop) or
# armed -> firing (deadline expired with no CPU progress) — never both.
_ARMED = "armed"
_DISARMED = "disarmed"
_FIRING = "firing"

# Module-level singleton: the arm sites (hermes_cli.main / hermes_cli.gateway
# / gateway.run.main / cli.py --gateway / scripts/hermes-gateway) and the
# disarm site (GatewayRunner, once the loop is live) have no shared object to
# hand a handle through, and only one gateway startup ever runs per process.
_handle_lock = threading.Lock()
_handle: Optional["StartupWatchdogHandle"] = None


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level diagnostic files.

    Stdlib-only replica of ``hermes_constants``' platform default — this
    module must not import application code (see module docstring). Hosted
    images always set ``HERMES_HOME`` explicitly.
    """
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


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


def _mark_lifecycle_exit(exit_code: int) -> None:
    """Record the watchdog exit in the NS-608 lifecycle sentinel.

    Runs on a dedicated helper thread (see ``_fire``): the ``import`` below
    can block indefinitely on the interpreter import lock if the wedged main
    thread holds it, and the fire path must reach ``os._exit`` regardless.
    """
    try:
        from gateway.lifecycle_ledger import mark_exited

        mark_exited(exit_code, reason="startup_liveness_watchdog")
    except Exception:
        pass


class StartupWatchdogHandle:
    """Disarm/inspect handle for the armed startup watchdog thread."""

    def __init__(self, timeout_s: float, exit_code: int):
        self.timeout_s = timeout_s
        self.exit_code = exit_code
        self.armed_at = time.monotonic()
        self._state = _ARMED
        self._state_lock = threading.Lock()
        self._deadline = self.armed_at + timeout_s
        self._disarmed_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._extensions = 0

    def disarm(self) -> None:
        """Startup reached a live event loop — stand down. Idempotent.

        Atomic with respect to firing: whichever of disarm/fire takes the
        state lock first wins, so a disarm that lands before the fire
        sequence begins is always honored (never lost to a deadline that
        expired concurrently).
        """
        with self._state_lock:
            if self._state == _ARMED:
                self._state = _DISARMED
        self._disarmed_event.set()

    def kick(self, extra_s: float = 0.0) -> None:
        """Push the deadline out to ``now + timeout + extra_s``.

        For call sites that are about to block intentionally with ~zero CPU
        activity (the respawn-storm breaker's backoff sleep), which would
        otherwise be indistinguishable from a parked deadlock.
        """
        try:
            extra = max(0.0, float(extra_s))
        except (TypeError, ValueError):
            extra = 0.0
        with self._state_lock:
            self._deadline = time.monotonic() + self.timeout_s + extra

    @property
    def disarmed(self) -> bool:
        return self._state == _DISARMED

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _process_cpu_seconds() -> Optional[float]:
        """Process-wide CPU time (user+system, all threads); None on failure."""
        try:
            return time.process_time()
        except Exception:
            return None

    def _fire(self) -> None:
        elapsed = time.monotonic() - self.armed_at
        try:
            logger.critical(
                "Gateway startup did not reach a live event loop within %.0fs "
                "(elapsed %.0fs, %d extension(s)) and shows no CPU progress; "
                "dumping all thread stacks and exiting with code %d so the "
                "service supervisor can restart it (OOF-298).",
                self.timeout_s,
                elapsed,
                self._extensions,
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
                "extensions": self._extensions,
                "exit_code": self.exit_code,
            }
        )
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:
            logger.debug("Startup watchdog faulthandler dump failed", exc_info=True)
        # Also dump stacks into the log file: on detached/windowless runs
        # (pythonw, some service managers) stderr may be absent, and the
        # whole point of firing is to leave forensics behind.
        try:
            path = get_startup_watchdog_dump_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except Exception:
            logger.debug(
                "Startup watchdog file-based faulthandler dump failed", exc_info=True
            )
        # Lifecycle-ledger write on a helper thread: it imports application
        # code, and the wedged main thread may hold the import lock. Bounded
        # join, then exit regardless (NS-608 classification is best-effort;
        # the respawn is not).
        try:
            ledger_thread = threading.Thread(
                target=_mark_lifecycle_exit,
                args=(self.exit_code,),
                daemon=True,
                name="gateway-startup-watchdog-ledger",
            )
            ledger_thread.start()
            ledger_thread.join(timeout=_LEDGER_JOIN_TIMEOUT_S)
        except Exception:
            pass
        self._exit(self.exit_code)

    @staticmethod
    def _exit(code: int) -> None:
        """Seam for tests; production is a bare ``os._exit``."""
        os._exit(code)

    def _run(self) -> None:
        last_cpu = self._process_cpu_seconds()
        while True:
            with self._state_lock:
                if self._state != _ARMED:
                    return
                deadline = self._deadline
            remaining = deadline - time.monotonic()
            if remaining > 0:
                if self._disarmed_event.wait(timeout=min(remaining, _POLL_SLICE_S)):
                    return
                continue
            # Deadline expired. A slow-but-alive startup (large state.db
            # schema migration inside SessionDB.__init__) burns CPU
            # continuously; a parked futex deadlock accrues ~none. Extend
            # for the former, fire only for the latter.
            cpu = self._process_cpu_seconds()
            if (
                cpu is not None
                and last_cpu is not None
                and (cpu - last_cpu) >= _CPU_PROGRESS_MIN_S
            ):
                window_delta = cpu - last_cpu
                last_cpu = cpu
                self._extensions += 1
                with self._state_lock:
                    if self._state != _ARMED:
                        return
                    self._deadline = time.monotonic() + self.timeout_s
                try:
                    logger.warning(
                        "Gateway startup exceeded %.0fs but is consuming CPU "
                        "(%.1fs this window) — likely a long schema migration; "
                        "extending the startup watchdog deadline (extension #%d).",
                        self.timeout_s,
                        window_delta,
                        self._extensions,
                    )
                except Exception:
                    pass
                continue
            # No progress: claim the fire transition atomically so a disarm
            # racing this exact moment can still win if it gets there first.
            with self._state_lock:
                if self._state != _ARMED:
                    return
                self._state = _FIRING
            self._fire()
            return

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
    exit_code: int = SERVICE_RESTART_EXIT_CODE,
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
    """Disarm the process-wide startup watchdog, if armed. Never raises.

    The handle's ``disarm()`` is called while still holding the singleton
    lock — it is non-blocking, and holding the lock closes the window where
    a concurrent re-arm could swap in a new handle that the disarm then
    misses.
    """
    global _handle
    try:
        with _handle_lock:
            handle = _handle
            _handle = None
            if handle is not None:
                handle.disarm()
    except Exception:
        logger.debug("Failed to disarm gateway startup watchdog", exc_info=True)


def kick_startup_watchdog(extra_s: float = 0.0) -> None:
    """Extend the armed watchdog's deadline. No-op when not armed; never raises.

    Call before intentionally blocking with ~zero CPU activity (e.g. the
    respawn-storm breaker's backoff sleep) so the idle wait is not mistaken
    for a parked deadlock.
    """
    try:
        with _handle_lock:
            handle = _handle
        if handle is not None:
            handle.kick(extra_s)
    except Exception:
        logger.debug("Failed to kick gateway startup watchdog", exc_info=True)


def _reset_for_tests() -> None:
    """Drop the module singleton (test isolation only)."""
    global _handle
    with _handle_lock:
        handle = _handle
        _handle = None
    if handle is not None:
        handle.disarm()
