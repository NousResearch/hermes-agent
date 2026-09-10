"""Startup-liveness watchdog — respawn a gateway that wedges before its loop runs (OOF-298).

Every other liveness backstop (loop-liveness watchdog, shutdown watchdog, heartbeat file)
assumes startup succeeded; none can fire if the process deadlocks before the event loop
is alive (OOF-298: ~30h with every thread in ``futex_wait_queue``, zero logs, s6 saw a
live PID). A daemon thread armed at process entry and disarmed once the loop is confirmed
live dumps all-thread stacks (``faulthandler``), records the exit in the lifecycle ledger
(NS-608) and ``os._exit``\\ s with the service-restart code so s6/systemd respawn.

Slow-but-alive startups are not killed. Order of authority: (1) phase-owned progress
leases (:func:`report_startup_progress`) — authoritative, prove the *startup path itself*
is alive, work for I/O-bound phases (schema migration, corruption repair in
``SessionDB.__init__``) with ~zero CPU; (2) process-wide CPU progress, capped at
``_MAX_CPU_EXTENSIONS`` since an unrelated thread burning CPU must not hide a parked
startup thread forever. Known limitation: a *spinning* startup deadlock earns the capped
extensions before firing; the observed class is parked-thread deadlocks. Idle-by-design
waits call :func:`kick_startup_watchdog` (respawn-storm backoff, up to 300s); MCP
discovery's 120s wait sits inside the 300s default.

IMPORT-LIGHTNESS IS A CORRECTNESS PROPERTY: top-level module, stdlib only. Arming must
precede importing ``gateway`` (hundreds of modules; an import-time deadlock is in scope),
and at fire time the wedged main thread may hold the import lock — so the fire path does
no imports on its own thread; the ledger write runs on a helper thread joined with a
timeout. Config is env-only (``HERMES_STARTUP_WATCHDOG=0``,
``HERMES_STARTUP_WATCHDOG_TIMEOUT_S``) because config.yaml parsing is itself in scope.
Everything is best-effort: a watchdog failure must never affect the startup it observes.
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

# Mirrors gateway.restart.GATEWAY_SERVICE_RESTART_EXIT_CODE (parity test in
# tests/gateway/test_startup_watchdog.py) — this module must not import gateway.
SERVICE_RESTART_EXIT_CODE = 75

ENV_STARTUP_WATCHDOG = "HERMES_STARTUP_WATCHDOG"
ENV_STARTUP_WATCHDOG_TIMEOUT_S = "HERMES_STARTUP_WATCHDOG_TIMEOUT_S"

_DUMP_RELATIVE = ("logs", "gateway-startup-watchdog.log")

_FALSEY = frozenset({"0", "false", "no", "off"})

# The waiter re-reads its deadline at most this often so kicks/extensions
# take effect promptly without busy-waiting.
_POLL_SLICE_S = 5.0

# Minimum process CPU-time delta within one expired window to count as
# progress. A parked futex deadlock accrues microseconds; a schema migration
# accrues orders of magnitude more per window even on slow disks.
_CPU_PROGRESS_MIN_S = 1.0

# Hard cap on CPU-fallback extensions: CPU is process-wide evidence, so it may
# only stretch the runway to (1 + cap) x timeout (3 x 300s = 20min); anything
# longer must hold an explicit phase lease.
_MAX_CPU_EXTENSIONS = 3

# Per-call clamp on progress leases; a phase that needs longer renews (the
# renewal is the liveness evidence). 15min covers the observed worst-case
# single migration step on multi-GB state.db files with margin.
_MAX_LEASE_S = 900.0

# Bounded wait for the lifecycle-ledger helper thread (import lock may be
# held by the wedged main thread).
_LEDGER_JOIN_TIMEOUT_S = 5.0

# Upper bound on the ENTIRE forensic fire path. A sibling escort thread that
# touches no logging/filesystem/locks hard-exits if forensics wedge (e.g. the
# wedged main thread holds the logging handler lock, disk full/hung). Must
# exceed _LEDGER_JOIN_TIMEOUT_S.
_FIRE_EXIT_BOUND_S = 10.0

# Handle states. Transitions are guarded by the handle's state lock so a
# disarm and a fire can never both "win": armed -> disarmed or armed -> firing.
# See #89750.
_ARMED = "armed"
_DISARMED = "disarmed"
_FIRING = "firing"

# Module singleton: the arm sites (hermes_cli.main / hermes_cli.gateway /
# gateway.run.main / cli.py --gateway) and the disarm site (GatewayRunner)
# share no object, and only one gateway startup ever runs per process.
_handle_lock = threading.Lock()
_handle: Optional["StartupWatchdogHandle"] = None


def _process_hermes_home() -> Path:
    """HERMES_HOME for diagnostic files — stdlib-only replica of the hermes_constants default."""
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
    return os.environ.get(ENV_STARTUP_WATCHDOG, "").strip().lower() in _FALSEY


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
            ENV_STARTUP_WATCHDOG_TIMEOUT_S, raw, DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S,
        )
        return DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S
    if value <= 0:
        return DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S
    return max(value, _MIN_TIMEOUT_S)


def _append_dump(write, failure_msg: str) -> None:
    """Open the dump file for append and hand it to *write*; failures only log at DEBUG."""
    try:
        path = get_startup_watchdog_dump_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            write(fh)
    except Exception:
        logger.debug(failure_msg, exc_info=True)


def _write_dump_record(record: Dict[str, Any]) -> None:
    """Append a one-line JSON metadata record beside the faulthandler dump."""
    _append_dump(
        lambda fh: fh.write(json.dumps(record, default=str) + "\n"),
        "Failed to write startup watchdog dump record",
    )


def _log_quietly(level: str, msg: str, *args) -> None:
    """Log at *level*; never raises (the wedged main thread may hold the logging lock)."""
    try:
        getattr(logger, level)(msg, *args)
    except Exception:
        pass


def _mark_lifecycle_exit(exit_code: int) -> None:
    """Record the watchdog exit in the NS-608 lifecycle sentinel.

    Runs on a helper thread (see ``_fire``): the import can block on the
    interpreter import lock, and the fire path must reach ``os._exit`` regardless.
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
        # Phase-owned progress lease (see lease()): monotonic deadline the
        # current startup phase has claimed for legitimately long sync work.
        self._lease_until = 0.0
        self._lease_phase: Optional[str] = None
        self._lease_count = 0
        # Set by _fire() once forensics complete so the exit escort stands down.
        self._fire_done = threading.Event()

    def disarm(self) -> None:
        """Startup reached a live event loop — stand down. Idempotent.

        Atomic with respect to firing: whichever of disarm/fire takes the state
        lock first wins, so a disarm landing before the fire sequence is never lost.
        """
        with self._state_lock:
            if self._state == _ARMED:
                self._state = _DISARMED
        self._disarmed_event.set()

    def kick(self, extra_s: float = 0.0) -> None:
        """Push the deadline out to ``now + timeout + extra_s``.

        For call sites about to block intentionally with ~zero CPU (the
        respawn-storm backoff sleep), otherwise indistinguishable from a parked deadlock.
        """
        try:
            extra = max(0.0, float(extra_s))
        except (TypeError, ValueError):
            extra = 0.0
        with self._state_lock:
            self._deadline = time.monotonic() + self.timeout_s + extra

    def lease(self, expected_s: float, phase: str = "") -> None:
        """Claim a progress lease: this phase expects up to ``expected_s`` more seconds of work.

        The authoritative "still making progress" signal — owned by the startup path,
        so it works for I/O-bound phases and cannot be counterfeited by unrelated
        threads. Clamped to ``_MAX_LEASE_S`` per call so one buggy caller cannot
        silence the watchdog indefinitely; long phases renew. Never raises.
        """
        try:
            expected = float(expected_s)
        except (TypeError, ValueError):
            return
        if expected <= 0:
            return
        expected = min(expected, _MAX_LEASE_S)
        with self._state_lock:
            self._lease_until = max(self._lease_until, time.monotonic() + expected)
            if phase:
                self._lease_phase = str(phase)
            self._lease_count += 1

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
        """Forensics, then exit — with the exit itself independently bounded.

        Everything producing forensics (logging, dump record, faulthandler, ledger)
        can block: the wedged main thread may hold the logging handler lock, the disk
        may be hung. None of that may stop the respawn, so the escort thread starts
        FIRST. ``os._exit`` is async-signal-safe and lock-free by design.
        """
        try:
            threading.Thread(
                target=self._exit_escort, daemon=True, name="gateway-startup-watchdog-exit-escort"
            ).start()
        except Exception:
            pass
        elapsed = time.monotonic() - self.armed_at
        _log_quietly(
            "critical",
            "Gateway startup did not reach a live event loop within %.0fs "
            "(elapsed %.0fs, %d extension(s)), holds no progress lease "
            "and shows no CPU progress; dumping all thread stacks and "
            "exiting with code %d so the service supervisor can restart "
            "it (OOF-298).",
            self.timeout_s, elapsed, self._extensions, self.exit_code,
        )
        _write_dump_record(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tag": "startup_watchdog.fired",
                "pid": os.getpid(),
                "timeout_s": self.timeout_s,
                "elapsed_s": round(elapsed, 3),
                "extensions": self._extensions,
                "lease_count": self._lease_count,
                "last_lease_phase": self._lease_phase,
                "exit_code": self.exit_code,
            }
        )
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:
            logger.debug("Startup watchdog faulthandler dump failed", exc_info=True)
        # Also dump into the log file: detached/windowless runs (pythonw, some
        # service managers) may have no stderr, and forensics are the point.
        _append_dump(
            lambda fh: faulthandler.dump_traceback(file=fh, all_threads=True),
            "Startup watchdog file-based faulthandler dump failed",
        )
        # Ledger write on a helper thread (it imports application code; the
        # wedged main thread may hold the import lock). Bounded join, then exit
        # regardless — NS-608 classification is best-effort; the respawn is not.
        try:
            ledger_thread = threading.Thread(
                target=_mark_lifecycle_exit, args=(self.exit_code,), daemon=True,
                name="gateway-startup-watchdog-ledger",
            )
            ledger_thread.start()
            ledger_thread.join(timeout=_LEDGER_JOIN_TIMEOUT_S)
        except Exception:
            pass
        self._fire_done.set()
        self._exit(self.exit_code)

    def _exit_escort(self) -> None:
        """Hard-exit if the forensic fire path wedges (bounded-exit seam).

        Deliberately free of log handlers, filesystem access, module loads and any
        lock shared with application code: only a sleep, an Event check and the exit seam.
        """
        self._sleep(_FIRE_EXIT_BOUND_S)
        if self._fire_done.is_set():
            return
        self._exit(self.exit_code)

    @staticmethod
    def _sleep(seconds: float) -> None:
        """Seam for tests; production is a bare ``time.sleep``."""
        time.sleep(seconds)

    @staticmethod
    def _exit(code: int) -> None:
        """Seam for tests; production is a bare ``os._exit``."""
        os._exit(code)

    def _extend_if_armed(self, deadline: float) -> bool:
        """Set a new deadline under the state lock; False when no longer armed."""
        with self._state_lock:
            if self._state != _ARMED:
                return False
            self._deadline = deadline
            return True

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
            # Deadline expired. Order of authority: (1) a phase lease is honored
            # outright; (2) CPU progress extends at most _MAX_CPU_EXTENSIONS
            # times (process-wide CPU proves activity, not startup progress).
            now = time.monotonic()
            with self._state_lock:
                lease_until = self._lease_until
                lease_phase = self._lease_phase
            if lease_until > now:
                if not self._extend_if_armed(max(lease_until, now + min(_POLL_SLICE_S, self.timeout_s))):
                    return
                _log_quietly(
                    "warning",
                    "Gateway startup exceeded %.0fs but phase %r holds a "
                    "progress lease for another %.0fs — honoring it.",
                    self.timeout_s, lease_phase or "unknown", lease_until - now,
                )
                # Leased work may be I/O-bound; reset the CPU baseline so the
                # post-lease window is judged on its own activity.
                last_cpu = self._process_cpu_seconds()
                continue
            cpu = self._process_cpu_seconds()
            window_delta = cpu - last_cpu if cpu is not None and last_cpu is not None else None
            if window_delta is not None and window_delta >= _CPU_PROGRESS_MIN_S and self._extensions < _MAX_CPU_EXTENSIONS:
                last_cpu = cpu
                self._extensions += 1
                if not self._extend_if_armed(time.monotonic() + self.timeout_s):
                    return
                _log_quietly(
                    "warning",
                    "Gateway startup exceeded %.0fs but is consuming CPU "
                    "(%.1fs this window); extending the startup watchdog "
                    "deadline (CPU-fallback extension %d of %d — phases "
                    "doing long legitimate work should call "
                    "report_startup_progress instead).",
                    self.timeout_s, window_delta, self._extensions, _MAX_CPU_EXTENSIONS,
                )
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
        thread = threading.Thread(target=self._run, daemon=True, name="gateway-startup-watchdog")
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

    ``disarm()`` runs while still holding the singleton lock — it is non-blocking,
    and this closes the window where a concurrent re-arm could swap in a new
    handle that the disarm then misses.
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


def _with_armed_handle(method: str, failure_msg: str, *args) -> None:
    """Call ``handle.<method>(*args)`` on the armed handle; no-op when unarmed, never raises."""
    try:
        with _handle_lock:
            handle = _handle
        if handle is not None:
            getattr(handle, method)(*args)
    except Exception:
        logger.debug(failure_msg, exc_info=True)


def kick_startup_watchdog(extra_s: float = 0.0) -> None:
    """Extend the armed watchdog's deadline. No-op when not armed; never raises.

    Call before intentionally blocking with ~zero CPU activity (e.g. the
    respawn-storm backoff sleep) so the idle wait is not mistaken for a parked deadlock.
    """
    _with_armed_handle("kick", "Failed to kick gateway startup watchdog", extra_s)


def report_startup_progress(expected_s: float, phase: str = "") -> None:
    """Declare a phase-owned progress lease on the armed startup watchdog.

    Call from startup phases about to do legitimately long synchronous work — most
    importantly ``state.db`` schema migrations and corruption repair/backup inside
    ``SessionDB.__init__`` — with an honest worst case, renewing for multi-step phases.
    Per-call duration is clamped to ``_MAX_LEASE_S``. No-op when not armed; never
    raises — safe to call unconditionally from application code.
    """
    _with_armed_handle("lease", "Failed to report startup progress", expected_s, phase)


def _reset_for_tests() -> None:
    """Drop the module singleton (test isolation only)."""
    global _handle
    with _handle_lock:
        handle = _handle
        _handle = None
    if handle is not None:
        handle.disarm()
