"""Out-of-loop shutdown backstop + event-loop liveness heartbeat (#66892).

When the asyncio loop freezes mid-drain, every asyncio-based recovery path is
structurally unable to fire: the drain deadline, status rewrites, and forensics
all need the same loop that is stuck. launchd/systemd KeepAlive only restarts a
*dead* process, so a wedged-but-alive gateway sits as a zombie until manual
SIGKILL.

This module provides:

1. A plain OS-thread shutdown watchdog armed at ``stop()``. If shutdown has not
   completed within ``restart_drain_timeout + grace``, it dumps all-thread
   stacks via ``faulthandler`` plus a metadata snapshot, then ``os._exit`` so
   the service manager can revive the process.
2. An event-loop heartbeat file at ``<HERMES_HOME>/state/gateway.heartbeat`` so
   external supervision can distinguish "process alive" from "loop frozen"
   (``gateway_state.json`` alone can't — it only rewrites on transitions/turns).
"""

from __future__ import annotations

import asyncio
import faulthandler
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Extra leash beyond ``agent.restart_drain_timeout`` so a slow-but-progressing
# drain is not cut short. Matches the issue #66892 suggested hardening.
DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S = 60.0
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
_HEARTBEAT_RELATIVE = ("state", "gateway.heartbeat")
_WATCHDOG_DUMP_RELATIVE = ("logs", "gateway-shutdown-watchdog.log")


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level identity files (ignore profile overrides)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    return get_hermes_home()


def get_loop_heartbeat_path(home: Optional[Path] = None) -> Path:
    """Return ``<HERMES_HOME>/state/gateway.heartbeat``."""
    base = home if home is not None else _process_hermes_home()
    return base.joinpath(*_HEARTBEAT_RELATIVE)


def get_shutdown_watchdog_dump_path(home: Optional[Path] = None) -> Path:
    """Return the faulthandler / metadata dump path for a fired watchdog."""
    base = home if home is not None else _process_hermes_home()
    return base.joinpath(*_WATCHDOG_DUMP_RELATIVE)


def write_loop_heartbeat(
    *,
    pid: Optional[int] = None,
    start_time: Optional[float] = None,
    home: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically rewrite the loop-liveness heartbeat file.

    ``start_time`` is the gateway process start (``time.time()`` epoch seconds)
    so supervisors can detect PID reuse. Best-effort — never raises.
    """
    path = get_loop_heartbeat_path(home)
    payload: Dict[str, Any] = {
        "pid": int(pid if pid is not None else os.getpid()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "monotonic": time.monotonic(),
    }
    if start_time is not None:
        payload["start_time"] = float(start_time)
    if extra:
        payload.update(extra)
    try:
        atomic_json_write(path, payload, indent=None)
    except Exception:
        logger.debug("Failed to write gateway loop heartbeat", exc_info=True)
    return path


def resolve_shutdown_watchdog_delay(
    drain_timeout: float,
    *,
    grace_s: float = DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S,
) -> float:
    """Return the wall-clock leash for the shutdown watchdog thread."""
    try:
        drain = max(float(drain_timeout), 0.0)
    except (TypeError, ValueError):
        drain = 0.0
    try:
        grace = max(float(grace_s), 0.0)
    except (TypeError, ValueError):
        grace = DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    return drain + grace


def _write_watchdog_dump(
    dump_path: Path,
    *,
    delay_s: float,
    snapshot: Optional[Dict[str, Any]],
) -> None:
    """Best-effort faulthandler + metadata dump before hard-exit."""
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    header = {
        "event": "shutdown_watchdog_fired",
        "pid": os.getpid(),
        "delay_s": delay_s,
        "fired_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot or {},
    }
    try:
        with open(dump_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(header, default=str) + "\n")
            fh.write("--- faulthandler dump (all threads) ---\n")
            fh.flush()
            try:
                faulthandler.dump_traceback(file=fh, all_threads=True)
            except Exception:
                fh.write("(faulthandler.dump_traceback failed)\n")
            fh.write("--- end dump ---\n")
            fh.flush()
    except Exception:
        pass

    # Also dump to stderr so journald/launchd capture it even if the file
    # write failed (wedged disk was one of the #66892 hypotheses).
    try:
        sys.stderr.write(
            f"Gateway shutdown watchdog fired after {delay_s:.0f}s "
            f"(pid={os.getpid()}); dumping all thread stacks.\n"
        )
        sys.stderr.flush()
        faulthandler.dump_traceback(all_threads=True)
    except Exception:
        pass


def arm_shutdown_watchdog(
    delay_s: float,
    *,
    done_event: Optional[threading.Event] = None,
    snapshot_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    exit_code: int = 1,
    dump_path: Optional[Path] = None,
    name: str = "gateway-shutdown-watchdog",
) -> threading.Event:
    """Arm a daemon-thread hard-exit backstop for a wedged shutdown path.

    If ``done_event`` is set before ``delay_s`` elapses, the thread exits
    quietly (normal / progressing shutdown completed). Otherwise it dumps
    diagnostics and calls ``os._exit(exit_code)``.

    Never raises. Returns the ``done_event`` (creating one when omitted) so
    the caller can disarm on successful completion.
    """
    done = done_event if done_event is not None else threading.Event()
    try:
        delay = max(float(delay_s), 0.0)
    except (TypeError, ValueError):
        delay = DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S

    if delay <= 0:
        return done

    def _watchdog() -> None:
        # Wait with interruptible chunks so a late disarm doesn't need the
        # full remaining sleep to observe done_event.
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if done.wait(timeout=min(remaining, 1.0)):
                return
        if done.is_set():
            return

        snapshot: Optional[Dict[str, Any]] = None
        if snapshot_fn is not None:
            try:
                snapshot = snapshot_fn()
            except Exception as exc:
                snapshot = {"snapshot_error": repr(exc)}

        target = dump_path if dump_path is not None else get_shutdown_watchdog_dump_path()
        _write_watchdog_dump(target, delay_s=delay, snapshot=snapshot)

        try:
            logger.critical(
                "Shutdown watchdog fired after %.0fs — forcing process exit "
                "(asyncio drain path appears wedged; see %s)",
                delay,
                target,
            )
        except Exception:
            pass

        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        # Mirror _exit_after_graceful_shutdown: release PID file + runtime
        # lock BEFORE the log drain (locks must never be stranded), then
        # drain the async log queue so the logger.critical above actually
        # reaches the file before os._exit bypasses atexit. (#66892)
        try:
            from gateway.status import remove_pid_file, release_gateway_runtime_lock
            remove_pid_file()
            release_gateway_runtime_lock()
        except Exception:
            pass
        try:
            from hermes_logging import drain_log_queue
            drain_log_queue(timeout=1.0)
        except Exception:
            pass
        os._exit(exit_code)

    try:
        threading.Thread(target=_watchdog, daemon=True, name=name).start()
    except Exception:
        logger.debug("Failed to arm shutdown watchdog", exc_info=True)
    return done


def arm_event_loop_health_watchdog(
    *,
    interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    start_time: Optional[float] = None,
    home: Optional[Path] = None,
    stale_factor: float = 3.0,
    exit_code: int = 1,
    name: str = "gateway-loop-health-watchdog",
) -> None:
    """Arm a daemon-thread event-loop health watchdog.

    Runs as an OS thread (NOT an asyncio task) so it can detect when
    the event loop has frozen — the async ``loop_heartbeat_forever``
    cannot run on a stuck loop.

    The thread checks ``<HERMES_HOME>/state/gateway.heartbeat`` on a
    cadence of ``interval_s / 2``. If the file's ``monotonic`` value
    has not advanced within ``interval_s * stale_factor`` seconds, the
    event loop is presumed frozen and the watchdog hard-exits the
    process so the service manager can restart it.

    Also registers a self-rescheduling ``loop.call_soon`` before
    returning, guaranteeing the loop always has at least one pending
    callback — preventing the ``select(forever)`` deadlock described
    in #69089.

    Best-effort — never raises.
    """
    try:
        interval = max(float(interval_s), 1.0)
    except (TypeError, ValueError):
        interval = DEFAULT_HEARTBEAT_INTERVAL_S

    try:
        stale_limit = interval * max(float(stale_factor), 2.0)
    except (TypeError, ValueError):
        stale_limit = interval * 3.0

    heartbeat_path = get_loop_heartbeat_path(home)
    _pid = os.getpid()

    # Helper: read the monotonic timestamp from the heartbeat file.
    def _read_heartbeat_monotonic() -> float:
        try:
            raw = heartbeat_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            val = payload.get("monotonic", 0.0)
            return float(val)
        except Exception:
            return 0.0

    def _watchdog_loop() -> None:
        # Ensure the heartbeat path's parent dir exists.
        try:
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Write an initial heartbeat so we have a baseline.
        write_loop_heartbeat(start_time=start_time, home=home)

        check_interval = max(interval / 2.0, 2.0)
        while True:
            try:
                time.sleep(check_interval)
            except Exception:
                return  # Thread interrupted — exit.

            try:
                last_monotonic = _read_heartbeat_monotonic()
                age = time.monotonic() - last_monotonic
            except Exception:
                # Can't read the file — skip this cycle.
                continue

            if age < stale_limit:
                continue  # Heartbeat still fresh — loop is alive.

            # Heartbeat is stale — event loop appears frozen.
            try:
                logger.critical(
                    "Event-loop health watchdog fired: heartbeat stale for "
                    "%.0fs (limit=%.0fs, pid=%d). The asyncio loop is "
                    "presumed frozen — hard-exiting so the service "
                    "manager can restart the gateway.",
                    age,
                    stale_limit,
                    _pid,
                )
            except Exception:
                pass

            # Dump diagnostics before the hard exit.
            dump_path = get_shutdown_watchdog_dump_path(home)
            snapshot: Optional[Dict[str, Any]] = {
                "event": "event_loop_health_watchdog_fired",
                "heartbeat_age_s": round(age, 1),
                "stale_limit_s": round(stale_limit, 1),
                "prompt": "Event loop frozen — no heartbeat in %.0fs" % age,
            }
            _write_watchdog_dump(dump_path, delay_s=age, snapshot=snapshot)

            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass

            try:
                from hermes_logging import drain_log_queue
                drain_log_queue(timeout=1.0)
            except Exception:
                pass

            os._exit(exit_code)

    try:
        t = threading.Thread(target=_watchdog_loop, daemon=True, name=name)
        t.start()
        logger.debug(
            "Event-loop health watchdog armed (check_interval=%.1fs, "
            "stale_limit=%.0fs, path=%s)",
            max(interval / 2.0, 2.0),
            stale_limit,
            heartbeat_path,
        )
    except Exception:
        logger.debug("Failed to arm event-loop health watchdog", exc_info=True)


async def loop_heartbeat_forever(
    *,
    interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    start_time: Optional[float] = None,
    home: Optional[Path] = None,
    should_continue: Optional[Callable[[], bool]] = None,
) -> None:
    """Rewrite the loop heartbeat file on a cadence until cancelled / gated off.

    Runs as an asyncio task on the gateway loop — if the loop freezes, this
    task stops and the file mtime/updated_at goes stale for external monitors.
    """
    try:
        interval = max(float(interval_s), 1.0)
    except (TypeError, ValueError):
        interval = DEFAULT_HEARTBEAT_INTERVAL_S

    # Immediate first write so monitors see a fresh file as soon as the
    # gateway is running, not after the first interval.
    write_loop_heartbeat(start_time=start_time, home=home)
    while True:
        if should_continue is not None and not should_continue():
            return
        await asyncio.sleep(interval)
        if should_continue is not None and not should_continue():
            return
        write_loop_heartbeat(start_time=start_time, home=home)
