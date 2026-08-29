"""Out-of-loop shutdown and event-loop liveness backstops (#66892, #69089).

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
3. A lifetime thread watchdog that can still diagnose and hard-exit when the
   event loop is too frozen to run its own heartbeat or timeout callbacks.
4. A self-rescheduling floor timer that keeps the loop selector's timeout
   finite, giving existing async recovery tasks a chance to resume.
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

from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Extra leash beyond ``agent.restart_drain_timeout`` so a slow-but-progressing
# drain is not cut short. Matches the issue #66892 suggested hardening.
DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S = 60.0
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S = 5.0
DEFAULT_LOOP_WATCHDOG_INTERVAL_S = 30.0
DEFAULT_LOOP_WATCHDOG_TIMEOUT_S = 10.0
# 3 sustained misses (~90-120s of loop block) escalate. The false-positive
# class that motivated raising this (the watchdog's own on-loop heartbeat
# fsync stalling the loop it monitors) is fixed at the root by the off-loop
# heartbeat write + two-witness probe (#90502), so the default stays tight
# for genuine wedges. Deployments with legitimately slow loops can tune via
# gateway.loop_watchdog_* in config.yaml.
DEFAULT_LOOP_WATCHDOG_MAX_STRIKES = 3
_HEARTBEAT_RELATIVE = ("state", "gateway.heartbeat")
_WATCHDOG_DUMP_RELATIVE = ("logs", "gateway-shutdown-watchdog.log")


class _LoopFloorTimerHandle:
    """Cancelable owner for the currently scheduled selector floor timer."""

    def __init__(self, loop: asyncio.AbstractEventLoop, interval: float):
        self._loop = loop
        self._interval = interval
        self._cancelled = False
        self._timer: Optional[asyncio.TimerHandle] = None
        self._schedule()

    def _schedule(self) -> None:
        self._timer = self._loop.call_later(self._interval, self._tick)

    def _tick(self) -> None:
        if not self._cancelled:
            self._schedule()

    def cancel(self) -> None:
        self._cancelled = True
        if self._timer is not None:
            self._timer.cancel()


class _LoopLivenessWatchdogHandle:
    """Small lifecycle handle for the daemon liveness thread."""

    def __init__(self, stop_event: threading.Event, thread: threading.Thread):
        self._stop_event = stop_event
        self._thread = thread

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def _arm_loop_floor_timer(
    loop: asyncio.AbstractEventLoop,
    interval: float = DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S,
) -> _LoopFloorTimerHandle:
    """Keep at least one timer pending so selector waits remain bounded."""
    try:
        resolved_interval = float(interval)
        if resolved_interval <= 0:
            raise ValueError
    except (TypeError, ValueError):
        resolved_interval = DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S
    return _LoopFloorTimerHandle(loop, resolved_interval)


def start_loop_liveness_watchdog(
    loop: asyncio.AbstractEventLoop,
    *,
    probe_interval: float = DEFAULT_LOOP_WATCHDOG_INTERVAL_S,
    probe_timeout: float = DEFAULT_LOOP_WATCHDOG_TIMEOUT_S,
    max_strikes: int = DEFAULT_LOOP_WATCHDOG_MAX_STRIKES,
    exit_code: int = GATEWAY_SERVICE_RESTART_EXIT_CODE,
) -> Optional[_LoopLivenessWatchdogHandle]:
    """Start an out-of-loop watchdog that hard-exits after missed probes.

    The guard is on by default; operators opt out with
    ``gateway.loop_watchdog: false`` in config.yaml (enforced by the caller,
    ``GatewayRunner._start_loop_liveness_guards`` — this module stays
    config-agnostic so bare-loop tests can drive it directly).
    """
    interval = probe_interval
    timeout = probe_timeout
    strikes_limit = max_strikes
    stop_event = threading.Event()

    def _wait_for_probe(probe_event: threading.Event) -> Optional[bool]:
        deadline = time.monotonic() + timeout
        while True:
            if stop_event.is_set():
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return probe_event.is_set()
            if probe_event.wait(timeout=min(remaining, 0.05)):
                return True

    def _watchdog() -> None:
        strikes = 0
        while not stop_event.wait(timeout=interval):
            probe_event = threading.Event()
            try:
                loop.call_soon_threadsafe(probe_event.set)
            except RuntimeError:
                # A normally closed loop cannot be probed and no longer needs
                # a process-liveness backstop.
                return
            except Exception:
                logger.debug(
                    "Failed to schedule gateway loop liveness probe", exc_info=True
                )
                return

            responded = _wait_for_probe(probe_event)
            if responded is None:
                return
            if responded:
                strikes = 0
                continue

            if stop_event.is_set():
                return
            strikes += 1
            if strikes < strikes_limit:
                continue

            if stop_event.is_set():
                return
            try:
                logger.critical(
                    "Gateway event loop missed %d consecutive liveness probes; "
                    "dumping all thread stacks and exiting with code %d so the "
                    "service supervisor can restart it.",
                    strikes,
                    exit_code,
                )
            except Exception:
                pass
            try:
                faulthandler.dump_traceback(all_threads=True)
            except Exception:
                logger.debug("Loop liveness faulthandler dump failed", exc_info=True)
            if stop_event.is_set():
                return
            # Record the watchdog exit in the lifecycle sentinel so the next
            # boot reports "watchdog hard-exit" instead of misclassifying
            # this as an unclean SIGKILL/OOM death (NS-608).
            try:
                from gateway.lifecycle_ledger import mark_exited
                mark_exited(exit_code, reason="loop_liveness_watchdog")
            except Exception:
                pass
            os._exit(exit_code)
            return

    thread = threading.Thread(
        target=_watchdog,
        daemon=True,
        name="gateway-loop-liveness-watchdog",
    )
    try:
        thread.start()
    except Exception:
        logger.debug("Failed to start gateway loop liveness watchdog", exc_info=True)
        return None
    return _LoopLivenessWatchdogHandle(stop_event, thread)


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


def get_loop_tick_socket_path(
    home: Optional[Path] = None, pid: Optional[int] = None
) -> Path:
    """Return the loop-scheduling witness socket for ``pid``.

    ``<HERMES_HOME>/state/gateway.loop-tick.<pid>.sock`` — PID-suffixed so a
    leftover node from a previous process can never be mistaken for this
    gateway's witness. Served by the gateway loop itself (see
    ``_tick_socket_handler``): an answer is direct proof that the loop is
    dispatching, which is exactly the property the heartbeat file lost when
    its write moved off-loop (#90502).
    """
    base = home if home is not None else _process_hermes_home()
    return base.joinpath(
        "state", f"gateway.loop-tick.{int(pid if pid is not None else os.getpid())}.sock"
    )


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
    # Embed a cheap memory sample (own RSS + MemAvailable + swap) so the
    # heartbeat doubles as a rolling pre-death telemetry snapshot: after an
    # unclean death (SIGKILL/OOM/VM loss) the last heartbeat is the closest
    # surviving record of memory pressure — see gateway.lifecycle_ledger
    # (NS-608).  Best-effort; <1ms of /proc reads on Linux, {} elsewhere.
    try:
        from gateway.lifecycle_ledger import sample_memory

        mem = sample_memory()
        if mem:
            payload["mem"] = mem
    except Exception:
        pass
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
        # Record the watchdog exit so the next boot's unclean-death detector
        # reports "shutdown watchdog fired" instead of SIGKILL/OOM (NS-608).
        try:
            from gateway.lifecycle_ledger import mark_exited
            mark_exited(exit_code, reason="shutdown_watchdog")
        except Exception:
            pass
        os._exit(exit_code)

    try:
        threading.Thread(target=_watchdog, daemon=True, name=name).start()
    except Exception:
        logger.debug("Failed to arm shutdown watchdog", exc_info=True)
    return done


async def _tick_socket_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Answer a liveness ping with one byte.

    Runs on the gateway loop: the reply is produced only while the loop is
    actually dispatching, so a successful read is a witness of loop
    schedulability that no executor thread and no filesystem stall can
    refresh. A UNIX-socket write is a socket-buffer copy — no fsync, no
    disk I/O — so the witness keeps working on the exact filesystem that
    stalls the heartbeat write. Best-effort; never raises.
    """
    try:
        writer.write(b"1")
        await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def loop_heartbeat_forever(
    *,
    interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    start_time: Optional[float] = None,
    home: Optional[Path] = None,
    should_continue: Optional[Callable[[], bool]] = None,
) -> None:
    """Rewrite the loop heartbeat file on a cadence until cancelled / gated off.

    Runs as an asyncio task on the gateway loop — if the loop freezes, this task
    stops and the file mtime/updated_at goes stale for external monitors. That
    property is load-bearing and is preserved below: the write is still
    *initiated* by the loop, so a frozen loop still lets the file age.

    The write itself is handed to a thread, because it is not free. It ends in
    ``atomic_json_write`` -> ``os.fsync``, and on a filesystem that stalls, that
    fsync blocks whatever thread runs it. Doing it inline meant the loop-liveness
    watchdog's own heartbeat could block the loop it exists to monitor: the probe
    times out at ``DEFAULT_LOOP_WATCHDOG_TIMEOUT_S`` (10s) and gives up after
    ``DEFAULT_LOOP_WATCHDOG_MAX_STRIKES`` (3), a ~90-120s budget, while a WSL2
    VHDX under io pressure was measured stalling a trivial stat-and-fsync probe
    at p99 31s and max 112s. So the watchdog killed the loop for being
    unresponsive at the moment it was blocked inside the watchdog's own write.

    Awaited, not fire-and-forget: an unawaited task would keep the file fresh
    while the loop was wedged, which is exactly the signal the docstring above
    promises. And a single in-flight write at a time, so a 112s stall cannot pile
    up one queued thread per interval behind it.

    Because the write is now off-loop, file freshness is no longer *proof* of
    loop schedulability: a stalled write or a saturated executor can age the file
    while the loop runs, and a write that lands after the loop froze can keep it
    fresh. The file therefore stops being sufficient authority on its own. This
    task also arms a loop-scheduling witness — a UNIX socket answered by the
    loop itself (``_tick_socket_handler``) — and records whether it is armed in
    the heartbeat payload (``loop_tick_socket``). External probes must require
    the witness to agree with file staleness before classifying a loop as
    wedged; see ``hermes_cli.gateway.probe_gateway_loop_liveness`` for the
    two-witness contract.
    """
    try:
        interval = max(float(interval_s), 1.0)
    except (TypeError, ValueError):
        interval = DEFAULT_HEARTBEAT_INTERVAL_S

    # Arm the loop-scheduling witness. Best-effort: a failed bind (permissions,
    # path length) must not abort the gateway or the file heartbeat — it only
    # disables the witness, and the payload flag tells probes that staleness is
    # no longer sufficient authority to escalate.
    #
    # Windows: asyncio.start_unix_server raises (no AF_UNIX event-loop
    # support), so the witness is PERMANENTLY absent there — the payload
    # records loop_tick_socket=False and every stale-file probe classifies
    # UNKNOWN, never WEDGED. That is deliberate fail-safe: a wedged native
    # Windows gateway keeps the graceful-drain backstop instead of an
    # escalation verdict built on a witness that cannot exist. (WSL2 — the
    # #90502 incident environment — is Linux and arms the socket normally.)
    tick_server = None
    tick_socket_path = None
    try:
        tick_socket_path = get_loop_tick_socket_path(home)
        tick_socket_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-bind over a leftover node from a dead process (os._exit(75) /
        # SIGKILL skip the finally-unlink; PID reuse re-lands on this
        # PID-suffixed path) is handled by asyncio itself:
        # create_unix_server os.remove()s an existing socket node before
        # binding — guarded by test_producer_rebinds_over_stale_socket_node.
        # What asyncio does NOT do is clean up SIBLING nodes from other
        # dead PIDs, so sweep those to keep state/ from accumulating
        # gateway.loop-tick.*.sock nodes across crash-restart cycles.
        # POSIX-only: os.kill(pid, 0) is a liveness probe here, but on
        # Windows os.kill calls TerminateProcess for non-CTRL signals —
        # and AF_UNIX server nodes are never created there anyway.
        if os.name == "posix":
            try:
                for _stale in tick_socket_path.parent.glob(
                    "gateway.loop-tick.*.sock"
                ):
                    if _stale == tick_socket_path:
                        continue
                    try:
                        _stale_pid = int(_stale.name.split(".")[-2])
                    except (ValueError, IndexError):
                        _stale.unlink(missing_ok=True)
                        continue
                    try:
                        os.kill(_stale_pid, 0)  # windows-footgun: ok — inside os.name == "posix" gate
                    except OSError:
                        _stale.unlink(missing_ok=True)
            except Exception:
                logger.debug(
                    "stale loop-tick socket sweep failed", exc_info=True
                )
        tick_server = await asyncio.start_unix_server(
            _tick_socket_handler, path=str(tick_socket_path)
        )
    except Exception:
        tick_server = None
        logger.warning(
            "Loop tick socket unavailable — liveness probes will have no "
            "loop-scheduling witness and will not escalate on a stale heartbeat",
            exc_info=True,
        )

    async def _write_off_loop() -> None:
        # write_loop_heartbeat never raises, so a failure here is an executor
        # problem (shutdown, saturation) and must not kill the heartbeat task.
        try:
            await asyncio.to_thread(
                write_loop_heartbeat,
                start_time=start_time,
                home=home,
                extra={"loop_tick_socket": tick_server is not None},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Loop heartbeat write failed off-loop", exc_info=True)

    try:
        # Immediate first write so monitors see a fresh file as soon as the
        # gateway is running, not after the first interval.
        await _write_off_loop()
        while True:
            if should_continue is not None and not should_continue():
                return
            await asyncio.sleep(interval)
            if should_continue is not None and not should_continue():
                return
            await _write_off_loop()
    finally:
        if tick_server is not None:
            tick_server.close()
            try:
                await tick_server.wait_closed()
            except Exception:
                pass
            if tick_socket_path is not None:
                try:
                    tick_socket_path.unlink(missing_ok=True)
                except Exception:
                    pass
