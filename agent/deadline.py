"""Unified deadline layer — one bounded-execution primitive, one timeout resolver (#85125).

* :func:`resolve_timeout` — ``timeouts:`` in config.yaml > legacy env var > default.
* :func:`clamp_timeout` — huge timeouts overflow ``time_t`` in ``Lock.acquire`` /
  ``Thread.join`` on macOS (#83220), so every timeout is capped.
* :func:`run_bounded_async` / :func:`run_bounded_sync` — wall-clock deadlines driven by
  a daemon ``threading.Timer`` / worker thread, so a blocked event loop cannot disable them.
* :func:`kill_process_tree` — portable whole-tree termination.

Invariants: operation exceptions propagate unchanged (only the *timeout* outcome is reified
as :class:`BoundedResult`); a timeout here is OUR deadline, not the provider's (classify
:class:`DeadlineExpired` distinctly from transport timeouts); ``None`` / non-positive means unbounded.
"""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
import faulthandler
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SAFE_TIMEOUT_S", "BoundedResult", "DeadlineExpired", "clamp_timeout", "resolve_timeout",
    "run_bounded_async", "run_bounded_sync", "kill_process_tree",
]

# Upper bound for any timeout handed to platform wait primitives.
#
# CPython converts ``threading.Lock.acquire(timeout=...)`` /
# ``Thread.join(timeout=...)`` deadlines to an absolute timestamp; very large
# relative timeouts can overflow the platform wait primitive and raise
# ``OverflowError`` (#83220). One year is semantically "unbounded" for every
# wait in this codebase, but Windows exposes a lower CPython primitive limit.
MAX_SAFE_TIMEOUT_S = min(31_536_000.0, threading.TIMEOUT_MAX)

# Grace after a deadline fires before concluding the loop thread is blocked and dumping stacks.
_LOOP_BLOCKED_DUMP_GRACE_S = 5.0

# ``Event.wait`` is a C-level block: KeyboardInterrupt / SetAsyncExc only land when the
# thread returns to Python, so the sync wait is sliced to observe /stop or SIGINT promptly.
# Slice the wait so a /stop or SIGINT during a bounded sync call is observed within this window rather than
# at the full deadline (#94285, tools/test_local_interrupt_cleanup).
_BOUNDED_SYNC_WAIT_SLICE_S = 0.2


class DeadlineExpired(TimeoutError):
    """A deadline enforced by this layer expired (Hermes's own bound, not the provider's)."""

    def __init__(self, label: str, timeout_s: float):
        super().__init__(f"deadline expired after {timeout_s:.1f}s: {label}")
        self.label = label
        self.timeout_s = timeout_s


class SuspectableBackend(Protocol):
    """Phase 3a (#85125): a stateful backend the deadline layer can flag.

    A timed-out stateful backend (MCP connection, browser session, LSP
    client) may be left wedged by the abandoned half-finished operation.
    ``run_bounded_*`` calls ``mark_suspect`` on timeout so the OWNER can
    health-check or recycle the backend before reuse (``ensure_healthy``)
    instead of returning a poisoned handle to the cache. Consumers adopt
    incrementally (Phase 3b, one backend per PR), so the layer fails open:
    backends without the protocol are simply never marked.

    Adopter contract: ``mark_suspect`` MUST be cheap, non-blocking, and
    must not acquire locks the guarded operation may hold. It runs inline —
    on the event loop in the async flavor, and on the caller's thread in
    the sync flavor while the wedged worker is still alive. Set a flag;
    do the expensive health-check/recycle work in ``ensure_healthy``.
    """

    def mark_suspect(self, reason: str) -> None: ...

    def ensure_healthy(self) -> bool: ...


def _mark_backend_suspect(backend: object | None, label: str, timeout_s: float) -> None:
    """Best-effort ``mark_suspect`` on a timed-out call's backend.

    Never raises: adoption state must not be able to weaken the deadline
    bound or corrupt the ``BoundedResult`` the caller is about to receive.
    A non-adopting backend (no ``mark_suspect``) is tolerated silently —
    Phase 3b lands per-backend, so absence is the norm during adoption.
    """
    if backend is None:
        return
    try:
        mark = getattr(backend, "mark_suspect", None)
        if callable(mark):
            mark(f"{label} timed out after {timeout_s:.1f}s")
    except Exception:
        logger.debug("deadline mark_suspect failed", exc_info=True)


@dataclass(frozen=True, kw_only=True)
class BoundedResult:
    """Outcome of a bounded operation; operation exceptions are never captured here."""

    timed_out: bool
    value: Any
    elapsed_s: float
    timeout_s: Optional[float]
    label: str


def _result(start: float, timeout_s: Optional[float], label: str, *, value: Any = None, timed_out: bool = False) -> BoundedResult:
    return BoundedResult(
        timed_out=timed_out, value=value, elapsed_s=time.monotonic() - start, timeout_s=timeout_s, label=label
    )


def clamp_timeout(timeout: Optional[float]) -> Optional[float]:
    """Normalize a timeout value for platform wait primitives.

    * ``None`` stays ``None`` (unbounded).
    * Non-positive values become ``None`` (unbounded) — matching the existing
      ``HERMES_CONCURRENT_TOOL_TIMEOUT_S`` "0 disables the bound" convention.
    * Values above :data:`MAX_SAFE_TIMEOUT_S` are capped so they can never
      overflow the platform primitive inside ``Lock.acquire`` /
      ``Thread.join`` (#83220).
    * Non-numeric values are treated as unset (``None``) with a warning
      rather than crashing the call path they were meant to protect.
    """
    if timeout is None:
        return None
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        logger.warning(
            "clamp_timeout: non-numeric timeout %r; treating as unbounded", timeout
        )
        return None
    if value != value:  # NaN
        logger.warning("clamp_timeout: NaN timeout; treating as unbounded")
        return None
    return None if value <= 0 else min(value, MAX_SAFE_TIMEOUT_S)


# --- Timeout resolution: config ``timeouts:`` > legacy env var > default ------



def _timeouts_section() -> dict:
    """Read the ``timeouts:`` root section from config.yaml (read-only, fail-open)."""
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("timeouts")
        return section if isinstance(section, dict) else {}
    except Exception:
        logger.debug("timeouts: config read failed; using defaults", exc_info=True)
        return {}


def _lookup_dotted(section: dict, key: str) -> Any:
    """Walk ``a.b.c`` through nested dicts; return None when absent."""
    node: Any = section
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def resolve_timeout(key: str, *, default: Optional[float], env_var: Optional[str] = None) -> Optional[float]:
    """Resolve a timeout (seconds): dotted ``timeouts.<key>`` > ``env_var`` > ``default``; the winner
    goes through :func:`clamp_timeout`, invalid config/env values fall through with a warning."""
    raw = _lookup_dotted(_timeouts_section(), key)
    if raw is not None:
        # Explicit float() so invalid config values FALL THROUGH to env/default instead of
        # resolving as unbounded. bool rejected (YAML `true` would become a 1s deadline);
        # NaN rejected for the same fall-through reason.
        if not isinstance(raw, bool):
            try:
                value = float(raw)
                if value == value:  # not NaN
                    return clamp_timeout(value)
            except (TypeError, ValueError):
                pass
        logger.warning(
            "timeouts.%s: invalid value %r in config.yaml; ignoring", key, raw
        )

    if env_var:
        env_raw = os.getenv(env_var, "").strip()
        if env_raw:
            try:
                return clamp_timeout(float(env_raw))
            except ValueError:
                logger.warning("invalid %s=%r; ignoring", env_var, env_raw)

    return clamp_timeout(default)


# --- Bounded execution — async flavor ------------------------------------------
# The deadline is a daemon threading.Timer so a blocked event loop cannot disable it; a
# second timer dumps all thread stacks when the loop provably failed to process the expiry.


def _consume_abandoned(task: "asyncio.Future[Any]") -> None:
    """Observe an abandoned task's outcome so it never logs 'never retrieved'."""
    try:
        if not task.cancelled():
            task.exception()
    except Exception:
        pass


def _abandon(task: "asyncio.Future[Any]") -> None:
    """Cancel ``task`` and never await it; its outcome is consumed so it stays unobserved-safe."""
    task.cancel()
    task.add_done_callback(_consume_abandoned)


async def _run_abandon_cleanup(on_abandon: Callable[[], Awaitable[Any]]) -> None:
    """Run abandonment cleanup fire-and-forget (its failures swallowed)."""
    try:
        await on_abandon()
    except Exception:
        logger.debug("deadline abandon-cleanup failed", exc_info=True)


def _dump_blocked_loop_diagnostics(label: str, timeout_s: float) -> None:
    logger.warning(
        "[deadline] %r deadline (%.0fs) expired but the event loop has not processed the expiry "
        "after a further %.0fs — the loop thread appears BLOCKED in a synchronous call, which is "
        "why no asyncio timeout can fire. Dumping all thread stacks to stderr to identify the "
        "blocking frame.",
        label, timeout_s, _LOOP_BLOCKED_DUMP_GRACE_S,
    )
    try:
        faulthandler.dump_traceback(all_threads=True)
    except Exception:
        logger.debug("faulthandler traceback dump failed", exc_info=True)


async def run_bounded_async(
    awaitable: Awaitable[Any],
    timeout: Optional[float],
    *,
    label: str = "operation",
    on_abandon: Optional[Callable[[], Awaitable[Any]]] = None,
    dump_on_blocked_loop: bool = True,
    backend: object | None = None,
) -> BoundedResult:
    """Await ``awaitable`` under a wall-clock deadline independent of loop timers.

    Operation exceptions (incl. ``CancelledError`` from a caller cancelling *us*) propagate
    unchanged. On timeout the task is cancelled and **abandoned** (never awaited —
    cancellation-shielded scopes are exactly the paths that wedge); ``on_abandon`` runs detached."""
    timeout_s = clamp_timeout(timeout)
    start = time.monotonic()
    if timeout_s is None:
        value = await awaitable
        return BoundedResult(
            timed_out=False,
            value=value,
            elapsed_s=time.monotonic() - start,
            timeout_s=None,
            label=label,
        )

    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline: "asyncio.Future[None]" = loop.create_future()
    loop_processed_expiry = threading.Event()

    def _mark_expired() -> None:
        loop_processed_expiry.set()
        if not deadline.done():
            deadline.set_result(None)

    def _watchdog_check() -> None:
        if not loop_processed_expiry.is_set():
            _dump_blocked_loop_diagnostics(label, timeout_s)

    timers = [threading.Timer(timeout_s, lambda: loop.call_soon_threadsafe(_mark_expired))]
    if dump_on_blocked_loop:
        timers.append(threading.Timer(timeout_s + _LOOP_BLOCKED_DUMP_GRACE_S, _watchdog_check))
    for t in timers:
        t.daemon = True
        t.start()
    try:
        try:
            done, _ = await asyncio.wait({task, deadline}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            _abandon(task)  # the CALLER cancelled us; `task` must not run unobserved
            raise
        if task in done:
            if not deadline.done():
                deadline.cancel()
            value = await task
            return BoundedResult(
                timed_out=False,
                value=value,
                elapsed_s=time.monotonic() - start,
                timeout_s=timeout_s,
                label=label,
            )

        _abandon(task)
        if on_abandon is not None:
            cleanup = asyncio.ensure_future(_run_abandon_cleanup(on_abandon))
            cleanup.add_done_callback(_consume_abandoned)
        # Phase 3a (#85125): the abandoned task may leave the backend
        # half-wedged; flag it so the owner recycles before reuse.
        # Deliberately INLINE on the loop (adopter contract: mark_suspect is
        # cheap and non-blocking). Running it synchronously guarantees the
        # mark happens-before this BoundedResult returns AND before the
        # ensure_future'd on_abandon cleanup can start (next loop tick) — an
        # offloaded mark would race both.
        _mark_backend_suspect(backend, label, timeout_s)
        logger.warning(
            "[deadline] %r timed out after %.1fs; task abandoned", label, timeout_s
        )
        return BoundedResult(
            timed_out=True,
            value=None,
            elapsed_s=time.monotonic() - start,
            timeout_s=timeout_s,
            label=label,
        )
    finally:
        for t in timers:
            t.cancel()
        # cancel() cannot stop a Timer whose callback is already running; setting the
        # event closes that race so a completed await is never misreported as blocked.
        loop_processed_expiry.set()


# --- Bounded execution — sync flavor -------------------------------------------



def run_bounded_sync(
    fn: Callable[[], Any],
    timeout: Optional[float],
    *,
    label: str = "operation",
    on_timeout: Optional[Callable[[], None]] = None,
    backend: object | None = None,
) -> BoundedResult:
    """Run ``fn`` in a daemon worker thread under a wall-clock deadline; exceptions re-raise in
    the caller. On expiry the worker is **abandoned** (every timeout leaks one daemon thread, so
    do NOT use per-item in hot loops) and ``on_timeout`` runs best-effort in the caller's thread.
    The worker runs under ``contextvars.copy_context()`` so secret scope / session id survive.

    See #94285.
    """
    timeout_s = clamp_timeout(timeout)
    start = time.monotonic()
    if timeout_s is None:
        return BoundedResult(
            timed_out=False,
            value=fn(),
            elapsed_s=time.monotonic() - start,
            timeout_s=None,
            label=label,
        )

    box: dict[str, Any] = {}
    done = threading.Event()
    ctx = contextvars.copy_context()

    def _worker() -> None:
        try:
            box["value"] = ctx.run(fn)
        except BaseException as exc:  # re-raised in caller; must not vanish
            box["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, name=f"deadline-{label}", daemon=True)
    thread.start()
    if not done.wait(timeout_s):
        logger.warning(
            "[deadline] %r timed out after %.1fs; worker abandoned", label, timeout_s
        )
        # Phase 3a (#85125), ordering: mark suspect BEFORE owner cleanup so a
        # recycle/re-init in on_timeout never gets a stale flag on the healed
        # replacement. The sync flavor runs the mark inline — the protocol
        # contract requires mark_suspect to be cheap.
        _mark_backend_suspect(backend, label, timeout_s)
        if on_timeout is not None:
            try:
                on_timeout()
            except Exception:
                logger.debug("deadline on_timeout callback failed", exc_info=True)
        return BoundedResult(
            timed_out=True,
            value=None,
            elapsed_s=time.monotonic() - start,
            timeout_s=timeout_s,
            label=label,
        )

    if "exc" in box:
        raise box["exc"]
    return BoundedResult(
        timed_out=False,
        value=box.get("value"),
        elapsed_s=time.monotonic() - start,
        timeout_s=timeout_s,
        label=label,
    )


# --- Whole-tree process termination --------------------------------------------


@contextmanager
def _process_tree_snapshot(pid: int, *, hard_kill: bool):
    """Stop each hard-kill target before discovering its children: a running
    parent can fork after psutil builds its PID map and escape the final signal.
    Resume anything we stopped if signalling fails. Graceful signals never stop
    their recipients, since their handlers must remain able to run.
    """
    descendants = []
    stopped = []
    try:
        try:
            import psutil
            root = psutil.Process(pid)
            descendants = root.children(recursive=True)
            if hard_kill:
                pending = [root]
                seen = {os.getpid()}
                known = {process.pid: process for process in descendants}
                stop_deadline = time.monotonic() + 1.0
                while pending:
                    for process in pending:
                        if process.pid in seen:
                            continue
                        seen.add(process.pid)
                        if time.monotonic() >= stop_deadline:
                            raise TimeoutError("process tree did not stop before snapshot deadline")
                        try:
                            status = process.status()
                            if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                                continue
                            if status != psutil.STATUS_STOPPED:
                                process.suspend()
                                stopped.append(process)
                            while process.status() != psutil.STATUS_STOPPED:
                                if time.monotonic() >= stop_deadline:
                                    raise TimeoutError("process did not stop before snapshot deadline")
                                time.sleep(0.001)
                        except psutil.NoSuchProcess:
                            continue
                    # Rescan after stopping the discovered generation. A child
                    # may have forked while that generation was being stopped.
                    for process in root.children(recursive=True):
                        known.setdefault(process.pid, process)
                    descendants = list(known.values())
                    pending = [process for process in descendants if process.pid not in seen]
        except Exception:
            # Preserve the existing best-effort group fallback when discovery or
            # stopping is unavailable; never strand a successfully stopped target.
            logger.debug("kill_process_tree: snapshot incomplete for pid %s", pid, exc_info=True)
        yield descendants
    finally:
        for process in stopped:
            try:
                process.resume()
            except Exception:
                logger.debug("kill_process_tree: target already gone or resume refused", exc_info=True)



def kill_process_tree(pid: int, *, sig: Optional[int] = None) -> bool:
    """Terminate ``pid`` and all its descendants, portably; True when anything was signalled.

    Windows: ``taskkill /F /T`` (``sig`` ignored). POSIX: snapshot descendants via
    psutil; for SIGKILL, stop and rescan the live tree so a concurrent fork cannot
    escape a stale snapshot. Signal identity-checked descendants before their
    parent, then its group when ``pid`` leads one. Stopping is best-effort with a
    bounded wait; unavailable psutil still leaves process-group cleanup. Other
    signals do not suspend recipients. ``sig`` defaults to ``SIGKILL``."""
    if sys.platform == "win32":
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags
            creationflags = windows_hide_flags()
        except Exception:
            creationflags = 0
        try:
            proc = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=15, check=False, creationflags=creationflags,
            )
            # taskkill exits non-zero for not-found / access-denied (False = nothing terminated).
            return proc.returncode == 0
        except Exception:
            logger.debug(
                "kill_process_tree: taskkill failed for pid %s", pid, exc_info=True
            )
            return False

    import signal as _signal
    if sig is None:
        sig = _signal.SIGKILL

    # Snapshot descendants while the parent is still alive — after it dies
    # they reparent to init/subreaper and a parent walk finds nothing.
    descendants: list = []
    try:
        import psutil

        descendants = psutil.Process(int(pid)).children(recursive=True)
    except Exception:
        # Already gone, or psutil unavailable in a stripped env — the
        # group-signal below still covers same-session descendants.
        descendants = []

    signalled = False
    try:
        # NOTE: getpgid→killpg has an inherent TOCTOU (pid could be reaped and
        # recycled between the calls). All existing killpg sites share it; the
        # psutil sweep below is identity-aware and does not.
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    try:
        if pgid is not None and pgid == pid:
            # pid leads its own group: one syscall covers the whole group.
            # (The == check guards against signalling the caller's own group
            # when pid is not a leader.)
            os.killpg(  # windows-footgun: ok — POSIX-only branch (win32 returns above)
                pgid, sig
            )
        else:
            os.kill(pid, sig)
        signalled = True
    except ProcessLookupError:
        pass
    except (PermissionError, OSError):
        logger.debug("kill_process_tree: signal failed for pid %s", pid, exc_info=True)

    # Sweep the snapshot: reaches descendants outside the parent's group
    # (their own setsid sessions) and the non-group-leader case.
    for child in descendants:
        try:
            if child.is_running():  # identity-aware: recycled PIDs skipped
                child.send_signal(sig)
                signalled = True
        except Exception:
            continue
    return signalled


class SuspectableBackend:
    """Protocol for backends whose connection state can be *poisoned* by a
    race (teardown-vs-keepalive, auth-lock corruption) without the backend
    itself being dead.

    The contract is **cheap-mark, lazy-verify**: noticing a poisoned state
    must never do I/O — ``mark_suspect`` just latches a reason string. The
    NEXT caller pays for verification once, via ``ensure_healthy``: a cheap
    health probe that either clears the suspicion (backend was fine) or
    forces a reconnect/recycle before the call proceeds. This is what keeps
    a single race from permanently parking a connection (#81051/#77765/
    #84132): instead of parking on the ambiguous event, the backend is
    marked suspect and recycled exactly once on next use.
    """

    def mark_suspect(self, reason: str) -> None:
        """Latch a suspicion about this backend. Must be cheap (no I/O)."""
        raise NotImplementedError

    async def ensure_healthy(self, timeout: float = 5.0) -> bool:
        """Verify a suspect backend before reuse.

        Returns True when the backend is healthy (clearing the suspicion);
        returns False after forcing a reconnect/recycle so the caller's
        normal no-session path handles the rebuild. Must not raise.
        """
        raise NotImplementedError
