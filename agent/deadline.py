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

# One year: semantically "unbounded" yet far below any platform time_t limit (#83220).
MAX_SAFE_TIMEOUT_S = 31_536_000.0

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
    """A stateful backend (MCP connection, browser session, LSP client) ``run_bounded_*`` flags via
    ``mark_suspect`` on timeout so the owner can health-check/recycle it before reuse. ``mark_suspect``
    MUST be cheap, non-blocking, and must not acquire locks the guarded operation may hold — it runs
    inline on the event loop / caller's thread while the wedged worker is still alive."""

    def mark_suspect(self, reason: str) -> None: ...

    def ensure_healthy(self) -> bool: ...


def _mark_backend_suspect(backend: object | None, label: str, timeout_s: float) -> None:
    """Best-effort ``mark_suspect``; never raises, non-adopting backends tolerated."""
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
    """Normalize a timeout: None/non-positive/non-numeric/NaN -> None (unbounded), else capped."""
    if timeout is None:
        return None
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        logger.warning("clamp_timeout: non-numeric timeout %r; treating as unbounded", timeout)
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
        logger.warning("timeouts.%s: invalid value %r in config.yaml; ignoring", key, raw)

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


# --------------------------------------------------------------------------- Bounded execution — async
# flavor. Generalizes plugins/platforms/telegram/adapter.py:_await_with_thread_deadline (the #63309 fix):
# the deadline is driven by a daemon threading.Timer so a blocked event loop cannot disable it, and a second
# timer dumps all thread stacks when the loop provably failed to process the expiry — the one piece of
# information loop-blocked hangs otherwise never surface.
# ---------------------------------------------------------------------------
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
        return _result(start, None, label, value=await awaitable)

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
            return _result(start, timeout_s, label, value=await task)

        _abandon(task)
        if on_abandon is not None:
            asyncio.ensure_future(_run_abandon_cleanup(on_abandon)).add_done_callback(_consume_abandoned)
        # Deliberately INLINE on the loop: the mark must happen-before this result returns
        # AND before the ensure_future'd on_abandon cleanup starts (next tick).
        _mark_backend_suspect(backend, label, timeout_s)
        logger.warning("[deadline] %r timed out after %.1fs; task abandoned", label, timeout_s)
        return _result(start, timeout_s, label, timed_out=True)
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
        return _result(start, None, label, value=fn())

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

    threading.Thread(target=_worker, name=f"deadline-{label}", daemon=True).start()
    deadline = start + timeout_s
    while not done.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        done.wait(min(_BOUNDED_SYNC_WAIT_SLICE_S, remaining))

    if not done.is_set():
        logger.warning("[deadline] %r timed out after %.1fs; worker abandoned", label, timeout_s)
        # Mark suspect BEFORE owner cleanup so a recycle in on_timeout never
        # inherits a stale flag on the healed replacement.
        _mark_backend_suspect(backend, label, timeout_s)
        if on_timeout is not None:
            try:
                on_timeout()
            except Exception:
                logger.debug("deadline on_timeout callback failed", exc_info=True)
        return _result(start, timeout_s, label, timed_out=True)

    if "exc" in box:
        raise box["exc"]
    return _result(start, timeout_s, label, value=box.get("value"))


# --- Whole-tree process termination --------------------------------------------


def kill_process_tree(pid: int, *, sig: Optional[int] = None) -> bool:
    """Terminate ``pid`` and all its descendants, portably; True when anything was signalled.

    Windows: ``taskkill /F /T`` (``sig`` ignored). POSIX: descendants are snapshotted via psutil
    BEFORE signalling (once the parent dies they reparent and a parent walk finds nothing), then
    the process group is signalled when ``pid`` leads one, and every snapshotted descendant
    individually — which also reaches ``setsid`` children. ``sig`` defaults to ``SIGKILL``."""
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
            logger.debug("kill_process_tree: taskkill failed for pid %s", pid, exc_info=True)
            return False

    import signal as _signal
    if sig is None:
        sig = _signal.SIGKILL

    try:
        import psutil
        descendants = psutil.Process(int(pid)).children(recursive=True)
    except Exception:
        # Already gone, or psutil unavailable — the group signal still covers same-session descendants.
        descendants = []

    signalled = False
    try:
        # getpgid→killpg has an inherent TOCTOU shared by every killpg site; the psutil
        # sweep below is identity-aware (PID + create time) and does not.
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    try:
        if pgid is not None and pgid == pid:
            # pid leads its own group (the == check avoids signalling the caller's group).
            os.killpg(pgid, sig)  # windows-footgun: ok — POSIX-only branch (win32 returns above)
        else:
            os.kill(pid, sig)
        signalled = True
    except ProcessLookupError:
        pass
    except (PermissionError, OSError):
        logger.debug("kill_process_tree: signal failed for pid %s", pid, exc_info=True)

    for child in descendants:
        try:
            if child.is_running():  # identity-aware: recycled PIDs skipped
                child.send_signal(sig)
                signalled = True
        except Exception:
            continue
    return signalled
