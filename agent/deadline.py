"""Unified deadline layer — one bounded-execution primitive, one timeout resolver.

Phase 1 of the architectural fix for the timeout/hang backlog
(https://github.com/NousResearch/hermes-agent/issues/85125).

The tree currently carries at least six site-local deadline mechanisms, each
built for one incident, none shared (tool_executor batch deadline, telegram
``_await_with_thread_deadline``, gateway turn lease, reasoning stale floors,
``human_wait_ceiling``, per-MCP-handler timeouts).  Every new stall report
grows that list by one.  This module is the shared foundation the call sites
migrate onto in later phases:

* :func:`resolve_timeout` — one config-first resolution path for timeout
  values (``timeouts:`` section in config.yaml > legacy env var > default),
  so new surfaces stop inventing ``HERMES_*_TIMEOUT`` env vars (".env is for
  secrets only") and hardcoded literals stop ignoring user config
  (#63302, #53161, #43272 class).

* :func:`clamp_timeout` — platform-safe clamping.  Large user-supplied
  timeouts overflow ``time_t`` inside ``threading.Lock.acquire(timeout=...)``
  / ``Thread.join(timeout=...)`` on macOS and kill whole tool batches
  (#83220).  Clamping at the shared boundary fixes that class once, for
  every consumer.

* :func:`run_bounded_async` — a wall-clock deadline for awaitables that does
  NOT depend on event-loop timers.  ``asyncio.wait_for`` schedules its expiry
  on the loop; when the loop thread itself is blocked in a synchronous call
  (family A of the #84047 stall triage), every asyncio-based timeout in the
  process is silently disabled.  This helper drives the deadline from a
  daemon ``threading.Timer`` (generalizing the proven telegram-adapter
  primitive) and abandons cancellation-shielded tasks instead of waiting for
  cancellation to complete.  The telegram adapter's private copy
  (``plugins/platforms/telegram/adapter.py:_await_with_thread_deadline``)
  migrates onto this in Phase 2 of #85125 — do not let the two drift in the
  meantime; fix bugs here first.

* :func:`run_bounded_sync` — the same contract for synchronous callables
  bounded from a synchronous context (daemon worker thread, abandoned on
  expiry).

* :func:`kill_process_tree` — portable whole-tree termination so
  kill-on-timeout stops orphaning descendants (#71148, #59549, #84967,
  #68139 class).  Existing site-local tree-kills that migrate onto this in
  Phase 4 of #85125: ``gateway/status.py`` (taskkill wrapper + psutil
  snapshot/reap pair) and ``tools/code_execution_tool.py`` (psutil
  recursive children kill).

Design invariants:

* Exceptions raised by the bounded operation propagate unchanged — callers
  keep their existing error handling.  Only the *timeout* outcome is
  reified (as :class:`BoundedResult`), because that is the outcome the
  call sites keep getting wrong.
* A timeout produced by this layer is OUR deadline, not the provider's.
  Callers that feed errors into ``agent/error_classifier.py`` should
  classify :class:`DeadlineExpired` distinctly from transport timeouts
  (the #59549 / #80323 misattribution class).
* ``None`` timeout means unbounded, and non-positive resolved values are
  normalized to ``None`` (matching the existing
  ``HERMES_CONCURRENT_TOOL_TIMEOUT_S`` convention).
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SAFE_TIMEOUT_S",
    "BoundedResult",
    "DeadlineExpired",
    "clamp_timeout",
    "resolve_timeout",
    "run_bounded_async",
    "run_bounded_sync",
    "kill_process_tree",
]

# Upper bound for any timeout handed to platform wait primitives.
#
# CPython converts ``threading.Lock.acquire(timeout=...)`` /
# ``Thread.join(timeout=...)`` deadlines to an absolute timestamp; very large
# relative timeouts overflow ``time_t`` on macOS and raise
# ``OverflowError: timestamp out of range for platform time_t`` (#83220).
# One year is semantically "unbounded" for every wait in this codebase while
# staying far below any platform conversion limit.
MAX_SAFE_TIMEOUT_S = 31_536_000.0  # 365 days

# Grace period after a deadline fires before concluding the event loop thread
# is blocked in a synchronous call and dumping stacks (family A diagnostics).
_LOOP_BLOCKED_DUMP_GRACE_S = 5.0


class DeadlineExpired(TimeoutError):
    """A deadline enforced by this layer expired.

    Distinct from transport/provider timeout types on purpose: when this is
    raised (or a :class:`BoundedResult` reports ``timed_out``), the timeout
    was Hermes's own bound — error classification must not attribute it to
    the provider (#59549 / #80323 misattribution class).
    """

    def __init__(self, label: str, timeout_s: float):
        super().__init__(f"deadline expired after {timeout_s:.1f}s: {label}")
        self.label = label
        self.timeout_s = timeout_s


@dataclass(frozen=True, kw_only=True)
class BoundedResult:
    """Outcome of a bounded operation.

    ``timed_out`` is the reified outcome; on completion ``value`` holds the
    operation's return value.  Operation exceptions are never captured here —
    they propagate to the caller unchanged.
    """

    timed_out: bool
    value: Any
    elapsed_s: float
    timeout_s: Optional[float]
    label: str

    def raise_if_timed_out(self) -> Any:
        """Return ``value``, raising :class:`DeadlineExpired` on timeout."""
        if self.timed_out:
            raise DeadlineExpired(self.label, float(self.timeout_s or 0.0))
        return self.value


def clamp_timeout(timeout: Optional[float]) -> Optional[float]:
    """Normalize a timeout value for platform wait primitives.

    * ``None`` stays ``None`` (unbounded).
    * Non-positive values become ``None`` (unbounded) — matching the existing
      ``HERMES_CONCURRENT_TOOL_TIMEOUT_S`` "0 disables the bound" convention.
    * Values above :data:`MAX_SAFE_TIMEOUT_S` are capped so they can never
      overflow ``time_t`` inside ``Lock.acquire`` / ``Thread.join`` on macOS
      (#83220).
    * Non-numeric values are treated as unset (``None``) with a warning
      rather than crashing the call path they were meant to protect.
    """
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
    if value <= 0:
        return None
    return min(value, MAX_SAFE_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Timeout resolution: config.yaml ``timeouts:`` section > legacy env var >
# registered default.
# ---------------------------------------------------------------------------

def _timeouts_section() -> dict:
    """Read the ``timeouts:`` root section from config.yaml (read-only).

    Isolated for testability and so a broken config read can never take down
    the call path the timeout was protecting.
    """
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


def resolve_timeout(
    key: str,
    *,
    default: Optional[float],
    env_var: Optional[str] = None,
) -> Optional[float]:
    """Resolve a timeout in seconds for a dotted config key.

    Precedence (established by the ``providers.*.request_timeout_seconds``
    pattern — config wins over the legacy env var):

    1. ``timeouts.<key>`` in config.yaml (dotted key walks nested maps, e.g.
       ``tools.concurrent_batch`` reads ``timeouts: {tools: {concurrent_batch: ...}}``)
    2. ``env_var`` when set and non-empty (legacy bridge — internal mechanism
       and back-compat only; new surfaces must not grow new user-facing
       ``HERMES_*`` timeout env vars)
    3. ``default``

    The winning value is passed through :func:`clamp_timeout`, so ``0`` or a
    negative value means "unbounded" and oversized values are made
    platform-safe.  Invalid (non-numeric) config/env values fall through to
    the next source with a warning instead of breaking the protected path.
    """
    raw = _lookup_dotted(_timeouts_section(), key)
    if raw is not None:
        # Explicit float() (clamp_timeout would also convert) so that invalid
        # config values FALL THROUGH to the env var / default instead of
        # resolving as unbounded — do not "simplify" this away. bool is
        # rejected because YAML `true` would silently become a 1-second
        # deadline; NaN is rejected for the same fall-through reason.
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


# ---------------------------------------------------------------------------
# Bounded execution — async flavor.
#
# Generalizes plugins/platforms/telegram/adapter.py:_await_with_thread_deadline
# (the #63309 fix): the deadline is driven by a daemon threading.Timer so a
# blocked event loop cannot disable it, and a second timer dumps all thread
# stacks when the loop provably failed to process the expiry — the one piece
# of information loop-blocked hangs otherwise never surface.
# ---------------------------------------------------------------------------

def _consume_abandoned(task: "asyncio.Future[Any]") -> None:
    """Observe an abandoned task's outcome so it never logs 'never retrieved'."""
    try:
        if not task.cancelled():
            task.exception()
    except Exception:
        pass


async def _run_abandon_cleanup(on_abandon: Callable[[], Awaitable[Any]]) -> None:
    """Run abandonment cleanup fully fire-and-forget (its failures swallowed)."""
    try:
        await on_abandon()
    except Exception:
        logger.debug("deadline abandon-cleanup failed", exc_info=True)


def _dump_blocked_loop_diagnostics(label: str, timeout_s: float) -> None:
    logger.warning(
        "[deadline] %r deadline (%.0fs) expired but the event loop has not "
        "processed the expiry after a further %.0fs — the loop thread appears "
        "BLOCKED in a synchronous call, which is why no asyncio timeout can "
        "fire. Dumping all thread stacks to stderr to identify the blocking "
        "frame.",
        label,
        timeout_s,
        _LOOP_BLOCKED_DUMP_GRACE_S,
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
) -> BoundedResult:
    """Await ``awaitable`` under a wall-clock deadline independent of loop timers.

    On completion returns ``BoundedResult(timed_out=False, value=...)``;
    exceptions from the operation (including ``asyncio.CancelledError`` from a
    caller cancelling *us*) propagate unchanged.

    On timeout the underlying task is cancelled and **abandoned** — we do not
    await cancellation completion, because cancellation-shielded scopes (anyio,
    httpcore init, MCP SDK teardown) are exactly the paths that wedge forever.
    ``on_abandon`` (zero-arg callable returning an awaitable) is scheduled as
    detached best-effort cleanup for the half-built state the abandoned task
    may leave behind.  Returns ``BoundedResult(timed_out=True, value=None)``.

    ``timeout=None`` (or a non-positive resolved value) awaits unbounded.
    """
    timeout_s = clamp_timeout(timeout)
    start = time.monotonic()
    if timeout_s is None:
        value = await awaitable
        return BoundedResult(timed_out=False, value=value, elapsed_s=time.monotonic() - start, timeout_s=None, label=label)

    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline: "asyncio.Future[None]" = loop.create_future()
    loop_processed_expiry = threading.Event()

    def _mark_expired() -> None:
        loop_processed_expiry.set()
        if not deadline.done():
            deadline.set_result(None)

    def _expire_from_thread() -> None:
        loop.call_soon_threadsafe(_mark_expired)

    def _watchdog_check() -> None:
        if not loop_processed_expiry.is_set():
            _dump_blocked_loop_diagnostics(label, timeout_s)

    timer = threading.Timer(timeout_s, _expire_from_thread)
    timer.daemon = True
    timer.start()
    watchdog: Optional[threading.Timer] = None
    if dump_on_blocked_loop:
        watchdog = threading.Timer(
            timeout_s + _LOOP_BLOCKED_DUMP_GRACE_S, _watchdog_check
        )
        watchdog.daemon = True
        watchdog.start()
    try:
        try:
            done, _ = await asyncio.wait(
                {task, deadline}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            # The CALLER cancelled us. Without this, `task` would keep running
            # unobserved (and later log "exception was never retrieved") —
            # a leak the telegram original also had. Cancel + abandon it, then
            # let the cancellation propagate.
            task.cancel()
            task.add_done_callback(_consume_abandoned)
            raise
        if task in done:
            if not deadline.done():
                deadline.cancel()
            value = await task
            return BoundedResult(timed_out=False, value=value, elapsed_s=time.monotonic() - start, timeout_s=timeout_s, label=label)

        task.cancel()
        task.add_done_callback(_consume_abandoned)
        if on_abandon is not None:
            cleanup = asyncio.ensure_future(_run_abandon_cleanup(on_abandon))
            cleanup.add_done_callback(_consume_abandoned)
        logger.warning("[deadline] %r timed out after %.1fs; task abandoned", label, timeout_s)
        return BoundedResult(timed_out=True, value=None, elapsed_s=time.monotonic() - start, timeout_s=timeout_s, label=label)
    finally:
        timer.cancel()
        if watchdog is not None:
            watchdog.cancel()
        # cancel() cannot stop a Timer whose callback is already running;
        # setting the event closes that race so a completed await can never
        # be misreported as a blocked loop.
        loop_processed_expiry.set()


# ---------------------------------------------------------------------------
# Bounded execution — sync flavor.
# ---------------------------------------------------------------------------

def run_bounded_sync(
    fn: Callable[[], Any],
    timeout: Optional[float],
    *,
    label: str = "operation",
    on_timeout: Optional[Callable[[], None]] = None,
) -> BoundedResult:
    """Run ``fn`` in a daemon worker thread under a wall-clock deadline.

    On completion returns its value (exceptions re-raised in the caller).
    On expiry the worker thread is **abandoned** (daemon, so it cannot block
    interpreter exit), ``on_timeout`` (if given) runs best-effort in the
    caller's thread — e.g. to mark a backend suspect or kill a subprocess —
    and ``BoundedResult(timed_out=True)`` is returned.

    Intended for infrequent, seconds-scale blocking backend calls. Do NOT
    use per-item in hot loops: each call spawns a thread, and every timeout
    permanently leaks an abandoned daemon thread — a wedged backend called
    in a retry loop would accumulate them.

    ``timeout=None`` (or non-positive) blocks until ``fn`` returns.
    """
    timeout_s = clamp_timeout(timeout)
    start = time.monotonic()
    if timeout_s is None:
        return BoundedResult(timed_out=False, value=fn(), elapsed_s=time.monotonic() - start, timeout_s=None, label=label)

    box: dict[str, Any] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised in caller; must not vanish
            box["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(
        target=_worker, name=f"deadline-{label}", daemon=True
    )
    thread.start()
    if not done.wait(timeout_s):
        logger.warning("[deadline] %r timed out after %.1fs; worker abandoned", label, timeout_s)
        if on_timeout is not None:
            try:
                on_timeout()
            except Exception:
                logger.debug("deadline on_timeout callback failed", exc_info=True)
        return BoundedResult(timed_out=True, value=None, elapsed_s=time.monotonic() - start, timeout_s=timeout_s, label=label)

    if "exc" in box:
        raise box["exc"]
    return BoundedResult(timed_out=False, value=box.get("value"), elapsed_s=time.monotonic() - start, timeout_s=timeout_s, label=label)


# ---------------------------------------------------------------------------
# Whole-tree process termination.
# ---------------------------------------------------------------------------

def kill_process_tree(pid: int, *, sig: Optional[int] = None) -> bool:
    """Terminate ``pid`` and all its descendants, portably.

    Kill-on-timeout that signals only the direct child orphans process trees
    (cron scripts, in-container shells, browser daemons — #71148 class).

    * Windows: ``taskkill /F /T`` terminates the tree (``sig`` ignored;
      Windows has no equivalent). Console-window flash is suppressed via
      ``windows_hide_flags`` and the exit code is checked, so a dead or
      inaccessible PID reports ``False`` like the POSIX path.
    * POSIX: the descendant set is snapshotted via psutil (a hard
      dependency) BEFORE any signal — once the parent dies its children are
      reparented and can no longer be found by a parent walk. Then the
      process group is signalled when ``pid`` leads one (covers
      grandchildren in the same session in one syscall), and every
      snapshotted descendant is signalled individually — which also reaches
      descendants that created their OWN sessions (a child that called
      ``setsid``, exactly what user shell commands do; see
      tools/environments/base.py). ``sig`` defaults to ``SIGKILL``.
      psutil's identity-aware ``Process`` (PID + create time) means a
      recycled PID is never signalled.

    Returns True when the target (or any of its tree) was signalled, False
    when the process was already gone or every termination call failed.
    """
    if sys.platform == "win32":
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            creationflags = windows_hide_flags()
        except Exception:
            creationflags = 0
        try:
            proc = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=15,
                check=False,
                creationflags=creationflags,
            )
            # taskkill exits non-zero for not-found / access-denied; keep the
            # cross-platform contract (False = nothing was terminated).
            return proc.returncode == 0
        except Exception:
            logger.debug("kill_process_tree: taskkill failed for pid %s", pid, exc_info=True)
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
            os.killpg(pgid, sig)  # windows-footgun: ok — POSIX-only branch (win32 returns above)
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
