"""Report what a stopping ``hermes serve`` is still draining on (#140411).

uvicorn is given a 3 s graceful window (Desktop escalates to SIGKILL ~5 s after SIGTERM), but the
phases that follow it have no bound of their own: cancelling the remaining ASGI tasks and awaiting
them, then lifespan shutdown with its cron/agent teardown. A request that cannot finish inside that
window keeps the process alive until systemd's ``TimeoutStopSec`` and is then SIGKILLed — for a
remote Desktop backend that is a multi-minute 502 outage — while the journal says nothing at all
about what held the drain. The only route left is ptrace/py-spy on a process that is already dying.

This reporter runs on its own thread, so it keeps reporting when the event loop is the thing that is
stuck: every ``interval`` seconds it logs how long the drain has run and what is still pending
(ASGI tasks, open connections, the loop's own named tasks, in-flight session turns), and once the
drain outlives ``stack_dump_after`` it dumps every thread's stack — the frames name the blocking
call, which is exactly what the operator otherwise has to reconstruct after the SIGKILL.

Diagnostics only: nothing here changes what the drain waits for, and every failure is swallowed so a
broken reporter can never turn a slow stop into a failed one.
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import logging
import signal
import sys
import threading
import time
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)

DRAIN_REPORT_INTERVAL_S = 2.0
# Well inside systemd's stop budget (180 s in the report) so the dump lands in the journal with room
# to read it, and past uvicorn's 3 s graceful window so a normal stop never dumps.
DRAIN_STACK_DUMP_AFTER_S = 20.0
# Names are for a human reading one log line; a whole-page list would bury the counts.
_MAX_LOGGED_TASK_NAMES = 6


def _task_name(task: Any) -> str:
    """Readable name for one asyncio task: its own name, plus the coroutine's qualified name.

    uvicorn's per-request tasks carry no name, so the coroutine is the only part that says which
    endpoint (or which harness: ``_wait_for_disconnect``, ``stream_response``) is stuck.
    """
    name = ""
    with contextlib.suppress(Exception):
        name = str(task.get_name())
    with contextlib.suppress(Exception):
        coro = task.get_coro()
        qualified = getattr(coro, "__qualname__", None) or getattr(coro, "__name__", None)
        if qualified and qualified not in name:
            name = f"{name}:{qualified}" if name else str(qualified)
    return name or "<unnamed>"


def pending_work(server: Any = None, loop: Any = None) -> dict[str, Any]:
    """Snapshot of what a stopping serve is still waiting on. Never raises.

    ``server`` is the uvicorn ``Server`` (its ``server_state`` holds the ASGI tasks and sockets it
    is about to cancel/close); ``loop`` is the serve event loop, whose tasks are the ones a wedged
    shutdown is usually inside. Either may be missing — a caller with no handle still gets counts
    from whatever it has.
    """
    state = getattr(server, "server_state", None)
    asgi_tasks = list(getattr(state, "tasks", []) or [])
    connections = list(getattr(state, "connections", []) or [])
    loop_tasks: list[Any] = []
    if loop is not None:
        with contextlib.suppress(Exception):
            loop_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    return {
        "asgi_tasks": len(asgi_tasks),
        "connections": len(connections),
        "loop_tasks": len(loop_tasks),
        "loop_task_names": [_task_name(task) for task in loop_tasks],
        "running_sessions": _running_session_turns(),
    }


def _running_session_turns() -> int:
    """In-memory sessions still mid-turn in this process (serve keeps them in the TUI gateway).

    That registry is the dashboard backend's own answer to "is a turn running?", so it is what the
    long-reporting stops in #140411 had: an empty serve with no turn at all stops promptly. Imported
    lazily — the reporter must stay importable from paths that never load the gateway.
    """
    with contextlib.suppress(Exception):
        from tui_gateway import server as tui_server

        with tui_server._sessions_lock:  # read-only snapshot of the live registry
            return sum(1 for session in tui_server._sessions.values() if session.get("running"))
    return 0


def format_drain_progress(elapsed_s: float, work: dict[str, Any]) -> str:
    """One journal line: how long the stop has been draining, and what is still pending."""
    parts = [
        f"{work.get('asgi_tasks', 0)} ASGI task(s)",
        f"{work.get('connections', 0)} open connection(s)",
        f"{work.get('running_sessions', 0)} running turn(s)",
    ]
    names = [name for name in (work.get("loop_task_names") or []) if name]
    if names:
        shown = ", ".join(names[:_MAX_LOGGED_TASK_NAMES])
        more = "" if len(names) <= _MAX_LOGGED_TASK_NAMES else f", +{len(names) - _MAX_LOGGED_TASK_NAMES} more"
        parts.append(f"{work.get('loop_tasks', len(names))} loop task(s): {shown}{more}")
    return f"serve stop: still draining after {int(elapsed_s)}s — " + "; ".join(parts)


def dump_all_thread_stacks() -> None:
    """Write every thread's stack to stderr (the journal, for a systemd unit)."""
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


class DrainReporter:
    """Thread-side reporter for one serve stop. ``start()`` is idempotent and never raises."""

    def __init__(
        self,
        *,
        server: Any = None,
        loop: Any = None,
        interval: float = DRAIN_REPORT_INTERVAL_S,
        stack_dump_after: float = DRAIN_STACK_DUMP_AFTER_S,
        dump_stacks: Optional[Callable[[], None]] = None,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self.server = server
        self.loop = loop
        self.interval = max(0.01, float(interval))
        self.stack_dump_after = max(0.0, float(stack_dump_after))
        self._dump_stacks = dump_stacks or dump_all_thread_stacks
        self._mono = mono
        self._started_at: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dumps = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_at = self._mono()
        # Spawned through the context-bound helper: the thread only reads and logs, but a bare
        # Thread would land on the default profile's HERMES_HOME if that ever changes.
        from agent.memory_provider import spawn_context_thread

        self._thread = spawn_context_thread(self._run, name="serve-drain-report", daemon=True)
        with contextlib.suppress(Exception):
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def elapsed_s(self) -> float:
        return 0.0 if self._started_at is None else max(0.0, self._mono() - self._started_at)

    def report_once(self) -> str:
        """Log (and, past the threshold, dump) one progress line; returns it for the tests."""
        line = format_drain_progress(self.elapsed_s(), pending_work(self.server, self.loop))
        _log.warning("%s", line)
        if self.dumps == 0 and self.elapsed_s() >= self.stack_dump_after:
            self.dumps += 1
            _log.warning(
                "serve stop: the drain has outlived %.0fs; dumping every thread's stack below "
                "(the frame that does not return is what systemd's TimeoutStopSec is waiting on)",
                self.stack_dump_after,
            )
            with contextlib.suppress(Exception):
                self._dump_stacks()
        return line

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with contextlib.suppress(Exception):  # never let the reporter kill the drain it reports
                self.report_once()


def report_serve_drain(
    server: Any,
    *,
    loop: Any = None,
    interval: float = DRAIN_REPORT_INTERVAL_S,
    stack_dump_after: float = DRAIN_STACK_DUMP_AFTER_S,
) -> Optional[DrainReporter]:
    """Wrap uvicorn's stop signals so the drain itself is reported; ``None`` when it cannot be.

    Must be called INSIDE ``server.capture_signals()``: uvicorn registers its own ``handle_exit``
    there, and the pre-capture handlers are only re-raised *after* a graceful shutdown — far too
    late to describe it. Chaining to ``server.handle_exit`` keeps the stop exactly as it was; only
    the reporting is added. Off the main thread (no signal delivery) this is a no-op.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    reporter = DrainReporter(server=server, loop=loop, interval=interval, stack_dump_after=stack_dump_after)
    handle_exit = getattr(server, "handle_exit", None)
    if not callable(handle_exit):
        return None

    def _handler(signum, frame) -> None:
        reporter.start()
        handle_exit(signum, frame)

    installed = False
    for signum in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError, RuntimeError):
            signal.signal(signum, _handler)
            installed = True
    if not installed:
        return None
    _log.debug("serve stop: drain progress reporting installed")
    return reporter
