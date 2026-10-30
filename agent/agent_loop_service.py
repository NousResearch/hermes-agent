"""The plugin-facing ``agent_loop`` capability: in-process agent runs on a bounded thread pool.

``PluginContext.agent_loop`` returns a :class:`PluginAgentLoop`. Each accepted call runs on its
own daemon thread — ``agent.memory_provider.spawn_context_thread`` copies the caller's
contextvars, so the profile home, secret scope and recursion depth follow the call — up to
``agent_loop.max_workers`` threads
at once. Beyond that, calls wait FIFO in a bounded queue (``agent_loop.max_queued``); nothing is
dropped silently. N concurrent calls therefore use N threads rather than one shared worker, and
each pool slot is a genuinely independent run.

Parameters mirror :func:`agent.agent_run.run`, which is what a slot executes. The facade returns
:class:`agent.agent_run.AgentRunResult`; :meth:`PluginAgentLoop.submit` returns an
:class:`AgentLoopJob` whose ``future`` settles with that result (or the failure).

A run that holds its slot past ``agent_loop.max_run_seconds`` is settled as stalled and its slot
released: a thread cannot be killed, but one wedged run must not refuse every later call. Only
the first settle of a job is delivered, so a late-finishing stalled run cannot report twice.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Bumped when the callable surface changes so a plugin can feature-detect.
CAPABILITY_VERSION = "agent_loop_v1"
#: Defaults; ``agent_loop:`` in config.yaml overrides each one.
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_QUEUED = 64
DEFAULT_MAX_RUN_SECONDS = 1800.0
DEFAULT_STALL_CHECK_SECONDS = 30.0

_SEQUENCE = itertools.count(1)


class AgentLoopQueueFull(RuntimeError):
    """The pool's waiting line is full: the caller gets an error instead of an unbounded queue."""


@dataclass
class AgentLoopJob:
    """One submitted run: its id, its handle, and when it took a slot."""

    job_id: str
    label: str
    session_key: str
    future: Future = field(default_factory=Future)
    dispatched_at: float = field(default_factory=time.time)
    #: When a worker thread picked the job up; 0 while it waits.
    started_at: float = 0.0
    #: 1-based place in the waiting line at submission; 0 when it started immediately.
    queue_position: int = 0
    #: Settle callback (see ``submit``), kept on the job so a stall or a failed thread start
    #: still delivers it — exactly once, like every other outcome.
    on_settle: Optional[Callable[..., None]] = field(default=None, repr=False, compare=False)

    @property
    def result(self) -> Callable[[Optional[float]], Any]:
        """``job.future.result([timeout])`` — the run's result, or the failure that ended it."""
        return self.future.result


@dataclass
class _Entry:
    job: AgentLoopJob
    thread: threading.Thread


class AgentLoopService:
    """A bounded pool of one-thread-per-run slots (see the module docstring)."""

    def __init__(self, *, max_workers: int = DEFAULT_MAX_WORKERS,
                 max_queued: int = DEFAULT_MAX_QUEUED,
                 max_run_seconds: float = DEFAULT_MAX_RUN_SECONDS,
                 stall_check_seconds: float = DEFAULT_STALL_CHECK_SECONDS) -> None:
        self._lock = threading.RLock()
        self._active: dict[str, _Entry] = {}
        self._pending: list[_Entry] = []
        self._max_workers = max(1, int(max_workers))
        self._max_queued = max(1, int(max_queued))
        self._max_run_seconds = max(0.0, float(max_run_seconds))
        self._stall_check_seconds = max(1.0, float(stall_check_seconds))
        self._watchdog: Optional[threading.Thread] = None
        self._stopping = threading.Event()

    # -- submission --------------------------------------------------------------

    def submit(self, task: Callable[[], Any], *, label: str = "", session_key: str = "",
               on_settle: Optional[Callable[[AgentLoopJob, Any, Optional[BaseException]], None]] = None
               ) -> AgentLoopJob:
        """Submit one callable; it starts now when a slot is free, else waits FIFO.

        ``on_settle`` runs exactly once per job — completion, failure or stall — after the
        future is settled. It is the notification seam for background callers; it must not raise
        (a raising callback is logged and ignored).
        """
        from agent.memory_provider import spawn_context_thread

        job = AgentLoopJob(job_id=f"agent-loop-{next(_SEQUENCE)}", label=str(label or ""),
                           session_key=str(session_key or ""), on_settle=on_settle)
        thread = spawn_context_thread(lambda: self._work(job, task),
                                      name=f"agent-loop-{job.label or job.job_id}"[:48])
        with self._lock:
            if len(self._active) < self._max_workers:
                self._register_locked(job, thread)
                start_now, queued = True, 0
            else:
                if len(self._pending) >= self._max_queued:
                    raise AgentLoopQueueFull(
                        f"agent_loop pool is full ({len(self._active)} running, "
                        f"{len(self._pending)} waiting; at most {self._max_queued} wait)")
                self._pending.append(_Entry(job, thread))
                start_now, queued = False, len(self._pending)
                job.queue_position = queued
        if start_now:
            self._start(job, thread)
        else:
            logger.info("[agent_loop] %s queued at position %d", job.job_id, queued)
        self._ensure_watchdog()
        return job

    def run(self, task: Callable[[], Any]) -> Any:
        """Submit ``task`` and wait for it; raises whatever the task raised."""
        return self.submit(task).future.result()

    def launch(self, message: str, **params: Any) -> AgentLoopJob:
        """Submit one :func:`agent.agent_run.run` call with *params* (its keyword surface)."""
        from . import agent_run

        label = str(params.get("layer") or (params.get("meta") or {}).get("agent") or "agent")
        session_key = str(params.get("history_session") or params.get("session_id") or "")
        return self.submit(lambda: agent_run.run(message, **params), label=label,
                           session_key=session_key)

    def run_agent(self, message: str, **params: Any) -> Any:
        """Run one agent on a pool thread and return its ``AgentRunResult``."""
        return self.launch(message, **params).future.result()

    # -- lifecycle ---------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"max_workers": self._max_workers, "max_queued": self._max_queued,
                    "running": len(self._active), "queued": len(self._pending),
                    "jobs": [entry.job.job_id for entry in self._active.values()]}

    def close(self, timeout: float = 5.0) -> None:
        """Stop the stall watchdog (queued/running jobs are left to finish on their threads)."""
        self._stopping.set()
        thread, self._watchdog = self._watchdog, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))

    # -- internals ---------------------------------------------------------------

    def _register_locked(self, job: AgentLoopJob, thread: threading.Thread) -> None:
        """Count a job as running; the caller holds the lock and starts the thread after it."""
        job.started_at = time.time()
        job.queue_position = 0
        self._active[job.job_id] = _Entry(job, thread)

    def _start(self, job: AgentLoopJob, thread: threading.Thread) -> None:
        """Start an already-registered job's thread; a start failure settles it and frees the slot."""
        try:
            thread.start()
        except RuntimeError as exc:
            logger.error("[agent_loop] %s could not start a worker thread: %s", job.job_id, exc)
            self._settle(job, error=exc)

    def _work(self, job: AgentLoopJob, task: Callable[[], Any]) -> None:
        try:
            value = task()
        except BaseException as exc:  # health: allow BLE001 -- the future is the boundary
            self._settle(job, error=exc)
        else:
            self._settle(job, value=value)

    def _settle(self, job: AgentLoopJob, *, value: Any = None,
                error: Optional[BaseException] = None,
                on_settle: Optional[Callable[..., None]] = None
                ) -> None:
        """Deliver the job once and start the next waiting one; later settles are no-ops.

        The callback defaults to the one stored at submission, so the stall path (no worker
        frame to carry it) and a failed thread start still notify their caller exactly once.
        """
        with self._lock:
            entry = self._active.pop(job.job_id, None)
            if entry is None:
                return  # a stalled job's slot was released and already reported
            starts = self._dispatch_locked()
        if error is None:
            job.future.set_result(value)
        else:
            job.future.set_exception(error)
        callback = on_settle if on_settle is not None else job.on_settle
        if callback is not None:
            try:
                callback(job, value, error)
            except Exception:  # health: allow BLE001 -- the run already happened; never lose it
                logger.warning("[agent_loop] %s settle callback failed", job.job_id, exc_info=True)
        for entry in starts:
            self._start(entry.job, entry.thread)

    def _dispatch_locked(self) -> list[_Entry]:
        starts: list[_Entry] = []
        while len(self._active) < self._max_workers and self._pending:
            entry = self._pending.pop(0)
            self._register_locked(entry.job, entry.thread)
            starts.append(entry)
        return starts

    # -- stall watchdog ----------------------------------------------------------

    def _ensure_watchdog(self) -> None:
        if self._max_run_seconds <= 0:
            return
        with self._lock:
            if self._watchdog is not None and self._watchdog.is_alive():
                return
            self._stopping.clear()
            watchdog = threading.Thread(  # health: allow HX012 -- process-wide, no profile state
                target=self._watch_stalls, name="agent-loop-stalls", daemon=True)
            self._watchdog = watchdog
        try:
            watchdog.start()
        except RuntimeError as exc:  # the watchdog is a safety net, not a precondition
            logger.warning("[agent_loop] could not start the stall watchdog: %s", exc)
            with self._lock:
                if self._watchdog is watchdog:
                    self._watchdog = None

    def _watch_stalls(self) -> None:
        while not self._stopping.wait(self._stall_check_seconds):
            try:
                self._scan_stalls()
            except Exception:  # health: allow BLE001 -- a dead watchdog would wedge the pool
                logger.warning("[agent_loop] stall scan failed", exc_info=True)

    def _scan_stalls(self) -> list[str]:
        now = time.time()
        with self._lock:
            expired = [entry.job for entry in self._active.values()
                       if entry.job.started_at and now - entry.job.started_at > self._max_run_seconds]
        for job in expired:
            self._settle(job, error=TimeoutError(
                f"no result after {int(self._max_run_seconds)}s; its slot was released and the run "
                "may still finish"))
        return [job.job_id for job in expired]


_SERVICE: Optional[AgentLoopService] = None
_SERVICE_LOCK = threading.Lock()


def _limits() -> dict[str, float]:
    """Configured pool limits; an unreadable config leaves the defaults in charge."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = (load_config_readonly() or {}).get("agent_loop") or {}
    except Exception:  # health: allow BLE001 -- unreadable config must not break submission
        logger.debug("agent_loop: config unreadable", exc_info=True)
        cfg = {}
    if not isinstance(cfg, Mapping):
        cfg = {}

    def _number(name: str, default: float) -> float:
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value)

    return {"max_workers": _number("max_workers", DEFAULT_MAX_WORKERS),
            "max_queued": _number("max_queued", DEFAULT_MAX_QUEUED),
            "max_run_seconds": _number("max_run_seconds", DEFAULT_MAX_RUN_SECONDS)}


def service() -> AgentLoopService:
    """The process-wide pool: one capacity budget shared by plugins and the ``agent`` tool."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            limits = _limits()
            _SERVICE = AgentLoopService(max_workers=int(limits["max_workers"]),
                                        max_queued=int(limits["max_queued"]),
                                        max_run_seconds=limits["max_run_seconds"])
        return _SERVICE


def reset_service() -> None:
    """Drop the shared pool (tests, or a config reload); its threads finish on their own."""
    global _SERVICE
    with _SERVICE_LOCK:
        old, _SERVICE = _SERVICE, None
    if old is not None:
        old.close()


class PluginAgentLoop:
    """What ``ctx.agent_loop`` returns: the host pool, plus this plugin's identity for labels."""

    capability_version = CAPABILITY_VERSION

    def __init__(self, plugin_id: str) -> None:
        self.plugin_id = str(plugin_id or "plugin")

    def run(self, message: str, **params: Any) -> Any:
        """Run one agent to completion on a pool thread and return its ``AgentRunResult``.

        Keyword parameters are :func:`agent.agent_run.run`'s: ``context``, ``model``,
        ``provider``, ``api_mode``, ``toolsets``, ``system_message``, ``max_iterations``,
        ``project_dir``, ``all_tools``, ``read_only``, ``history``, ``session_id``, ``layer``,
        ``history_session``, ``account_task``, ``skip_memory``, ``skip_context_files``,
        ``skip_background_review``, ``load_soul_identity``, ``extra_tools``, ``cache_scope``,
        ``tool_guardrails``, ``meta`` and ``record_history``.
        """
        params.setdefault("layer", self.plugin_id)
        return service().run_agent(message, **params)

    def submit(self, message: str, **params: Any) -> AgentLoopJob:
        """Submit a run and return immediately; ``job.future`` settles with the result."""
        params.setdefault("layer", self.plugin_id)
        return service().launch(message, **params)

    def stats(self) -> dict[str, Any]:
        """Running/queued counts and the pool's capacity (for UI and diagnostics)."""
        return service().stats()


__all__ = [
    "CAPABILITY_VERSION",
    "AgentLoopJob",
    "AgentLoopQueueFull",
    "AgentLoopService",
    "PluginAgentLoop",
    "reset_service",
    "service",
]
