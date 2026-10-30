"""Per-process turn/step scheduling engine: registration, wheels, bounded dispatch.

One engine per process (CLI, gateway, TUI each construct their own). Registration is
process-wide; wheels are per session and created lazily on the session's first tick. Logical
time only: a refunded iteration never commits, so it advances nothing and fires no post task.

Design: website/docs/developer-guide/scheduled-tasks.md.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import Any, Callable, Iterable, Optional

from scheduler.runner import BoundedCallbackRunner, CallbackOutcome, DEFAULT_MAX_ABANDONED
from scheduler.tasks import StepView, TaskContext, TaskSpec, validate_spec
from scheduler.wheel import TimingWheel, WheelNode

logger = logging.getLogger("hermes.scheduler")

DEFAULT_HANDLER_TIMEOUT_MS = 50
DEFAULT_MAX_TASKS_PER_PLUGIN = 32


def _scheduler_config() -> dict[str, Any]:
    """``scheduler`` config section; an unreadable config must never break a turn."""
    try:
        from hermes_cli.config import load_config

        section = load_config().get("scheduler")
        return section if isinstance(section, dict) else {}
    except Exception:  # pragma: no cover - config is best-effort for observers
        return {}


class _Session:
    __slots__ = (
        "agent_name", "job_heaps", "jobs", "jobs_loaded", "profile_name", "session_id",
        "step", "step_nodes", "turn", "turn_nodes",
    )

    def __init__(
        self, session_id: str, turn_seq: int, step_seq: int, *,
        profile_name: str = "", agent_name: str = "",
    ) -> None:
        self.session_id = session_id
        self.turn = TimingWheel(now=max(0, int(turn_seq)))
        self.step = TimingWheel(now=max(0, int(step_seq)))
        self.turn_nodes: dict[tuple[str, str], WheelNode] = {}
        self.step_nodes: dict[tuple[str, str], WheelNode] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.job_heaps: dict[str, list[tuple[int, str]]] = {"turn": [], "step": []}
        self.jobs_loaded = False
        self.profile_name = profile_name
        self.agent_name = agent_name

    def load_jobs(self, jobs: Iterable[dict[str, Any]]) -> None:
        self.jobs = {str(job["id"]): job for job in jobs if job.get("enabled")}
        self.job_heaps = {"turn": [], "step": []}
        for job in self.jobs.values():
            kind = job.get("kind")
            if kind in self.job_heaps:
                self.job_heaps[kind].append((int(job["next_due_seq"]), str(job["id"])))
        for heap in self.job_heaps.values():
            heapq.heapify(heap)
        self.jobs_loaded = True

    def due_jobs(self, kind: str, seq: int) -> list[dict[str, Any]]:
        heap = self.job_heaps.get(kind, [])
        due: list[dict[str, Any]] = []
        while heap and heap[0][0] <= seq:
            due_seq, job_id = heapq.heappop(heap)
            job = self.jobs.get(job_id)
            if not job or job.get("kind") != kind:
                continue
            if int(job.get("next_due_seq", 0)) != due_seq:
                continue  # stale heap entry (updated/removed elsewhere)
            due.append(dict(job))
        return due

    def reschedule_job(self, job_id: str, next_due: int, *, enabled: bool = True) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        job["next_due_seq"] = int(next_due)
        if not enabled:
            job["enabled"] = 0
            return
        kind = job.get("kind")
        if kind in self.job_heaps:
            heapq.heappush(self.job_heaps[kind], (int(next_due), job_id))

    def wheel(self, period: str) -> TimingWheel:
        return self.turn if period == "turn" else self.step

    def nodes(self, period: str) -> dict[tuple[str, str], WheelNode]:
        return self.turn_nodes if period == "turn" else self.step_nodes

    def add(self, spec: TaskSpec) -> None:
        wheel = self.wheel(spec.period)
        self.nodes(spec.period)[spec.key] = wheel.insert(spec, wheel.now + spec.every)

    def drop(self, spec: TaskSpec, *, cancel: bool = True) -> None:
        node = self.nodes(spec.period).pop(spec.key, None)
        if node is not None and cancel:
            self.wheel(spec.period).cancel(node)


class TaskHandle:
    """Cancellation handle returned by :meth:`SchedulingEngine.register`."""

    def __init__(self, engine: SchedulingEngine, key: tuple[str, str]) -> None:
        self._engine = engine
        self.key = key

    def cancel(self) -> None:
        self._engine.unregister(*self.key)


class SchedulingEngine:
    def __init__(
        self, *, register_builtins: bool = True, runner: Optional[BoundedCallbackRunner] = None,
        handler_timeout_ms: Optional[int] = None, max_tasks_per_plugin: Optional[int] = None,
        run_sink: Optional[Callable[[TaskSpec, TaskContext, CallbackOutcome], None]] = None,
        job_loader: Optional[Callable[[str], list[dict[str, Any]]]] = None,
        job_runner: Optional[Callable[[dict[str, Any], int], None]] = None,
        job_claim: Optional[Callable[[str, int], bool]] = None,
    ) -> None:
        config = _scheduler_config()
        raw_turn_step = config.get("turn_step")
        turn_step = raw_turn_step if isinstance(raw_turn_step, dict) else {}
        if handler_timeout_ms is None:
            handler_timeout_ms = int(turn_step.get("handler_timeout_ms", DEFAULT_HANDLER_TIMEOUT_MS) or DEFAULT_HANDLER_TIMEOUT_MS)
        if max_tasks_per_plugin is None:
            max_tasks_per_plugin = int(turn_step.get("max_tasks_per_plugin", DEFAULT_MAX_TASKS_PER_PLUGIN) or DEFAULT_MAX_TASKS_PER_PLUGIN)
        max_abandoned = int(turn_step.get("max_abandoned_workers", DEFAULT_MAX_ABANDONED) or DEFAULT_MAX_ABANDONED)
        self.handler_timeout_ms = max(1, min(60_000, handler_timeout_ms))
        self.max_tasks_per_plugin = max(1, max_tasks_per_plugin)
        self.runner = runner or BoundedCallbackRunner(
            timeout_s=self.handler_timeout_ms / 1000.0, max_abandoned=max_abandoned)
        self.run_sink = run_sink
        self._job_loader = job_loader
        self._job_runner = job_runner
        self._job_claim = job_claim
        self._lock = threading.RLock()
        self._specs: dict[tuple[str, str], TaskSpec] = {}
        self._sessions: dict[str, _Session] = {}
        self._builtins_registered = False
        if register_builtins:
            from scheduler.builtins import register_builtin_tasks

            self._builtins_registered = True
            register_builtin_tasks(self)

    # ── registration ──────────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        return bool(self._specs) or any(session.jobs for session in self._sessions.values())

    def specs(self) -> tuple[TaskSpec, ...]:
        with self._lock:
            return tuple(self._specs.values())

    def task_count(self, plugin_id: str, *, exclude: Optional[tuple[str, str]] = None) -> int:
        with self._lock:
            return sum(1 for key in self._specs if key[0] == plugin_id and key != exclude)

    def register(self, spec: TaskSpec) -> TaskHandle:
        with self._lock:
            existing = sum(1 for key in self._specs if key[0] == spec.plugin_id and key != spec.key)
            validate_spec(spec, max_tasks_per_plugin=self.max_tasks_per_plugin, existing=existing)
            old = self._specs.get(spec.key)
            if old is not None:
                for session in self._sessions.values():
                    session.drop(old)
            self._specs[spec.key] = spec
            for session in self._sessions.values():
                session.add(spec)
        return TaskHandle(self, spec.key)

    def unregister(self, plugin_id: str, name: str) -> bool:
        with self._lock:
            spec = self._specs.pop((plugin_id, name), None)
            if spec is None:
                return False
            for session in self._sessions.values():
                session.drop(spec)
            return True

    # ── sessions ──────────────────────────────────────────────────────────
    def ensure_session(
        self, session_id: str, *, turn_seq: int = 0, step_seq: Optional[int] = None,
        profile_name: str = "", agent_name: str = "",
    ) -> None:
        if not session_id:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session(session_id, turn_seq, 0 if step_seq is None else step_seq,
                                   profile_name=profile_name, agent_name=agent_name)
                for spec in self._specs.values():
                    session.add(spec)
                self._load_jobs_locked(session)
                self._sessions[session_id] = session
                return
            if profile_name:
                session.profile_name = profile_name
            if agent_name:
                session.agent_name = agent_name
            if not session.jobs_loaded:
                self._load_jobs_locked(session)
            # First observation on a resumed session: re-derive the schedule from the durable
            # counter (crash-safe by design — the wheel itself is never persisted).
            if step_seq is not None and 0 <= session.step.now < step_seq:
                self._rebase(session, "step", int(step_seq))
            if 0 <= session.turn.now < turn_seq - 1:
                self._rebase(session, "turn", max(0, int(turn_seq) - 1))

    def _load_jobs_locked(self, session: _Session) -> None:
        if session.jobs_loaded or self._job_loader is None:
            return
        try:
            session.load_jobs(self._job_loader(session.session_id) or [])
        except Exception:  # loading is best-effort; never wedge a turn
            logger.debug("scheduler: session job load failed", exc_info=True)
            session.jobs_loaded = True

    def refresh_session_jobs(self, session_id: str) -> None:
        """Re-read a live session's jobs after a create/update/remove."""
        if self._job_loader is None:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.jobs_loaded = False
            self._load_jobs_locked(session)

    def _rebase(self, session: _Session, period: str, seq: int) -> None:
        wheel = session.wheel(period)
        wheel.now = max(0, int(seq))
        for spec in self._specs.values():
            if spec.period != period:
                continue
            session.drop(spec)
            session.add(spec)

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    # ── ticks ─────────────────────────────────────────────────────────────
    def turn_pre(self, session_id: str, *, turn_seq: int, turn_id: Optional[str] = None,
                 profile_name: str = "", agent_name: str = "", result_status: Optional[str] = None) -> list[CallbackOutcome]:
        if not self.active:
            return []
        session = self._live_session(session_id, turn_seq=max(0, int(turn_seq) - 1), step_seq=None)
        return self._fire_pre(session, "turn", int(turn_seq), turn_id=turn_id, turn_seq=int(turn_seq),
                              step_index=None, result_status=result_status)

    def step_pre(self, session_id: str, *, step_index: Optional[int] = None,
                 turn_id: Optional[str] = None, turn_seq: Optional[int] = None,
                 profile_name: str = "", agent_name: str = "") -> list[CallbackOutcome]:
        if not self.active:
            return []
        session = self._live_session(session_id, turn_seq=None, step_seq=None)
        seq = session.step.now + 1
        return self._fire_pre(session, "step", seq, turn_id=turn_id, turn_seq=turn_seq,
                              step_index=step_index, result_status=None)

    def turn_committed(self, session_id: str, *, turn_seq: int, turn_id: Optional[str] = None,
                       profile_name: str = "", agent_name: str = "",
                       result_status: str = "ok") -> list[CallbackOutcome]:
        if not self.active:
            return []
        session = self._live_session(session_id, turn_seq=max(0, int(turn_seq) - 1), step_seq=None)
        return self._advance(session, "turn", int(turn_seq), turn_id=turn_id, turn_seq=int(turn_seq),
                             step_index=None, result_status=result_status, step=None)

    def step_committed(self, session_id: str, *, step_seq: Optional[int] = None,
                       step_index: Optional[int] = None, usage_status: str = "ok",
                       status: str = "ok", total_tokens: int = 0, input_tokens: int = 0,
                       output_tokens: int = 0, cache_read_tokens: int = 0,
                       cache_write_tokens: int = 0, reasoning_tokens: int = 0,
                       duration_ms: int = 0, finish_reason: Optional[str] = None,
                       turn_id: Optional[str] = None, turn_seq: Optional[int] = None,
                       profile_name: str = "", agent_name: str = "") -> list[CallbackOutcome]:
        if not self.active:
            return []
        session = self._live_session(session_id, turn_seq=None, step_seq=None)
        seq = session.step.now + 1 if step_seq is None else max(session.step.now + 1, int(step_seq))
        view = StepView(
            step_seq=seq, step_index=step_index, usage_status=usage_status, status=status,
            total_tokens=total_tokens, input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
            reasoning_tokens=reasoning_tokens, duration_ms=duration_ms, finish_reason=finish_reason,
        )
        return self._advance(session, "step", seq, turn_id=turn_id, turn_seq=turn_seq,
                             step_index=step_index, result_status=status, step=view)

    # ── internals ─────────────────────────────────────────────────────────
    def _live_session(self, session_id: str, *, turn_seq: Optional[int], step_seq: Optional[int]) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session(session_id, 0, 0)
                for spec in self._specs.values():
                    session.add(spec)
                self._load_jobs_locked(session)
                self._sessions[session_id] = session
            if step_seq is not None and 0 <= session.step.now < step_seq:
                self._rebase(session, "step", int(step_seq))
            if turn_seq is not None and 0 <= session.turn.now < int(turn_seq):
                self._rebase(session, "turn", int(turn_seq))
            return session

    def _context(
        self, session: _Session, period: str, phase: str, *, turn_id: Optional[str],
        turn_seq: Optional[int], step_seq: Optional[int], step_index: Optional[int],
        result_status: Optional[str], step: Optional[StepView], profile_name: str = "",
        agent_name: str = "",
    ) -> TaskContext:
        return TaskContext(
            session_id=session.session_id,
            profile_name=profile_name or session.profile_name,
            agent_name=agent_name or session.agent_name,
            phase=phase,
            now=time.time(),
            turn_id=turn_id,
            turn_seq=turn_seq,
            step_seq=step_seq,
            step_index=step_index,
            result_status=result_status,
            step=step,
        )

    def _fire_pre(self, session: _Session, period: str, seq: int, **ctx_kwargs: Any) -> list[CallbackOutcome]:
        wheel = session.wheel(period)
        outcomes: list[CallbackOutcome] = []
        for node in wheel.due_nodes(seq):
            if node.cancelled or node.spec.phase != "pre":
                continue
            if node.spec.once:
                node.cancelled = True
            ctx = self._context(session, period, "pre", step_seq=None if period == "turn" else seq,
                                step=None, **ctx_kwargs)
            outcomes.append(self._fire(node.spec, ctx, seq))
        return outcomes

    def _advance(self, session: _Session, period: str, seq: int, *, step: Optional[StepView],
                 **ctx_kwargs: Any) -> list[CallbackOutcome]:
        wheel = session.wheel(period)
        outcomes: list[CallbackOutcome] = []

        def _on_due(node: WheelNode) -> None:
            if node.spec.phase == "post":
                ctx = self._context(session, period, "post",
                                    step_seq=None if period == "turn" else node.due,
                                    step=step, **ctx_kwargs)
                outcomes.append(self._fire(node.spec, ctx, node.due))
            if node.spec.once:
                node.cancelled = True
                return
            wheel.reschedule(node, wheel.now + node.spec.every)

        wheel.advance_to(seq, _on_due)
        self._fire_due_jobs(session, period, seq)
        return outcomes

    def _fire_due_jobs(self, session: _Session, kind: str, seq: int) -> None:
        """Claim and dispatch session jobs whose logical due point was reached."""
        if self._job_runner is None:
            return
        for job in session.due_jobs(kind, seq):
            fired_seq = int(job["next_due_seq"])
            if self._job_claim is not None:
                try:
                    if not self._job_claim(job["id"], fired_seq):
                        continue
                except Exception:
                    logger.warning("scheduler: session job claim failed for %s", job["id"], exc_info=True)
                    session.reschedule_job(job["id"], fired_seq)
                    continue
            remaining = job.get("remaining")
            enabled = remaining is None or int(remaining) > 1
            session.reschedule_job(job["id"], fired_seq + int(job["every"]), enabled=enabled)
            try:
                self._job_runner(job, fired_seq)
            except Exception:
                logger.warning("scheduler: session job dispatch failed for %s", job["id"], exc_info=True)

    def _fire(self, spec: TaskSpec, ctx: TaskContext, seq: Any) -> CallbackOutcome:
        timeout = (spec.timeout_ms or self.handler_timeout_ms) / 1000.0
        outcome = self.runner.run(
            task_key=spec.key, session_id=ctx.session_id, seq=seq,
            fn=lambda: spec.handler(ctx), timeout_s=timeout,
        )
        if outcome.status == "error":
            logger.warning("scheduler task %s failed in session %s: %s",
                           spec.name, ctx.session_id, outcome.error)
        if self.run_sink is not None:
            try:
                self.run_sink(spec, ctx, outcome)
            except Exception:  # observer persistence never breaks a turn
                logger.debug("scheduler: task_runs sink failed", exc_info=True)
        return outcome

    def sessions_known(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._sessions)


_ENGINE: Optional[SchedulingEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> SchedulingEngine:
    """The process's scheduling engine (built-ins registered once)."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                from scheduler.session_jobs import claim_job, load_jobs, run_job
                from scheduler.store import record_task_run

                _ENGINE = SchedulingEngine(
                    run_sink=record_task_run, job_loader=load_jobs,
                    job_runner=run_job, job_claim=claim_job,
                )
    return _ENGINE


def peek_engine() -> Optional[SchedulingEngine]:
    """The engine only if this process already built one (never creates it)."""
    return _ENGINE


def reset_engine() -> None:
    """Drop the process engine (tests and profile teardown)."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = None
