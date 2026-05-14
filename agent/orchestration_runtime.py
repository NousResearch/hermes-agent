"""Runtime orchestration context for Hermes (orchestrator Phase 7).

Phase 6 built the read-only *presentation* layer (:mod:`agent.orchestration_status`):
given a :class:`~agent.task_registry.TaskRegistry` and/or a
:class:`~agent.worker_lanes.WorkerLaneRegistry` it produces concise, Telegram-
friendly status text.  But it deliberately stopped short of wiring ``/tasks`` /
``/agents`` because there was no *live* place for a command handler to find those
registries -- the formatter was a library, not an observable runtime.

This module is the smallest bridge across that gap.  :class:`OrchestrationRuntime`
is a tiny container that simply holds one :class:`~agent.task_registry.TaskRegistry`
and one :class:`~agent.worker_lanes.WorkerLaneRegistry` and forwards
``snapshot`` / ``format_tasks`` / ``format_agents`` / ``format_overview`` to the
Phase 6 functions against them.  The module-level helpers
(:func:`get_orchestration_runtime`, :func:`get_or_create_orchestration_runtime`,
:func:`set_orchestration_runtime`, :func:`format_runtime_tasks`,
:func:`format_runtime_agents`, :func:`format_runtime_overview`) let a *CLI / gateway
runner / session object / test dummy* carry such a runtime on a private
``_orchestration_runtime`` attribute -- so a later thin command handler can do
``format_runtime_overview(self)`` without this module knowing or caring what
``self`` is, and without any global mutable state.

What this module is -- and isn't:

* It is a *container + accessor* substrate.  ``OrchestrationRuntime.create()``
  makes a fresh, empty, in-memory :class:`~agent.task_registry.TaskRegistry`
  (``path=None`` -- no file persistence) and an empty
  :class:`~agent.worker_lanes.WorkerLaneRegistry` (no lanes registered).  The
  per-owner helpers store/fetch a runtime on a private attribute named by
  :data:`RUNTIME_ATTR`; "create if absent" attaches a fresh empty runtime so an
  observability surface always has *something* truthful to read (an empty board),
  never a fabricated one.
* It now includes the first *library-level* frontdesk loop:
  :meth:`OrchestrationRuntime.handle_frontdesk_input` consumes the Phase 2
  control-plane verdict and performs the minimal injected-registry side effects
  needed for STOP / STATUS / STEER / WORKER.  This loop is still not live
  CLI/Gateway wiring and still starts only lanes that a caller explicitly
  registered on the runtime.
* It is **not** the Ralph / focused-agent runtime, **not** a
  ``delegate_task(background=True)`` mechanism, **not** automatic
  Telegram/gateway routing of arbitrary natural-language messages, **not** a
  durable routing DB / SQLite schema, **not** an LLM classifier, and **not** a
  global singleton -- there is no module-level registry; every runtime lives on
  the object that owns it.  It also does **not** register or rewire the existing
  ``/tasks`` / ``/agents`` slash commands here: those names are already taken by
  an unrelated background-process / subagent listing in ``cli.py`` and
  ``gateway/run.py``, so repurposing them -- or threading a runtime through that
  dispatch -- remains a focused command-wiring phase.

Scope discipline (mirrors the Phase 2-6 leaf/presentation modules):

* Status/advisory helpers remain read-only: they never mutate a task, a worker,
  a queue, or a follow-up.  The explicit frontdesk loop is the only mutating
  path in this module, and it mutates only the injected task/worker registries.
* Everything :meth:`OrchestrationRuntime.snapshot` returns is the Phase 6
  :class:`~agent.orchestration_status.OrchestrationSnapshot`, whose ``to_dict``
  is plain JSON-safe data (strings, ints, ``None``, lists/dicts of those).
* The per-owner helpers are duck-typed about the owner: they only ever
  ``getattr`` / ``setattr`` the single private attribute, so a ``HermesCLI``, a
  gateway runner, a session object, or a hand-built test dummy all work
  identically.  ``OrchestrationRuntime`` itself takes its two registries by
  position/keyword (the "explicit injection" path used by tests) with
  ``create()`` as the only sugar -- there is no implicit shared default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.control_plane import (
    ControlPlaneDecision,
    Intent,
    Recommendation,
    classify as _classify_frontdesk,
)
from agent.orchestration_status import (
    OrchestrationSnapshot,
    build_snapshot,
    format_agents as _format_agents,
    format_overview as _format_overview,
    format_tasks as _format_tasks,
)
from agent.task_registry import (
    STATUS_CANCELLED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TaskRegistry,
)
from agent.worker_lanes import (
    WorkerLaneRegistry,
    WorkerSpec,
    link_worker_to_task,
)

__all__ = [
    "RUNTIME_ATTR",
    "FrontdeskTurnResult",
    "OrchestrationRuntime",
    "get_orchestration_runtime",
    "get_or_create_orchestration_runtime",
    "set_orchestration_runtime",
    "format_runtime_tasks",
    "format_runtime_agents",
    "format_runtime_overview",
    "advise_frontdesk_for_owner",
]

# The private attribute name a runtime is stored under on an owner object.  One
# place so every helper agrees; underscore-prefixed so it reads as internal state
# on whatever ``HermesCLI`` / gateway runner / session it lands on.
RUNTIME_ATTR = "_orchestration_runtime"


@dataclass(frozen=True, slots=True)
class FrontdeskTurnResult:
    """Result of the minimal frontdesk control loop for one input fragment.

    ``action`` is deliberately a small string vocabulary rather than an enum so
    adapters can render it without importing another type.  The runtime returns
    this object instead of raising for ordinary routing outcomes; caller-visible
    errors (for example, no worker lane registered) are represented as a control
    message and a task note.
    """

    decision: ControlPlaneDecision
    action: str
    message: str
    task_id: str | None = None
    worker_id: str | None = None
    cancelled_tasks: int = 0
    cancelled_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "action": self.action,
            "message": self.message,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "cancelled_tasks": self.cancelled_tasks,
            "cancelled_workers": self.cancelled_workers,
        }


# --------------------------------------------------------------------------
# OrchestrationRuntime
# --------------------------------------------------------------------------
@dataclass
class OrchestrationRuntime:
    """A live home for one :class:`~agent.task_registry.TaskRegistry` and one
    :class:`~agent.worker_lanes.WorkerLaneRegistry`, plus thin pass-throughs to
    the Phase 6 status formatter.

    Construct it directly with explicit registries -- ``OrchestrationRuntime(
    task_registry=tr, worker_registry=wr)`` -- when you want to share existing
    ones (tests, or a future phase that already owns them); use
    :meth:`create` for the common case of "give me a fresh empty pair".  The
    object is intentionally tiny: it owns no threads, no locks, no background
    work; the registries it holds bring their own semantics.
    """

    task_registry: TaskRegistry
    worker_registry: WorkerLaneRegistry

    @classmethod
    def create(cls) -> "OrchestrationRuntime":
        """Return a runtime with a fresh, empty, in-memory task registry
        (``path=None`` -- no file persistence) and a fresh, empty worker-lane
        registry (no lanes registered)."""
        return cls(task_registry=TaskRegistry(), worker_registry=WorkerLaneRegistry())

    # -- read-only views (delegate to Phase 6) ---------------------------
    def snapshot(self, *, session_key: str | None = None) -> OrchestrationSnapshot:
        """Build an :class:`~agent.orchestration_status.OrchestrationSnapshot` of
        the held registries (all tasks for *session_key*, the active/total split
        living in ``snapshot.counts``).  ``snapshot.to_dict()`` is JSON-safe."""
        return build_snapshot(
            self.task_registry, self.worker_registry, session_key=session_key
        )

    def format_tasks(self, *, session_key: str | None = None, compact: bool = True) -> str:
        """Render the focused-task board (active tasks for *session_key*); the
        graceful empty-state line when there are none."""
        return _format_tasks(self.task_registry, compact=compact, session_key=session_key)

    def format_agents(self, *, compact: bool = True) -> str:
        """Render the worker/agent board; the graceful empty-state line when
        there are none."""
        return _format_agents(self.worker_registry, compact=compact)

    def format_overview(self, *, session_key: str | None = None, compact: bool = True) -> str:
        """Render the combined board (summary, then tasks, then workers); the
        graceful empty-state line when there is neither."""
        return _format_overview(
            self.task_registry, self.worker_registry, session_key=session_key, compact=compact
        )

    # -- frontdesk policy advisory (read-only) ---------------------------
    def advise_frontdesk(
        self, request_text: str, *, frontdesk_mode_active: bool = False
    ) -> ControlPlaneDecision:
        """Return a :class:`~agent.control_plane.ControlPlaneDecision` for
        *request_text*.

        Side-effect-free: this is a thin pass-through to
        :func:`agent.control_plane.classify`, including its mode-gating
        invariant.  Defaulting ``frontdesk_mode_active`` to ``False`` preserves
        legacy UX: worker/steer-shaped policy verdicts are advisories only and
        downgrade to ``MAIN`` until a future phase explicitly enables
        frontdesk mode.  The runtime's own task/worker state is *not* consulted.
        """
        return _classify_frontdesk(
            request_text, frontdesk_mode_active=frontdesk_mode_active
        )

    # -- minimal frontdesk control loop ----------------------------------
    def handle_frontdesk_input(
        self,
        request_text: str,
        *,
        frontdesk_mode_active: bool = False,
        session_key: str | None = None,
        source_surface: str = "cli",
        main_in_flight: bool = False,
        steer_callback: Callable[[str], Any] | None = None,
    ) -> FrontdeskTurnResult:
        """Route one input fragment through the first functional frontdesk loop.

        This is intentionally still a *library* loop, not live CLI/Gateway
        wiring.  It consumes the Phase 2 control-plane decision and performs the
        smallest safe side effects on the injected registries:

        * STOP cancels active tasks/workers for the session and returns a local
          control line; the stopped text is never converted into a follow-up.
        * STATUS returns the local runtime overview.
        * STEER calls an explicit ``steer_callback`` only when the caller says a
          main turn is in flight; otherwise it falls back to ``MAIN``.
        * NEW_TASK_WORKER creates a focused task, starts a registered worker lane,
          and links task <-> worker.  If no lane is registered, the task is
          cancelled with a local control message instead of fabricating work.
        """
        decision = self.advise_frontdesk(
            request_text, frontdesk_mode_active=frontdesk_mode_active
        )

        if decision.intent is Intent.STOP:
            cancelled_tasks, cancelled_workers = self._cancel_active_frontdesk_work(
                session_key=session_key
            )
            return FrontdeskTurnResult(
                decision=decision,
                action="stopped",
                message=(
                    f"control: stopped {cancelled_tasks} task(s), "
                    f"{cancelled_workers} worker(s)"
                ),
                cancelled_tasks=cancelled_tasks,
                cancelled_workers=cancelled_workers,
            )

        if decision.intent is Intent.STATUS:
            return FrontdeskTurnResult(
                decision=decision,
                action="status",
                message=self.format_overview(session_key=session_key),
            )

        if decision.intent is Intent.STEER:
            if main_in_flight and steer_callback is not None:
                try:
                    accepted = bool(steer_callback(decision.raw_text))
                except Exception:
                    accepted = False
                if accepted:
                    return FrontdeskTurnResult(
                        decision=decision,
                        action="steered",
                        message="control: steered active main turn",
                    )
                return FrontdeskTurnResult(
                    decision=decision,
                    action="main",
                    message="control: steer not accepted; route as main input",
                )
            return FrontdeskTurnResult(
                decision=decision,
                action="main",
                message="control: no active main turn to steer; route as main input",
            )

        if decision.intent is Intent.NEW_TASK_WORKER:
            return self._start_worker_task(
                decision,
                session_key=session_key,
                source_surface=source_surface,
            )

        if decision.recommendation is Recommendation.CONTROL:
            return FrontdeskTurnResult(
                decision=decision,
                action=decision.intent.value,
                message=f"control: {decision.intent.value}",
            )

        if (
            main_in_flight
            and steer_callback is not None
            and decision.intent is Intent.NEW_TASK_MAIN
        ):
            try:
                accepted = bool(steer_callback(decision.raw_text))
            except Exception:
                accepted = False
            if accepted:
                return FrontdeskTurnResult(
                    decision=decision,
                    action="steered",
                    message="control: steered active main turn",
                )

        return FrontdeskTurnResult(
            decision=decision,
            action="main",
            message="route: main",
        )

    def _start_worker_task(
        self,
        decision: ControlPlaneDecision,
        *,
        session_key: str | None,
        source_surface: str,
    ) -> FrontdeskTurnResult:
        task = self.task_registry.create_task(
            decision.raw_text,
            session_key=session_key,
            origin={"platform": source_surface, "session_key": session_key},
            status=STATUS_QUEUED,
        )
        lane_names = self.worker_registry.lane_names()
        if not lane_names:
            self.task_registry.update_status(
                task.task_id,
                STATUS_CANCELLED,
                note="frontdesk worker requested but no worker lane is registered",
            )
            return FrontdeskTurnResult(
                decision=decision,
                action="worker_unavailable",
                message="control: worker lane unavailable",
                task_id=task.task_id,
            )

        lane_name = lane_names[0]
        spec = WorkerSpec(
            goal=decision.raw_text,
            task_id=task.task_id,
            lane=lane_name,
            metadata={
                "frontdesk_fingerprint": decision.fingerprint,
                "source_surface": source_surface,
                "session_key": session_key,
            },
        )
        handle = self.worker_registry.start(spec)
        link_worker_to_task(self.task_registry, task.task_id, handle)
        self.task_registry.update_status(
            task.task_id,
            STATUS_RUNNING,
            note=f"worker started: {handle.worker_id}",
        )
        return FrontdeskTurnResult(
            decision=decision,
            action="worker_started",
            message=f"control: worker started {handle.worker_id}",
            task_id=task.task_id,
            worker_id=handle.worker_id,
        )

    def _cancel_active_frontdesk_work(
        self, *, session_key: str | None = None
    ) -> tuple[int, int]:
        cancelled_workers = 0
        for worker in self.snapshot(session_key=session_key).workers:
            if worker.get("status") in {"queued", "running"}:
                worker_id = worker.get("worker_id")
                if isinstance(worker_id, str) and self.worker_registry.cancel(worker_id):
                    cancelled_workers += 1

        cancelled_tasks = 0
        for task in self.task_registry.list_tasks(
            session_key=session_key, active_only=True
        ):
            if task.status != STATUS_CANCELLED:
                self.task_registry.cancel_task(task.task_id, reason="frontdesk stop")
                cancelled_tasks += 1
        return cancelled_tasks, cancelled_workers


# --------------------------------------------------------------------------
# Per-owner attach / fetch helpers (no global state)
# --------------------------------------------------------------------------
def get_orchestration_runtime(owner: Any) -> OrchestrationRuntime | None:
    """Return the :class:`OrchestrationRuntime` carried by *owner*, or ``None``.

    Only an actual :class:`OrchestrationRuntime` (or subclass) at the private
    :data:`RUNTIME_ATTR` slot counts -- a missing attribute, ``None``, or some
    other value all read as "no runtime here".  This never creates or attaches
    anything.
    """
    existing = getattr(owner, RUNTIME_ATTR, None)
    return existing if isinstance(existing, OrchestrationRuntime) else None


def set_orchestration_runtime(owner: Any, runtime: OrchestrationRuntime) -> OrchestrationRuntime:
    """Attach *runtime* to *owner* at the private :data:`RUNTIME_ATTR` slot and
    return it.  Replaces any prior value.  Raises ``TypeError`` if *runtime* is
    not an :class:`OrchestrationRuntime` -- the slot is ours and a later
    :func:`get_orchestration_runtime` would otherwise silently ignore a bad
    value."""
    if not isinstance(runtime, OrchestrationRuntime):
        raise TypeError("runtime must be an OrchestrationRuntime")
    setattr(owner, RUNTIME_ATTR, runtime)
    return runtime


def get_or_create_orchestration_runtime(owner: Any) -> OrchestrationRuntime:
    """Return *owner*'s :class:`OrchestrationRuntime`, attaching a fresh empty one
    (via :meth:`OrchestrationRuntime.create`) if it does not already carry one.

    The "create if absent" path attaches an *empty* runtime -- a status surface
    built on it reads as a graceful empty board, never as fabricated state -- and
    leaves it in place so later work that populates the registries is visible
    through the same handle.  Two different owners get two independent runtimes;
    there is no shared/global instance.
    """
    existing = get_orchestration_runtime(owner)
    if existing is not None:
        return existing
    return set_orchestration_runtime(owner, OrchestrationRuntime.create())


# --------------------------------------------------------------------------
# Owner-facing formatting convenience (get-or-create, then delegate)
# --------------------------------------------------------------------------
def format_runtime_tasks(
    owner: Any, *, session_key: str | None = None, compact: bool = True
) -> str:
    """The focused-task board for *owner*'s runtime, creating an empty runtime on
    *owner* first if needed -- so a thin ``/tasks``-style handler can be a one-liner
    and an owner with no runtime yet still gets the graceful empty-state line."""
    return get_or_create_orchestration_runtime(owner).format_tasks(
        session_key=session_key, compact=compact
    )


def format_runtime_agents(owner: Any, *, compact: bool = True) -> str:
    """The worker/agent board for *owner*'s runtime, creating an empty runtime on
    *owner* first if needed."""
    return get_or_create_orchestration_runtime(owner).format_agents(compact=compact)


def format_runtime_overview(
    owner: Any, *, session_key: str | None = None, compact: bool = True
) -> str:
    """The combined orchestration board for *owner*'s runtime, creating an empty
    runtime on *owner* first if needed."""
    return get_or_create_orchestration_runtime(owner).format_overview(
        session_key=session_key, compact=compact
    )


def advise_frontdesk_for_owner(
    owner: Any, request_text: str, *, frontdesk_mode_active: bool = False
) -> ControlPlaneDecision:
    """Return the frontdesk policy verdict for *request_text* against *owner*'s
    runtime, creating an empty runtime on *owner* first if needed.

    Read-only: no task, worker, or queue state is mutated.  Callers that only
    need the verdict (and not the runtime) can call
    :func:`agent.frontdesk_policy.classify_request` directly; this helper
    matches the per-owner shape of the other ``*_runtime_*`` formatters so a
    thin command handler can stay a one-liner.
    """
    return get_or_create_orchestration_runtime(owner).advise_frontdesk(
        request_text, frontdesk_mode_active=frontdesk_mode_active
    )
