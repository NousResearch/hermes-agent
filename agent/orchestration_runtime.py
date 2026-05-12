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
* It is **not** the Ralph / focused-agent runtime, **not** a worker-dispatch or
  ``delegate_task(background=True)`` mechanism (it starts, polls, kills no
  workers -- a future phase that wants a lane calls
  ``runtime.worker_registry.register(lane)`` itself), **not** a follow-up
  classifier or natural-language router (that is :mod:`agent.followup_router`),
  **not** automatic Telegram/gateway routing of status queries, **not** a
  cancel/stop/force-kill surface, **not** a durable routing DB / SQLite schema,
  **not** an LLM classifier, and **not** a global singleton -- there is no
  module-level registry; every runtime lives on the object that owns it.  It also
  does **not** register or rewire the existing ``/tasks`` / ``/agents`` slash
  commands here: those names are already taken by an unrelated background-process
  / subagent listing in ``cli.py`` and ``gateway/run.py``, so repurposing them --
  or threading a runtime through that dispatch -- is exactly the broad
  CLI/gateway refactor this phase stops short of; the helpers below are what a
  later, focused command-wiring phase will build on.  See the Phase 7 notes doc.

Scope discipline (mirrors the Phase 2-6 leaf/presentation modules):

* This module is read-only with respect to user/task state: it never mutates a
  task, a worker, a queue, or a follow-up.  It only *holds* the registries and
  *reads* them via the Phase 6 formatter, which itself counts
  ``pending_followups`` (``len(...)``) and never iterates their payloads -- so
  :attr:`~agent.pending_turn_queue.PendingTurnItem.raw` is never serialised,
  copied, deep-copied, or otherwise touched anywhere on this path.
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
from typing import Any

from agent.orchestration_status import (
    OrchestrationSnapshot,
    build_snapshot,
    format_agents as _format_agents,
    format_overview as _format_overview,
    format_tasks as _format_tasks,
)
from agent.task_registry import TaskRegistry
from agent.worker_lanes import WorkerLaneRegistry

__all__ = [
    "RUNTIME_ATTR",
    "OrchestrationRuntime",
    "get_orchestration_runtime",
    "get_or_create_orchestration_runtime",
    "set_orchestration_runtime",
    "format_runtime_tasks",
    "format_runtime_agents",
    "format_runtime_overview",
]

# The private attribute name a runtime is stored under on an owner object.  One
# place so every helper agrees; underscore-prefixed so it reads as internal state
# on whatever ``HermesCLI`` / gateway runner / session it lands on.
RUNTIME_ATTR = "_orchestration_runtime"


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
