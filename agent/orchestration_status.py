"""Read-only orchestration observatory for Hermes (orchestrator Phase 6).

Hermes is a concierge / front-desk / butler orchestrator: the user should be
able to *speak naturally* while Hermes coordinates work behind the scenes -- the
desired UX is not command-heavy.  But once there are focused tasks and worker
lanes in flight, the user still needs a lightweight status board: *what tasks are
active, which workers are running, what follow-ups are queued, is anything
blocked or waiting for me?*  This module is that read-only presentation layer
over the Phase 2-5 substrates -- it answers those four questions and nothing more.

What this module is -- and isn't:

* :class:`OrchestrationSnapshot` is a flat, JSON-safe point-in-time view: a list
  of compact task dicts, a list of compact worker dicts, a ``counts`` summary,
  and a ``warnings`` list of cheap cross-consistency notes (a task pointing at a
  worker no lane knows about, a worker pointing at a task no registry knows
  about).  :class:`OrchestrationStatusFormatter` turns a snapshot -- or, directly,
  a :class:`~agent.task_registry.TaskRegistry` / a
  :class:`~agent.worker_lanes.WorkerLaneRegistry` / a single lane / a plain list
  of handle-ish objects -- into concise, deterministic, Telegram-friendly text:
  :func:`format_tasks`, :func:`format_agents`, :func:`format_overview`.  Bullets
  and labels, never markdown tables.  :func:`looks_like_orchestration_status_query`
  is a deterministic Korean/English predicate that *only* says "this message reads
  like a request for the status board"; it routes nothing by itself.
* It is **not** the Ralph / focused-agent runtime, **not** an LLM/model
  classifier, **not** automatic Telegram/gateway routing of natural-language
  status queries, **not** a live worker process dashboard (it reads whatever a
  :class:`~agent.worker_lanes.WorkerLaneRegistry` snapshot already exposes -- it
  starts, polls, kills nothing), **not** force-kill / force-cancel, **not** the
  public ``delegate_task(background=True)`` API, **not** a durable routing DB, and
  **not** a global singleton registry.  It registers no ``/tasks`` / ``/agents``
  slash commands here: there is no long-lived task/worker runtime in the CLI or
  gateway for such a command to read yet, and wiring one in would be exactly the
  broad CLI/gateway refactor this phase stops short of.  When that runtime
  exists, a thin command handler can call :func:`format_tasks` /
  :func:`format_agents` directly (or, until then, return a graceful "no focused
  tasks are currently registered in this session" message); see the Phase 6
  notes doc for why this is deferred.

Scope discipline (mirrors the Phase 2-5 modules):

* This module is read-only.  Nothing here mutates a task, a worker, a registry,
  or a queue: it only *reads* and *formats*.  It never serialises, copies, deep-
  copies, or otherwise touches :attr:`~agent.pending_turn_queue.PendingTurnItem.raw`
  -- it counts ``pending_followups`` (``len(...)``) and never iterates their
  payloads; the one place a worker's appended follow-ups are touched at all is
  through ``WorkerLane.snapshot(worker_id)``, whose existing contract serialises
  them via :meth:`PendingTurnItem.to_dict` (which drops ``raw`` without reading
  it) -- and even that call is best-effort and wrapped, so a worker carrying a
  non-JSON-safe spec ``metadata`` degrades to a goal-less worker line instead of
  raising.
* Everything :class:`OrchestrationSnapshot` carries -- and everything in
  :meth:`OrchestrationSnapshot.to_dict` -- is plain JSON-safe data (strings,
  ints, ``None``, lists and dicts of those).  Compact goal/result/error strings
  are stored *untruncated* in the snapshot; truncation is a formatting concern.
* Inputs are used purely duck-typed: ``TaskRegistry`` is reached via
  ``list_tasks(...)`` and per-task attributes (``task_id`` / ``status`` /
  ``user_goal`` / ``active_worker_id`` / ``worker_kind`` / ``pending_followups``
  / ``notes`` / ``artifacts`` / ``session_key``); ``WorkerLaneRegistry`` via
  ``lane_names()`` / ``get_lane()``; a lane via ``name`` / ``worker_ids()`` /
  ``status()`` / ``result()`` / ``snapshot()``; a handle/result via attributes or
  dict keys.  So the formatter works against the real Phase 3/4 objects, a future
  injected runtime registry, or hand-built fixtures alike.  The small task/worker
  *status* string sets are imported from the substrates so there is one source of
  truth, but nothing requires the inputs to be those concrete classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from agent.task_registry import (
    STATUS_BLOCKED as _TASK_STATUS_BLOCKED,
    STATUS_ERROR as _TASK_STATUS_ERROR,
    TERMINAL_STATUSES as _TERMINAL_TASK_STATUSES,
)
from agent.worker_lanes import (
    ACTIVE_WORKER_STATUSES as _ACTIVE_WORKER_STATUSES,
    TERMINAL_WORKER_STATUSES as _TERMINAL_WORKER_STATUSES,
    WorkerStatus as _WorkerStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; never imported at runtime
    from agent.task_registry import FocusedTask, TaskRegistry
    from agent.worker_lanes import WorkerHandle, WorkerLane, WorkerLaneRegistry, WorkerResult

__all__ = [
    "OrchestrationSnapshot",
    "OrchestrationStatusFormatter",
    "build_snapshot",
    "format_tasks",
    "format_agents",
    "format_overview",
    "looks_like_orchestration_status_query",
]

# Truncation budgets for the *formatted* text.  ``compact=True`` (the default)
# keeps goal/result/error short for a glanceable Telegram board; ``compact=False``
# gives more room without becoming a developer dump.
_GOAL_COMPACT = 60
_GOAL_FULL = 140
_DETAIL_COMPACT = 60
_DETAIL_FULL = 200

# Empty-state lines (kept identical between the class and the module functions).
_EMPTY_TASKS = "No active tasks are currently registered."
_EMPTY_AGENTS = "No active workers are currently registered."
_EMPTY_OVERVIEW = "No active tasks or workers are currently registered."


# --------------------------------------------------------------------------
# Tiny helpers
# --------------------------------------------------------------------------
def _truncate(text: Any, limit: int) -> str:
    """Strip *text*, then clip to *limit* chars with a trailing ``…`` if needed."""
    s = text.strip() if isinstance(text, str) else ("" if text is None else str(text).strip())
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + "…"


def _str_or_none(value: Any) -> str | None:
    """Return *value* if it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value else None


def _plural(n: int, word: str) -> str:
    """``"1 task"`` / ``"2 tasks"`` -- naive ``+ "s"`` (fine for our nouns)."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _is_plain_iterable(value: Any) -> bool:
    """True for a non-string, non-mapping iterable (a list/tuple/generator of items)."""
    if isinstance(value, (str, bytes, dict)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _task_is_active(status: Any) -> bool:
    """Active == not terminal (so an unknown status counts as active, like FocusedTask)."""
    return status not in _TERMINAL_TASK_STATUSES


def _worker_is_terminal(status: Any) -> bool:
    return status in _TERMINAL_WORKER_STATUSES


def _safe_call(fn: Any) -> Any:
    """Call a zero-arg callable, swallowing *any* exception (best-effort probes)."""
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001 - duck-typed probe; a failure just means "no data"
        return None


def _safe_call1(fn: Any, arg: Any) -> Any:
    if not callable(fn):
        return None
    try:
        return fn(arg)
    except Exception:  # noqa: BLE001 - see _safe_call
        return None


# --------------------------------------------------------------------------
# Building compact task / worker dicts
# --------------------------------------------------------------------------
def _count_value(value: Any) -> int:
    """Return a safe non-negative count for list-ish or integer-ish values."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if value is None:
        return 0
    try:
        return max(0, len(value))  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - foreign fixture values become zero, not fatal
        return 0


def _task_to_dict(task: Any) -> dict[str, Any]:
    """A flat, JSON-safe presentation dict for one focused task.

    Goal is stored *untruncated*.  Only follow-up / note / artifact *counts* are
    captured -- the follow-up payloads (and their ``raw``) are never touched.
    Dict inputs are normalized too, so a caller-provided object in ``goal`` or
    other fields cannot leak into the advertised JSON-safe snapshot.
    """
    if isinstance(task, dict):
        status = task.get("status")
        goal = _str_or_none(task.get("goal")) or _str_or_none(task.get("user_goal")) or ""
        worker_id = _str_or_none(task.get("worker_id")) or _str_or_none(task.get("active_worker_id"))
        followups = _count_value(task.get("followups", task.get("pending_followups")))
        notes = _count_value(task.get("notes"))
        artifacts = _count_value(task.get("artifacts"))
        st = status if isinstance(status, str) else "?"
        return {
            "task_id": _str_or_none(task.get("task_id")) or "?",
            "status": st,
            "goal": goal,
            "session_key": _str_or_none(task.get("session_key")),
            "worker_id": worker_id,
            "worker_kind": _str_or_none(task.get("worker_kind")),
            "followups": followups,
            "notes": notes,
            "artifacts": artifacts,
            "active": _task_is_active(st),
            "blocked": st == _TASK_STATUS_BLOCKED,
            "error": st == _TASK_STATUS_ERROR,
        }

    status = getattr(task, "status", None)
    goal = _str_or_none(getattr(task, "user_goal", None)) or ""
    worker_id = _str_or_none(getattr(task, "active_worker_id", None))
    followups = getattr(task, "pending_followups", None) or []
    notes = getattr(task, "notes", None) or []
    artifacts = getattr(task, "artifacts", None) or []
    return {
        "task_id": _str_or_none(getattr(task, "task_id", None)) or "?",
        "status": status if isinstance(status, str) else "?",
        "goal": goal,
        "session_key": _str_or_none(getattr(task, "session_key", None)),
        "worker_id": worker_id,
        "worker_kind": _str_or_none(getattr(task, "worker_kind", None)),
        "followups": len(followups),
        "notes": len(notes),
        "artifacts": len(artifacts),
        "active": _task_is_active(status),
        "blocked": status == _TASK_STATUS_BLOCKED,
        "error": status == _TASK_STATUS_ERROR,
    }


def _worker_dict(
    *,
    worker_id: Any,
    status: Any,
    lane: Any,
    task_id: Any,
    cancel_requested: Any,
    goal: Any = None,
    result: Any = None,
    error: Any = None,
) -> dict[str, Any]:
    """A flat, JSON-safe presentation dict for one worker (strings/ints/None only)."""
    st = status if isinstance(status, str) else "?"
    return {
        "worker_id": _str_or_none(worker_id) or "?",
        "status": st,
        "lane": _str_or_none(lane),
        "task_id": _str_or_none(task_id),
        "goal": _str_or_none(goal),
        "result": _str_or_none(result),
        "error": _str_or_none(error),
        "cancel_requested": bool(cancel_requested),
        "active": st in _ACTIVE_WORKER_STATUSES or not _worker_is_terminal(st),
    }


def _worker_goal_from_lane(lane: Any, worker_id: str) -> str | None:
    """Best-effort goal lookup for *worker_id* via ``lane.snapshot(worker_id)``.

    ``snapshot(worker_id)`` is the only documented way to reach a worker's
    originating :class:`~agent.worker_lanes.WorkerSpec` (and therefore its goal);
    that call serialises the worker's appended follow-ups via
    :meth:`PendingTurnItem.to_dict` (which drops ``raw`` without reading it) and
    may raise ``TypeError`` if the spec carries non-JSON-safe ``metadata``.  Both
    are tolerated here: on any failure the worker line just goes goal-less.
    """
    snap_fn = getattr(lane, "snapshot", None)
    snap = _safe_call1(snap_fn, worker_id)
    if isinstance(snap, dict):
        spec = snap.get("spec")
        if isinstance(spec, dict):
            return _str_or_none(spec.get("goal"))
        # Some snapshot shapes inline the goal at the top level.
        return _str_or_none(snap.get("goal"))
    return None


def _worker_from_lane_by_id(lane: Any, worker_id: str, lane_name: str | None) -> dict[str, Any]:
    handle = _safe_call1(getattr(lane, "status", None), worker_id)
    result_obj = _safe_call1(getattr(lane, "result", None), worker_id)
    status = getattr(handle, "status", None)
    task_id = getattr(handle, "task_id", None)
    cancel_requested = getattr(handle, "cancel_requested", False)
    handle_lane = _str_or_none(getattr(handle, "lane", None))
    result_str = getattr(result_obj, "result", None) if result_obj is not None else None
    error_str = getattr(result_obj, "error", None) if result_obj is not None else None
    return _worker_dict(
        worker_id=getattr(handle, "worker_id", None) or worker_id,
        status=status,
        lane=handle_lane or lane_name,
        task_id=task_id,
        cancel_requested=cancel_requested,
        goal=_worker_goal_from_lane(lane, worker_id),
        result=result_str,
        error=error_str,
    )


def _workers_from_lane(lane: Any) -> list[dict[str, Any]]:
    """All workers a single lane knows about, via ``worker_ids()`` then per-id probes.

    Falls back to the lane's whole-lane ``snapshot()`` when ``worker_ids()`` is not
    available; that fallback can raise on a non-JSON-safe spec, so it too is
    best-effort.
    """
    lane_name = _str_or_none(getattr(lane, "name", None))
    ids = _safe_call(getattr(lane, "worker_ids", None))
    if ids is not None:
        try:
            id_list = list(ids)
        except TypeError:
            id_list = []
        return [_worker_from_lane_by_id(lane, str(wid), lane_name) for wid in id_list]
    snap = _safe_call(getattr(lane, "snapshot", None))
    if isinstance(snap, dict) and _is_plain_iterable(snap.get("workers")):
        return [
            _worker_item_to_dict(w, default_lane=_str_or_none(snap.get("lane")) or lane_name)
            for w in snap["workers"]
        ]
    return []


def _worker_item_to_dict(item: Any, *, default_lane: str | None = None) -> dict[str, Any]:
    """Coerce a single handle-ish thing (a dict, a :class:`WorkerHandle`, a
    :class:`WorkerResult`, a ``snapshot_dict``) into a presentation worker dict."""
    if isinstance(item, dict):
        spec = item.get("spec") if isinstance(item.get("spec"), dict) else {}
        task_id = item.get("task_id")
        if task_id is None:
            task_id = spec.get("task_id")
        goal = item.get("goal")
        if not isinstance(goal, str) or not goal:
            goal = spec.get("goal")
        return _worker_dict(
            worker_id=item.get("worker_id"),
            status=item.get("status"),
            lane=item.get("lane") or default_lane,
            task_id=task_id,
            cancel_requested=item.get("cancel_requested", False),
            goal=goal,
            result=item.get("result"),
            error=item.get("error"),
        )
    spec = getattr(item, "spec", None)
    spec_goal = getattr(spec, "goal", None)
    return _worker_dict(
        worker_id=getattr(item, "worker_id", None),
        status=getattr(item, "status", None),
        lane=getattr(item, "lane", None) or default_lane,
        task_id=getattr(item, "task_id", None),
        cancel_requested=getattr(item, "cancel_requested", False),
        goal=getattr(item, "goal", None) or spec_goal,
        result=getattr(item, "result", None),
        error=getattr(item, "error", None),
    )


def _collect_tasks(source: Any, *, session_key: str | None, active_only: bool) -> list[dict[str, Any]]:
    """Resolve *source* into a list of presentation task dicts.

    *source* may be an :class:`OrchestrationSnapshot`, a duck-typed task registry
    (anything with ``list_tasks``), a list of :class:`~agent.task_registry.FocusedTask`-ish
    objects or task dicts, or ``None``.  ``active_only`` only applies when *source*
    is a registry (a registry is the "live" handle, so a status command reading one
    shows the active set by default); a snapshot or explicit list is shown as given.
    """
    if source is None:
        return []
    if isinstance(source, OrchestrationSnapshot):
        return [dict(t) for t in source.tasks]
    list_tasks = getattr(source, "list_tasks", None)
    if callable(list_tasks):
        try:
            tasks = list_tasks(session_key=session_key, active_only=active_only)
        except TypeError:
            # A registry whose list_tasks does not take those kwargs.
            tasks = list_tasks()
        return [_task_to_dict(t) for t in (tasks or [])]
    if _is_plain_iterable(source):
        out: list[dict[str, Any]] = []
        for t in source:
            out.append(_task_to_dict(t))
        return out
    return []


def _collect_workers(source: Any) -> list[dict[str, Any]]:
    """Resolve *source* into a list of presentation worker dicts.

    *source* may be an :class:`OrchestrationSnapshot`, a :class:`WorkerLaneRegistry`
    (``lane_names`` + ``get_lane``), a single lane (``name`` + ``worker_ids`` /
    ``snapshot``), a ``{"lane": ..., "workers": [...]}`` dict, a list of handle-ish
    objects / dicts, or ``None``.
    """
    if source is None:
        return []
    if isinstance(source, OrchestrationSnapshot):
        return [dict(w) for w in source.workers]
    # WorkerLaneRegistry: enumerate lanes, then workers within each.
    lane_names = getattr(source, "lane_names", None)
    get_lane = getattr(source, "get_lane", None)
    if callable(lane_names) and callable(get_lane):
        out: list[dict[str, Any]] = []
        for name in (_safe_call(lane_names) or []):
            lane = _safe_call1(get_lane, name)
            if lane is not None:
                out.extend(_workers_from_lane(lane))
        return out
    # A single lane: it can enumerate its own workers.
    if callable(getattr(source, "worker_ids", None)) or (
        _str_or_none(getattr(source, "name", None)) and callable(getattr(source, "snapshot", None))
    ):
        return _workers_from_lane(source)
    # A lane snapshot dict.
    if isinstance(source, dict) and _is_plain_iterable(source.get("workers")):
        default_lane = _str_or_none(source.get("lane"))
        return [_worker_item_to_dict(w, default_lane=default_lane) for w in source["workers"]]
    # A plain list of handle-ish things.
    if _is_plain_iterable(source):
        return [_worker_item_to_dict(w) for w in source]
    return []


# --------------------------------------------------------------------------
# OrchestrationSnapshot
# --------------------------------------------------------------------------
@dataclass
class OrchestrationSnapshot:
    """A flat, JSON-safe point-in-time view of the orchestration state.

    ``tasks`` and ``workers`` are lists of compact presentation dicts (see
    :func:`_task_to_dict` / :func:`_worker_dict`).  ``counts`` is a small summary
    (``tasks_total`` / ``tasks_active`` / ``tasks_blocked`` / ``tasks_error`` /
    ``workers_total`` / ``workers_active`` / ``workers_running`` / ``workers_error``
    / ``workers_cancel_requested`` / ``followups_pending``).  ``warnings`` are
    cheap cross-consistency notes (a task pointing at an unknown worker; a worker
    pointing at an unknown task) -- empty when only one side was supplied or
    everything lines up.  Every value is a plain string / int / ``None`` / list /
    dict, so :meth:`to_dict` round-trips through ``json.dumps`` cleanly.
    """

    tasks: list[dict[str, Any]] = field(default_factory=list)
    workers: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe copy (fresh lists/dicts; nested values are already plain)."""
        return {
            "tasks": [dict(t) for t in self.tasks],
            "workers": [dict(w) for w in self.workers],
            "counts": dict(self.counts),
            "warnings": list(self.warnings),
        }


def _count_state(tasks: list[dict[str, Any]], workers: list[dict[str, Any]]) -> dict[str, int]:
    running = _WorkerStatus.RUNNING
    return {
        "tasks_total": len(tasks),
        "tasks_active": sum(1 for t in tasks if t.get("active")),
        "tasks_blocked": sum(1 for t in tasks if t.get("blocked")),
        "tasks_error": sum(1 for t in tasks if t.get("error")),
        "workers_total": len(workers),
        "workers_active": sum(1 for w in workers if w.get("active")),
        "workers_running": sum(1 for w in workers if w.get("status") == running),
        "workers_error": sum(1 for w in workers if w.get("status") == _WorkerStatus.ERROR),
        "workers_cancel_requested": sum(1 for w in workers if w.get("cancel_requested")),
        "followups_pending": sum(int(t.get("followups") or 0) for t in tasks),
    }


def _consistency_warnings(tasks: list[dict[str, Any]], workers: list[dict[str, Any]]) -> list[str]:
    """Cheap, deterministic cross-checks -- only when *both* sides were supplied."""
    if not tasks or not workers:
        return []
    worker_ids = {w.get("worker_id") for w in workers if w.get("worker_id")}
    task_ids = {t.get("task_id") for t in tasks if t.get("task_id")}
    warnings: list[str] = []
    for t in tasks:
        wid = t.get("worker_id")
        if wid and wid not in worker_ids:
            warnings.append(
                f"task {t.get('task_id')} references worker {wid} which is not in any lane"
            )
    for w in workers:
        tid = w.get("task_id")
        if tid and tid not in task_ids:
            warnings.append(
                f"worker {w.get('worker_id')} references task {tid} which is not registered"
            )
    return warnings


def build_snapshot(
    task_registry: Any = None,
    worker_registry: Any = None,
    *,
    session_key: str | None = None,
) -> OrchestrationSnapshot:
    """Build an :class:`OrchestrationSnapshot` from a task source and/or a worker source.

    *task_registry* is read via ``list_tasks(session_key=..., active_only=False)``
    when it is a registry (so the snapshot captures *all* tasks for the session,
    terminal ones included -- the active/total split lives in ``counts``); it may
    also be an :class:`OrchestrationSnapshot`, a list of tasks, or ``None``.
    *worker_registry* may be a :class:`WorkerLaneRegistry`, a single lane, a list
    of handle-ish objects, or ``None``.  Either may be omitted; an all-``None``
    call yields an empty snapshot.
    """
    tasks = _collect_tasks(task_registry, session_key=session_key, active_only=False)
    workers = _collect_workers(worker_registry)
    return OrchestrationSnapshot(
        tasks=tasks,
        workers=workers,
        counts=_count_state(tasks, workers),
        warnings=_consistency_warnings(tasks, workers),
    )


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def _goal_limit(compact: bool) -> int:
    return _GOAL_COMPACT if compact else _GOAL_FULL


def _detail_limit(compact: bool) -> int:
    return _DETAIL_COMPACT if compact else _DETAIL_FULL


def _format_task_block(task: dict[str, Any], *, compact: bool, worker_index: dict[str, dict[str, Any]]) -> str:
    """One task rendered as a header line plus optional worker / counts lines."""
    task_id = task.get("task_id") or "?"
    status = task.get("status") or "?"
    head = f"- {task_id} [{status}]"
    goal = _truncate(task.get("goal"), _goal_limit(compact))
    if goal:
        head += f" {goal}"
    if task.get("blocked"):
        head += " — needs you"
    elif task.get("error"):
        head += " — failed"
    lines = [head]

    worker_id = task.get("worker_id")
    if worker_id:
        wline = f"  worker: {worker_id}"
        kind = task.get("worker_kind")
        if kind:
            wline += f" ({kind})"
        live = worker_index.get(worker_id)
        if live is not None:
            wline += f" [{live.get('status') or '?'}]"
            if live.get("cancel_requested"):
                wline += " (cancel requested)"
        lines.append(wline)

    bits: list[str] = []
    if task.get("followups"):
        bits.append(_plural(int(task["followups"]), "follow-up") + " queued")
    if task.get("notes"):
        bits.append(_plural(int(task["notes"]), "note"))
    if not compact and task.get("artifacts"):
        bits.append(_plural(int(task["artifacts"]), "artifact"))
    if bits:
        lines.append("  " + " · ".join(bits))
    return "\n".join(lines)


def _format_worker_line(worker: dict[str, Any], *, compact: bool) -> str:
    worker_id = worker.get("worker_id") or "?"
    status = worker.get("status") or "?"
    line = f"- {worker_id} [{status}]"
    lane = worker.get("lane")
    if lane:
        line += f" lane={lane}"
    task_id = worker.get("task_id")
    if task_id:
        line += f" task={task_id}"
    limit = _detail_limit(compact)
    if status == _WorkerStatus.ERROR:
        detail = _truncate(worker.get("error"), limit) or _truncate(worker.get("goal"), limit)
        if detail:
            line += f' error="{detail}"' if worker.get("error") else f' goal="{detail}"'
    elif status == _WorkerStatus.DONE:
        result = _truncate(worker.get("result"), limit)
        if result:
            line += f' result="{result}"'
        else:
            goal = _truncate(worker.get("goal"), limit)
            if goal:
                line += f' goal="{goal}"'
    elif status == _WorkerStatus.CANCELLED:
        goal = _truncate(worker.get("goal"), limit)
        if goal:
            line += f' goal="{goal}"'
    else:  # queued / running / unknown-active
        goal = _truncate(worker.get("goal"), limit)
        if goal:
            line += f' goal="{goal}"'
    if worker.get("cancel_requested"):
        line += " (cancel requested)"
    return line


def _tasks_header(tasks: list[dict[str, Any]]) -> str:
    return "Active tasks:" if all(t.get("active") for t in tasks) else "Tasks:"


def _workers_header(workers: list[dict[str, Any]]) -> str:
    return "Active workers:" if all(w.get("active") for w in workers) else "Workers:"


def format_tasks(
    snapshot_or_registry: Any = None,
    *,
    compact: bool = True,
    session_key: str | None = None,
) -> str:
    """Render the focused-task board.

    *snapshot_or_registry* may be an :class:`OrchestrationSnapshot` (rendered as
    captured, terminal tasks included), a duck-typed task registry (rendered as
    its *active* tasks for *session_key*), a list of tasks / task dicts, or
    ``None`` (empty board).  When given a snapshot, a task's recorded worker is
    annotated with that worker's *live* status if the snapshot also captured it.
    The result is bullets-and-labels text, never a markdown table; an empty board
    is the single line ``"No active tasks are currently registered."``.
    """
    is_snapshot = isinstance(snapshot_or_registry, OrchestrationSnapshot)
    tasks = _collect_tasks(snapshot_or_registry, session_key=session_key, active_only=not is_snapshot)
    if not tasks:
        return _EMPTY_TASKS
    worker_index: dict[str, dict[str, Any]] = {}
    if is_snapshot:
        for w in snapshot_or_registry.workers:
            wid = w.get("worker_id")
            if wid:
                worker_index[wid] = w
    blocks = [_format_task_block(t, compact=compact, worker_index=worker_index) for t in tasks]
    return _tasks_header(tasks) + "\n" + "\n".join(blocks)


def format_agents(
    snapshot_or_registry: Any = None,
    *,
    compact: bool = True,
) -> str:
    """Render the worker/agent board.

    *snapshot_or_registry* may be an :class:`OrchestrationSnapshot`, a
    :class:`WorkerLaneRegistry`, a single lane, a list of handle-ish objects /
    worker dicts, or ``None`` (empty board).  Each worker line carries its id,
    status, lane, linked task, a compact goal/result/error, and a cancel-requested
    marker when set; an empty board is
    ``"No active workers are currently registered."``.
    """
    if isinstance(snapshot_or_registry, OrchestrationSnapshot):
        workers = [dict(w) for w in snapshot_or_registry.workers]
    else:
        workers = _collect_workers(snapshot_or_registry)
    if not workers:
        return _EMPTY_AGENTS
    lines = [_format_worker_line(w, compact=compact) for w in workers]
    return _workers_header(workers) + "\n" + "\n".join(lines)


def _overview_header(counts: dict[str, int]) -> str:
    parts = [_plural(int(counts.get("tasks_active", 0)), "active task")]
    blocked = int(counts.get("tasks_blocked", 0))
    if blocked:
        parts.append(f"{blocked} blocked")
    if int(counts.get("workers_total", 0)) > 0:
        parts.append(_plural(int(counts.get("workers_running", 0)), "running worker"))
    cancelling = int(counts.get("workers_cancel_requested", 0))
    if cancelling:
        parts.append(f"{cancelling} cancelling")
    followups = int(counts.get("followups_pending", 0))
    if followups:
        parts.append(_plural(followups, "follow-up") + " queued")
    return "Orchestration status — " + ", ".join(parts) + "."


def format_overview(
    snapshot_or_task_registry: Any = None,
    worker_registry: Any = None,
    *,
    session_key: str | None = None,
    compact: bool = True,
) -> str:
    """Render the combined status board: a one-line summary, then tasks, then workers.

    The first positional argument may already be an :class:`OrchestrationSnapshot`
    (in which case *worker_registry* is ignored); otherwise it is treated as a task
    source and a snapshot is built from ``(task_source, worker_registry,
    session_key)``.  Sections are separated by blank lines; any consistency
    ``warnings`` (and a callout when something is blocked) follow at the end.  When
    there are neither tasks nor workers the whole thing collapses to
    ``"No active tasks or workers are currently registered."``.
    """
    if isinstance(snapshot_or_task_registry, OrchestrationSnapshot):
        snap = snapshot_or_task_registry
    else:
        snap = build_snapshot(
            task_registry=snapshot_or_task_registry,
            worker_registry=worker_registry,
            session_key=session_key,
        )
    if not snap.tasks and not snap.workers:
        return _EMPTY_OVERVIEW

    sections = [_overview_header(snap.counts)]

    if snap.tasks:
        blocks = [_format_task_block(t, compact=compact, worker_index={w["worker_id"]: w for w in snap.workers if w.get("worker_id")}) for t in snap.tasks]
        sections.append(_tasks_header(snap.tasks) + "\n" + "\n".join(blocks))
    if snap.workers:
        lines = [_format_worker_line(w, compact=compact) for w in snap.workers]
        sections.append(_workers_header(snap.workers) + "\n" + "\n".join(lines))

    callouts: list[str] = []
    blocked = int(snap.counts.get("tasks_blocked", 0))
    if blocked:
        verb = "is" if blocked == 1 else "are"
        callouts.append(f"⚠ {_plural(blocked, 'task')} {verb} blocked and waiting on you.")
    for w in snap.warnings:
        callouts.append(f"⚠ {w}.")
    if callouts:
        sections.append("\n".join(callouts))

    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# OrchestrationStatusFormatter -- the namespaced façade the packet sketches
# --------------------------------------------------------------------------
class OrchestrationStatusFormatter:
    """Thin namespace over :func:`build_snapshot` / :func:`format_tasks` /
    :func:`format_agents` / :func:`format_overview`.

    Stateless (every method is static); kept as a class so a later phase can give
    it configuration -- a default ``compact`` mode, a locale, an injected runtime
    registry -- without changing call sites.
    """

    @staticmethod
    def snapshot(
        task_registry: Any = None,
        worker_registry: Any = None,
        *,
        session_key: str | None = None,
    ) -> OrchestrationSnapshot:
        return build_snapshot(task_registry, worker_registry, session_key=session_key)

    @staticmethod
    def format_tasks(
        snapshot_or_registry: Any = None,
        *,
        compact: bool = True,
        session_key: str | None = None,
    ) -> str:
        return format_tasks(snapshot_or_registry, compact=compact, session_key=session_key)

    @staticmethod
    def format_agents(snapshot_or_registry: Any = None, *, compact: bool = True) -> str:
        return format_agents(snapshot_or_registry, compact=compact)

    @staticmethod
    def format_overview(
        snapshot_or_task_registry: Any = None,
        worker_registry: Any = None,
        *,
        session_key: str | None = None,
        compact: bool = True,
    ) -> str:
        return format_overview(
            snapshot_or_task_registry,
            worker_registry,
            session_key=session_key,
            compact=compact,
        )


# --------------------------------------------------------------------------
# Natural-language status-query predicate (a hint only -- routes nothing).
#
# Like agent.followup_router's trigger lists this is the documented set, not an
# exhaustive grammar: a later LLM phase can refine the genuinely ambiguous text.
# Matching is plain case-insensitive substring (Korean is unaffected by casing) --
# deliberately a touch generous, because this predicate decides nothing on its
# own; it only lets a later gateway/CLI layer choose to call the formatter.
# --------------------------------------------------------------------------
_STATUS_QUERY_TRIGGERS: tuple[str, ...] = (
    # --- Korean: "what are you doing / what's running / which tasks / agents" ---
    "뭐 하고 있", "뭐하고 있", "뭐 하고있", "뭐하고있", "뭐 하는 중", "뭐하는 중",
    "뭐 하는중", "뭐하는중", "뭐 하는 거", "뭐 하고 계", "지금 뭐 해", "지금 뭐 하",
    "지금 뭐하", "뭐 하니", "뭐 하고 있니", "뭐 하는 중이", "뭐 진행", "뭐 작업",
    "무슨 작업", "어떤 작업", "뭐 돌아가", "뭐 돌고", "뭐가 돌아", "뭐가 돌고",
    "돌고 있는 작업", "돌고있는 작업", "돌아가는 작업", "돌고 있는 거", "돌고있는 거",
    "진행 중인 작업", "진행중인 작업", "진행 중인 거", "진행중인 거", "진행 중인 일", "진행중인 일",
    "작업 목록", "작업 리스트", "작업 있어", "작업 있나", "작업 뭐", "작업 몇 개", "작업 몇개",
    "활성 작업", "활성화된 작업", "남은 작업",
    "에이전트 뭐", "에이전트 목록", "에이전트 리스트", "에이전트 돌", "에이전트 있", "에이전트 몇",
    "무슨 에이전트", "어떤 에이전트", "워커 뭐", "워커 목록", "워커 돌", "워커 있", "일꾼",
    "대기 중인", "대기중인", "막힌 거", "막혀 있는", "막혀있는", "기다리는 거", "내가 봐야",
    "내가 확인해야", "후속 작업", "후속 요청", "팔로업", "팔로우업", "follow-up 뭐",
    # --- English ---
    "what are you working on", "what're you working on", "what you working on",
    "what are you doing", "what're you doing", "what you doing",
    "are you working on anything", "anything you're working on", "anything you are working on",
    "what's running", "whats running", "what is running", "anything running",
    "anything active", "what's active", "whats active",
    "active task", "active tasks", "active worker", "active workers",
    "any active task", "any tasks", "any task running", "tasks running", "tasks active",
    "what tasks", "which tasks", "what task is", "list tasks", "show tasks",
    "show me the tasks", "task list", "the task board",
    "any agents", "any agent running", "agents running", "what agents", "which agents",
    "agent list", "list agents", "show agents", "any workers", "what workers",
    "which workers", "workers running", "worker list",
    "what's in progress", "whats in progress", "what is in progress", "in progress right now",
    "what's queued", "whats queued", "anything queued", "follow-ups queued",
    "followups queued", "follow ups queued",
    "anything blocked", "what's blocked", "whats blocked", "is anything blocked",
    "anything waiting for me", "waiting for me", "waiting on me",
    "anything need me", "anything needs me", "need my input", "needs my input",
    "status board", "orchestration status", "orchestrator status",
)


def looks_like_orchestration_status_query(text: Any) -> bool:
    """True when *text* reads like a request for the orchestration status board.

    Examples that match: ``"지금 뭐 하고 있어?"``, ``"돌고 있는 작업 있어?"``,
    ``"에이전트 뭐 돌아가?"``, ``"what are you working on?"``, ``"active tasks?"``,
    ``"any agents running?"``.  This is a *hint*: it neither parses nor routes the
    message -- a later gateway/CLI layer may use it to decide to call
    :func:`format_overview` (or :func:`format_tasks` / :func:`format_agents`).
    Non-strings, empty strings, and ordinary prose all return ``False``.
    """
    if not isinstance(text, str):
        return False
    low = text.strip().lower()
    if not low:
        return False
    return any(trigger in low for trigger in _STATUS_QUERY_TRIGGERS)
