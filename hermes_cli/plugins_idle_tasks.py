"""Plugin idle tasks (``ctx.register_idle_task``): the skill curator's scheduler, for plugins.

A plugin registers a session-less job over its own state. Every host that ticks the curator also
polls the registered idle tasks of the active profile's plugin manager (:func:`run_idle_jobs`):
gateway housekeeping (once per served profile), CLI/TUI startup maintenance and ``hermes serve``'s
maintenance ticker. A task runs when the host has been idle ``min_idle_hours``, the task is not
paused, ``interval_hours`` have passed since its last run (the first observation seeds
``last_run_at`` and defers one interval, like the curator, unless ``first_run="immediate"``) and no
other run of it holds the claim. Runs are synchronous on the polling thread; an exception is
logged and recorded, never raised.

State is per (profile, plugin, task): ``<plugin data dir>/idle-tasks/<name>.json`` with
``last_run_at``, ``run_count``, ``paused``, ``last_summary``, ``last_status``, ``last_error``.
``hermes plugins idle-tasks`` lists it and pauses/resumes a task.

No capability gates this surface: it calls nothing privileged itself; model calls go through
``ctx.llm`` and its existing gates.
"""

from __future__ import annotations

import logging
import math
import re
import time
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger("hermes_cli.plugins")

MIN_INTERVAL_HOURS = 1.0
_FIRST_RUN = ("defer", "immediate")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_STATE_DIR = "idle-tasks"
_SUMMARY_MAX_CHARS = 2000


@dataclass(frozen=True)
class IdleTaskRun:
    """What the task function receives: plain data, so it crosses the isolated plugin host too (the
    plugin reaches its own ``ctx`` through the closure that registered the task)."""

    name: str
    idle_for_seconds: float
    last_run_at: Optional[str]  # ISO-8601, None on an immediate first run
    run_count: int


@dataclass(frozen=True)
class IdleTaskSpec:
    ctx: Any  # PluginContext
    name: str
    fn: Callable[[IdleTaskRun], Optional[str]]
    interval_hours: float
    min_idle_hours: float
    first_run: str

    @property
    def plugin_id(self) -> str:
        return self.ctx.plugin_id

    def state_path(self) -> Path:
        """Resolved per call: the active profile's plugin data dir."""
        return self.ctx.state.data_dir / _STATE_DIR / f"{self.name}.json"


# Registered tasks per plugin manager (one manager per profile home), keyed "<plugin id>:<name>".
_registries: weakref.WeakKeyDictionary[Any, dict[str, IdleTaskSpec]] = weakref.WeakKeyDictionary()


def registered_idle_tasks(manager: Any) -> dict[str, IdleTaskSpec]:
    return _registries.setdefault(manager, {})


def build_idle_task_spec(
    ctx: Any, name: Any, fn: Any, *, interval_hours: Any, min_idle_hours: Any, first_run: Any,
) -> IdleTaskSpec:
    """The registration, or ``ValueError`` for a bad name, callable, interval or ``first_run``."""
    me = ctx.manifest.name
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"Plugin '{me}' idle task name {name!r} must match {_NAME_RE.pattern}")
    if not callable(fn):
        raise ValueError(f"Plugin '{me}' idle task {name!r}: fn must be callable")
    if first_run not in _FIRST_RUN:
        raise ValueError(f"Plugin '{me}' idle task {name!r}: first_run must be one of {_FIRST_RUN}")
    try:
        interval, idle = float(interval_hours), float(min_idle_hours)
    except (TypeError, ValueError):
        raise ValueError(f"Plugin '{me}' idle task {name!r}: interval_hours and min_idle_hours must be numbers") from None
    if not (math.isfinite(interval) and math.isfinite(idle)):  # inf would overflow timedelta; NaN never compares
        raise ValueError(f"Plugin '{me}' idle task {name!r}: interval_hours and min_idle_hours must be finite")
    if interval < MIN_INTERVAL_HOURS:
        logger.warning("Plugin '%s' idle task %r: interval_hours=%r floored to %s", me, name, interval_hours,
                       MIN_INTERVAL_HOURS)
        interval = MIN_INTERVAL_HOURS
    return IdleTaskSpec(ctx, name, fn, interval, idle if idle >= 0 else 0.0, first_run)


# --- state ------------------------------------------------------------------------------------


def _base_state(plugin_id: str, name: str) -> dict[str, Any]:
    return {"plugin": plugin_id, "task": name, "last_run_at": None, "run_count": 0, "paused": False,
            "last_summary": None, "last_status": None, "last_error": None, "last_run_duration_seconds": None,
            "interval_hours": None}


def load_state(path: Path, plugin_id: str = "", name: str = "") -> dict[str, Any]:
    import json

    state = _base_state(plugin_id, name or path.stem)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    except (OSError, ValueError) as e:
        logger.debug("Unreadable idle task state %s: %s", path, e)
        return state
    if isinstance(data, dict):
        state.update({k: v for k, v in data.items() if k in state})
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    from utils import atomic_json_write

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, state, indent=2, sort_keys=True)
    except Exception as e:
        logger.debug("Failed to save idle task state %s: %s", path, e, exc_info=True)


def set_paused(path: Path, paused: bool) -> dict[str, Any]:
    state = load_state(path)
    state["paused"] = bool(paused)
    save_state(path, state)
    return state


def iter_state_files(plugin_id: Optional[str] = None) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Every idle task state of the active profile (optionally one plugin's), sorted."""
    from hermes_constants import get_hermes_home

    for path in sorted((get_hermes_home() / "plugin-data").glob(f"*/{_STATE_DIR}/*.json")):
        state = load_state(path)
        if plugin_id is None or state.get("plugin") == plugin_id:
            yield path, state


def next_due(state: dict[str, Any]) -> Optional[str]:
    from agent.idle_schedule import parse_iso

    last, interval = parse_iso(state.get("last_run_at")), state.get("interval_hours")
    if last is None or not isinstance(interval, (int, float)) or not math.isfinite(interval):
        return None
    return (last + timedelta(hours=interval)).isoformat(timespec="seconds")


# --- scheduler --------------------------------------------------------------------------------


def _due(spec: IdleTaskSpec, state: dict[str, Any], path: Path, now: datetime) -> bool:
    from agent.idle_schedule import interval_due

    if state["paused"]:
        return False
    if state["last_run_at"] is None and spec.first_run == "immediate":
        return True
    state["interval_hours"] = spec.interval_hours
    return interval_due(state, spec.interval_hours, now=now, save=lambda s: save_state(path, s),
                        seed_summary="deferred first run: runs after one interval")


def _run_one(spec: IdleTaskSpec, idle_for_seconds: float, now: datetime) -> Optional[str]:
    """Run ``spec`` if every gate passes; its status (``ok``/``error``) or None when not run."""
    from agent.idle_schedule import claim_run, release_run

    if idle_for_seconds < spec.min_idle_hours * 3600.0:
        return None
    path = spec.state_path()
    state = load_state(path, spec.plugin_id, spec.name)
    if not _due(spec, state, path, now):
        return None
    lock = path.with_name(f".{spec.name}.lock")
    if not claim_run(lock):  # another poller (thread or process) is running it
        return None
    started, status, summary, error = time.monotonic(), "ok", None, None
    try:
        result = spec.fn(IdleTaskRun(spec.name, idle_for_seconds, state["last_run_at"], int(state["run_count"] or 0)))
        if isinstance(result, str) and result.strip():
            summary = result.strip()[:_SUMMARY_MAX_CHARS]
    except Exception as exc:
        # Boundary: a plugin's job never breaks the host's housekeeping.
        status, error = "error", f"{type(exc).__name__}: {exc}"[:500]
        logger.warning("Plugin %s idle task %s failed", spec.plugin_id, spec.name, exc_info=True)
    finally:
        fresh = load_state(path, spec.plugin_id, spec.name)  # keep a pause set while it ran
        fresh.update(last_run_at=now.isoformat(), run_count=int(fresh["run_count"] or 0) + 1, last_status=status,
                     last_summary=summary, last_error=error, interval_hours=spec.interval_hours,
                     last_run_duration_seconds=round(time.monotonic() - started, 3))
        save_state(path, fresh)
        release_run(lock)
    logger.info("plugin idle task plugin=%s task=%s status=%s", spec.plugin_id, spec.name, status)
    return status


def run_plugin_idle_tasks(
    *, idle_for_seconds: float, manager: Any = None, now: Optional[datetime] = None,
) -> dict[str, str]:
    """Poll the active profile's registered idle tasks; ``{"<plugin>:<task>": status}`` for those
    that ran. A manager polled outside its own profile runs nothing. Never raises."""
    from hermes_constants import hermes_home_key
    from hermes_cli.plugins import get_plugin_manager

    ran: dict[str, str] = {}
    try:
        manager = manager if manager is not None else get_plugin_manager()
        if manager.scope_key != hermes_home_key():
            return ran
        now = now or datetime.now(UTC)
        for key, spec in list(registered_idle_tasks(manager).items()):
            if (status := _run_one(spec, idle_for_seconds, now)) is not None:
                ran[key] = status
    except Exception:
        logger.debug("plugin idle task poll failed", exc_info=True)
    return ran


def run_idle_jobs(*, idle_for_seconds: float, on_summary: Optional[Callable[[str], None]] = None) -> None:
    """The idle poll every curator host runs: the curator, then the plugins' idle tasks."""
    from agent.curator import maybe_run_curator

    maybe_run_curator(idle_for_seconds=idle_for_seconds, on_summary=on_summary)
    run_plugin_idle_tasks(idle_for_seconds=idle_for_seconds)


def cmd_idle_tasks(plugin: Optional[str] = None, *, pause: Optional[str] = None, resume: Optional[str] = None) -> None:
    """``hermes plugins idle-tasks [plugin] [--pause TASK | --resume TASK]``: list the active
    profile's idle task states, or pause/resume one task (the plugin name is required then)."""
    target = pause or resume
    if target is not None:
        if not plugin:
            raise SystemExit("hermes plugins idle-tasks: name the plugin to pause or resume a task")
        hit = next((path for path, state in iter_state_files(plugin) if state["task"] == target), None)
        if hit is None:
            raise SystemExit(f"No idle task {target!r} of plugin {plugin!r} has run or been seeded in this profile")
        set_paused(hit, pause is not None)
        print(f"{plugin}:{target} {'paused' if pause else 'resumed'}")
        return
    rows = list(iter_state_files(plugin))
    if not rows:
        print("No plugin idle tasks have been seeded in this profile.")
        return
    for _path, s in rows:
        status = "paused" if s["paused"] else (s["last_status"] or "seeded")
        print(f"{s['plugin']}:{s['task']}  {status}  runs={s['run_count']}  last={s['last_run_at'] or '-'}  "
              f"next={next_due(s) or '-'}")
        if s["last_summary"] or s["last_error"]:
            print(f"    {s['last_error'] or s['last_summary']}")


class PluginIdleTaskCtxMixin:
    """``PluginContext.register_idle_task``."""

    def register_idle_task(
        self, name: str, fn: Callable[[IdleTaskRun], Optional[str]], *, interval_hours: float,
        min_idle_hours: float = 2.0, first_run: str = "defer",
    ):
        """Run ``fn`` in the background when Hermes is idle, at most once per ``interval_hours``
        (floored to 1) and only after ``min_idle_hours`` without activity. ``fn`` receives an
        :class:`IdleTaskRun` (``name``, ``idle_for_seconds``, ``last_run_at``, ``run_count``) and
        may return a short summary string. The first run waits one interval
        unless ``first_run="immediate"``. Returns the :class:`PluginRegistration` (dispose to stop
        polling). Raises ``ValueError`` for a bad name (``[a-z0-9][a-z0-9_-]*``), a non-callable
        ``fn``, a non-numeric interval, or an unknown ``first_run``."""
        spec = build_idle_task_spec(self, name, fn, interval_hours=interval_hours,
                                    min_idle_hours=min_idle_hours, first_run=first_run)
        tasks = registered_idle_tasks(self._manager)
        key = f"{self.plugin_id}:{name}"
        return self._register_entry("idle_task", key, tasks, spec, "Plugin %s registered idle task: %s", name,
                                    previous=tasks.get(key))
