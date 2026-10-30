"""``ctx.schedule`` registration: spec comparison, cadence reset, and teardown.

The spec lives in ``task_state`` (per profile), not in memory, so process boot, hot reload and a
version upgrade all take the same path: an identical spec keeps its cadence (handler reference
swapped atomically); a changed one re-bases it and writes one ``rescheduled`` audit row.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from scheduler.engine import SchedulingEngine, TaskHandle, get_engine
from scheduler.tasks import TaskSpec, validate_spec

logger = logging.getLogger("hermes.scheduler")


def spec_hash(spec: TaskSpec) -> str:
    material = "|".join(str(value) for value in (
        spec.period, spec.every, spec.phase, spec.timeout_ms or "", int(bool(spec.once)),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _open_db() -> Any:
    try:
        from hermes_state_registry import acquire

        return acquire()
    except Exception:  # registration must not fail because accounting is unavailable
        logger.debug("scheduler: task_state unavailable", exc_info=True)
        return None


def _close_db(db: Any) -> None:
    if db is None:
        return
    try:
        from hermes_state_registry import release_or_close

        release_or_close(db)
    except Exception:  # pragma: no cover - teardown path
        pass


class ScheduledTaskHandle:
    """Plugin-facing handle: cancellation also disables the task durably."""

    def __init__(self, engine: SchedulingEngine, spec: TaskSpec, handle: TaskHandle,
                 db: Any = None) -> None:
        self._engine = engine
        self._spec = spec
        self._handle = handle
        self._db = db  # caller-owned DB during registration; opened per release otherwise

    @property
    def name(self) -> str:
        return self._spec.name

    def cancel(self) -> None:
        self.release("admin")

    def release(self, reason: str) -> None:
        """Remove the wheel nodes and disable the task durably with *reason*."""
        self._handle.cancel()
        db = self._db if self._db is not None else _open_db()
        if db is None:
            return
        try:
            db.set_task_enabled(self._spec.plugin_id, self._spec.name, enabled=False, disabled_by=reason)
        except Exception:
            logger.debug("scheduler: disabling cancelled task failed", exc_info=True)
        finally:
            if self._db is None:
                _close_db(db)


def register_scheduled_task(
    spec: TaskSpec, *, engine: Optional[SchedulingEngine] = None, db: Any = None,
) -> Optional[ScheduledTaskHandle]:
    """Register *spec* through the durable ``task_state`` contract; ``None`` when disabled."""
    engine = engine or get_engine()
    existing = engine.task_count(spec.plugin_id, exclude=spec.key)
    validate_spec(spec, max_tasks_per_plugin=engine.max_tasks_per_plugin, existing=existing)
    handle = engine.register(spec)
    task_spec_hash = spec_hash(spec)
    owns_db = db is None
    db = db if db is not None else _open_db()
    if db is None:
        return ScheduledTaskHandle(engine, spec, handle, None if owns_db else db)
    try:
        state = db.get_task_state(spec.plugin_id, spec.name)
        if state is None:
            db.save_task_state(
                spec.plugin_id, spec.name, period=spec.period, every=spec.every, phase=spec.phase,
                spec_hash=task_spec_hash, next_due_hint=spec.every,
            )
            return ScheduledTaskHandle(engine, spec, handle, None if owns_db else db)
        if state.get("spec_hash") == task_spec_hash:
            if not int(state.get("enabled") or 0):
                if state.get("disabled_by") == "admin":
                    handle.cancel()
                    return None
                db.set_task_enabled(spec.plugin_id, spec.name, enabled=True)
            return ScheduledTaskHandle(engine, spec, handle, None if owns_db else db)
        admin_disabled = state.get("disabled_by") == "admin"
        old = f"every {state.get('every')}→{spec.every}; phase {state.get('phase')}→{spec.phase}"
        db.save_task_state(
            spec.plugin_id, spec.name, period=spec.period, every=spec.every, phase=spec.phase,
            spec_hash=task_spec_hash, next_due_hint=spec.every, enabled=not admin_disabled,
            disabled_by="admin" if admin_disabled else None, reset=True, last_status="rescheduled",
        )
        db.record_task_run(
            plugin_id=spec.plugin_id, task_name=spec.name, period=spec.period, phase=spec.phase,
            status="rescheduled", error=old,
        )
        if admin_disabled:
            handle.cancel()
            return None
        return ScheduledTaskHandle(engine, spec, handle, None if owns_db else db)
    except Exception:
        logger.warning("scheduler: task_state comparison failed for %s", spec.name, exc_info=True)
        return ScheduledTaskHandle(engine, spec, handle, None if owns_db else db)
    finally:
        if owns_db:
            _close_db(db)
