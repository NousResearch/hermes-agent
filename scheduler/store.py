"""Persist scheduler dispatch outcomes to ``task_runs`` in the caller's profile DB.

Called from the engine's run sink on the turn thread; the tick's contextvars carry the
profile, so ``acquire()`` opens the same ``state.db`` the turn writes.
"""

from __future__ import annotations

import logging
import os

from scheduler.runner import CallbackOutcome
from scheduler.tasks import TaskContext, TaskSpec

logger = logging.getLogger("hermes.scheduler")

_STATUSES = frozenset(("ok", "error", "timeout", "skipped"))


def record_task_run(spec: TaskSpec, ctx: TaskContext, outcome: CallbackOutcome) -> None:
    from hermes_state_registry import acquire, release_or_close

    status = outcome.status if outcome.status in _STATUSES else "skipped"
    detail = None
    if status != "ok":
        detail = outcome.error or outcome.reason or None
    db = acquire()
    try:
        db.record_task_run(
            plugin_id=spec.plugin_id, task_name=spec.name, period=spec.period, phase=spec.phase,
            status=status, owner_pid=str(os.getpid()), session_id=ctx.session_id or None,
            turn_id=ctx.turn_id, turn_seq=ctx.turn_seq, step_seq=ctx.step_seq,
            step_index=ctx.step_index, profile_name=ctx.profile_name, agent_name=ctx.agent_name,
            duration_ms=outcome.duration_ms, error=detail, patience_s=0.5,
        )
    finally:
        release_or_close(db)
