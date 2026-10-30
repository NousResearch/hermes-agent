"""Built-in turn/step observers (``plugin_id="core"``).

Each task has a named consumer; no speculative surface. Handlers run on the bounded runner,
never inline, and use the profile bound in the tick's copied context to open the same
``state.db`` the turn is writing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from scheduler.engine import _scheduler_config
from scheduler.tasks import StepView, TaskContext, TaskSpec

logger = logging.getLogger("hermes.scheduler")

BUILTIN_PLUGIN_ID = "core"


def _with_db(fn: Callable[[Any], Any]) -> Any:
    from hermes_state_registry import acquire, release_or_close

    db = acquire()
    try:
        return fn(db)
    finally:
        release_or_close(db)


def _steps_config() -> dict[str, Any]:
    section = _scheduler_config().get("steps")
    return section if isinstance(section, dict) else {}


def prune_steps(ctx: TaskContext) -> None:
    """Trim the session's ledger past the configured cap (newest + every non-ok row kept)."""
    section = _steps_config()
    max_rows = int(section.get("max_rows_per_session", 2000) or 2000)
    retention_days = int(section.get("retention_days", 30) or 30)
    _with_db(lambda db: db.prune_turn_steps(
        ctx.session_id, max_rows=max_rows, retention_days=retention_days))


def audit_cache(ctx: TaskContext) -> None:
    """Warn when the newest window's cache-hit ratio regresses against the previous window."""
    recent = _with_db(lambda db: db.turn_step_cache_stats(ctx.session_id, last_n=10))
    wider = _with_db(lambda db: db.turn_step_cache_stats(ctx.session_id, last_n=20))
    prev_cached = wider["cache_read_tokens"] - recent["cache_read_tokens"]
    prev_uncached = wider["input_tokens"] - recent["input_tokens"]
    prev_total = prev_cached + prev_uncached
    if prev_total <= 0 or recent["steps"] < 5:
        return
    previous_ratio = prev_cached / prev_total
    if previous_ratio >= 0.2 and recent["hit_ratio"] < previous_ratio * 0.5:
        logger.warning(
            "scheduler: cache-hit ratio regressed for session %s: %.0f%% -> %.0f%% over the last %d step(s)",
            ctx.session_id, previous_ratio * 100, recent["hit_ratio"] * 100, int(recent["steps"]),
        )


def rollup_usage(ctx: TaskContext) -> None:
    """Force-drain the accounting queue during long turns so usage stays current."""
    _with_db(lambda db: db.flush_token_counts())


def register_builtin_tasks(engine: Any) -> None:
    from scheduler.plugin_api import register_scheduled_task

    specs = (
        TaskSpec(plugin_id=BUILTIN_PLUGIN_ID, name="core.steps.prune", period="turn",
                 every=50, phase="post", handler=prune_steps),
        TaskSpec(plugin_id=BUILTIN_PLUGIN_ID, name="core.cache.audit", period="step",
                 every=10, phase="post", handler=audit_cache),
        TaskSpec(plugin_id=BUILTIN_PLUGIN_ID, name="core.usage.rollup", period="step",
                 every=32, phase="post", handler=rollup_usage),
    )
    for spec in specs:
        register_scheduled_task(spec, engine=engine)
