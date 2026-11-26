"""Leaf constants shared by ``gateway/run.py`` and its ``run_*`` mixin modules.

Kept import-cycle free (imports nothing from ``gateway.run``) because these values
are used as default-argument sentinels, which must resolve at ``def`` time.
"""

from __future__ import annotations

import asyncio
from typing import Any

# Sentinel for "caller did not pass metadata" vs "caller passed None".
_UNSET = object()


def _csv_or_list_to_set(raw: Any) -> set[str]:
    """Normalize a config list or comma-separated scalar into a string set."""
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def startup_should_abort(runner: Any) -> bool:
    """Single predicate for whether shutdown, drain, or restart must abort startup."""
    shutdown_event = getattr(runner, "_shutdown_event", None)
    return bool(
        getattr(runner, "_stop_task", None) is not None
        or getattr(runner, "_restart_requested", False)
        or getattr(runner, "_draining", False)
        or getattr(runner, "_stop_requested_by_signal", False)
        or getattr(runner, "_planned_takeover_shutdown", False)
        or (isinstance(shutdown_event, asyncio.Event) and shutdown_event.is_set())
    )


async def abort_startup_if_shutdown_requested(runner: Any, *, check: bool = True) -> bool:
    """Await any active stop task or invoke ``runner.stop`` when startup must abort."""
    if check and not startup_should_abort(runner):
        return False
    stop_task = getattr(runner, "_stop_task", None)
    if stop_task is not None and stop_task is not asyncio.current_task():
        await stop_task
        return True
    shutdown_event = getattr(runner, "_shutdown_event", None)
    if isinstance(shutdown_event, asyncio.Event) and shutdown_event.is_set():
        return True
    stop = getattr(runner, "stop", None)
    if callable(stop):
        await stop(
            restart=bool(getattr(runner, "_restart_requested", False)),
            detached_restart=bool(getattr(runner, "_restart_detached", False)),
            service_restart=bool(getattr(runner, "_restart_via_service", False)),
        )
    return True
