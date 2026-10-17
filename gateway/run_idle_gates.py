"""Idle gates for the gateway's per-profile pollers (heartbeat restore, handoff watcher, loop wakeup).

Entering ``_profile_runtime_scope`` costs a config.yaml load, a ``.env`` parse, secret hydration and a
terminal-policy build; on a multiplex gateway the pollers paid that per profile per tick with nothing
to do. Each gate answers "does this profile's store hold work?" from the goals-cached SessionDB with
ONLY the HERMES_HOME contextvar installed. Every gate fails OPEN: an unavailable store, a failing
read or a corrupt row is "cannot prove emptiness", never "idle".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("gateway.run")


def _profile_session_db_probe(profile_home: Path) -> Optional[Any]:
    """The goals-cached SessionDB for *profile_home*; None when unavailable."""
    from hermes_cli.goals import _get_session_db
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(profile_home))
    try:
        return _get_session_db()
    except Exception:
        logger.debug("session-db probe failed for %s", profile_home, exc_info=True)
        return None
    finally:
        reset_hermes_home_override(token)


def _gate(profile_home: Path, store_has_work: Callable[[Any], bool]) -> bool:
    db = _profile_session_db_probe(profile_home)
    if db is None:
        return True
    try:
        return store_has_work(db)
    except Exception:
        logger.debug("idle probe failed for %s; keeping the full sweep", profile_home, exc_info=True)
        return True


def profile_has_active_heartbeat(profile_home: Path) -> bool:
    from hermes_cli.heartbeat import store_has_active_heartbeat

    return _gate(profile_home, store_has_active_heartbeat)


def profile_has_active_loop(profile_home: Path) -> bool:
    from hermes_cli.loops import store_has_active_loop

    return _gate(profile_home, store_has_active_loop)


def profile_has_pending_handoff(profile_home: Path) -> bool:
    return _gate(profile_home, lambda db: db.has_pending_handoffs())


async def off_loop_gate(runner: object, probe: Callable[[], bool]) -> bool:
    """Run a sync gate through the runner's executor hop. Runners without one (bare test stand-ins
    for the handoff watcher) keep the historical always-enter behaviour."""
    offload = getattr(runner, "_run_in_executor_with_context", None)
    if not callable(offload):
        return True
    return bool(await offload(probe))
