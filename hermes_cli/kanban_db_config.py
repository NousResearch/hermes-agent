"""Shared ``kanban.*`` config resolvers for every dispatch entry point.

The gateway-embedded dispatcher, the standalone daemon and
``hermes kanban dispatch`` must agree on what the operator configured;
when one entry point resolves a key and another silently falls back to a
parameter default, the setting looks honoured while doing nothing (the
``max_in_progress`` parity gap, and ``dispatch_stale_timeout_seconds``
never reaching the daemon/CLI paths at all). Resolvers read the merged
config (``load_config_readonly`` — defaults already applied) and never
raise: an unreadable config resolves to the conservative value.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def configured_max_in_progress() -> Optional[int]:
    """Read ``kanban.max_in_progress`` from config, or None when unset/invalid.

    Shared so every dispatch entry point agrees on "explicitly configured": a
    positive integer wins, anything else falls through to the derived default.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get("max_in_progress")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def configured_stale_timeout_seconds() -> int:
    """Read ``kanban.dispatch_stale_timeout_seconds`` (merged default 14400).

    ``0`` is a meaningful value (stale detection off) and is honoured;
    missing/invalid/negative resolves to ``0`` so a broken config never
    invents a reclaim policy the operator did not configure.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get(
            "dispatch_stale_timeout_seconds")
    except Exception:
        logger.debug("kanban stale-timeout config unreadable; stale detection off", exc_info=True)
        return 0
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(ival, 0)
