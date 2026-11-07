"""Kanban worker lane registry.

Populated by plugins via :meth:`PluginContext.register_kanban_worker_lane`; the dispatcher asks
:func:`matching_lanes` which lanes may run a task (``hermes_cli/kanban_db_lanes.py``). Lanes are
registered per profile like every provider registry, but a dispatcher serves all of its process's
profiles: the gateway's embedded dispatcher ticks every board for every profile it serves, so the
lookup reads the lanes of every profile loaded in this process, not only the active one.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.provider_registry import ProviderRegistry
from hermes_cli.kanban_worker_lanes import KanbanWorkerLane

logger = logging.getLogger(__name__)

_registry: ProviderRegistry[KanbanWorkerLane] = ProviderRegistry(
    label="Kanban worker lane", provider_cls=KanbanWorkerLane, logger=logger,
)
_registry.export(globals())


def check_lane(lane: Any) -> None:
    """Refuse (``ValueError``) a lane whose filters are malformed or match every task."""
    patterns = {}
    for field in ("assignees", "boards"):
        value = getattr(lane, field, ())
        if isinstance(value, str) or not isinstance(value, (tuple, list, frozenset, set)):
            raise ValueError(f"{field} must be a tuple of names or globs, got {type(value).__name__}")
        if not all(isinstance(p, str) and p.strip() for p in value):
            raise ValueError(f"{field} must hold non-empty strings")
        patterns[field] = value
    if not patterns["assignees"] and not patterns["boards"]:
        raise ValueError("a lane must name at least one assignee or board")


def matching_lanes(assignee: Any, board: Optional[str]) -> list[tuple[Optional[str], KanbanWorkerLane]]:
    """``(scope, lane)`` for every registered lane whose ``assignees`` and ``boards`` match, in a
    stable order (lane name, then scope) so the same task always reaches the same lane first."""
    rows = [(scope, lane) for scope, lane in _registry.all_scopes() if _matches(lane, assignee, board)]
    return sorted(rows, key=lambda row: (row[1].name, row[0] or ""))


def _matches(lane: KanbanWorkerLane, assignee: Any, board: Optional[str]) -> bool:
    try:
        return bool(lane.matches(assignee, board))
    except Exception:  # health: allow BLE001 -- a broken plugin lane matches nothing; the native spawn still runs
        logger.warning("Kanban worker lane %r: matches() raised; skipping it", lane.name, exc_info=True)
        return False
