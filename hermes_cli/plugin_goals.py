"""Plugin-safe ownership-aware access to persistent session goals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import time

from hermes_cli.goals import DEFAULT_MAX_TURNS, GoalContract, GoalManager, GoalState
from hermes_cli.goals_storage import save_if_unchanged


@dataclass(frozen=True)
class GoalSnapshot:
    """Read-only subset of a session goal exposed to plugins."""

    objective: str
    status: str
    turns_used: int
    max_turns: int
    owner: Optional[str]


class PluginGoals:
    """Goal operations for one plugin; ownership is always this plugin's id."""

    def __init__(self, plugin_id: str):
        self._plugin_id = plugin_id

    def _owner(self, owner: Optional[str]) -> str:
        if owner is not None and owner != self._plugin_id:
            raise PermissionError("plugins may only manage goals they own")
        return self._plugin_id

    @staticmethod
    def _snapshot(manager: GoalManager) -> Optional[GoalSnapshot]:
        state = manager.state
        if state is None or state.status == "cleared":
            return None
        return GoalSnapshot(state.goal, state.status, state.turns_used, state.max_turns, state.owner)

    def get(self, session_id: str) -> Optional[GoalSnapshot]:
        """Return the current goal without exposing its mutable state object."""
        return self._snapshot(GoalManager(session_id))

    def set(
        self, session_id: str, objective: str, *, contract: GoalContract | Mapping[str, Any] | None = None,
        max_turns: Optional[int] = None, owner: Optional[str] = None, if_absent: bool = True,
    ) -> bool:
        """Create a plugin-owned goal, preserving an active or paused goal by default."""
        manager = GoalManager(session_id)
        state = manager.state
        if if_absent and state is not None and state.status in {"active", "paused"}:
            return False
        if isinstance(contract, Mapping):
            contract = GoalContract.from_dict(dict(contract))
        if contract is not None and not isinstance(contract, GoalContract):
            raise TypeError("contract must be a GoalContract, mapping, or None")
        objective = objective.strip()
        if not objective:
            raise ValueError("goal text is empty")
        candidate = GoalState(
            goal=objective, status="active", created_at=time.time(),
            max_turns=int(max_turns) if max_turns else DEFAULT_MAX_TURNS,
            contract=contract if contract is not None else GoalContract(), owner=self._owner(owner),
        )
        return save_if_unchanged(session_id, candidate, state.to_json() if state else None) is True

    def clear(self, session_id: str, *, owner: Optional[str] = None) -> bool:
        """Clear only a goal that this plugin created."""
        owner = self._owner(owner)
        manager = GoalManager(session_id)
        state = manager.state
        if state is None or state.owner != owner:
            return False
        expected = state.to_json()
        state.status = "cleared"
        return save_if_unchanged(session_id, state, expected) is True
