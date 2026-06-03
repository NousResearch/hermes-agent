"""Hermes direct Ultragoal runtime adapter.

The runtime does not spawn Kanban dispatcher workers.  It materializes a bounded
Hermes `/goal` handoff inside the run root so the active Hermes agent can execute
or resume the next tick with durable state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_direct_goal_action(*, run_id: str, tick: int, root_objective: str, target_mode: str, scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "stepId": f"{run_id}:tick-{tick + 1}:hermes-direct-goal-loop",
        "executor": "hermes-direct-goal-loop",
        "dispatcherUsed": False,
        "phase": "prepared",
        "targetMode": target_mode,
        "goalPrompt": (
            f"Execute Ultragoal run {run_id} directly through Hermes /goal. "
            f"Objective: {root_objective}. Do not call kanban dispatch; use Kanban only as authority."
        ),
        "scope": scope,
        "sideEffects": {
            "kanbanDispatchCalled": False,
            "branchCreated": False,
            "commitSha": None,
            "pushedRef": None,
            "prUrl": None,
            "gatewayRestarted": False,
            "deployed": False,
        },
    }


def write_direct_goal_handoff(run_root: Path, action: dict[str, Any]) -> None:
    import json

    path = run_root / "direct-goal-handoff.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(action, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


__all__ = ["build_direct_goal_action", "write_direct_goal_handoff"]
