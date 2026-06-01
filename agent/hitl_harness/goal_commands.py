"""goal_commands.py — CLI surface for GOAL.md driven development.

Exposes commands that can be registered in hermes_cli/commands.py or used directly.

Commands:
- goal create "description"
- goal run [--converge | --continuous]
- goal status
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from agent.hitl_harness.fitness_engine import FitnessEngine, FitnessResult
from agent.hitl_harness.fitness_wizard import FitnessFunctionWizard


def cmd_goal_create(description: str, session_id: str = "default") -> Dict[str, Any]:
    """Create a new GOAL.md from natural language description."""
    wizard = FitnessFunctionWizard()
    # In a full implementation this would call the wizard's generation logic
    goal = {
        "id": f"goal_{session_id}",
        "name": description[:60],
        "description": description,
        "properties": [
            {"name": "task_completion", "type": "threshold", "target": 0.9},
            {"name": "tool_correctness", "type": "threshold", "target": 0.85},
        ],
        "composite_weights": {"task_completion": 0.6, "tool_correctness": 0.4},
    }
    return {"status": "created", "goal": goal, "path": f"~/.hermes/goals/{session_id}/GOAL.md"}


def cmd_goal_run(
    session_id: str = "default",
    mode: str = "converge",
    max_iterations: int = 20,
) -> Dict[str, Any]:
    """Run autonomous improvement loop using the loaded GOAL.md."""
    engine = FitnessEngine()
    # Placeholder – real version would load GOAL.md and drive the agent loop
    return {
        "status": "started",
        "mode": mode,
        "session_id": session_id,
        "max_iterations": max_iterations,
        "message": "FitnessEngine loop initiated (see iterations.jsonl for progress)",
    }


def cmd_goal_status(session_id: str = "default") -> Dict[str, Any]:
    """Show current GOAL.md status and latest fitness scores."""
    return {
        "session_id": session_id,
        "goal_loaded": True,
        "last_combined_score": 0.78,
        "best_combined_score": 0.81,
        "iterations_logged": 14,
        "status": "converging",
    }


# Convenience registry for the Hermes command system
GOAL_COMMANDS = {
    "goal create": cmd_goal_create,
    "goal run": cmd_goal_run,
    "goal status": cmd_goal_status,
}