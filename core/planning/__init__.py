"""
Hermes Core Planning Package.
Handles intent analysis, task decomposition, and plan execution.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_name: str
    completed: bool = False


@dataclass
class AgentPlan:
    goal: str
    steps: List[PlanStep] = field(default_factory=list)


class TaskPlanner:
    def create_plan(self, goal: str) -> AgentPlan:
        return AgentPlan(
            goal=goal,
            steps=[PlanStep(step_id=1, description=f"Analyze goal: {goal}", tool_name="system")]
        )


__all__ = ["TaskPlanner", "AgentPlan", "PlanStep"]
