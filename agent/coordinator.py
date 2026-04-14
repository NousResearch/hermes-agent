"""Coordinator planning layer for structured task execution."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from uuid import uuid4

from agent.task_manager import TaskManager


class Coordinator:
    """Create and execute simple serial plans."""

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    @staticmethod
    def _step_title(subtask: dict, index: int) -> str:
        for key in ("title", "description", "task_description", "name"):
            value = subtask.get(key)
            if value:
                return str(value).strip()
        return f"Subtask {index + 1}"

    def plan(self, task_description: str, subtasks: list[dict] | None = None) -> dict:
        source_task = str(task_description).strip()
        if not source_task:
            raise ValueError("task_description is required")

        normalized_subtasks = [item for item in (subtasks or []) if isinstance(item, dict)]
        steps: list[dict] = []

        if not normalized_subtasks:
            steps.append(
                {
                    "step_id": str(uuid4()),
                    "title": source_task,
                    "execution_mode": "local",
                    "depends_on": [],
                    "status": "pending",
                }
            )
        else:
            previous_step_id: str | None = None
            for index, subtask in enumerate(normalized_subtasks):
                step_id = str(uuid4())
                depends_on = [previous_step_id] if previous_step_id else []
                execution_mode = "delegated" if subtask.get("needs_external_reasoning") else "local"
                steps.append(
                    {
                        "step_id": step_id,
                        "title": self._step_title(subtask, index),
                        "execution_mode": execution_mode,
                        "depends_on": depends_on,
                        "status": "pending",
                    }
                )
                previous_step_id = step_id

        return {
            "plan_id": str(uuid4()),
            "source_task": source_task,
            "steps": steps,
            "created_at": self._now_iso(),
        }

    def execute_plan(self, plan: dict, task_manager: TaskManager) -> dict:
        updated_plan = copy.deepcopy(plan)

        for step in updated_plan.get("steps", []):
            execution_mode = step.get("execution_mode", "local")
            if execution_mode == "skip":
                step["status"] = "completed"
                step["task_id"] = None
                continue

            task = task_manager.create(
                title=step.get("title"),
                kind=execution_mode,
                metadata={
                    "plan_id": updated_plan.get("plan_id"),
                    "step_id": step.get("step_id"),
                    "source_task": updated_plan.get("source_task"),
                },
            )
            step["task_id"] = task["task_id"]
            step["status"] = "running"
            task_manager.update(task["task_id"], status="running")
            step["status"] = "completed"
            task_manager.update(task["task_id"], status="completed")

        return updated_plan

    def summarize(self, plan: dict) -> str:
        steps = plan.get("steps", [])
        lines = [
            f"Plan {plan.get('plan_id')} for: {plan.get('source_task')}",
            f"Created at: {plan.get('created_at')}",
            f"Steps: {len(steps)}",
        ]
        for index, step in enumerate(steps, start=1):
            lines.append(
                f"{index}. [{step.get('status', 'pending')}] "
                f"{step.get('title')} ({step.get('execution_mode', 'local')})"
            )
        return "\n".join(lines)
