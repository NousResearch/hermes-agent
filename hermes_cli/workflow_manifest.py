"""Declarative workflow manifests for repeatable sub-agent orchestration.

Issue context: competing agents (Claude Code dynamic workflows, OpenCode agent
teams) let a user save a multi-step fan-out as a script with a token budget,
a parallelism cap and a resumable checkpoint. Hermes already has the engine
(``tools/delegate_tool.py`` with ``DEFAULT_MAX_ITERATIONS`` and
``max_concurrent_children``), but no way to save and replay a plan.

This module is the missing declarative layer. It is deliberately pure: it
parses and validates a workflow manifest, tracks which steps completed, and
answers scheduling questions. It does not start agents. That keeps it
reviewable and testable without touching the delegation engine.

A workflow is a JSON manifest::

    {
      "name": "review-pr",
      "steps": [
        {"id": "scan", "goal": "..."},
        {"id": "test", "goal": "...", "depends_on": ["scan"]}
      ],
      "max_iterations": 250,
      "max_children": 4
    }

The state file records completed step ids so ``resume`` skips them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class WorkflowError(ValueError):
    """Raised for an invalid manifest or an impossible resume."""


@dataclass(frozen=True)
class Step:
    id: str
    goal: str
    depends_on: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        step_id = str(data.get("id", "")).strip()
        goal = str(data.get("goal", "")).strip()
        if not step_id or not goal:
            raise WorkflowError("each step needs a non-empty id and goal")
        raw_deps = data.get("depends_on", []) or []
        if isinstance(raw_deps, str):
            raw_deps = [raw_deps]
        deps = tuple(str(d).strip() for d in raw_deps if str(d).strip())
        return cls(id=step_id, goal=goal, depends_on=deps)


@dataclass(frozen=True)
class Workflow:
    name: str
    steps: tuple[Step, ...]
    max_iterations: int = 250
    max_children: int = 4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workflow:
        name = str(data.get("name", "")).strip()
        if not name:
            raise WorkflowError("workflow needs a name")
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            raise WorkflowError("workflow needs a non-empty steps list")
        steps = tuple(Step.from_dict(s) for s in raw_steps)
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            raise WorkflowError("step ids must be unique")
        known = set(ids)
        for step in steps:
            unknown = set(step.depends_on) - known
            if unknown:
                raise WorkflowError(
                    f"step {step.id} depends on unknown step(s): {', '.join(sorted(unknown))}")
        iterations = int(data.get("max_iterations", 250))
        children = int(data.get("max_children", 4))
        if iterations < 1 or children < 1:
            raise WorkflowError("max_iterations and max_children must be positive")
        return cls(name=name, steps=steps, max_iterations=iterations, max_children=children)

    @classmethod
    def load(cls, path: Path) -> Workflow:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class WorkflowState:
    """Checkpoint of completed steps, saved next to the manifest."""

    completed: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> WorkflowState:
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(completed=set(data.get("completed", [])))

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"completed": sorted(self.completed)}, indent=2), encoding="utf-8")

    def mark(self, step_id: str) -> None:
        self.completed.add(step_id)


def next_steps(workflow: Workflow, state: WorkflowState) -> list[Step]:
    """Steps ready to run: dependencies satisfied and not yet completed."""
    ready: list[Step] = []
    for step in workflow.steps:
        if step.id in state.completed:
            continue
        if all(dep in state.completed for dep in step.depends_on):
            ready.append(step)
    return ready


def is_complete(workflow: Workflow, state: WorkflowState) -> bool:
    return all(step.id in state.completed for step in workflow.steps)


def resume_state_path(manifest_path: Path) -> Path:
    """Checkpoint sits beside the manifest, named after it."""
    p = Path(manifest_path)
    return p.with_name(p.stem + ".state.json")
