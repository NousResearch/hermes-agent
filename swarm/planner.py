from __future__ import annotations

import re
from collections import deque
from typing import Any

from swarm.types import SwarmTask
from swarm.roles import PLANNER, get_role

_ROLE_KEYWORDS: dict[str, list[str]] = {
    "critic": ["review", "critique", "check quality", "code review", "feedback"],
    "verifier": ["test", "verify", "validate", "assert", "confirm", "check"],
    "executor": ["implement", "build", "create", "write", "code", "develop", "fix", "refactor"],
    "researcher": ["research", "search", "find", "investigate", "explore", "gather", "read docs"],
    "planner": ["plan", "design", "architect", "outline", "decompose", "break down"],
    "merger": ["merge", "combine", "integrate", "consolidate", "unify", "resolve conflicts"],
}

_DEP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:then|after(?:\s+that)?|once\s+(?:that|step\s*\d+)|next|finally|lastly)\b", re.IGNORECASE),
    re.compile(r"\b(?:depends?\s+on|requires?|blocked\s+by|waiting\s+(?:for|on))\b", re.IGNORECASE),
    re.compile(r"\b(?:after\s+step\s*(\d+)|once\s+step\s*(\d+))\b", re.IGNORECASE),
]

_STEP_SPLIT = re.compile(
    r"(?:^|\n)\s*(?:"
    r"\d+[\.\)]\s+"        # "1. " or "1) "
    r"|[-*+]\s+"           # "- " or "* "
    r"|step\s+\d+[:\s]+"   # "step 1: "
    r")",
    re.IGNORECASE,
)


def _split_steps(text: str) -> list[str]:
    parts = _STEP_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip()]


def _infer_role(text: str) -> str:
    lower = text.lower()
    best_role = "executor"
    best_count = 0
    for role, keywords in _ROLE_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lower)
        if count > best_count:
            best_count = count
            best_role = role
    return best_role


def _has_dep_signal(text: str) -> bool:
    return any(p.search(text) for p in _DEP_PATTERNS)


def _extract_explicit_dep(text: str) -> list[int]:
    refs: list[int] = []
    for m in re.finditer(r"(?:after|once|depends?\s+on)\s+step\s*(\d+)", text, re.IGNORECASE):
        refs.append(int(m.group(1)) - 1)  # 0-indexed
    return refs


class TaskPlanner:
    def __init__(self, model: str | None = None):
        self.model = model or PLANNER.default_model

    def decompose(self, goal: str, context: str = "") -> list[SwarmTask]:
        full_text = f"{goal}\n{context}" if context else goal
        steps = _split_steps(full_text)
        if not steps:
            steps = [goal]

        # If the goal is a single sentence (no explicit steps), auto-decompose
        # into parallel subtasks so the swarm actually uses multiple workers.
        if len(steps) == 1:
            steps = self._auto_decompose(steps[0])

        tasks: list[SwarmTask] = []
        for i, step_text in enumerate(steps):
            role = _infer_role(step_text)
            role_obj = get_role(role)

            deps: list[str] = []
            explicit = _extract_explicit_dep(step_text)
            if explicit:
                for dep_idx in explicit:
                    if 0 <= dep_idx < len(tasks):
                        deps.append(tasks[dep_idx].id)
            elif i > 0 and _has_dep_signal(step_text):
                deps.append(tasks[i - 1].id)

            task = SwarmTask(
                name=f"step_{i}",
                prompt=step_text,
                role=role,
                model=role_obj.default_model,
                deps=deps,
            )
            tasks.append(task)

        return tasks

    def _auto_decompose(self, goal: str) -> list[str]:
        """Break a single goal sentence into parallel subtasks."""
        # Extract requested agent count if present (e.g. "5 agents to build X")
        count_match = re.search(r'(\d+)\s*agents?', goal, re.IGNORECASE)
        n = int(count_match.group(1)) if count_match else 3
        n = max(2, min(n, 10))

        # Strip the "N agents to" prefix to get the actual goal
        clean_goal = re.sub(r'(?:swarm\s+)?\d+\s*agents?\s*(?:to\s+)?', '', goal, flags=re.IGNORECASE).strip()
        clean_goal = re.sub(r'^swarm\s+', '', clean_goal, flags=re.IGNORECASE).strip()
        if not clean_goal:
            clean_goal = goal

        # Each subtask is a concrete, actionable piece — no vague "design" tasks
        subtasks = []
        for i in range(n):
            subtasks.append(
                f"You are worker {i+1} of {n} building: {clean_goal}\n"
                f"Write your part as working code. Be concrete — create files, not plans.\n"
                f"Focus on part {i+1}: split the work roughly into {n} equal pieces and do piece {i+1}."
            )

        return subtasks

    def plan_from_steps(
        self,
        steps: list[str],
        deps: dict[int, list[int]] | None = None,
    ) -> list[SwarmTask]:
        tasks: list[SwarmTask] = []
        for i, step_text in enumerate(steps):
            role = _infer_role(step_text)
            role_obj = get_role(role)

            task_deps: list[str] = []
            if deps and i in deps:
                for dep_idx in deps[i]:
                    if 0 <= dep_idx < len(tasks):
                        task_deps.append(tasks[dep_idx].id)

            task = SwarmTask(
                name=f"step_{i}",
                prompt=step_text,
                role=role,
                model=role_obj.default_model,
                deps=task_deps,
            )
            tasks.append(task)

        return tasks

    def estimate_parallelism(self, tasks: list[SwarmTask]) -> int:
        if not tasks:
            return 0

        id_to_idx = {t.id: i for i, t in enumerate(tasks)}
        in_degree = [0] * len(tasks)
        children: list[list[int]] = [[] for _ in tasks]

        for i, t in enumerate(tasks):
            for dep_id in t.deps:
                if dep_id in id_to_idx:
                    parent = id_to_idx[dep_id]
                    children[parent].append(i)
                    in_degree[i] += 1

        queue = deque(i for i, d in enumerate(in_degree) if d == 0)
        max_width = len(queue) if queue else 1

        while queue:
            next_level: list[int] = []
            for _ in range(len(queue)):
                node = queue.popleft()
                for child in children[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_level.append(child)
            if next_level:
                max_width = max(max_width, len(next_level))
                queue.extend(next_level)

        return max_width

    def format_plan(self, tasks: list[SwarmTask]) -> str:
        if not tasks:
            return "(empty plan)"

        id_to_idx = {t.id: i for i, t in enumerate(tasks)}
        lines: list[str] = []
        lines.append(f"Plan ({len(tasks)} tasks, max parallelism={self.estimate_parallelism(tasks)}):")
        lines.append("")

        for i, t in enumerate(tasks):
            dep_labels = []
            for dep_id in t.deps:
                if dep_id in id_to_idx:
                    dep_labels.append(f"step_{id_to_idx[dep_id]}")
                else:
                    dep_labels.append(dep_id)

            dep_str = f" (after {', '.join(dep_labels)})" if dep_labels else " (parallel)"
            prompt_short = t.prompt[:80] + ("..." if len(t.prompt) > 80 else "")
            lines.append(f"  [{i}] {t.name} [{t.role}]{dep_str}")
            lines.append(f"       {prompt_short}")

        return "\n".join(lines)

    def validate_plan(self, tasks: list[SwarmTask]) -> list[str]:
        errors: list[str] = []
        if not tasks:
            return errors

        id_set = {t.id for t in tasks}
        id_to_idx = {t.id: i for i, t in enumerate(tasks)}

        # Check for missing deps
        for i, t in enumerate(tasks):
            for dep_id in t.deps:
                if dep_id not in id_set:
                    errors.append(f"step_{i} depends on unknown task {dep_id!r}")

        # Check for self-deps
        for i, t in enumerate(tasks):
            if t.id in t.deps:
                errors.append(f"step_{i} depends on itself")

        # Cycle detection via Kahn's algorithm
        in_degree = {t.id: 0 for t in tasks}
        children: dict[str, list[str]] = {t.id: [] for t in tasks}

        for t in tasks:
            for dep_id in t.deps:
                if dep_id in id_set:
                    children[dep_id].append(t.id)
                    in_degree[t.id] += 1

        queue = deque(tid for tid, d in in_degree.items() if d == 0)
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if visited < len(tasks):
            cycle_members = [
                f"step_{id_to_idx[tid]}"
                for tid, d in in_degree.items()
                if d > 0
            ]
            errors.append(f"Cycle detected involving: {', '.join(cycle_members)}")

        # Check for duplicate task ids
        if len(id_set) < len(tasks):
            errors.append("Duplicate task IDs found")

        return errors
