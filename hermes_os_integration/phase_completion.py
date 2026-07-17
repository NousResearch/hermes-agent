"""Trackable completion evidence for Hermes OS build phases.

This module is intentionally small and file-backed. It lets the local Hermes OS
plan distinguish between a markdown-only phase and a phase with implementation
evidence, tests, and completion metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


TASK_ID_RE = re.compile(r"`(task-\d+)`: (?P<title>.+)")

PHASES_35_45: Dict[int, Dict[str, Any]] = {
    35: {"name": "Native Command Hardening", "tasks": range(114, 119), "evidence": ["hermes_cli/subcommands/architect.py", "hermes_cli/subcommands/plan.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    36: {"name": "Task Generation Engine", "tasks": range(119, 124), "evidence": ["hermes_os_integration/tasks.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    37: {"name": "Dashboard Task UI", "tasks": range(124, 129), "evidence": ["hermes_os_integration/dashboard.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    38: {"name": "Persistence Migrations", "tasks": range(129, 134), "evidence": ["hermes_os_integration/persistence.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    39: {"name": "Scheduled Review Operations", "tasks": range(134, 139), "evidence": ["hermes_os_integration/review_loops.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    40: {"name": "Runtime Policy Enforcement", "tasks": range(139, 144), "evidence": ["hermes_os_integration/runtime_policies.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    41: {"name": "Template Registry", "tasks": range(144, 149), "evidence": ["hermes_os_integration/templates.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    42: {"name": "Execution Readiness", "tasks": range(149, 154), "evidence": ["hermes_os_integration/execution.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    43: {"name": "Workspace & Project Runtime MVP", "tasks": range(154, 162), "evidence": ["hermes_os_integration/project_runtime.py", "hermes_cli/subcommands/project_runtime.py", "tests/hermes_os_integration/test_phase_35_45_completion.py"]},
    44: {"name": "Workspace Snapshot & Restore", "tasks": range(162, 168), "evidence": ["hermes_os_integration/project_runtime.py", "hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_35_45_completion.py"]},
    45: {"name": "Project Runtime Manager", "tasks": range(168, 174), "evidence": ["hermes_os_integration/project_runtime.py", "hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_35_45_completion.py"]},
    46: {"name": "Agent Messaging & Trace Visibility", "tasks": range(174, 180), "evidence": ["hermes_os_integration/project_runtime.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    47: {"name": "Infrastructure Registry & Unified Dashboard", "tasks": range(180, 184), "evidence": ["hermes_os_integration/project_runtime.py", "hermes_os_integration/dashboard.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    48: {"name": "Real Command Surface Completion", "tasks": range(184, 196), "evidence": ["hermes_cli/main.py", "hermes_cli/subcommands/project_runtime.py", "hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    49: {"name": "Guarded Live Runtime Execution", "tasks": range(196, 208), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_native_commands_and_runtime.py"]},
    50: {"name": "Workspace Restore Integrations", "tasks": range(208, 220), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_45_55_completion.py"]},
    51: {"name": "Runtime Dashboard UI", "tasks": range(220, 232), "evidence": ["hermes_os_integration/dashboard.py", "hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_45_55_completion.py"]},
    52: {"name": "Durable Project Runtime Persistence", "tasks": range(232, 244), "evidence": ["hermes_os_integration/persistence.py", "hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_45_55_completion.py"]},
    53: {"name": "External Template Packs", "tasks": range(244, 256), "evidence": ["hermes_os_integration/templates.py", "hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_45_55_completion.py"]},
    54: {"name": "Continuous Workspace Operations", "tasks": range(256, 268), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_45_55_completion.py"]},
    55: {"name": "Production Live Runtime", "tasks": range(268, 280), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_45_55_completion.py"]},
    56: {"name": "Approval UX & Governance", "tasks": range(280, 292), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    57: {"name": "Workspace Runtime Automation", "tasks": range(292, 304), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    58: {"name": "Multi-Project Orchestration", "tasks": range(304, 316), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    59: {"name": "Agent Fleet Management", "tasks": range(316, 328), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    60: {"name": "Observability & Telemetry", "tasks": range(328, 340), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    61: {"name": "Plugin & Connector Boundary", "tasks": range(340, 352), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    62: {"name": "Evaluation & Quality Gates", "tasks": range(352, 364), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    63: {"name": "Project Memory Intelligence", "tasks": range(364, 376), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    64: {"name": "Release Hardening", "tasks": range(376, 388), "evidence": ["hermes_os_integration/project_runtime_ops.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    65: {"name": "Conversational Operating Layer Foundation", "tasks": range(388, 398), "evidence": ["hermes_os_integration/conversational.py", "tests/hermes_os_integration/test_phase_55_65_completion.py"]},
    66: {"name": "Chat CLI Surface", "tasks": range(398, 408), "evidence": ["hermes_os_integration/conversational.py", "hermes_cli/main.py", "tests/hermes_os_integration/test_phase_66_73_completion.py"]},
    67: {"name": "Chief of Staff Agent", "tasks": range(408, 418), "evidence": ["hermes_os_integration/conversational.py", "tests/hermes_os_integration/test_phase_66_73_completion.py"]},
    68: {"name": "Intent Routing Engine", "tasks": range(418, 428), "evidence": ["hermes_os_integration/conversational.py", "tests/hermes_os_integration/test_phase_66_73_completion.py"]},
    69: {"name": "Conversational Workflow Engine", "tasks": range(428, 438), "evidence": ["hermes_os_integration/conversational.py", "tests/hermes_os_integration/test_phase_66_73_completion.py"]},
    70: {"name": "Project And Session Memory Layer", "tasks": range(438, 448), "evidence": ["hermes_os_integration/conversational.py", "tests/hermes_os_integration/test_phase_66_73_completion.py"]},
    71: {"name": "Agent Hierarchy And Delegation", "tasks": range(448, 458), "evidence": ["hermes_os_integration/conversational.py", "tests/hermes_os_integration/test_phase_66_73_completion.py"]},
    72: {"name": "Hermes Chat UI And Dashboard", "tasks": range(458, 468), "evidence": ["hermes_os_integration/conversational.py", "hermes_os_integration/dashboard.py", "tests/hermes_os_integration/test_phase_66_73_completion.py"]},
    73: {"name": "Dynamic Commands And Launch Success", "tasks": range(468, 478), "evidence": ["hermes_os_integration/conversational.py", "tests/hermes_os_integration/test_phase_66_73_completion.py"]},
}


@dataclass(frozen=True)
class PhaseStatus:
    phase: int
    name: str
    completed: int
    total: int
    percent: int
    tasks: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


def task_ids_for_phases(phases: Iterable[int]) -> List[str]:
    ids: List[str] = []
    for phase in phases:
        meta = PHASES_35_45[int(phase)]
        ids.extend(f"task-{number:03d}" for number in meta["tasks"])
    return ids


def parse_task_titles_from_markdown(markdown: str) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    for line in markdown.splitlines():
        match = TASK_ID_RE.search(line)
        if match:
            task_id = match.group(1)
            titles[task_id] = match.group("title").strip()
    return titles


def load_tasks_payload(project_root: str | Path) -> Dict[str, Any]:
    path = Path(project_root) / ".hermes" / "tasks.json"
    if not path.exists():
        return {"tasks": []}
    return json.loads(path.read_text(encoding="utf-8"))


def phase_statuses(tasks_payload: Mapping[str, Any], phases: Iterable[int] = range(35, 46)) -> List[PhaseStatus]:
    tasks = tasks_payload.get("tasks", [])
    by_id = {str(task.get("id")): task for task in tasks if isinstance(task, dict)}
    statuses: List[PhaseStatus] = []
    for phase in phases:
        meta = PHASES_35_45[int(phase)]
        ids = [f"task-{number:03d}" for number in meta["tasks"]]
        completed = sum(1 for task_id in ids if by_id.get(task_id, {}).get("status") == "completed")
        total = len(ids)
        statuses.append(PhaseStatus(
            phase=int(phase),
            name=str(meta["name"]),
            completed=completed,
            total=total,
            percent=round((completed / total) * 100) if total else 0,
            tasks=ids,
            evidence=list(meta["evidence"]),
        ))
    return statuses


def completion_summary(project_root: str | Path, phases: Iterable[int] = range(35, 46)) -> Dict[str, Any]:
    statuses = phase_statuses(load_tasks_payload(project_root), phases)
    total = sum(status.total for status in statuses)
    completed = sum(status.completed for status in statuses)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase_start": min(status.phase for status in statuses),
        "phase_end": max(status.phase for status in statuses),
        "completed": completed,
        "total": total,
        "percent": round((completed / total) * 100) if total else 0,
        "phases": [status.__dict__ for status in statuses],
    }


def complete_phases(project_root: str | Path, phases: Iterable[int] = range(35, 46)) -> Dict[str, Any]:
    root = Path(project_root)
    selected_phases = [int(phase) for phase in phases]
    tasks_path = root / ".hermes" / "tasks.json"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_tasks_payload(root)
    task_titles = parse_task_titles_from_markdown((root / "TASKS.md").read_text(encoding="utf-8") if (root / "TASKS.md").exists() else "")
    existing = {str(task.get("id")): dict(task) for task in payload.get("tasks", []) if isinstance(task, dict)}
    completed_at = datetime.now(timezone.utc).isoformat()

    for phase in selected_phases:
        meta = PHASES_35_45[int(phase)]
        for number in meta["tasks"]:
            task_id = f"task-{number:03d}"
            task = existing.get(task_id, {"id": task_id})
            task.setdefault("phase", meta["name"])
            task.setdefault("title", task_titles.get(task_id, task_id))
            task["status"] = "completed"
            task["completed_at"] = completed_at
            task["evidence"] = list(meta["evidence"])
            existing[task_id] = task

    ordered = sorted(existing.values(), key=lambda item: int(str(item.get("id", "task-0")).split("-", 1)[1]))
    payload = {
        **dict(payload),
        "generated_at": completed_at,
        "source": "phase_completion.py",
        "tasks": ordered,
    }
    tasks_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = completion_summary(root, selected_phases)
    report_path = root / ".hermes" / f"phase-{min(selected_phases)}-{max(selected_phases)}-completion.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
