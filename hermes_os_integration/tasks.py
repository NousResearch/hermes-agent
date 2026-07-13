"""Task generation for Hermes OS architecture and work graph outputs."""

import json
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence


TASK_ID_RE = re.compile(r"task-(\d+)")


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    title: str
    phase: str
    status: str = "planned"
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    risk: str = "medium"
    source_ref: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


def generate_tasks_from_review(report, start_at: int = 1) -> List[TaskDefinition]:
    allocator = TaskIdAllocator(start_at=start_at)
    tasks: List[TaskDefinition] = []
    for doc in getattr(report, "missing_documents", []):
        tasks.append(TaskDefinition(
            id=allocator.next(),
            title="Create " + doc,
            phase="Architecture Documents",
            risk="high" if doc in {"PROJECT.md", "DOMAIN.md", "WORKFLOWS.md"} else "medium",
            source_ref="architecture-review:" + report.project_id,
            acceptance_criteria=[
                doc + " exists in the project root or docs directory.",
                doc + " includes owner, purpose, and traceability back to the architecture review.",
            ],
            metadata={"missing_document": doc},
        ))
    for schema in getattr(report, "missing_schemas", []):
        tasks.append(TaskDefinition(
            id=allocator.next(),
            title="Define " + schema + " schema",
            phase="Schemas",
            risk="medium",
            source_ref="architecture-review:" + report.project_id,
            acceptance_criteria=[
                schema + " has a validated schema definition.",
                "Invalid agent artifacts are rejected before persistence.",
            ],
            metadata={"missing_schema": schema},
        ))
    for dashboard in getattr(report, "missing_dashboards", []):
        tasks.append(TaskDefinition(
            id=allocator.next(),
            title="Define dashboard signal: " + dashboard,
            phase="Dashboards",
            risk="medium",
            source_ref="architecture-review:" + report.project_id,
            acceptance_criteria=[
                "Dashboard signal is documented with source data and refresh cadence.",
                "Failure and opportunity states are visible to users.",
            ],
            metadata={"missing_dashboard": dashboard},
        ))
    for approval in getattr(report, "missing_approvals", []):
        tasks.append(TaskDefinition(
            id=allocator.next(),
            title="Define approval gate: " + approval,
            phase="Approvals",
            risk="high",
            source_ref="architecture-review:" + report.project_id,
            acceptance_criteria=[
                "Approval owner and decision criteria are documented.",
                "High-risk execution is blocked without approval.",
            ],
            metadata={"missing_approval": approval},
        ))
    return tasks


def generate_tasks_from_work_graph(graph, start_at: int = 1) -> List[TaskDefinition]:
    allocator = TaskIdAllocator(start_at=start_at)
    tasks: List[TaskDefinition] = []
    for node in getattr(graph, "nodes", []):
        if getattr(node, "status", "") != "blocked" and getattr(node, "type", "") != "task":
            continue
        tasks.append(TaskDefinition(
            id=allocator.next(),
            title=getattr(node, "title", getattr(node, "id", "Work graph task")),
            phase="Work Graph",
            status="blocked" if getattr(node, "status", "") == "blocked" else "planned",
            risk="high" if getattr(node, "status", "") == "blocked" else "medium",
            source_ref=getattr(node, "source_ref", "") or "work-graph:" + graph.project_id,
            acceptance_criteria=[
                "Work graph node is unblocked or completed.",
                "Validation rules for the node have passing evidence.",
            ],
            metadata={"node_id": getattr(node, "id", "")},
        ))
    for finding in getattr(graph, "findings", []):
        title = "Resolve finding: " + str(finding.get("missing") or finding.get("recommended_fix") or "work graph issue")
        tasks.append(TaskDefinition(
            id=allocator.next(),
            title=title,
            phase="Work Graph Findings",
            risk=str(finding.get("severity", "medium")),
            source_ref="work-graph:" + graph.project_id,
            acceptance_criteria=[str(finding.get("recommended_fix", "Finding is resolved."))],
            metadata=dict(finding),
        ))
    return dedupe_tasks(tasks)


def dedupe_tasks(tasks: Sequence[TaskDefinition]) -> List[TaskDefinition]:
    seen = set()
    result = []
    for task in tasks:
        key = (task.title, task.phase, task.source_ref)
        if key in seen:
            continue
        seen.add(key)
        result.append(task)
    return result


class TaskIdAllocator:
    def __init__(self, start_at: int = 1):
        self.current = int(start_at) - 1

    def next(self) -> str:
        self.current += 1
        return "task-%03d" % self.current


def next_task_number_from_text(text: str, default: int = 1) -> int:
    numbers = [int(match.group(1)) for match in TASK_ID_RE.finditer(text or "")]
    return max(numbers + [default - 1]) + 1


def next_task_number_from_files(paths: Iterable[str], default: int = 1) -> int:
    highest = default - 1
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        numbers = [int(match.group(1)) for match in TASK_ID_RE.finditer(text)]
        if numbers:
            highest = max(highest, max(numbers))
    return highest + 1


def write_task_artifacts(project_path: str, tasks: Sequence[TaskDefinition], overwrite: bool = True):
    os.makedirs(project_path, exist_ok=True)
    hermes_dir = os.path.join(project_path, ".hermes")
    os.makedirs(hermes_dir, exist_ok=True)
    md_path = os.path.join(project_path, "TASKS.md")
    json_path = os.path.join(hermes_dir, "tasks.json")
    if not overwrite and (os.path.exists(md_path) or os.path.exists(json_path)):
        return {"status": "skipped", "paths": [md_path, json_path]}
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_tasks_markdown(tasks))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "hermes-os-task-generation",
        "tasks": [_to_jsonable(task) for task in tasks],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"status": "written", "paths": [md_path, json_path]}


def render_tasks_markdown(tasks: Sequence[TaskDefinition]) -> str:
    lines = ["# Hermes OS Task Backlog", ""]
    phases: Dict[str, List[TaskDefinition]] = {}
    for task in tasks:
        phases.setdefault(task.phase, []).append(task)
    for phase, phase_tasks in phases.items():
        lines.extend(["## " + phase, ""])
        for task in phase_tasks:
            lines.append(f"- `{task.id}`: {task.title}")
            if task.acceptance_criteria:
                lines.append("  Acceptance: " + "; ".join(task.acceptance_criteria))
            if task.dependencies:
                lines.append("  Depends on: " + ", ".join(task.dependencies))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def task_summary(tasks: Sequence[TaskDefinition]):
    by_status: Dict[str, int] = {}
    by_phase: Dict[str, int] = {}
    for task in tasks:
        by_status[task.status] = by_status.get(task.status, 0) + 1
        by_phase[task.phase] = by_phase.get(task.phase, 0) + 1
    return {
        "task_count": len(tasks),
        "by_status": by_status,
        "by_phase": by_phase,
        "blocked_count": by_status.get("blocked", 0),
        "approval_required_count": len([task for task in tasks if task.risk == "high"]),
    }


def _to_jsonable(value: Any):
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
