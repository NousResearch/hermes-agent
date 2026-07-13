"""Project runtime control-plane primitives for Hermes OS.

This module keeps Hermes as a coordinator. It records project definitions,
workspace snapshots, memory summaries, runtime service status, agent messages,
agent traces, and infrastructure references without taking ownership of domain
databases or production systems.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MEMORY_FILES = [
    "architecture.md",
    "decisions.md",
    "progress.md",
    "experiments.md",
    "lessons.md",
    "backlog.md",
    "agents.md",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("project definition must be a mapping")
        return data
    except ImportError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the simple project.yaml shape used by the runtime spec.

    This is not a full YAML parser. It is a fallback for environments without
    PyYAML and supports top-level scalars, nested mappings, and string lists.
    """

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            value = line[2:].strip()
            if not isinstance(parent, list):
                raise ValueError("list item without list parent")
            parent.append(_coerce_scalar(value))
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"invalid line: {raw}")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _coerce_scalar(value)
        else:
            next_container: Any = []
            parent[key] = next_container
            stack.append((indent, next_container))
    return root


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "~"}:
        return None
    return value.strip("\"'")


@dataclass(frozen=True)
class RuntimeServiceDefinition:
    name: str
    command: str
    cwd: str = ""
    dashboard_url: str = ""
    health_check: str = ""


@dataclass(frozen=True)
class ProjectDefinition:
    name: str
    type: str
    path: str
    repository: Dict[str, Any] = field(default_factory=dict)
    startup: List[str] = field(default_factory=list)
    dashboards: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    infrastructure: Dict[str, Any] = field(default_factory=dict)
    vector_databases: Dict[str, Any] = field(default_factory=dict)
    runtime_services: List[RuntimeServiceDefinition] = field(default_factory=list)
    enabled: bool = True


@dataclass(frozen=True)
class ProjectValidation:
    ok: bool
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkspaceSnapshot:
    project_id: str
    open_files: List[str] = field(default_factory=list)
    vscode_layout: Dict[str, Any] = field(default_factory=dict)
    browser_urls: List[str] = field(default_factory=list)
    active_terminals: List[str] = field(default_factory=list)
    running_services: List[str] = field(default_factory=list)
    current_branch: str = ""
    open_tasks: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class RuntimeServiceStatus:
    name: str
    command: str
    status: str
    pid: Optional[int] = None
    started_at: str = ""
    last_error: str = ""


@dataclass(frozen=True)
class AgentMessage:
    project_id: str
    source: str
    target: str
    type: str
    priority: str
    message: str
    correlation_id: str
    timestamp: str = field(default_factory=_now)


@dataclass(frozen=True)
class AgentTraceRecord:
    project_id: str
    sender: str
    receiver: str
    message_type: str
    content: str
    correlation_id: str
    timestamp: str = field(default_factory=_now)


class ProjectRuntimeManager:
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).expanduser()
        self.hermes_dir = self.workspace_root / ".hermes"
        self.projects_dir = self.hermes_dir / "projects"

    def load_project(self, project_id: str) -> ProjectDefinition:
        path = self.projects_dir / project_id / "project.yaml"
        if not path.exists():
            raise FileNotFoundError(f"project definition not found: {path}")
        return project_definition_from_dict(_load_yaml_or_json(path), default_name=project_id)

    def list_projects(self) -> List[ProjectDefinition]:
        if not self.projects_dir.exists():
            return []
        projects = []
        for entry in sorted(self.projects_dir.iterdir()):
            if not entry.is_dir():
                continue
            config = entry / "project.yaml"
            if config.exists():
                projects.append(project_definition_from_dict(_load_yaml_or_json(config), default_name=entry.name))
        return projects

    def validate_project(self, definition: ProjectDefinition) -> ProjectValidation:
        errors: List[str] = []
        if not definition.name:
            errors.append("name is required")
        if not definition.path:
            errors.append("path is required")
        if not isinstance(definition.infrastructure, dict):
            errors.append("infrastructure must be a mapping")
        return ProjectValidation(ok=not errors, errors=errors)

    def scaffold_memory(self, definition: ProjectDefinition) -> List[str]:
        memory_dir = Path(definition.path).expanduser() / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name in MEMORY_FILES:
            target = memory_dir / name
            if not target.exists():
                target.write_text(f"# {name.removesuffix('.md').title()}\n\n", encoding="utf-8")
            written.append(str(target))
        return written

    def load_memory(self, definition: ProjectDefinition) -> Dict[str, str]:
        memory_dir = Path(definition.path).expanduser() / "memory"
        result: Dict[str, str] = {}
        for name in MEMORY_FILES:
            target = memory_dir / name
            result[name] = target.read_text(encoding="utf-8") if target.exists() else ""
        return result

    def task_summary(self, definition: ProjectDefinition) -> Dict[str, int]:
        candidates = [Path(definition.path).expanduser() / "TASKS.md", Path(definition.path).expanduser() / ".hermes" / "tasks.json"]
        if candidates[1].exists():
            try:
                data = json.loads(candidates[1].read_text(encoding="utf-8"))
                tasks = data if isinstance(data, list) else data.get("tasks", [])
                open_count = sum(1 for item in tasks if item.get("status") not in {"done", "completed"})
                return {"open": open_count, "total": len(tasks)}
            except Exception:
                return {"open": 0, "total": 0}
        if candidates[0].exists():
            lines = [line for line in candidates[0].read_text(encoding="utf-8").splitlines() if line.strip().startswith("-")]
            return {"open": len(lines), "total": len(lines)}
        return {"open": 0, "total": 0}

    def project_status(self, definition: ProjectDefinition) -> Dict[str, Any]:
        memory = self.load_memory(definition)
        return {
            "project": asdict(definition),
            "validation": asdict(self.validate_project(definition)),
            "memory_files": {key: bool(value.strip()) for key, value in memory.items()},
            "tasks": self.task_summary(definition),
            "dashboards": definition.dashboards,
            "agents": [{"name": agent, "health": "unknown"} for agent in definition.agents],
            "infrastructure": definition.infrastructure,
            "vector_databases": definition.vector_databases,
            "runtime": {"services": [asdict(service) for service in definition.runtime_services]},
        }

    def switch_project(self, project_id: str, *, dry_run: bool = True) -> Dict[str, Any]:
        definition = self.load_project(project_id)
        self.scaffold_memory(definition)
        status = self.project_status(definition)
        return {
            "project_id": definition.name,
            "dry_run": dry_run,
            "steps": [
                "project_definition_loaded",
                "workspace_restore_planned",
                "dashboard_urls_loaded",
                "runtime_services_planned",
                "project_memory_loaded",
                "agents_connected_planned",
                "active_tasks_loaded",
                "status_displayed",
            ],
            "status": status,
        }

    def save_snapshot(self, project_id: str, snapshot: WorkspaceSnapshot) -> str:
        target = self.hermes_dir / "snapshots" / project_id / f"{snapshot.created_at.replace(':', '-')}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(snapshot), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return str(target)

    def latest_snapshot_path(self, project_id: str) -> Optional[Path]:
        root = self.hermes_dir / "snapshots" / project_id
        if not root.exists():
            return None
        snapshots = sorted(root.glob("*.json"))
        return snapshots[-1] if snapshots else None

    def restore_snapshot(self, project_id: str, *, dry_run: bool = True) -> Dict[str, Any]:
        path = self.latest_snapshot_path(project_id)
        if not path:
            raise FileNotFoundError(f"no snapshot found for {project_id}")
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        return {
            "project_id": project_id,
            "dry_run": dry_run,
            "snapshot": snapshot,
            "restore_contracts": {
                "vscode": snapshot.get("open_files", []),
                "browser_urls": snapshot.get("browser_urls", []),
                "services": snapshot.get("running_services", []),
            },
        }

    def start_project(self, project_id: str, *, dry_run: bool = True) -> Dict[str, Any]:
        definition = self.load_project(project_id)
        statuses = []
        for service in definition.runtime_services:
            if dry_run:
                statuses.append(RuntimeServiceStatus(name=service.name, command=service.command, status="planned"))
                continue
            try:
                proc = subprocess.Popen(service.command, shell=True, cwd=service.cwd or definition.path)
                statuses.append(RuntimeServiceStatus(
                    name=service.name,
                    command=service.command,
                    status="running",
                    pid=proc.pid,
                    started_at=_now(),
                ))
            except Exception as exc:
                statuses.append(RuntimeServiceStatus(name=service.name, command=service.command, status="failed", last_error=str(exc)))
        payload = {"project_id": definition.name, "dry_run": dry_run, "services": [asdict(item) for item in statuses]}
        self._write_runtime_status(definition.name, payload)
        return payload

    def _write_runtime_status(self, project_id: str, payload: Dict[str, Any]) -> None:
        target = self.hermes_dir / "runtime" / f"{project_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def record_agent_message(self, message: AgentMessage) -> AgentTraceRecord:
        msg_path = self.hermes_dir / "agent_messages" / f"{message.project_id}.jsonl"
        trace_path = self.hermes_dir / "agent_traces" / f"{message.project_id}.jsonl"
        msg_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        msg_path.write_text("", encoding="utf-8") if not msg_path.exists() else None
        with msg_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(message), sort_keys=True) + "\n")
        trace = AgentTraceRecord(
            project_id=message.project_id,
            sender=message.source,
            receiver=message.target,
            message_type=message.type,
            content=message.message,
            correlation_id=message.correlation_id,
            timestamp=message.timestamp,
        )
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trace), sort_keys=True) + "\n")
        return trace

    def agent_trace_view(self, project_id: str) -> Dict[str, Any]:
        path = self.hermes_dir / "agent_traces" / f"{project_id}.jsonl"
        records = []
        if path.exists():
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {"project_id": project_id, "timeline": records}

    def unified_dashboard(self) -> Dict[str, Any]:
        projects = [self.project_status(project) for project in self.list_projects()]
        return {
            "projects": projects,
            "agent_health": [agent for project in projects for agent in project["agents"]],
            "costs": {"status": "not_configured"},
            "experiments": {"status": "project_scoped"},
            "tasks": {project["project"]["name"]: project["tasks"] for project in projects},
            "alerts": [],
            "infrastructure": {project["project"]["name"]: project["infrastructure"] for project in projects},
            "queues": {"status": "not_configured"},
            "activity_feed": [],
        }


def project_definition_from_dict(data: Dict[str, Any], *, default_name: str = "") -> ProjectDefinition:
    runtime = data.get("runtime", {}) or {}
    services = runtime.get("services", []) if isinstance(runtime, dict) else []
    startup = data.get("startup") or []
    runtime_services: List[RuntimeServiceDefinition] = []
    for item in services:
        if isinstance(item, str):
            runtime_services.append(RuntimeServiceDefinition(name=item, command=item))
        elif isinstance(item, dict):
            runtime_services.append(RuntimeServiceDefinition(
                name=str(item.get("name") or item.get("command") or "service"),
                command=str(item.get("command") or item.get("name") or ""),
                cwd=str(item.get("cwd") or ""),
                dashboard_url=str(item.get("dashboard_url") or ""),
                health_check=str(item.get("health_check") or ""),
            ))
    for command in startup:
        runtime_services.append(RuntimeServiceDefinition(name=str(command).split()[0], command=str(command)))
    infrastructure = data.get("infrastructure") or {}
    vector_databases = {}
    if isinstance(infrastructure, dict):
        for key in ("vector_db", "vector_databases", "vectors"):
            if key in infrastructure:
                vector_databases[key] = infrastructure[key]
    status = data.get("status") or {}
    return ProjectDefinition(
        name=str(data.get("name") or default_name),
        type=str(data.get("type") or "unknown"),
        path=os.path.expanduser(str(data.get("path") or "")),
        repository=data.get("repository") or {},
        startup=[str(item) for item in startup],
        dashboards=[str(item) for item in data.get("dashboards") or []],
        documents=[str(item) for item in data.get("documents") or []],
        agents=[str(item) for item in data.get("agents") or []],
        infrastructure=infrastructure if isinstance(infrastructure, dict) else {},
        vector_databases=vector_databases,
        runtime_services=runtime_services,
        enabled=bool(status.get("enabled", True)) if isinstance(status, dict) else True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes workspace-runtime")
    parser.add_argument("--workspace-root", default=".")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("projects")
    switch = sub.add_parser("switch")
    switch.add_argument("project")
    switch.add_argument("--live", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("project")
    start = sub.add_parser("start")
    start.add_argument("project")
    start.add_argument("--live", action="store_true")
    snap = sub.add_parser("snapshot")
    snap.add_argument("action", choices=["save", "restore"])
    snap.add_argument("project")
    snap.add_argument("--live", action="store_true")
    dash = sub.add_parser("dashboard")
    dash.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    manager = ProjectRuntimeManager(args.workspace_root)

    if args.command == "projects":
        payload = {"projects": [asdict(project) for project in manager.list_projects()]}
    elif args.command == "switch":
        payload = manager.switch_project(args.project, dry_run=not args.live)
    elif args.command == "status":
        payload = manager.project_status(manager.load_project(args.project))
    elif args.command == "start":
        payload = manager.start_project(args.project, dry_run=not args.live)
    elif args.command == "snapshot":
        if args.action == "save":
            definition = manager.load_project(args.project)
            payload = {"snapshot_path": manager.save_snapshot(args.project, WorkspaceSnapshot(
                project_id=args.project,
                browser_urls=definition.dashboards,
                running_services=[service.name for service in definition.runtime_services],
            ))}
        else:
            payload = manager.restore_snapshot(args.project, dry_run=not args.live)
    elif args.command == "dashboard":
        payload = manager.unified_dashboard()
    else:
        parser.print_help()
        return 2
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
