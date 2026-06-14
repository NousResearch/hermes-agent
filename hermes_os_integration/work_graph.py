"""Work graph compiler and orchestration contracts for Hermes OS v3."""

import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .architecture_first import REQUIRED_PROJECT_DOCS
from .errors import STATE_CONFLICT, VALIDATION_ERROR, adapter_error
from .gates import evaluate_task_execution
from .registry import get_agent, select_agent_kind
from .scanners import scan_project


GRAPH_NODE_TYPES = {
    "project",
    "epic",
    "workflow",
    "task",
    "subtask",
    "approval",
    "artifact",
    "metric",
    "agent_assignment",
    "execution_result",
    "validation_result",
}


@dataclass(frozen=True)
class WorkGraphNode:
    id: str
    type: str
    title: str
    project_id: str
    status: str = "planned"
    source_ref: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Dependency:
    source_id: str
    target_id: str
    reason: str = ""


@dataclass(frozen=True)
class AgentAssignment:
    node_id: str
    agent_kind: str
    confidence: float
    reason: str
    fallback: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    node_id: str
    status: str
    runtime_provider: str = ""
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ValidationResult:
    node_id: str
    status: str
    rules: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkGraph:
    project_id: str
    nodes: List[WorkGraphNode] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    assignments: List[AgentAssignment] = field(default_factory=list)
    execution_results: List[ExecutionResult] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    findings: List[Dict[str, object]] = field(default_factory=list)


def read_architecture_artifacts(project_path: str):
    artifacts = {}
    missing = []
    for doc in REQUIRED_PROJECT_DOCS:
        path = _find_doc(project_path, doc)
        if path:
            with open(path, "r", encoding="utf-8") as handle:
                artifacts[doc] = {
                    "path": path,
                    "content": handle.read(),
                }
        else:
            missing.append(doc)
    return artifacts, missing


def compile_work_graph(project: str, projects_root: Optional[str] = None):
    scan = scan_project(project, projects_root)
    artifacts, missing = read_architecture_artifacts(scan.project_path)
    nodes = [WorkGraphNode(
        id="project:" + scan.project_id,
        type="project",
        title=scan.profile.canonical_name if scan.profile else scan.project_id,
        project_id=scan.project_id,
        source_ref=scan.project_path,
        metadata={"path": scan.project_path},
    )]

    dependencies = []
    for doc, artifact in artifacts.items():
        node = _node_from_doc(scan.project_id, doc, artifact["path"])
        nodes.append(node)
        dependencies.append(Dependency(source_id="project:" + scan.project_id, target_id=node.id, reason="architecture artifact"))

    for doc in missing:
        nodes.append(WorkGraphNode(
            id="task:create-" + doc.lower().replace(".md", ""),
            type="task",
            title="Create " + doc,
            project_id=scan.project_id,
            status="blocked" if doc in {"PROJECT.md", "DOMAIN.md", "WORKFLOWS.md"} else "planned",
            source_ref="architecture-review",
            metadata={"missing_document": doc, "task_type": "documentation"},
        ))

    findings = detect_missing_work(scan.project_id, missing)
    assignments = assign_agents(nodes)
    validation_results = generate_validation_rules(nodes)
    graph = WorkGraph(
        project_id=scan.project_id,
        nodes=nodes,
        dependencies=dependencies,
        assignments=assignments,
        validation_results=validation_results,
        findings=findings,
    )
    graph, error = validate_work_graph(graph)
    if error:
        raise ValueError(error.message)
    return graph


def detect_missing_work(project_id: str, missing_documents: Iterable[str]):
    findings = []
    for doc in missing_documents:
        findings.append({
            "project_id": project_id,
            "severity": "high" if doc in {"PROJECT.md", "DOMAIN.md", "WORKFLOWS.md"} else "medium",
            "missing": doc,
            "recommended_fix": "Create " + doc,
        })
    return findings


def assign_agents(nodes: Iterable[WorkGraphNode]):
    assignments = []
    for node in nodes:
        task_type = str(node.metadata.get("task_type", node.type))
        agent_kind = select_agent_kind(task_type)
        agent = get_agent(agent_kind)
        confidence = 0.85 if agent else 0.35
        assignments.append(AgentAssignment(
            node_id=node.id,
            agent_kind=agent_kind if agent else "research",
            confidence=confidence,
            reason="Matched task type " + task_type,
            fallback=agent is None,
        ))
    return assignments


def generate_validation_rules(nodes: Iterable[WorkGraphNode]):
    results = []
    for node in nodes:
        rules = ["has-stable-id", "has-project-reference"]
        failures = []
        if not node.id:
            failures.append("missing-id")
        if not node.project_id:
            failures.append("missing-project-id")
        results.append(ValidationResult(
            node_id=node.id,
            status="failed" if failures else "pending",
            rules=rules,
            failures=failures,
        ))
    return results


def resolve_dependencies(graph: WorkGraph):
    node_ids = {node.id for node in graph.nodes}
    incoming = {node.id: set() for node in graph.nodes}
    outgoing = {node.id: set() for node in graph.nodes}
    for dependency in graph.dependencies:
        if dependency.source_id not in node_ids or dependency.target_id not in node_ids:
            continue
        outgoing[dependency.source_id].add(dependency.target_id)
        incoming[dependency.target_id].add(dependency.source_id)
    ready = sorted([node_id for node_id, deps in incoming.items() if not deps])
    ordered = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_id)
        for target in sorted(outgoing[node_id]):
            incoming[target].discard(node_id)
            if not incoming[target]:
                ready.append(target)
    remaining = sorted([node_id for node_id, deps in incoming.items() if deps])
    return {
        "ordered": ordered,
        "cycles": remaining,
        "blocked": [node.id for node in graph.nodes if node.status == "blocked"] + remaining,
    }


def build_execution_queue(graph: WorkGraph, dry_run: bool = True):
    resolution = resolve_dependencies(graph)
    nodes_by_id = {node.id: node for node in graph.nodes}
    assignments = {assignment.node_id: assignment for assignment in graph.assignments}
    queue = []
    for node_id in resolution["ordered"]:
        node = nodes_by_id[node_id]
        if node.status == "blocked":
            continue
        gate = evaluate_graph_node_gate(node, dry_run=dry_run)
        if not gate.allowed:
            continue
        assignment = assignments.get(node_id)
        queue.append({
            "node_id": node_id,
            "title": node.title,
            "type": node.type,
            "agent_kind": assignment.agent_kind if assignment else "research",
            "approval_required": node.type == "approval",
            "dry_run": dry_run,
        })
    return queue


def evaluate_graph_node_gate(node: WorkGraphNode, dry_run: bool = True):
    task_type = str(node.metadata.get("task_type", node.type))
    completed = ["business_system", "control_plane", "domain_models", "workflows", "dashboards", "metrics", "approval_gates", "agents"]
    return evaluate_task_execution(task_type, completed, dry_run=dry_run)


def ingest_execution_result(graph: WorkGraph, result: ExecutionResult):
    nodes = []
    for node in graph.nodes:
        if node.id == result.node_id:
            nodes.append(WorkGraphNode(
                id=node.id,
                type=node.type,
                title=node.title,
                project_id=node.project_id,
                status=result.status,
                source_ref=node.source_ref,
                metadata=node.metadata,
            ))
        else:
            nodes.append(node)
    return WorkGraph(
        project_id=graph.project_id,
        nodes=nodes,
        dependencies=graph.dependencies,
        assignments=graph.assignments,
        execution_results=graph.execution_results + [result],
        validation_results=graph.validation_results,
        findings=graph.findings,
    )


def validate_work_graph(graph: WorkGraph):
    if not graph.project_id:
        return None, adapter_error(VALIDATION_ERROR, "Work graph missing project_id")
    seen = set()
    for node in graph.nodes:
        if node.type not in GRAPH_NODE_TYPES:
            return None, adapter_error(VALIDATION_ERROR, "Unknown node type: " + node.type)
        if node.id in seen:
            return None, adapter_error(STATE_CONFLICT, "Duplicate node id: " + node.id)
        seen.add(node.id)
    return graph, None


def serialize_work_graph(graph: WorkGraph):
    return json.dumps(_to_jsonable(graph), indent=2, sort_keys=True) + "\n"


def deserialize_work_graph(payload: str):
    data = json.loads(payload)
    nodes = [WorkGraphNode(**item) for item in data.get("nodes", [])]
    dependencies = [Dependency(**item) for item in data.get("dependencies", [])]
    assignments = [AgentAssignment(**item) for item in data.get("assignments", [])]
    execution_results = [ExecutionResult(**item) for item in data.get("execution_results", [])]
    validation_results = [ValidationResult(**item) for item in data.get("validation_results", [])]
    graph = WorkGraph(
        project_id=data.get("project_id", ""),
        nodes=nodes,
        dependencies=dependencies,
        assignments=assignments,
        execution_results=execution_results,
        validation_results=validation_results,
        findings=data.get("findings", []),
    )
    return validate_work_graph(graph)


def save_work_graph(project_path: str, graph: WorkGraph):
    path = os.path.join(project_path, "workgraph.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(serialize_work_graph(graph))
    return path


def _find_doc(project_path: str, doc: str):
    for root in [project_path, os.path.join(project_path, "docs")]:
        path = os.path.join(root, doc)
        if os.path.exists(path):
            return path
    return None


def _node_from_doc(project_id: str, doc: str, path: str):
    mapping = {
        "PROJECT.md": ("epic", "Project Architecture"),
        "DOMAIN.md": ("workflow", "Domain Model"),
        "WORKFLOWS.md": ("workflow", "Workflow Design"),
        "DASHBOARD.md": ("metric", "Dashboard Requirements"),
        "METRICS.md": ("metric", "Metrics"),
        "APPROVALS.md": ("approval", "Approvals"),
        "AGENTS.md": ("agent_assignment", "Agent Boundaries"),
    }
    node_type, title = mapping.get(doc, ("artifact", doc))
    return WorkGraphNode(
        id=doc.lower().replace(".md", ""),
        type=node_type,
        title=title,
        project_id=project_id,
        source_ref=path,
    )


def _to_jsonable(value):
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
