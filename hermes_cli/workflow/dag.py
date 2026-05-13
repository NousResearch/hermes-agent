"""Workflow DAG normalization and validation.

The workflow DAG is a human-readable YAML/JSON artifact before materialization.
This module keeps validation deterministic so Kanban remains an execution
substrate rather than the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import WorkflowValidationIssue
from .policy import CANONICAL_ROLES, VALID_SCALES

SUPPORTED_DAG_SCHEMA_VERSION = 1
VALID_NODE_STATUSES = {"waiting", "ready", "running", "blocked", "review", "publish", "done", "failed", "cancelled"}
_VALID_NODE_ID_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")


@dataclass(frozen=True)
class DagValidationResult:
    """Structured result for DAG validation/normalization."""

    dag: dict[str, Any] | None
    issues: list[WorkflowValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[WorkflowValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[WorkflowValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dag": self.dag,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def validate_dag(dag: dict[str, Any], *, policy: dict[str, Any]) -> DagValidationResult:
    """Validate a workflow DAG and return structured issues."""

    return normalize_dag(dag, policy=policy)


def normalize_dag(dag: dict[str, Any], *, policy: dict[str, Any]) -> DagValidationResult:
    """Validate and normalize a workflow DAG.

    Normalization fills node defaults, denormalizes child edges, and emits a
    deterministic ``edges`` list. If validation fails, ``dag`` is ``None``.
    """

    issues: list[WorkflowValidationIssue] = []
    if not isinstance(dag, dict):
        return DagValidationResult(
            dag=None,
            issues=[_issue("dag_not_mapping", "Workflow DAG must be a mapping/object.", "")],
        )

    _validate_top_level(dag, issues)
    raw_nodes = dag.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        issues.append(_issue("nodes_not_list", "nodes must be a non-empty list.", "nodes"))
        return DagValidationResult(dag=None, issues=issues)

    seen: dict[str, int] = {}
    normalized_nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        node_path = f"nodes[{index}]"
        if not isinstance(raw_node, dict):
            issues.append(_issue("node_not_mapping", "Each node must be a mapping/object.", node_path))
            continue
        node = _normalize_node(raw_node, index, issues)
        node_id = node.get("id")
        if isinstance(node_id, str):
            if node_id in seen:
                issues.append(_issue("duplicate_node_id", f"Duplicate node id: {node_id!r}.", f"{node_path}.id"))
            else:
                seen[node_id] = index
                node_ids.add(node_id)
        normalized_nodes.append(node)

    policy_roles = policy.get("roles") if isinstance(policy.get("roles"), dict) else {}
    children_by_parent: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    edges: list[dict[str, str]] = []
    engineer_count = 0
    integrator_count = 0

    for index, node in enumerate(normalized_nodes):
        node_path = f"nodes[{index}]"
        node_id = node.get("id")
        role = node.get("role")
        if role not in CANONICAL_ROLES:
            issues.append(_issue("unknown_role", f"Unknown canonical workflow role: {role!r}.", f"{node_path}.role"))
        elif role == "engineer":
            engineer_count += 1
        elif role == "integrator":
            integrator_count += 1

        if role in CANONICAL_ROLES and policy_roles.get(role) is None:
            issues.append(_issue("unmapped_profile", f"No workflow profile is mapped for role {role!r}.", f"{node_path}.profile"))

        parents = node.get("parents", [])
        for parent_index, parent_id in enumerate(parents):
            if parent_id not in node_ids:
                issues.append(
                    _issue("unknown_parent", f"Parent node {parent_id!r} does not exist.", f"{node_path}.parents[{parent_index}]")
                )
                continue
            children_by_parent.setdefault(parent_id, []).append(node_id)
            edges.append({"source": parent_id, "target": node_id, "kind": "depends_on"})

        if role == "engineer" and not node.get("definition_of_done"):
            issues.append(
                _issue(
                    "missing_definition_of_done",
                    "Engineer nodes must include at least one definition_of_done item.",
                    f"{node_path}.definition_of_done",
                )
            )
        scope = node.get("scope")
        if not isinstance(scope, dict) or not isinstance(scope.get("summary"), str) or not scope["summary"].strip():
            issues.append(_issue("missing_scope_summary", "Every node must include scope.summary.", f"{node_path}.scope.summary"))

        gate = node.get("gate")
        if gate is not None:
            _validate_gate(gate, node_path, issues)

    _detect_cycle(normalized_nodes, issues)
    scale = dag.get("scale")
    dag_policy = policy.get("dag") if isinstance(policy.get("dag"), dict) else {}
    if scale in {"large", "xl"} and engineer_count > 1 and dag_policy.get("require_integrator_for_large", True):
        if integrator_count != 1:
            issues.append(
                _issue(
                    "missing_integrator",
                    "Large/XL DAGs with multiple engineer nodes must include exactly one integrator node.",
                    "nodes",
                )
            )

    if issues:
        return DagValidationResult(dag=None, issues=issues)

    for node in normalized_nodes:
        node["children"] = children_by_parent.get(node["id"], [])

    normalized = {
        "schema_version": SUPPORTED_DAG_SCHEMA_VERSION,
        "workflow_id": dag["workflow_id"],
        "name": dag.get("name", ""),
        "scale": dag["scale"],
        "nodes": normalized_nodes,
        "edges": edges,
    }
    return DagValidationResult(dag=normalized, issues=[])


def _validate_top_level(dag: dict[str, Any], issues: list[WorkflowValidationIssue]) -> None:
    version = dag.get("schema_version")
    if version != SUPPORTED_DAG_SCHEMA_VERSION:
        issues.append(_issue("unsupported_schema_version", f"Unsupported DAG schema version: {version!r}.", "schema_version"))
    workflow_id = dag.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        issues.append(_issue("invalid_workflow_id", "workflow_id must be a non-empty string.", "workflow_id"))
    scale = dag.get("scale")
    if scale not in VALID_SCALES:
        issues.append(_issue("invalid_scale", f"scale must be one of {sorted(VALID_SCALES)!r}.", "scale"))


def _normalize_node(raw_node: dict[str, Any], index: int, issues: list[WorkflowValidationIssue]) -> dict[str, Any]:
    node_path = f"nodes[{index}]"
    node_id = raw_node.get("id")
    if not isinstance(node_id, str) or not node_id.strip():
        issues.append(_issue("invalid_node_id", "Node id must be a non-empty string.", f"{node_path}.id"))
        node_id = ""
    elif not _is_slug_safe(node_id):
        issues.append(_issue("invalid_node_id", "Node id must be slug-safe lowercase text.", f"{node_path}.id"))

    parents = raw_node.get("parents", [])
    if parents is None:
        parents = []
    if not isinstance(parents, list):
        issues.append(_issue("parents_not_list", "parents must be a list of node ids.", f"{node_path}.parents"))
        parents = []
    normalized_parents: list[str] = []
    for parent_index, parent_id in enumerate(parents):
        if not isinstance(parent_id, str) or not parent_id.strip():
            issues.append(_issue("invalid_parent", "Parent references must be non-empty strings.", f"{node_path}.parents[{parent_index}]"))
            continue
        normalized_parents.append(parent_id)

    status = raw_node.get("status", "waiting")
    if status not in VALID_NODE_STATUSES:
        issues.append(_issue("invalid_node_status", f"Unknown node status: {status!r}.", f"{node_path}.status"))
        status = "waiting"

    definition_of_done = raw_node.get("definition_of_done", [])
    if definition_of_done is None:
        definition_of_done = []
    if not isinstance(definition_of_done, list):
        issues.append(_issue("definition_of_done_not_list", "definition_of_done must be a list.", f"{node_path}.definition_of_done"))
        definition_of_done = []

    gate = raw_node.get("gate")
    gate_level = 1
    if isinstance(gate, dict) and isinstance(gate.get("level"), int):
        gate_level = gate["level"]

    return {
        "id": node_id,
        "title": raw_node.get("title", ""),
        "role": raw_node.get("role"),
        "profile": raw_node.get("profile"),
        "status": status,
        "parents": normalized_parents,
        "children": [],
        "gate_level": gate_level,
        "scope": raw_node.get("scope", {}),
        "definition_of_done": definition_of_done,
        "workspace": raw_node.get("workspace"),
        "gate": gate,
    }


def _validate_gate(gate: Any, node_path: str, issues: list[WorkflowValidationIssue]) -> None:
    if not isinstance(gate, dict):
        issues.append(_issue("gate_not_mapping", "gate must be a mapping/object.", f"{node_path}.gate"))
        return
    level = gate.get("level")
    if not isinstance(level, int) or level < 1 or level > 3:
        issues.append(_issue("invalid_gate_level", "gate.level must be an integer from 1 to 3.", f"{node_path}.gate.level"))
    gate_type = gate.get("type")
    if not isinstance(gate_type, str) or not gate_type.strip():
        issues.append(_issue("invalid_gate_type", "gate.type must be a non-empty string.", f"{node_path}.gate.type"))


def _detect_cycle(nodes: list[dict[str, Any]], issues: list[WorkflowValidationIssue]) -> None:
    graph = {node["id"]: list(node.get("parents", [])) for node in nodes if node.get("id")}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for parent_id in graph.get(node_id, []):
            if parent_id in graph and visit(parent_id):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    for node_id in graph:
        if visit(node_id):
            issues.append(_issue("cycle_detected", "Workflow DAG must be acyclic.", "nodes"))
            return


def _is_slug_safe(value: str) -> bool:
    return bool(value) and value == value.lower() and not value.startswith("-") and not value.endswith("-") and set(value) <= _VALID_NODE_ID_CHARS


def _issue(code: str, message: str, path: str) -> WorkflowValidationIssue:
    return WorkflowValidationIssue(code=code, message=message, path=path)
