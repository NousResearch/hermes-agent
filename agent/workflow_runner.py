"""Minimal Hermes workflow compiler.

Workflows are declarative graphs that compile to Kanban task specs.  The runner
keeps execution durable by letting the existing Kanban dispatcher own workers,
dependencies, retries, comments, and blocking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import yaml


@dataclass
class WorkflowNode:
    name: str
    assignee: str
    body: str = ""
    depends_on: list[str] = field(default_factory=list)
    approval_required: bool = False
    skills: list[str] = field(default_factory=list)


@dataclass
class WorkflowSpec:
    name: str
    nodes: list[WorkflowNode]

    @classmethod
    def from_yaml(cls, text: str) -> "WorkflowSpec":
        raw = yaml.safe_load(text) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("workflow YAML must be a mapping")
        name = str(raw.get("name") or "workflow")
        raw_nodes = raw.get("nodes") or {}
        nodes: list[WorkflowNode] = []
        if isinstance(raw_nodes, Mapping):
            iterable = raw_nodes.items()
        elif isinstance(raw_nodes, Sequence):
            iterable = ((n.get("name"), n) for n in raw_nodes if isinstance(n, Mapping))
        else:
            raise ValueError("workflow nodes must be a mapping or list")
        for node_name, node_cfg in iterable:
            if not isinstance(node_cfg, Mapping):
                raise ValueError(f"workflow node {node_name!r} must be a mapping")
            assignee = str(node_cfg.get("assignee") or "").strip()
            if not assignee:
                raise ValueError(f"workflow node {node_name!r} missing assignee")
            depends = node_cfg.get("depends_on") or node_cfg.get("parents") or []
            if isinstance(depends, str):
                depends = [depends]
            nodes.append(
                WorkflowNode(
                    name=str(node_name),
                    assignee=assignee,
                    body=str(node_cfg.get("body") or node_cfg.get("prompt") or ""),
                    depends_on=[str(d) for d in depends],
                    approval_required=bool(node_cfg.get("approval_required", False)),
                    skills=[str(s) for s in (node_cfg.get("skills") or [])],
                )
            )
        return cls(name=name, nodes=nodes)


@dataclass
class KanbanTaskSpec:
    title: str
    assignee: str
    body: str
    parents: list[str] = field(default_factory=list)
    parent_node_names: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)


def compile_workflow_to_kanban_tasks(
    spec: WorkflowSpec,
    *,
    parent_task_id: str | None = None,
) -> list[KanbanTaskSpec]:
    node_names = {n.name for n in spec.nodes}
    tasks: list[KanbanTaskSpec] = []
    for node in spec.nodes:
        unknown = [d for d in node.depends_on if d not in node_names]
        if unknown:
            raise ValueError(f"workflow node {node.name!r} depends on unknown nodes: {unknown}")
        parents = [parent_task_id] if parent_task_id and not node.depends_on else []
        body = node.body
        if node.approval_required:
            body = (body + "\n\nApproval required before side effects.").strip()
        tasks.append(
            KanbanTaskSpec(
                title=f"{spec.name}: {node.name}",
                assignee=node.assignee,
                body=body,
                parents=parents,
                parent_node_names=list(node.depends_on),
                skills=list(node.skills),
            )
        )
    return tasks
