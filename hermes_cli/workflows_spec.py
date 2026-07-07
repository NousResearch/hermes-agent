"""Pydantic workflow definition models and cheap graph validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NODE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

TriggerType = Literal["manual", "schedule", "webhook", "kanban_event"]
NodeType = Literal[
    "pass",
    "switch",
    "agent_task",
    "wait",
    "parallel",
    "join",
    "send_message",
    "fail",
    "subworkflow",
]


class TriggerSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: TriggerType
    id: str | None = None
    cron: str | None = None
    schedule: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)


class RetrySpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_attempts: int = Field(default=1, ge=1)
    delay_seconds: float = Field(default=0, ge=0)
    backoff_seconds: float | None = Field(default=None, ge=0)
    multiplier: float = Field(default=1, ge=0)


class WorkspaceSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: NodeType
    output: Any = None
    cases: list[Any] = Field(default_factory=list)
    default: str | None = None
    profile: str | None = None
    prompt: Any = None
    result_contract: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    workspace_kind: str | None = None
    workspace_path: str | None = None
    skills: list[str] = Field(default_factory=list)
    model_override: str | None = None
    max_retries: int | None = Field(default=None, ge=1)
    goal_mode: bool = False
    goal_max_turns: int | None = Field(default=None, ge=1)
    retry: RetrySpec | None = None
    catch: str | None = None
    workspace: WorkspaceSpec | None = None
    seconds: int = Field(default=0, ge=0)


class EdgeSpec(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    from_: str = Field(alias="from", min_length=1)
    to: str = Field(min_length=1)


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    name: str
    version: int = Field(ge=1)
    max_node_runs: int = Field(default=500, ge=1)
    enabled: bool = True
    triggers: list[TriggerSpec] = Field(default_factory=list)
    nodes: dict[str, NodeSpec] = Field(default_factory=dict)
    edges: list[EdgeSpec] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _validate_node_ids(cls, nodes: Any) -> Any:
        if isinstance(nodes, dict):
            for node_id in nodes:
                if not isinstance(node_id, str) or not _NODE_ID_RE.fullmatch(node_id):
                    raise ValueError(f"invalid node id: {node_id!r}")
        return nodes


def _blank_prompt(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    return False


def validate_graph(spec: WorkflowSpec) -> None:
    for trigger in spec.triggers:
        expr = trigger.cron or trigger.schedule or getattr(trigger, "expr", None)
        if trigger.type == "schedule" and not expr:
            raise ValueError("schedule trigger requires cron or schedule")

    if not spec.nodes:
        raise ValueError("workflow must define at least one node")

    node_ids = set(spec.nodes)
    outgoing_sources: set[str] = set()
    edge_sources: set[str] = set()
    incoming_targets: set[str] = set()

    for edge in spec.edges:
        source_base = edge.from_
        branch = None
        if "." in edge.from_:
            source_base, branch = edge.from_.split(".", 1)
        if source_base not in node_ids:
            raise ValueError(f"unknown edge source: {edge.from_}")
        if branch is None and spec.nodes[source_base].type == "parallel":
            raise ValueError(f"parallel edge source {edge.from_} requires branch suffix")
        if branch is not None:
            if not branch:
                raise ValueError(f"edge source {edge.from_} requires branch suffix")
            if spec.nodes[source_base].type not in {"switch", "parallel"}:
                raise ValueError(f"dotted edge source {edge.from_} requires switch or parallel source")
        if edge.to not in node_ids:
            raise ValueError(f"unknown edge target: {edge.to}")
        outgoing_sources.add(source_base)
        edge_sources.add(edge.from_)
        incoming_targets.add(edge.to)

    for node_id, node in spec.nodes.items():
        if node.catch is not None:
            if node.catch == node_id:
                raise ValueError(f"node {node_id} cannot catch itself")
            if node.catch not in node_ids:
                raise ValueError(f"unknown catch target for node {node_id}: {node.catch}")
            incoming_targets.add(node.catch)
        if node.type == "switch":
            if node.default is not None:
                if node.default not in node_ids:
                    raise ValueError(f"unknown switch default target: {node.default}")
                incoming_targets.add(node.default)
            elif node_id not in outgoing_sources:
                raise ValueError(f"switch node {node_id} must define outgoing edges or default")
            for case in node.cases:
                if not isinstance(case, Mapping):
                    raise ValueError(f"switch case for node {node_id} must be a mapping")
                name = case.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"switch case for node {node_id} requires name")
                if f"{node_id}.{name}" not in edge_sources:
                    raise ValueError(
                        f"switch case {node_id}.{name} requires matching outgoing edge"
                    )
        if node.type == "agent_task":
            if not str(node.profile or "").strip():
                raise ValueError(f"agent_task node {node_id} requires a non-blank profile")
            if _blank_prompt(node.prompt):
                raise ValueError(f"agent_task node {node_id} requires a non-empty prompt")

    if not any(node_id not in incoming_targets for node_id in node_ids):
        raise ValueError("workflow must define at least one entry node")
