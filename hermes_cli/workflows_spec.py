"""Pydantic workflow definition models and cheap graph validation."""

from __future__ import annotations

import re
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


def validate_graph(spec: WorkflowSpec) -> None:
    for trigger in spec.triggers:
        expr = trigger.cron or trigger.schedule or getattr(trigger, "expr", None)
        if trigger.type == "schedule" and not expr:
            raise ValueError("schedule trigger requires cron or schedule")

    if not spec.nodes:
        raise ValueError("workflow must define at least one node")

    node_ids = set(spec.nodes)
    outgoing_sources: set[str] = set()

    for edge in spec.edges:
        source_base = edge.from_
        branch = None
        if "." in edge.from_:
            source_base, branch = edge.from_.split(".", 1)
        if source_base not in node_ids:
            raise ValueError(f"unknown edge source: {edge.from_}")
        if branch is not None:
            if not branch:
                raise ValueError(f"edge source {edge.from_} requires branch suffix")
            if spec.nodes[source_base].type != "switch":
                raise ValueError(f"dotted edge source {edge.from_} requires switch source")
        if edge.to not in node_ids:
            raise ValueError(f"unknown edge target: {edge.to}")
        outgoing_sources.add(source_base)

    for node_id, node in spec.nodes.items():
        if node.catch is not None and node.catch not in node_ids:
            raise ValueError(f"unknown catch target for node {node_id}: {node.catch}")
        if node.type == "switch":
            if node.default is not None:
                if node.default not in node_ids:
                    raise ValueError(f"unknown switch default target: {node.default}")
            elif node_id not in outgoing_sources:
                raise ValueError(f"switch node {node_id} must define outgoing edges or default")
        missing_prompt = node.prompt is None or (
            isinstance(node.prompt, str) and not node.prompt.strip()
        )
        if node.type == "agent_task" and (not (node.profile or "").strip() or missing_prompt):
            raise ValueError(f"agent_task node {node_id} requires profile and prompt")
