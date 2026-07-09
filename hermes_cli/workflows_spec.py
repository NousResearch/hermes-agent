"""Pydantic workflow definition models and cheap graph validation."""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from typing import Any, Literal

from croniter import croniter
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

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
    expr: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


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


def _optional_clean_string(value: Any, name: str = "value") -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    return text or None


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

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
    provider_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices("provider", "provider_override"),
        serialization_alias="provider",
    )
    model_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model", "model_override"),
        serialization_alias="model",
    )
    max_retries: int | None = Field(default=None, ge=1)
    goal_mode: bool = False
    goal_max_turns: int | None = Field(default=None, ge=1)
    retry: RetrySpec | None = None
    catch: str | None = None
    workspace: WorkspaceSpec | None = None
    seconds: int = Field(default=0, ge=0)
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_conflicting_routing_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        data = dict(data)

        provider = _optional_clean_string(data.get("provider"), "provider")
        provider_override = _optional_clean_string(data.get("provider_override"), "provider_override")
        if provider and provider_override and provider != provider_override:
            raise ValueError("provider and provider_override must match when both are set")
        chosen_provider = provider or provider_override
        if chosen_provider:
            data["provider"] = chosen_provider
        else:
            data.pop("provider", None)
        data.pop("provider_override", None)

        model = _optional_clean_string(data.get("model"), "model")
        model_override = _optional_clean_string(data.get("model_override"), "model_override")
        if model and model_override and model != model_override:
            raise ValueError("model and model_override must match when both are set")
        chosen_model = model or model_override
        if chosen_model:
            data["model"] = chosen_model
        else:
            data.pop("model", None)
        data.pop("model_override", None)

        return data

    @model_validator(mode="after")
    def _normalize_routing_overrides(self) -> "NodeSpec":
        self.provider_override = _optional_clean_string(self.provider_override)
        self.model_override = _optional_clean_string(self.model_override)
        return self


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
    description: str | None = None

    @field_validator("nodes", mode="before")
    @classmethod
    def _validate_node_ids(cls, nodes: Any) -> Any:
        if isinstance(nodes, dict):
            for node_id in nodes:
                if not isinstance(node_id, str) or not _NODE_ID_RE.fullmatch(node_id):
                    raise ValueError(f"invalid node id: {node_id!r}")
        return nodes


def _model_field_names(model_cls: type[BaseModel]) -> set[str]:
    names: set[str] = set()
    for name, field in model_cls.model_fields.items():
        names.add(name)
        if field.alias:
            names.add(field.alias)
        alias = field.validation_alias
        if isinstance(alias, str):
            names.add(alias)
        elif isinstance(alias, AliasChoices):
            names.update(str(choice) for choice in alias.choices)
    return names


_WORKFLOW_FIELDS = _model_field_names(WorkflowSpec)
_TRIGGER_FIELDS = _model_field_names(TriggerSpec)
_NODE_FIELDS = _model_field_names(NodeSpec)
_EDGE_FIELDS = _model_field_names(EdgeSpec)
_RETRY_FIELDS = _model_field_names(RetrySpec)
_WORKSPACE_FIELDS = _model_field_names(WorkspaceSpec)


def _collect_unknown_keys(
    raw: Mapping,
    allowed: set[str],
    where: str,
    errors: list[str],
) -> None:
    for key in raw:
        key_text = str(key)
        if key_text in allowed:
            continue
        matches = difflib.get_close_matches(key_text, sorted(allowed), n=1)
        hint = f"; did you mean {matches[0]!r}?" if matches else ""
        errors.append(f"unknown field {key_text!r} {where}{hint}")


def unknown_spec_field_errors(raw: Any) -> list[str]:
    """Return errors for unrecognized keys anywhere in a raw workflow object.

    The Pydantic models keep ``extra="allow"`` so previously-stored specs
    always load; strictness is enforced here, at ingestion time only
    (validate/deploy/draft), so a typo like ``result_contarct`` fails loudly
    instead of silently no-opping the user's intent.
    """
    errors: list[str] = []
    if not isinstance(raw, Mapping):
        return errors
    _collect_unknown_keys(raw, _WORKFLOW_FIELDS, "on workflow", errors)

    triggers = raw.get("triggers")
    if isinstance(triggers, list):
        for index, trigger in enumerate(triggers):
            if isinstance(trigger, Mapping):
                _collect_unknown_keys(trigger, _TRIGGER_FIELDS, f"on trigger [{index}]", errors)

    nodes = raw.get("nodes")
    if isinstance(nodes, Mapping):
        for node_id, node in nodes.items():
            if not isinstance(node, Mapping):
                continue
            _collect_unknown_keys(node, _NODE_FIELDS, f"on node {node_id!r}", errors)
            retry = node.get("retry")
            if isinstance(retry, Mapping):
                _collect_unknown_keys(retry, _RETRY_FIELDS, f"on node {node_id!r} retry", errors)
            workspace = node.get("workspace")
            if isinstance(workspace, Mapping):
                _collect_unknown_keys(workspace, _WORKSPACE_FIELDS, f"on node {node_id!r} workspace", errors)

    edges = raw.get("edges")
    if isinstance(edges, list):
        for index, edge in enumerate(edges):
            if isinstance(edge, Mapping):
                _collect_unknown_keys(edge, _EDGE_FIELDS, f"on edge [{index}]", errors)
    return errors


def reject_unknown_spec_fields(raw: Any) -> None:
    """Raise ValueError when a raw workflow object contains unknown fields."""
    errors = unknown_spec_field_errors(raw)
    if errors:
        raise ValueError("; ".join(errors))


def load_spec_from_object(raw: Any) -> "WorkflowSpec":
    """Strictly parse and validate a raw workflow object (shared ingestion path)."""
    if not isinstance(raw, Mapping):
        raise ValueError("workflow spec must be an object")
    reject_unknown_spec_fields(raw)
    spec = WorkflowSpec.model_validate(raw)
    validate_graph(spec)
    return spec


def _blank_prompt(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    return False


def _cycle_path(spec: WorkflowSpec) -> list[str] | None:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in spec.nodes}
    for edge in spec.edges:
        source_base = edge.from_.split(".", 1)[0]
        adjacency.setdefault(source_base, []).append(edge.to)
    for node_id, node in spec.nodes.items():
        if node.catch:
            adjacency.setdefault(node_id, []).append(node.catch)
        if node.type == "switch" and node.default:
            adjacency.setdefault(node_id, []).append(node.default)

    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node_id: str) -> list[str] | None:
        if node_id in visiting:
            start = stack.index(node_id) if node_id in stack else 0
            return stack[start:] + [node_id]
        if node_id in visited:
            return None
        visiting.add(node_id)
        stack.append(node_id)
        for next_id in adjacency.get(node_id, []):
            found = visit(next_id)
            if found:
                return found
        stack.pop()
        visiting.remove(node_id)
        visited.add(node_id)
        return None

    for node_id in spec.nodes:
        found = visit(node_id)
        if found:
            return found
    return None


def validate_graph(spec: WorkflowSpec) -> None:
    for trigger in spec.triggers:
        expr = trigger.cron or trigger.schedule or trigger.expr
        if trigger.type == "schedule":
            if not expr:
                raise ValueError("schedule trigger requires cron or schedule")
            trigger_label = trigger.id or "schedule"
            try:
                croniter(expr, 0)
            except ValueError as exc:
                raise ValueError(
                    f"invalid cron expression on trigger {trigger_label!r}: {expr!r} ({exc})"
                ) from exc

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

    cycle = _cycle_path(spec)
    if cycle:
        raise ValueError("workflow graph contains cycle: " + " -> ".join(cycle))

    if not any(node_id not in incoming_targets for node_id in node_ids):
        raise ValueError("workflow must define at least one entry node")
