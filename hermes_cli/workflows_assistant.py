"""Validate assistant-authored workflow drafts."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from hermes_cli.workflows_capabilities import (
    IMPLEMENTED_NODE_TYPES,
    IMPLEMENTED_TRIGGER_TYPES,
)
from hermes_cli.workflows_spec import WorkflowSpec, validate_graph

_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class AssistantValidationError(ValueError):
    """Raised when assistant workflow output cannot be accepted."""


class WorkflowDraftResult(BaseModel):
    spec: WorkflowSpec
    summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)
    valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def _extract_json_object(text: str) -> dict[str, Any]:
    match = _JSON_FENCE_RE.search(text)
    raw = match.group(1) if match else text
    data = json.loads(raw.strip())
    if not isinstance(data, dict):
        raise AssistantValidationError("assistant payload must be a JSON object")
    return data


def _ensure_supported_primitives(spec: WorkflowSpec) -> None:
    errors: list[str] = []
    for trigger in spec.triggers:
        if trigger.type not in IMPLEMENTED_TRIGGER_TYPES:
            errors.append(f"unsupported trigger type {trigger.type}")
    for node_id, node in spec.nodes.items():
        if node.type not in IMPLEMENTED_NODE_TYPES:
            errors.append(f"unsupported node type {node.type} on node {node_id}")
    if errors:
        raise AssistantValidationError("; ".join(errors))


def parse_assistant_payload(payload: dict[str, Any] | str) -> WorkflowDraftResult:
    try:
        data = _extract_json_object(payload) if isinstance(payload, str) else payload
        if not isinstance(data, dict):
            raise AssistantValidationError("assistant payload must be a JSON object")
        if "spec" not in data:
            raise AssistantValidationError("assistant payload requires spec")
        spec = WorkflowSpec.model_validate(data["spec"])
        validate_graph(spec)
        _ensure_supported_primitives(spec)
        return WorkflowDraftResult(
            spec=spec,
            summary=data.get("summary", ""),
            assumptions=data.get("assumptions", []),
            questions=data.get("questions", []),
            warnings=data.get("warnings", []),
            unsupported_requests=data.get("unsupported_requests", []),
        )
    except AssistantValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise AssistantValidationError(f"invalid JSON: {exc.msg}") from exc
    except (ValidationError, ValueError, TypeError) as exc:
        raise AssistantValidationError(str(exc)) from exc
