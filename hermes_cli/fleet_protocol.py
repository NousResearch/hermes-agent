"""Wire-safe records shared by federated Hermes runners.

The protocol deliberately contains task requirements and runner capabilities,
not credentials or profile state.  A runner advertises what it can do; the
coordinator remains responsible for choosing an eligible runner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping


PROTOCOL_SCHEMA = "hermes.fleet.v1"
MAX_MESSAGE_BYTES = 1024 * 1024
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ProtocolError(ValueError):
    """Raised when a fleet message is malformed or unsafe to process."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _normalized_values(values: Any, field: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise ProtocolError(f"{field} must be a list of strings")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ProtocolError(f"{field} must contain non-empty strings")
        result.add(value)
    return tuple(sorted(result))


def _body_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    body = value.get("body")
    return body if isinstance(body, Mapping) else value


@dataclass(frozen=True)
class TaskRequirement:
    models: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    project: str | None = None
    workspace_kind: str | None = None

    _KIND: ClassVar[str] = "task_requirement"

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", _normalized_values(self.models, "models"))
        object.__setattr__(self, "tools", _normalized_values(self.tools, "tools"))
        if self.project is not None:
            _required_text(self.project, "project")
        if self.workspace_kind is not None:
            _required_text(self.workspace_kind, "workspace_kind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "tools": list(self.tools),
            "project": self.project,
            "workspace_kind": self.workspace_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskRequirement":
        if not isinstance(value, Mapping):
            raise ProtocolError("requirement must be an object")
        value = _body_mapping(value)
        return cls(
            models=value.get("models", ()),
            tools=value.get("tools", ()),
            project=value.get("project"),
            workspace_kind=value.get("workspace_kind"),
        )


@dataclass(frozen=True)
class RunnerCapability:
    node_id: str
    profile: str
    models: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()
    platform: str = ""

    _KIND: ClassVar[str] = "runner_capability"

    def __post_init__(self) -> None:
        _required_text(self.node_id, "node_id")
        _required_text(self.profile, "profile")
        _required_text(self.platform, "platform")
        object.__setattr__(self, "models", _normalized_values(self.models, "models"))
        object.__setattr__(self, "tools", _normalized_values(self.tools, "tools"))
        object.__setattr__(self, "projects", _normalized_values(self.projects, "projects"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "profile": self.profile,
            "models": list(self.models),
            "tools": list(self.tools),
            "projects": list(self.projects),
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunnerCapability":
        if not isinstance(value, Mapping):
            raise ProtocolError("runner capability must be an object")
        value = _body_mapping(value)
        return cls(
            node_id=_required_text(value.get("node_id"), "node_id"),
            profile=_required_text(value.get("profile"), "profile"),
            models=value.get("models", ()),
            tools=value.get("tools", ()),
            projects=value.get("projects", ()),
            platform=_required_text(value.get("platform"), "platform"),
        )


@dataclass(frozen=True)
class FleetTask:
    task_id: str
    title: str
    body: str
    requirement: TaskRequirement
    idempotency_key: str

    _KIND: ClassVar[str] = "fleet_task"

    def __post_init__(self) -> None:
        _required_text(self.task_id, "task_id")
        _required_text(self.title, "title")
        _required_text(self.body, "body")
        if not isinstance(self.requirement, TaskRequirement):
            raise ProtocolError("requirement must be a TaskRequirement")
        if not isinstance(self.idempotency_key, str) or not _IDEMPOTENCY_KEY.fullmatch(
            self.idempotency_key
        ):
            raise ValueError("idempotency_key must contain only safe identifier characters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "body": self.body,
            "requirement": self.requirement.to_dict(),
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FleetTask":
        if not isinstance(value, Mapping):
            raise ProtocolError("fleet task must be an object")
        value = _body_mapping(value)
        return cls(
            task_id=_required_text(value.get("task_id"), "task_id"),
            title=_required_text(value.get("title"), "title"),
            body=_required_text(value.get("body"), "body"),
            requirement=TaskRequirement.from_dict(value.get("requirement", {})),
            idempotency_key=value.get("idempotency_key", ""),
        )


_MESSAGE_TYPES = {
    TaskRequirement._KIND: TaskRequirement,
    RunnerCapability._KIND: RunnerCapability,
    FleetTask._KIND: FleetTask,
}


def _message_value(value: Any) -> tuple[str, Mapping[str, Any]]:
    for kind, message_type in _MESSAGE_TYPES.items():
        if isinstance(value, message_type):
            return kind, value.to_dict()
    raise ProtocolError(f"unsupported fleet message type: {type(value).__name__}")


def encode_message(value: Any) -> bytes:
    kind, body = _message_value(value)
    payload = json.dumps(
        {"schema": PROTOCOL_SCHEMA, "kind": kind, "body": body},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message too large")
    return payload


def decode_message(payload: bytes, *, expected_kind: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, (bytes, bytearray)):
        raise ProtocolError("message payload must be bytes")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message too large")
    try:
        message = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid fleet message JSON") from exc
    if not isinstance(message, dict):
        raise ProtocolError("fleet message must be an object")
    if message.get("schema") != PROTOCOL_SCHEMA:
        raise ProtocolError(f"unsupported fleet protocol schema: {message.get('schema')!r}")
    kind = message.get("kind")
    if kind not in _MESSAGE_TYPES:
        raise ProtocolError(f"unsupported fleet message kind: {kind!r}")
    if expected_kind is not None and kind != expected_kind:
        raise ProtocolError(f"expected {expected_kind}, received {kind}")
    body = message.get("body")
    if not isinstance(body, dict):
        raise ProtocolError("fleet message body must be an object")
    message["body"] = body
    return message
