"""Data models for the Hermes eval lab.

The models are intentionally dependency-light dataclasses so the eval lab stays
local, deterministic, and separate from any training framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


_MISSING = object()


def _require(data: dict[str, Any], field_name: str, expected_type: type | tuple[type, ...]) -> Any:
    value = data.get(field_name, _MISSING)
    if value is _MISSING:
        raise ValueError(f"Missing required field: {field_name}")
    if not isinstance(value, expected_type):
        expected = getattr(expected_type, "__name__", str(expected_type))
        raise ValueError(f"Invalid field {field_name}: expected {expected}")
    return value


def _optional(data: dict[str, Any], field_name: str, expected_type: type | tuple[type, ...]) -> Any:
    value = data.get(field_name)
    if value is not None and not isinstance(value, expected_type):
        expected = getattr(expected_type, "__name__", str(expected_type))
        raise ValueError(f"Invalid field {field_name}: expected {expected} or None")
    return value


def _string_list(data: dict[str, Any], field_name: str) -> list[str]:
    value = _require(data, field_name, list)
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Invalid field {field_name}: expected list[str]")
    return list(value)


@dataclass(frozen=True)
class EvalScenario:
    id: str
    title: str
    prompt: str
    tags: list[str] = field(default_factory=list)
    expected_artifacts: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalScenario":
        if not isinstance(data, dict):
            raise ValueError("EvalScenario payload must be a mapping")
        required_fields = (
            "id",
            "title",
            "prompt",
            "tags",
            "expected_artifacts",
            "blocked_actions",
            "success_criteria",
        )
        missing = [field_name for field_name in required_fields if field_name not in data]
        if missing:
            raise ValueError("Missing required fields: " + ", ".join(missing))
        return cls(
            id=_require(data, "id", str),
            title=_require(data, "title", str),
            prompt=_require(data, "prompt", str),
            tags=_string_list(data, "tags"),
            expected_artifacts=_string_list(data, "expected_artifacts"),
            blocked_actions=_string_list(data, "blocked_actions"),
            success_criteria=_string_list(data, "success_criteria"),
        )


@dataclass(frozen=True)
class TrajectoryStep:
    role: str
    content: str | None = None
    tool_name: str | None = None
    tool_args_redacted: dict[str, Any] | None = None
    duration_ms: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryStep":
        if not isinstance(data, dict):
            raise ValueError("TrajectoryStep payload must be a mapping")
        return cls(
            role=_require(data, "role", str),
            content=_optional(data, "content", str),
            tool_name=_optional(data, "tool_name", str),
            tool_args_redacted=_optional(data, "tool_args_redacted", dict),
            duration_ms=_optional(data, "duration_ms", int),
            error=_optional(data, "error", str),
        )


@dataclass(frozen=True)
class TrajectoryAttempt:
    attempt_id: str
    scenario_id: str
    started_at: str
    finished_at: str | None
    status: str
    final_response: str | None
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryAttempt":
        if not isinstance(data, dict):
            raise ValueError("TrajectoryAttempt payload must be a mapping")
        steps = _require(data, "steps", list)
        return cls(
            attempt_id=_require(data, "attempt_id", str),
            scenario_id=_require(data, "scenario_id", str),
            started_at=_require(data, "started_at", str),
            finished_at=_optional(data, "finished_at", str),
            status=_require(data, "status", str),
            final_response=_optional(data, "final_response", str),
            steps=[TrajectoryStep.from_dict(step) for step in steps],
            metadata=_require(data, "metadata", dict),
        )


@dataclass(frozen=True)
class TrajectoryGroup:
    group_id: str
    scenario_id: str
    attempts: list[TrajectoryAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryGroup":
        if not isinstance(data, dict):
            raise ValueError("TrajectoryGroup payload must be a mapping")
        attempts = _require(data, "attempts", list)
        return cls(
            group_id=_require(data, "group_id", str),
            scenario_id=_require(data, "scenario_id", str),
            attempts=[TrajectoryAttempt.from_dict(attempt) for attempt in attempts],
        )


@dataclass(frozen=True)
class EvalScore:
    attempt_id: str
    total: float
    criteria: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalScore":
        if not isinstance(data, dict):
            raise ValueError("EvalScore payload must be a mapping")
        criteria = _require(data, "criteria", dict)
        notes = _string_list(data, "notes")
        if not all(isinstance(key, str) and isinstance(value, (int, float)) for key, value in criteria.items()):
            raise ValueError("Invalid field criteria: expected dict[str, float]")
        return cls(
            attempt_id=_require(data, "attempt_id", str),
            total=float(_require(data, "total", (int, float))),
            criteria={key: float(value) for key, value in criteria.items()},
            notes=notes,
        )
