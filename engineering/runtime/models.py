"""Provider-neutral models for one Engineering-owned runtime invocation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from uuid import uuid4


class TurnStatus(str, Enum):
    """Outcome of an agent runtime turn, not an Engineering workflow state."""

    RETURNED = "RETURNED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeUsage:
    """Optional provider-neutral token counts reported by a runtime."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnRequest:
    """Minimum immutable input required to execute one agent turn."""

    workflow_run_id: str
    attempt: int
    message: str
    request_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    task_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("request_id", self.request_id)
        _require_non_empty_string("workflow_run_id", self.workflow_run_id)
        _require_positive_attempt(self.attempt)
        _require_non_empty_string("message", self.message)
        _validate_optional_identifier("session_id", self.session_id)
        _validate_optional_identifier("task_id", self.task_id)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnResult:
    """Immutable result of one runtime turn.

    ``RETURNED`` means that the runtime produced a normal non-empty response.
    It does not imply verification, review, or Engineering workflow completion.
    ``FAILED`` and ``INTERRUPTED`` are expected structured outcomes and may omit
    a response.
    """

    request_id: str
    workflow_run_id: str
    attempt: int
    status: TurnStatus
    response: str | None
    turn_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    task_id: str | None = None
    provider: str | None = None
    model: str | None = None
    usage: RuntimeUsage | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("request_id", self.request_id)
        _require_non_empty_string("turn_id", self.turn_id)
        _require_non_empty_string("workflow_run_id", self.workflow_run_id)
        _require_positive_attempt(self.attempt)
        if not isinstance(self.status, TurnStatus):
            raise TypeError("status must be a TurnStatus")
        if self.status is TurnStatus.RETURNED:
            _require_non_empty_string("response", self.response)
        elif self.response is not None:
            _require_non_empty_string("response", self.response)
        for name in ("session_id", "task_id", "provider", "model"):
            _validate_optional_identifier(name, getattr(self, name))
        if self.usage is not None and not isinstance(self.usage, RuntimeUsage):
            raise TypeError("usage must be a RuntimeUsage or None")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


def _require_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive_attempt(attempt: object) -> None:
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be an integer greater than zero")


def _validate_optional_identifier(name: str, value: object) -> None:
    if value is not None:
        _require_non_empty_string(name, value)


def _freeze_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        frozen[key] = _freeze_metadata_value(value)
    return MappingProxyType(frozen)


def _freeze_metadata_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _freeze_metadata(value)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(_freeze_metadata_value(item) for item in value)
    raise TypeError(
        "metadata values must contain only provider-neutral structured data"
    )
