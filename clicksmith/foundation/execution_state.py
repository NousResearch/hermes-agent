"""FND-03 — Canonical execution state model.

This module owns Clicksmith execution-state semantics only.

It does not authorize, execute, persist, recover, verify, deliver
artifacts, schedule work, or modify Hermes lifecycle semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from .execution_identity import ExecutionIdentity


SCHEMA_VERSION = "1.0"

PENDING = "PENDING"
AUTHORIZED = "AUTHORIZED"
RUNNING = "RUNNING"
PAUSED = "PAUSED"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

CANONICAL_STATES = frozenset(
    {
        PENDING,
        AUTHORIZED,
        RUNNING,
        PAUSED,
        SUCCEEDED,
        FAILED,
        CANCELLED,
    }
)

TERMINAL_STATES = frozenset(
    {
        SUCCEEDED,
        FAILED,
        CANCELLED,
    }
)

VALID_TRANSITIONS = {
    PENDING: frozenset({AUTHORIZED, CANCELLED}),
    AUTHORIZED: frozenset({RUNNING, CANCELLED}),
    RUNNING: frozenset({PAUSED, SUCCEEDED, FAILED, CANCELLED}),
    PAUSED: frozenset({RUNNING, CANCELLED}),
    SUCCEEDED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}

ALLOWED_KEYS = frozenset(
    {
        "identity",
        "schema_version",
        "state",
    }
)


class ExecutionStateError(ValueError):
    """Base error for FND-03 execution-state operations."""


class InvalidStateError(ExecutionStateError):
    """The supplied state is invalid or unknown."""


class InvalidTransitionError(ExecutionStateError):
    """The requested state transition is not permitted."""


class InvalidIdentityError(ExecutionStateError):
    """The supplied execution identity is invalid."""


class StateSchemaError(ExecutionStateError):
    """Serialized execution-state data violates the strict schema."""


class TerminalStateMutationError(ExecutionStateError):
    """A terminal execution state was asked to transition."""


@dataclass
class ExecutionState:
    """Canonical FND-03 execution state associated with FND-02 identity."""

    identity: ExecutionIdentity
    state: str = PENDING
    schema_version: str = SCHEMA_VERSION

    SCHEMA_VERSION: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_schema_version()
        self._validate_state(self.state)

    def _validate_identity(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise InvalidIdentityError(
                "identity must be an ExecutionIdentity."
            )

        try:
            self.identity.validate()
        except Exception as exc:
            raise InvalidIdentityError(
                "Execution Identity is invalid."
            ) from exc

    def _validate_schema_version(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise StateSchemaError(
                f"schema_version must be {SCHEMA_VERSION!r}."
            )

    @staticmethod
    def _validate_state(state: Any) -> None:
        if not isinstance(state, str):
            raise InvalidStateError("state must be a string.")

        if state not in CANONICAL_STATES:
            raise InvalidStateError(
                f"Unknown or invalid execution state: {state!r}."
            )

    def transition(self, target_state: str) -> "ExecutionState":
        """Atomically transition to a permitted state.

        Validation occurs before mutation. Failed transitions leave the
        current state unchanged.
        """
        self._validate_state(target_state)

        current_state = self.state

        if current_state in TERMINAL_STATES:
            raise TerminalStateMutationError(
                f"Terminal state {current_state!r} cannot transition."
            )

        if target_state not in VALID_TRANSITIONS[current_state]:
            raise InvalidTransitionError(
                f"Invalid transition: {current_state!r} -> "
                f"{target_state!r}."
            )

        self.state = target_state
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic FND-03 representation."""
        self._validate_identity()
        self._validate_schema_version()
        self._validate_state(self.state)

        return {
            "identity": self.identity.to_dict(),
            "schema_version": self.schema_version,
            "state": self.state,
        }

    def serialize(self) -> str:
        """Serialize deterministically as canonical UTF-8 JSON text."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionState":
        """Strictly deserialize a canonical execution-state mapping."""
        if not isinstance(value, Mapping):
            raise StateSchemaError(
                "Execution State must be a mapping."
            )

        unknown_keys = set(value) - ALLOWED_KEYS
        if unknown_keys:
            raise StateSchemaError(
                "Unknown Execution State fields: "
                + ", ".join(sorted(map(str, unknown_keys)))
            )

        required_keys = ALLOWED_KEYS
        missing_keys = required_keys - set(value)
        if missing_keys:
            raise StateSchemaError(
                "Missing Execution State fields: "
                + ", ".join(sorted(missing_keys))
            )

        schema_version = value["schema_version"]
        if not isinstance(schema_version, str):
            raise StateSchemaError(
                "schema_version must be a string."
            )

        if schema_version != SCHEMA_VERSION:
            raise StateSchemaError(
                f"Unsupported schema_version: {schema_version!r}."
            )

        try:
            identity = ExecutionIdentity.from_dict(value["identity"])
        except Exception as exc:
            raise InvalidIdentityError(
                "Invalid Execution Identity in Execution State."
            ) from exc

        state = value["state"]
        cls._validate_state(state)

        return cls(
            identity=identity,
            state=state,
            schema_version=schema_version,
        )

    @classmethod
    def deserialize(cls, payload: str) -> "ExecutionState":
        """Strictly deserialize canonical JSON text."""
        if not isinstance(payload, str):
            raise StateSchemaError(
                "Serialized Execution State must be a string."
            )

        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StateSchemaError(
                "Invalid Execution State JSON."
            ) from exc

        return cls.from_dict(value)


__all__ = [
    "AUTHORIZED",
    "CANCELLED",
    "CANONICAL_STATES",
    "ExecutionState",
    "ExecutionStateError",
    "FAILED",
    "InvalidIdentityError",
    "InvalidStateError",
    "InvalidTransitionError",
    "PENDING",
    "PAUSED",
    "RUNNING",
    "SCHEMA_VERSION",
    "StateSchemaError",
    "SUCCEEDED",
    "TERMINAL_STATES",
    "TerminalStateMutationError",
    "VALID_TRANSITIONS",
]
