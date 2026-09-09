"""Canonical Clicksmith Execution Identity — FND-02 Phase 1.

This module defines execution identity independently from execution state,
authorization, evidence, recovery, persistence, retry engines, and
infrastructure.

TASK_ID remains owned by FND-01 TaskContract.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Mapping, Optional, Protocol


SCHEMA_VERSION = "1.0"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

_REJECTED_FIELDS = frozenset(
    {
        "password",
        "api_key",
        "apikey",
        "token",
        "secret",
        "private_key",
        "private_key_data",
        "recovery_code",
        "authorization",
        "authorization_context",
        "approval",
        "approval_context",
        "approval_decision",
        "execution_state",
        "state",
        "status",
    }
)


class ExecutionIdentityError(ValueError):
    """Raised when an Execution Identity is invalid."""


class IdentityCollisionError(ExecutionIdentityError):
    """Raised when an identity conflicts with an existing identity."""


class IdentityResolver(Protocol):
    """Minimal boundary for checking existing execution identities."""

    def resolve_run_task(self, run_id: str) -> Optional[str]:
        """Return the TASK_ID associated with RUN_ID, if known."""

    def resolve_attempt_run(self, attempt_id: str) -> Optional[str]:
        """Return the RUN_ID associated with ATTEMPT_ID, if known."""


@dataclasses.dataclass(frozen=True)
class ExecutionIdentity:
    """Canonical execution identity.

    TASK_ID identifies the logical task and remains owned by FND-01.
    RUN_ID identifies one execution run of that task.
    ATTEMPT_ID identifies one concrete attempt within that run.
    SESSION_ID identifies an execution/session context.
    REVISION identifies the task-contract revision associated with
    this execution identity.

    This object does not authorize, execute, persist, recover, or
    transition execution state.
    """

    task_id: str
    run_id: str
    attempt_id: str
    session_id: Optional[str]
    revision: int
    schema_version: str = SCHEMA_VERSION

    def validate(
        self,
        *,
        identity_resolver: Optional[IdentityResolver] = None,
    ) -> None:
        _validate_task_id(self.task_id)
        _validate_identity_id("run_id", self.run_id)
        _validate_identity_id("attempt_id", self.attempt_id)
        if self.session_id is not None:
            _validate_identity_id("session_id", self.session_id)

        if type(self.revision) is not int or self.revision < 0:
            raise ExecutionIdentityError(
                "revision must be a non-negative integer."
            )

        if self.schema_version != SCHEMA_VERSION:
            raise ExecutionIdentityError(
                f"schema_version must be {SCHEMA_VERSION!r}."
            )

        if identity_resolver is not None:
            _validate_collisions(self, identity_resolver)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic identity representation."""
        self.validate()
        return {
            "attempt_id": self.attempt_id,
            "revision": self.revision,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "task_id": self.task_id,
        }

    def serialize(self) -> str:
        """Serialize deterministically as canonical JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        identity_resolver: Optional[IdentityResolver] = None,
    ) -> "ExecutionIdentity":
        if not isinstance(value, Mapping):
            raise ExecutionIdentityError(
                "Execution Identity must be a mapping."
            )

        _reject_unknown_keys(value)

        try:
            identity = cls(
                task_id=value["task_id"],
                run_id=value["run_id"],
                attempt_id=value["attempt_id"],
                session_id=value.get("session_id"),
                revision=value["revision"],
                schema_version=value.get(
                    "schema_version",
                    SCHEMA_VERSION,
                ),
            )
        except ExecutionIdentityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionIdentityError(
                "Malformed Execution Identity."
            ) from exc

        identity.validate(identity_resolver=identity_resolver)
        return identity

    @classmethod
    def deserialize(
        cls,
        payload: str,
        *,
        identity_resolver: Optional[IdentityResolver] = None,
    ) -> "ExecutionIdentity":
        if not isinstance(payload, str):
            raise ExecutionIdentityError(
                "Serialized Execution Identity must be a string."
            )

        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ExecutionIdentityError(
                "Invalid Execution Identity JSON."
            ) from exc

        return cls.from_dict(
            value,
            identity_resolver=identity_resolver,
        )


def validate_task_association(
    identity: ExecutionIdentity,
    task_id: str,
) -> None:
    """Validate that identity TASK_ID matches the supplied FND-01 TASK_ID."""
    if not isinstance(identity, ExecutionIdentity):
        raise ExecutionIdentityError("identity must be an ExecutionIdentity.")

    _validate_task_id(task_id)

    if identity.task_id != task_id:
        raise ExecutionIdentityError(
            "Execution Identity task_id does not match the FND-01 TASK_ID."
        )


def validate_retry_identity(
    previous: ExecutionIdentity,
    retry: ExecutionIdentity,
) -> None:
    """Validate retry identity semantics.

    A retry continuing the same run must preserve TASK_ID, RUN_ID,
    and REVISION, while creating a distinct ATTEMPT_ID.

    A different RUN_ID represents a new run / re-execution, not a retry
    of the previous run.
    """
    if not isinstance(previous, ExecutionIdentity):
        raise ExecutionIdentityError(
            "previous must be an ExecutionIdentity."
        )

    if not isinstance(retry, ExecutionIdentity):
        raise ExecutionIdentityError(
            "retry must be an ExecutionIdentity."
        )

    previous.validate()
    retry.validate()

    if retry.task_id != previous.task_id:
        raise ExecutionIdentityError(
            "Retry must preserve TASK_ID."
        )

    if retry.run_id != previous.run_id:
        raise ExecutionIdentityError(
            "Retry continuing the same run must preserve RUN_ID."
        )

    if retry.revision != previous.revision:
        raise ExecutionIdentityError(
            "Retry must preserve task-contract revision."
        )

    if retry.attempt_id == previous.attempt_id:
        raise ExecutionIdentityError(
            "Retry must use a distinct ATTEMPT_ID."
        )


def _validate_task_id(value: Any) -> None:
    """Validate TASK_ID at the FND-01 ownership boundary.

    FND-02 must not narrow the canonical TASK_ID domain defined by FND-01.
    """
    if not isinstance(value, str) or not value.strip():
        raise ExecutionIdentityError(
            "task_id must be a non-empty string."
        )


def _validate_identity_id(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionIdentityError(
            f"{name} must be a non-empty string."
        )

    if not _ID_PATTERN.fullmatch(value):
        raise ExecutionIdentityError(
            f"{name} contains invalid identity characters."
        )


def _validate_collisions(
    identity: ExecutionIdentity,
    resolver: IdentityResolver,
) -> None:
    existing_task = resolver.resolve_run_task(identity.run_id)

    if existing_task is not None and existing_task != identity.task_id:
        raise IdentityCollisionError(
            "RUN_ID is already associated with another TASK_ID."
        )

    existing_run = resolver.resolve_attempt_run(identity.attempt_id)

    if existing_run is not None and existing_run != identity.run_id:
        raise IdentityCollisionError(
            "ATTEMPT_ID is already associated with another RUN_ID."
        )


def _reject_unknown_keys(value: Mapping[str, Any]) -> None:
    allowed = {
        "task_id",
        "run_id",
        "attempt_id",
        "session_id",
        "revision",
        "schema_version",
    }

    rejected = {
        str(key).lower()
        for key in value
        if str(key).lower() in _REJECTED_FIELDS
    }

    if rejected:
        names = ", ".join(sorted(rejected))
        raise ExecutionIdentityError(
            f"execution_identity contains prohibited field(s): {names}."
        )

    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise ExecutionIdentityError(
            f"execution_identity contains unknown field(s): {names}."
        )
