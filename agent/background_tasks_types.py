"""Public contract types for the external background-task lifecycle API.

Holders of the stable, plugin-facing dataclasses and the handle signing /
validation helpers for :mod:`agent.background_tasks`. Kept separate so the
service module stays focused on orchestration and size/schema validation.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping
from typing import Any, Dict, Optional

from agent.background_tasks_store import sign_handle

PUBLIC_CONTRACT_VERSION = 1

# Bounded text/schema sizes enforced BEFORE any persistence or queue write.
MAX_EXTERNAL_ID_CHARS = 200
MAX_IDEMPOTENCY_KEY_CHARS = 200
MAX_EVENT_ID_CHARS = 200
MAX_LABEL_CHARS = 500
MAX_SUMMARY_CHARS = 32_000
MAX_ERROR_CHARS = 32_000
MAX_PAYLOAD_BYTES = 32_000


class BackgroundTaskError(ValueError):
    """A request cannot be safely accepted by the external background-task API."""


class ExternalTaskState(str, enum.Enum):
    """Durable lifecycle state of an external background task."""

    REGISTERED = "registered"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class ExternalTaskHandle:
    """Opaque, tamper-evident reference to one registered external task.

    The ``signature`` is an HMAC over the other fields using a persisted
    profile-local key; plugins never see the key and cannot forge handles for
    tasks they do not own. Handles remain valid across process restarts.
    """

    contract_version: int
    task_id: str
    plugin_id: str
    parent_session_id: str
    created_at: float
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ExternalTaskStatus:
    """Read-only snapshot of one external task for the owning plugin."""

    handle: ExternalTaskHandle
    state: ExternalTaskState
    delivery_state: str
    external_id: str
    created_at: float
    updated_at: float
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ExternalTaskResult:
    """Outcome of a lifecycle operation; all fields are JSON-safe."""

    handle: Optional[ExternalTaskHandle] = None
    accepted: bool = False
    state: str = ""
    already_terminal: bool = False
    conflict: bool = False
    unknown_handle: bool = False
    cancel_already_requested: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def sign(
    key: bytes, task_id: str, plugin_id: str, parent_session_id: str, created_at: float
) -> str:
    return sign_handle(key, task_id, plugin_id, parent_session_id, created_at)


def handle_from_row(key: bytes, row: Mapping[str, Any]) -> ExternalTaskHandle:
    return ExternalTaskHandle(
        contract_version=PUBLIC_CONTRACT_VERSION,
        task_id=row["task_id"],
        plugin_id=row["plugin_id"],
        parent_session_id=row["parent_session_id"],
        created_at=row["created_at"],
        signature=sign(
            key,
            row["task_id"],
            row["plugin_id"],
            row["parent_session_id"],
            row["created_at"],
        ),
    )


def valid_handle_shape(handle: ExternalTaskHandle) -> bool:
    return (
        isinstance(handle, ExternalTaskHandle)
        and type(handle.contract_version) is int
        and handle.contract_version == PUBLIC_CONTRACT_VERSION
        and isinstance(handle.task_id, str)
        and bool(handle.task_id)
        and isinstance(handle.plugin_id, str)
        and bool(handle.plugin_id)
        and isinstance(handle.parent_session_id, str)
        and isinstance(handle.created_at, (int, float))
        and not isinstance(handle.created_at, bool)
        and math.isfinite(handle.created_at)
        and isinstance(handle.signature, str)
        and bool(handle.signature)
    )


def coerce_handle(value: Any) -> Optional[ExternalTaskHandle]:
    """Parse a handle passed as an object or a JSON-safe mapping."""
    if isinstance(value, ExternalTaskHandle):
        handle = value
    elif isinstance(value, Mapping):
        try:
            handle = ExternalTaskHandle(**dict(value))
        except (TypeError, ValueError):
            return None
    else:
        return None
    return handle if valid_handle_shape(handle) else None
