"""Append-only audit-record validation and deterministic chain hashing."""

from __future__ import annotations

from typing import Any, Mapping

from plugins.agentops.control.events import canonical_hash, validate_string_value
from plugins.agentops.control.models import AuditEvent


class AuditValidationError(ValueError):
    """Raised without serializing untrusted audit metadata."""


def validate_audit_fields(
    *,
    actor_type: object,
    actor_id: object,
    action: object,
    object_type: object,
    object_id: object,
    timestamp: object,
    metadata: object,
    before_hash: object,
    after_hash: object,
) -> None:
    required = (actor_type, actor_id, action, object_type, object_id, timestamp)
    try:
        for value in required:
            validate_string_value(value, required=True)
        if before_hash is not None:
            validate_string_value(before_hash, required=True)
        if after_hash is not None:
            validate_string_value(after_hash, required=True)
    except ValueError as exc:
        raise AuditValidationError("audit validation failed")
    if not isinstance(metadata, Mapping):
        raise AuditValidationError("audit validation failed")
    try:
        canonical_hash(metadata)
    except ValueError as exc:
        raise AuditValidationError("audit validation failed") from exc


def validate_audit_event(event: AuditEvent) -> None:
    validate_audit_fields(
        actor_type=event.actor_type,
        actor_id=event.actor_id,
        action=event.action,
        object_type=event.object_type,
        object_id=event.object_id,
        timestamp=event.timestamp,
        metadata=event.metadata,
        before_hash=event.before_hash,
        after_hash=event.after_hash,
    )


def audit_entry_hash(*, sequence: int, payload: Mapping[str, Any]) -> str:
    return canonical_hash({"sequence": sequence, "payload": dict(payload)})
