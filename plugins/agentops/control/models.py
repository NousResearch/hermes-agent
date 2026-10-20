"""Stable, side-effect-free data models for the AgentOps control plane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class AuthorityMode(str, Enum):
    """The only active authority available in Phase 1."""

    OBSERVE_ONLY = "observe_only"


@dataclass(frozen=True)
class EventEnvelope:
    """Schema-v1, redacted event persisted by the local control plane."""

    schema_version: int
    event_id: str
    event_type: str
    occurred_at: datetime
    producer: str
    target_id: str
    correlation_id: str | None
    payload: Mapping[str, Any]
    redaction_version: int

    @classmethod
    def create(
        cls,
        *,
        schema_version: int,
        event_id: str,
        event_type: str,
        occurred_at: datetime,
        producer: str,
        target_id: str,
        correlation_id: str | None,
        payload: Mapping[str, Any],
        redaction_version: int,
    ) -> "EventEnvelope":
        from plugins.agentops.control.events import validate_event_fields

        validate_event_fields(
            schema_version=schema_version,
            event_id=event_id,
            event_type=event_type,
            occurred_at=occurred_at,
            producer=producer,
            target_id=target_id,
            correlation_id=correlation_id,
            payload=payload,
            redaction_version=redaction_version,
        )
        return cls(
            schema_version=schema_version,
            event_id=event_id,
            event_type=event_type,
            occurred_at=occurred_at,
            producer=producer,
            target_id=target_id,
            correlation_id=correlation_id,
            payload=dict(payload),
            redaction_version=redaction_version,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventEnvelope":
        if not isinstance(value, Mapping):
            from plugins.agentops.control.events import EventValidationError

            raise EventValidationError("event validation failed")
        occurred_at = value.get("occurred_at")
        if not isinstance(occurred_at, str):
            from plugins.agentops.control.events import EventValidationError

            raise EventValidationError("event validation failed")
        try:
            parsed_time = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError as exc:
            from plugins.agentops.control.events import EventValidationError

            raise EventValidationError("event validation failed") from exc
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            from plugins.agentops.control.events import EventValidationError

            raise EventValidationError("event validation failed")
        return cls.create(
            schema_version=value.get("schema_version"),
            event_id=value.get("event_id"),
            event_type=value.get("event_type"),
            occurred_at=parsed_time,
            producer=value.get("producer"),
            target_id=value.get("target_id"),
            correlation_id=value.get("correlation_id"),
            payload=payload,
            redaction_version=value.get("redaction_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "producer": self.producer,
            "target_id": self.target_id,
            "correlation_id": self.correlation_id,
            "payload": dict(self.payload),
            "redaction_version": self.redaction_version,
        }

    @property
    def content_hash(self) -> str:
        from plugins.agentops.control.events import canonical_hash

        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AuditEvent:
    """Intent record for an append-only local audit chain."""

    actor_type: str
    actor_id: str
    action: str
    object_type: str
    object_id: str
    timestamp: str
    metadata: Mapping[str, Any]
    before_hash: str | None = None
    after_hash: str | None = None

    @classmethod
    def create(
        cls,
        *,
        actor_type: str,
        actor_id: str,
        action: str,
        object_type: str,
        object_id: str,
        timestamp: str,
        metadata: Mapping[str, Any],
        before_hash: str | None = None,
        after_hash: str | None = None,
    ) -> "AuditEvent":
        from plugins.agentops.control.audit import validate_audit_fields

        validate_audit_fields(
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            object_type=object_type,
            object_id=object_id,
            timestamp=timestamp,
            metadata=metadata,
            before_hash=before_hash,
            after_hash=after_hash,
        )
        return cls(
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            object_type=object_type,
            object_id=object_id,
            timestamp=timestamp,
            metadata=dict(metadata),
            before_hash=before_hash,
            after_hash=after_hash,
        )

    def to_dict(self, *, previous_hash: str | None = None) -> dict[str, Any]:
        return {
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "action": self.action,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "previous_hash": previous_hash,
        }


@dataclass(frozen=True)
class AppendResult:
    event_id: str
    inserted: bool
    content_hash: str


@dataclass(frozen=True)
class SpoolReplayResult:
    appended: int = 0
    duplicates: int = 0
    quarantined: int = 0
    dropped: int = 0
    failed: int = 0


@dataclass(frozen=True)
class ControlPlaneHealth:
    ready: bool
    authority_mode: AuthorityMode
    safe_start_reasons: tuple[str, ...]
    store_available: bool
    audit_chain_valid: bool | None
    event_count: int
    spool_depth: int
    spool_bytes: int = 0
    spool_quarantine_bytes: int = 0
    spool_healthy: bool = True
    global_write_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "authority_mode": self.authority_mode.value,
            "safe_start_reasons": list(self.safe_start_reasons),
            "store_available": self.store_available,
            "audit_chain_valid": self.audit_chain_valid,
            "event_count": self.event_count,
            "spool_depth": self.spool_depth,
            "spool_bytes": self.spool_bytes,
            "spool_quarantine_bytes": self.spool_quarantine_bytes,
            "spool_healthy": self.spool_healthy,
            "global_write_enabled": self.global_write_enabled,
        }


@dataclass(frozen=True)
class StoreInspection:
    exists: bool
    schema_version: int | None
    audit_chain_valid: bool | None
    event_count: int | None
    integrity_ok: bool | None = None
    error: str | None = None
