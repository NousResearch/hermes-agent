"""Internal agent-event envelopes for Discord Native Multi-Bot Protocol v2.

Agent-agent collaboration in protocol v2 is authoritative in the local SQLite
store, not in Discord messages.  This module defines the small validated envelope
used by the handoff/consult/review API before rows are persisted to
``agent_events``.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from gateway.secret_refs import redact_sensitive_data

REQUEST_EVENT_TYPES = frozenset(
    {
        "handoff.requested",
        "consult.requested",
        "review.requested",
    }
)
HANDOFF_STATE_EVENT_TYPES = frozenset(
    {
        "handoff.accepted",
        "handoff.declined",
        "handoff.completed",
        "handoff.cancelled",
    }
)
AGENT_EVENT_TYPES = REQUEST_EVENT_TYPES | HANDOFF_STATE_EVENT_TYPES
HANDOFF_STATES = ("requested", "accepted", "declined", "completed", "cancelled")

_AGENT_EVENT_ID_RE = re.compile(r"^evt_[A-Za-z0-9_-]{16,64}$")


@dataclass(frozen=True)
class AgentEventEnvelope:
    """Validated persisted-event envelope for local agent-agent collaboration."""

    agent_event_id: str
    event_type: str
    source_agent_id: str | None
    target_agent_id: str
    topic_id: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_event_id": self.agent_event_id,
            "event_type": self.event_type,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "topic_id": self.topic_id,
            "payload": self.payload,
        }

    def to_row_payload(self) -> dict[str, Any]:
        return self.to_dict()


def new_agent_event_id(seed: str | None = None) -> str:
    """Return an opaque ``evt_`` id.

    When ``seed`` is supplied the generated id is deterministic, which gives
    callers a restart/replay-safe idempotency key without exposing semantic data
    in the primary key.  Without a seed it is random and stable once persisted.
    """

    if seed is not None and str(seed):
        return f"evt_{uuid.uuid5(uuid.NAMESPACE_URL, str(seed)).hex}"
    return f"evt_{uuid.uuid4().hex}"


def is_valid_agent_event_id(agent_event_id: str) -> bool:
    return bool(_AGENT_EVENT_ID_RE.fullmatch(str(agent_event_id or "")))


def safe_event_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return JSON-safe, redacted payload data for DB/audit persistence."""

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    redacted = redact_sensitive_data(dict(payload))
    # Fail early on unserializable values instead of relying on sqlite helpers to
    # raise later, possibly after other rows were written.
    json.dumps(redacted, sort_keys=True, ensure_ascii=False)
    return redacted


def create_agent_event_envelope(
    *,
    event_type: str,
    source_agent_id: str | None,
    target_agent_id: str,
    topic_id: str,
    payload: Mapping[str, Any] | None = None,
    agent_event_id: str | None = None,
    idempotency_seed: str | None = None,
) -> AgentEventEnvelope:
    """Validate and build an internal event envelope."""

    effective_event_id = agent_event_id or new_agent_event_id(idempotency_seed)
    envelope = AgentEventEnvelope(
        agent_event_id=str(effective_event_id),
        event_type=str(event_type or ""),
        source_agent_id=str(source_agent_id) if source_agent_id else None,
        target_agent_id=str(target_agent_id or ""),
        topic_id=str(topic_id or ""),
        payload=safe_event_payload(payload),
    )
    validate_agent_event_envelope(envelope)
    return envelope


def validate_agent_event_envelope(envelope: AgentEventEnvelope) -> None:
    """Raise ``ValueError`` if an internal event envelope is not persistable."""

    if not is_valid_agent_event_id(envelope.agent_event_id):
        raise ValueError("agent_event_id must be opaque evt_[A-Za-z0-9_-]{16,64}")
    if envelope.event_type not in AGENT_EVENT_TYPES:
        raise ValueError("event_type must be a supported internal collaboration event")
    if not envelope.target_agent_id:
        raise ValueError("target_agent_id is required")
    if not envelope.topic_id:
        raise ValueError("topic_id is required")
    json.dumps(envelope.payload, sort_keys=True, ensure_ascii=False)
