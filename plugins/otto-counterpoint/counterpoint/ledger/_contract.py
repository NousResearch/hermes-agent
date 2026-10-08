# -*- coding: utf-8 -*-
"""Versioned event contract for the Phase 2 execution ledger."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping, Optional, Sequence

CURRENT_SCHEMA_VERSION = "v1"
schema_version = CURRENT_SCHEMA_VERSION


class EventType:
    CREATED = "run.created"
    STARTED = "run.started"
    BLOCKED = "run.blocked"
    FINISHED = "run.finished"


# Lowercase aliases keep the public contract convenient for callers that use
# the same names as the event stream.
event_type = EventType()
VALID_EVENT_TYPES = frozenset(
    {
        EventType.CREATED,
        EventType.STARTED,
        EventType.BLOCKED,
        EventType.FINISHED,
    }
)
VALID_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
VALID_TRANSITIONS = {
    EventType.CREATED: frozenset({EventType.STARTED, EventType.BLOCKED}),
    EventType.STARTED: frozenset({EventType.FINISHED, EventType.BLOCKED}),
    EventType.BLOCKED: frozenset({EventType.STARTED, EventType.FINISHED}),
    EventType.FINISHED: frozenset(),
}
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class LedgerContractError(ValueError):
    """The event violates the versioned ledger contract."""


class LedgerConflictError(LedgerContractError):
    """An idempotency key was reused with a different business payload."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_timestamp(value: str, field: str) -> None:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise LedgerContractError(f"{field} must be UTC ISO-8601 seconds: {value!r}")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise LedgerContractError(f"{field} is not a valid timestamp: {value!r}") from exc


def _event_id_for_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return f"evt-{digest}"


@dataclass(frozen=True)
class EventEnvelope:
    """Immutable in-memory representation of one ledger event."""

    event_id: str
    event_type: str
    schema_version: str
    project_id: str
    linear_issue_id: Optional[str]
    run_id: str
    agent_id: str
    occurred_at: str
    recorded_at: str
    sequence: int
    idempotency_key: str
    evidence_refs: tuple[str, ...]
    blocked_reason: Optional[str]
    terminal_state: Optional[str]
    payload: dict[str, Any]
    previous_event_id: Optional[str]

    _fingerprint_ignored_fields: ClassVar[frozenset[str]] = frozenset(
        {"event_id", "recorded_at"}
    )

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        project_id: str,
        linear_issue_id: Optional[str],
        run_id: str,
        agent_id: str,
        occurred_at: str,
        idempotency_key: str,
        evidence_refs: Sequence[str] = (),
        blocked_reason: Optional[str] = None,
        terminal_state: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        previous_event_id: Optional[str] = None,
        schema_version: str = CURRENT_SCHEMA_VERSION,
        event_id: Optional[str] = None,
        recorded_at: Optional[str] = None,
        sequence: int = 1,
    ) -> "EventEnvelope":
        if payload is None:
            payload = {}
        try:
            payload_copy = json.loads(json.dumps(dict(payload), sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise LedgerContractError("payload must be JSON-serializable") from exc
        envelope = cls(
            event_id=event_id or _event_id_for_key(idempotency_key),
            event_type=event_type,
            schema_version=schema_version,
            project_id=project_id,
            linear_issue_id=linear_issue_id,
            run_id=run_id,
            agent_id=agent_id,
            occurred_at=occurred_at,
            recorded_at=recorded_at or _now_utc(),
            sequence=sequence,
            idempotency_key=idempotency_key,
            evidence_refs=tuple(evidence_refs),
            blocked_reason=blocked_reason,
            terminal_state=terminal_state,
            payload=payload_copy,
            previous_event_id=previous_event_id,
        )
        envelope.validate()
        return envelope

    def validate(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise LedgerContractError(
                f"schema_version must be {CURRENT_SCHEMA_VERSION!r}"
            )
        if self.event_type not in VALID_EVENT_TYPES:
            raise LedgerContractError(f"unknown event_type: {self.event_type!r}")
        for field, value in (
            ("event_id", self.event_id),
            ("project_id", self.project_id),
            ("run_id", self.run_id),
            ("agent_id", self.agent_id),
            ("idempotency_key", self.idempotency_key),
        ):
            if not isinstance(value, str) or not value.strip():
                raise LedgerContractError(f"{field} must be a non-empty string")
        if self.linear_issue_id is not None and (
            not isinstance(self.linear_issue_id, str) or not self.linear_issue_id.strip()
        ):
            raise LedgerContractError("linear_issue_id must be null or non-empty string")
        _validate_timestamp(self.occurred_at, "occurred_at")
        _validate_timestamp(self.recorded_at, "recorded_at")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1:
            raise LedgerContractError("sequence must be an integer >= 1")
        if any(not isinstance(ref, str) or not ref.strip() for ref in self.evidence_refs):
            raise LedgerContractError("evidence_refs must contain non-empty strings")
        if self.event_type == EventType.BLOCKED:
            if not isinstance(self.blocked_reason, str) or not self.blocked_reason.strip():
                raise LedgerContractError("run.blocked requires blocked_reason")
        elif self.blocked_reason is not None:
            raise LedgerContractError("blocked_reason is only valid for run.blocked")
        if self.event_type == EventType.FINISHED:
            if self.terminal_state not in VALID_TERMINAL_STATES:
                raise LedgerContractError("run.finished requires a valid terminal_state")
        elif self.terminal_state is not None:
            raise LedgerContractError("terminal_state is only valid for run.finished")
        if not isinstance(self.payload, dict):
            raise LedgerContractError("payload must be an object")
        if self.previous_event_id is not None and (
            not isinstance(self.previous_event_id, str) or not self.previous_event_id.strip()
        ):
            raise LedgerContractError("previous_event_id must be null or non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible snapshot."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "linear_issue_id": self.linear_issue_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "occurred_at": self.occurred_at,
            "recorded_at": self.recorded_at,
            "sequence": self.sequence,
            "idempotency_key": self.idempotency_key,
            "evidence_refs": list(self.evidence_refs),
            "blocked_reason": self.blocked_reason,
            "terminal_state": self.terminal_state,
            "payload": copy.deepcopy(self.payload),
            "previous_event_id": self.previous_event_id,
        }

    def idempotency_fingerprint(self) -> dict[str, Any]:
        body = self.to_dict()
        for field in self._fingerprint_ignored_fields:
            body.pop(field, None)
        return body
