"""Pure, read-only invariants for the Roadmaps Phase 0 contract.

This module deliberately has no Hermes imports, database access, filesystem writes,
or network surface. Repository implementations in this module are in-memory test
fixtures only; ``projects.db`` remains the durable backend authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, TypeAlias


class DuplicateEventConflict(ValueError):
    """An event id was reused for a different event payload or identity."""


class StaleEventError(ValueError):
    """An event would move an aggregate to an older or equal version."""


_IDENTIFIER_FIELDS = frozenset({
    "profile_id", "project_id", "roadmap_id", "node_id", "event_id",
    "aggregate_id", "actor", "causation_id", "correlation_id",
})
_ALLOWED_AGGREGATE_TYPES = frozenset({"roadmap", "node", "todo", "report", "proof"})
_ALLOWED_RELATION_KINDS = frozenset({"depends_on", "validates", "blocks", "contains", "derived_from"})
_EVENT_TYPE = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+$")


def _identifier(value: object, field: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"invalid {field}: must be a non-empty trimmed string")
    if "/" in value or "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"invalid {field}: path separators and control characters are forbidden")
    return value


def _freeze(value: Any, path: str = "payload") -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze(item, f"{path}[]") for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item, f"{path}[]") for item in value)
    if isinstance(value, (Scope, QualifiedNodeKey, Relation, EventEnvelope)):
        return value
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise TypeError(f"{path} must contain finite JSON values")
        return value
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _payload_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class Scope:
    profile_id: str
    project_id: str
    roadmap_id: str

    def __post_init__(self) -> None:
        for field in ("profile_id", "project_id", "roadmap_id"):
            _identifier(getattr(self, field), field)

    def as_tuple(self) -> tuple[str, str, str]:
        return self.profile_id, self.project_id, self.roadmap_id


@dataclass(frozen=True, slots=True)
class QualifiedNodeKey:
    scope: Scope
    node_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope, Scope):
            raise TypeError("scope must be a Scope")
        _identifier(self.node_id, "node_id")

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (*self.scope.as_tuple(), self.node_id)


@dataclass(frozen=True, slots=True)
class Relation:
    from_node: QualifiedNodeKey
    to_node: QualifiedNodeKey
    kind: str = "depends_on"

    def __post_init__(self) -> None:
        if not isinstance(self.from_node, QualifiedNodeKey) or not isinstance(self.to_node, QualifiedNodeKey):
            raise TypeError("relations must reference QualifiedNodeKey values")
        if self.from_node.scope != self.to_node.scope:
            raise ValueError("relations must reference nodes in the same scope")
        if not isinstance(self.kind, str):
            raise TypeError("relation kind must be a string")
        if self.kind not in _ALLOWED_RELATION_KINDS:
            raise ValueError(f"invalid relation kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    schema_version: int
    event_id: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    scope: Scope
    actor: str
    occurred_at: datetime
    received_at: datetime
    causation_id: str | None
    correlation_id: str | None
    aggregate_version: int
    payload_hash: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.scope, Scope):
            raise TypeError("scope must be a Scope")
        for field in ("event_id", "aggregate_id", "actor"):
            _identifier(getattr(self, field), field)
        if not isinstance(self.event_type, str) or _EVENT_TYPE.fullmatch(self.event_type) is None:
            raise ValueError("event_type must be a lowercase dot-qualified string")
        if not isinstance(self.aggregate_type, str):
            raise TypeError("aggregate_type must be a string")
        if self.aggregate_type not in _ALLOWED_AGGREGATE_TYPES:
            raise ValueError(f"invalid aggregate_type: {self.aggregate_type!r}")
        _identifier(self.causation_id, "causation_id", optional=True)
        _identifier(self.correlation_id, "correlation_id", optional=True)
        if type(self.aggregate_version) is not int or self.aggregate_version < 1:
            raise ValueError("aggregate_version must be a positive integer")
        occurred = _utc(self.occurred_at, "occurred_at")
        received = _utc(self.received_at, "received_at")
        if not isinstance(self.payload_hash, str) or re.fullmatch(r"[0-9a-f]{64}", self.payload_hash) is None:
            raise ValueError("payload_hash must be a sha256 hex digest")
        object.__setattr__(self, "occurred_at", occurred)
        object.__setattr__(self, "received_at", received)


    @classmethod
    def create(
        cls, *, event_id: str, scope: Scope, aggregate_id: str, payload: Mapping[str, object],
        event_type: str,
        aggregate_type: str = "roadmap", actor: str = "contract-test", aggregate_version: int = 1,
        causation_id: str | None = None, correlation_id: str | None = None,
        occurred_at: datetime | None = None, received_at: datetime | None = None,
    ) -> "EventEnvelope":
        now = datetime.now(timezone.utc)
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        frozen_payload = _freeze(payload)
        return cls(
            schema_version=1, event_id=event_id, event_type=event_type, aggregate_type=aggregate_type,
            aggregate_id=aggregate_id, scope=scope, actor=actor,
            occurred_at=occurred_at if occurred_at is not None else now,
            received_at=received_at if received_at is not None else now,
            causation_id=causation_id, correlation_id=correlation_id,
            aggregate_version=aggregate_version, payload_hash=_payload_hash(frozen_payload),
        )

    def ensure_same_identity(self, other: "EventEnvelope") -> None:
        if not isinstance(other, EventEnvelope):
            raise TypeError("other must be an EventEnvelope")
        if self.event_id != other.event_id:
            raise ValueError("events have different event_id values")
        # Timestamps are part of identity: a replay is idempotent only when the
        # complete envelope metadata is byte-for-byte equivalent after UTC
        # normalization. Payload content is represented by its required hash.
        comparable = (
            self.scope, self.event_type, self.aggregate_type, self.aggregate_id, self.actor,
            self.aggregate_version, self.payload_hash, self.causation_id,
            self.correlation_id, self.occurred_at, self.received_at,
        )
        other_comparable = (
            other.scope, other.event_type, other.aggregate_type, other.aggregate_id, other.actor,
            other.aggregate_version, other.payload_hash, other.causation_id,
            other.correlation_id, other.occurred_at, other.received_at,
        )
        if comparable != other_comparable:
            raise DuplicateEventConflict(f"event_id {self.event_id!r} has conflicting payload or identity")


class RoadmapRepository(Protocol):
    """Read-only repository boundary; this protocol contains no persistence behavior."""

    def list(self, *, profile_id: str, project_id: str) -> tuple[Scope, ...]: ...
    def get(self, scope: Scope) -> Mapping[str, object]: ...
    def get_snapshot(self, scope: Scope) -> Mapping[str, object]: ...
    def get_events_after(self, scope: Scope, cursor: int) -> tuple[EventEnvelope, ...]: ...


class FixtureRoadmapRepository:
    """Injected, immutable in-memory repository used only by contract tests."""

    def __init__(self, scope: Scope, snapshot: Mapping[str, object], events: Sequence[EventEnvelope] = ()) -> None:
        if not isinstance(scope, Scope):
            raise TypeError("scope must be a Scope")
        self._scope = scope
        frozen_snapshot = _freeze(snapshot)
        if not isinstance(frozen_snapshot, Mapping):
            raise TypeError("snapshot must be a mapping")
        self._snapshot = frozen_snapshot
        self._events = tuple(events)
        for event in self._events:
            if not isinstance(event, EventEnvelope):
                raise TypeError("events must contain only EventEnvelope values")
            if event.scope != self._scope:
                raise ValueError("events must belong to the repository scope")

    def list(self, *, profile_id: str, project_id: str) -> tuple[Scope, ...]:
        return (self._scope,) if (profile_id, project_id) == self._scope.as_tuple()[:2] else ()

    def get(self, scope: Scope) -> Mapping[str, object]:
        if scope != self._scope:
            raise KeyError(scope)
        return self._snapshot

    def get_snapshot(self, scope: Scope) -> Mapping[str, object]:
        return self.get(scope)

    def get_events_after(self, scope: Scope, cursor: int) -> tuple[EventEnvelope, ...]:
        if scope != self._scope:
            raise KeyError(scope)
        if type(cursor) is not int or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        return replay_events(self._events, cursor=cursor)


PLAN_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"proposed"}), "proposed": frozenset({"validated", "revision_requested"}),
    "validated": frozenset({"in_progress", "revision_requested"}),
    "in_progress": frozenset({"blocked", "completed", "revision_requested"}),
    "blocked": frozenset({"in_progress", "revision_requested"}),
    "completed": frozenset({"archived", "revision_requested"}),
    "revision_requested": frozenset({"proposed", "archived"}), "archived": frozenset(),
}
NODE_TRANSITIONS: dict[str, frozenset[str]] = {
    "planned": frozenset({"ready"}), "ready": frozenset({"in_progress"}),
    "in_progress": frozenset({"blocked", "completed"}), "blocked": frozenset({"in_progress", "completed"}),
    "completed": frozenset({"archived"}), "archived": frozenset(),
}


def _transition(table: Mapping[str, frozenset[str]], kind: str, current: str, target: str) -> str:
    if target not in table.get(current, frozenset()):
        raise ValueError(f"invalid {kind} transition: {current!r} -> {target!r}")
    return target


def transition_plan(current: str, target: str) -> str:
    return _transition(PLAN_TRANSITIONS, "plan", current, target)


def transition_node(current: str, target: str) -> str:
    return _transition(NODE_TRANSITIONS, "node", current, target)


def replay_events(events: Sequence[EventEnvelope], cursor: int = 0) -> tuple[EventEnvelope, ...]:
    """Validate the complete stream, then return the suffix after ``cursor``."""
    if type(cursor) is not int or cursor < 0 or cursor > len(events):
        raise ValueError("cursor must be within the event stream")
    versions: dict[tuple[Scope, str, str], int] = {}
    identities: dict[str, EventEnvelope] = {}
    for event in events:
        if not isinstance(event, EventEnvelope):
            raise TypeError("events must contain only EventEnvelope values")
        previous = identities.get(event.event_id)
        if previous is not None:
            previous.ensure_same_identity(event)
            # An exactly identical duplicate is an idempotent replay, not a
            # second application of the same aggregate version.
            continue
        identities[event.event_id] = event
        key = (event.scope, event.aggregate_type, event.aggregate_id)
        if event.aggregate_version <= versions.get(key, 0):
            raise StaleEventError(
                f"aggregate {event.aggregate_id!r} received non-increasing version {event.aggregate_version}"
            )
        versions[key] = event.aggregate_version
    emitted_ids: set[str] = {event.event_id for event in events[:cursor]}
    unique_suffix: list[EventEnvelope] = []
    for event in events[cursor:]:
        if event.event_id not in emitted_ids:
            unique_suffix.append(event)
            emitted_ids.add(event.event_id)
    return tuple(unique_suffix)


def build_fixture(scope: Scope) -> Mapping[str, object]:
    """Return a small in-memory fixture; never persist or mutate external state."""
    nodes = (QualifiedNodeKey(scope, "objective"), QualifiedNodeKey(scope, "step-1"), QualifiedNodeKey(scope, "proof-1"))
    return MappingProxyType({
        "scope": scope, "plan_snapshot": MappingProxyType({"version": 1, "state": "validated"}),
        "execution_state": "ready", "reported_state": "none", "verification_state": "unverified",
        "todos": (), "nodes": nodes, "relations": (Relation(nodes[1], nodes[2], "validates"),),
    })


FixtureRepository: TypeAlias = FixtureRoadmapRepository
