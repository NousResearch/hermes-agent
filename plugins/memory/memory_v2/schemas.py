"""Canonical schema types for the Memory v2 provider.

These dataclasses are intentionally lightweight and dependency-free. They form
Memory v2's in-process contract; later storage/index layers can serialize them
as YAML/JSON without depending on opaque dict shapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, cast


class ValidationError(ValueError):
    """Raised when a Memory v2 schema record is invalid."""


class _StrEnum(str, Enum):
    """String enum with ergonomic coercion from raw strings."""

    @classmethod
    def coerce(cls, value: Any, field_name: str):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except Exception as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValidationError(f"Invalid {field_name}: {value!r}; expected one of: {allowed}") from exc


class SourceType(_StrEnum):
    SESSION = "session"
    MESSAGE = "message"
    FILE = "file"
    TOOL_RESULT = "tool_result"
    MEMORY = "memory"
    SKILL = "skill"
    WEB = "web"
    MANUAL = "manual"


class MemoryType(_StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    BELIEF = "belief"
    CONSTRAINT = "constraint"
    ENVIRONMENT = "environment"
    PROJECT_STATE = "project_state"
    EPISODE = "episode"
    PROCEDURE_REF = "procedure_ref"


class MemoryStatus(_StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    UNCERTAIN = "uncertain"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class ProjectStatus(_StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"

class CoreMemoryCategory(_StrEnum):
    USER = "user"
    ASSISTANT_IDENTITY = "assistant_identity"
    ENVIRONMENT = "environment"
    OPERATING_RULE = "operating_rule"


class GateDecision(_StrEnum):
    PENDING = "pending"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ARCHIVED_ONLY = "archived_only"
    SUPERSEDED = "superseded"


def utc_now_iso() -> str:
    """Return a compact UTC ISO timestamp with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_nonblank(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValidationError(f"{field_name} is required")
    return text


def _validate_unit_interval(value: float, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a number between 0.0 and 1.0") from exc
    if number < 0.0 or number > 1.0:
        raise ValidationError(f"{field_name} must be between 0.0 and 1.0")
    return number


def _list_of_strings(values: Optional[List[Any]]) -> List[str]:
    if values is None:
        return []
    return [str(value) for value in values]


def normalize_project_id(value: str) -> str:
    """Normalize a project name/id into ``project:<slug>`` form."""
    text = _require_nonblank(value, "project id")
    if text.startswith("project:"):
        raw = text[len("project:") :]
    else:
        raw = text
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    if not slug:
        raise ValidationError("project id must contain at least one alphanumeric character")
    return f"project:{slug}"


@dataclass
class SourceRef:
    id: str
    type: SourceType | str
    uri: str
    title: str = ""
    observed_at: str = ""
    quote: Optional[str] = None

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.type = SourceType.coerce(self.type, "type")
        self.uri = _require_nonblank(self.uri, "uri")
        self.title = str(self.title or "")
        self.observed_at = str(self.observed_at or "")
        if self.quote is not None:
            self.quote = str(self.quote)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "type": cast(SourceType, self.type).value,
            "uri": self.uri,
            "title": self.title,
            "observed_at": self.observed_at,
        }
        if self.quote is not None:
            data["quote"] = self.quote
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceRef":
        return cls(**data)


@dataclass
class MemoryItem:
    id: str
    type: MemoryType | str
    subject: str
    predicate: Optional[str] = None
    value: Optional[str] = None
    body: Optional[str] = None
    summary: Optional[str] = None
    status: MemoryStatus | str = MemoryStatus.ACTIVE
    confidence: float = 0.7
    importance: float = 0.5
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    expires_at: Optional[str] = None
    source_refs: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    superseded_at: Optional[str] = None
    supersession_reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.type = MemoryType.coerce(self.type, "type")
        self.subject = _require_nonblank(self.subject, "subject")
        self.status = MemoryStatus.coerce(self.status, "status")
        self.confidence = _validate_unit_interval(self.confidence, "confidence")
        self.importance = _validate_unit_interval(self.importance, "importance")
        self.source_refs = _list_of_strings(self.source_refs)
        self.supersedes = _list_of_strings(self.supersedes)
        self.tags = _list_of_strings(self.tags)
        if self.status == MemoryStatus.SUPERSEDED and not self.superseded_by:
            raise ValidationError("superseded_by is required when status is superseded")
        if self.status == MemoryStatus.ACTIVE and self.superseded_by:
            raise ValidationError("active memories cannot set superseded_by")
        if self.superseded_by is not None:
            self.superseded_by = str(self.superseded_by)
        if self.superseded_at is not None:
            self.superseded_at = str(self.superseded_at)
        if self.supersession_reason is not None:
            self.supersession_reason = str(self.supersession_reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": getattr(self.type, "value", str(self.type)),
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "body": self.body,
            "summary": self.summary,
            "status": getattr(self.status, "value", str(self.status)),
            "confidence": self.confidence,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "expires_at": self.expires_at,
            "source_refs": list(self.source_refs),
            "supersedes": list(self.supersedes),
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at,
            "supersession_reason": self.supersession_reason,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(**data)


@dataclass
class ProjectCard:
    id: str
    name: str
    status: ProjectStatus | str = ProjectStatus.ACTIVE
    importance: float = 0.5
    updated_at: str = field(default_factory=utc_now_iso)
    goal: str = ""
    why_it_matters: str = ""
    current_state: str = ""
    decisions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    injection_policy: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = normalize_project_id(self.id)
        self.name = _require_nonblank(self.name, "name")
        self.status = ProjectStatus.coerce(self.status, "status")
        self.importance = _validate_unit_interval(self.importance, "importance")
        self.decisions = _list_of_strings(self.decisions)
        self.open_questions = _list_of_strings(self.open_questions)
        self.next_actions = _list_of_strings(self.next_actions)
        self.source_refs = _list_of_strings(self.source_refs)
        self.related_entities = _list_of_strings(self.related_entities)
        self.injection_policy = dict(self.injection_policy or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": cast(ProjectStatus, self.status).value,
            "importance": self.importance,
            "updated_at": self.updated_at,
            "goal": self.goal,
            "why_it_matters": self.why_it_matters,
            "current_state": self.current_state,
            "decisions": list(self.decisions),
            "open_questions": list(self.open_questions),
            "next_actions": list(self.next_actions),
            "source_refs": list(self.source_refs),
            "related_entities": list(self.related_entities),
            "injection_policy": dict(self.injection_policy),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectCard":
        return cls(**data)


@dataclass
class CoreMemoryRecord:
    id: str
    category: CoreMemoryCategory | str
    statement: str
    layer: str = "core"
    priority: float = 0.8
    confidence: float = 0.9
    updated_at: str = field(default_factory=utc_now_iso)
    source_refs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.category = CoreMemoryCategory.coerce(self.category, "category")
        self.statement = _require_nonblank(self.statement, "statement")
        self.layer = str(self.layer or "core")
        if self.layer != "core":
            raise ValidationError("layer must be core")
        self.priority = _validate_unit_interval(self.priority, "priority")
        self.confidence = _validate_unit_interval(self.confidence, "confidence")
        self.source_refs = _list_of_strings(self.source_refs)
        self.tags = _list_of_strings(self.tags)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "layer": "core",
            "category": cast(CoreMemoryCategory, self.category).value,
            "statement": self.statement,
            "priority": self.priority,
            "confidence": self.confidence,
            "updated_at": self.updated_at,
            "source_refs": list(self.source_refs),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoreMemoryRecord":
        return cls(**data)


@dataclass
class CandidateMemory:
    id: str
    type: MemoryType | str
    claim: str
    created_at: str = field(default_factory=utc_now_iso)
    proposed_destination: str = ""
    importance: float = 0.5
    confidence: float = 0.7
    promotion_reason: str = ""
    source_refs: List[str] = field(default_factory=list)
    gate_decision: GateDecision | str = GateDecision.PENDING
    decision_reason: str = ""

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.type = MemoryType.coerce(self.type, "type")
        self.claim = _require_nonblank(self.claim, "claim")
        self.importance = _validate_unit_interval(self.importance, "importance")
        self.confidence = _validate_unit_interval(self.confidence, "confidence")
        self.source_refs = _list_of_strings(self.source_refs)
        self.gate_decision = GateDecision.coerce(self.gate_decision, "gate_decision")
        if self.gate_decision != GateDecision.PENDING and not str(self.decision_reason or "").strip():
            raise ValidationError("decision_reason is required for non-pending gate decisions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "type": cast(MemoryType, self.type).value,
            "claim": self.claim,
            "proposed_destination": self.proposed_destination,
            "importance": self.importance,
            "confidence": self.confidence,
            "promotion_reason": self.promotion_reason,
            "source_refs": list(self.source_refs),
            "gate_decision": cast(GateDecision, self.gate_decision).value,
            "decision_reason": self.decision_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateMemory":
        return cls(**data)


@dataclass
class WorkingMemory:
    session_id: str
    updated_at: str = field(default_factory=utc_now_iso)
    focus: Dict[str, Any] = field(default_factory=dict)
    scratchpad: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session_id = _require_nonblank(self.session_id, "session_id")
        self.focus = dict(self.focus or {})
        default_scratchpad = {
            "relevant_paths": [],
            "relevant_commands": [],
            "retrieved_memory_ids": [],
        }
        merged = dict(default_scratchpad)
        merged.update(dict(self.scratchpad or {}))
        self.scratchpad = merged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "updated_at": self.updated_at,
            "focus": dict(self.focus),
            "scratchpad": dict(self.scratchpad),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        return cls(**data)


@dataclass
class MemoryPacket:
    route: str
    confidence: str
    token_budget: int
    items: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sections: Dict[str, Any] = field(default_factory=dict)
    retrieval_plan: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.route = _require_nonblank(self.route, "route")
        self.confidence = _require_nonblank(self.confidence, "confidence")
        try:
            self.token_budget = int(self.token_budget)
        except (TypeError, ValueError) as exc:
            raise ValidationError("token_budget must be a non-negative integer") from exc
        if self.token_budget < 0:
            raise ValidationError("token_budget must be a non-negative integer")
        self.items = [dict(item) for item in (self.items or [])]
        self.warnings = _list_of_strings(self.warnings)
        self.sections = dict(self.sections or {})
        self.retrieval_plan = dict(self.retrieval_plan or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "token_budget": self.token_budget,
            "items": [dict(item) for item in self.items],
            "warnings": list(self.warnings),
            "sections": dict(self.sections),
            "retrieval_plan": dict(self.retrieval_plan),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryPacket":
        return cls(**data)
