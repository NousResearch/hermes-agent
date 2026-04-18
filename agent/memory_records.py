from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping, Optional, Sequence
import uuid


class MemoryType(str, Enum):
    PROFILE = "profile"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    TRANSCRIPT_REFERENCE = "transcript_reference"


class MemoryScope(str, Enum):
    OPERATOR = "operator"
    PROFILE = "profile"
    WORKSPACE = "workspace"
    SESSION = "session"


class TrustTier(str, Enum):
    USER_ASSERTED = "user_asserted"
    OBSERVED = "observed"
    USER_APPROVED = "user_approved"
    INFERRED = "inferred"
    UNVERIFIED = "unverified"


class SalienceTier(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecordStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    EXPIRED = "expired"
    ARCHIVED = "archived"


@dataclass
class MemoryRecord:
    record_id: str
    memory_type: MemoryType
    scope: MemoryScope
    topic_key: Optional[str]
    content: str
    source: str
    source_kind: str
    trust_tier: TrustTier
    salience_tier: SalienceTier
    status: RecordStatus
    summary: Optional[str] = None
    created_at: Optional[str] = None
    last_confirmed_at: Optional[str] = None
    last_used_at: Optional[str] = None
    review_after: Optional[str] = None
    expires_at: Optional[str] = None
    supersedes: Optional[str] = None
    conflicts_with: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    revision: int = 1

    def __post_init__(self) -> None:
        self.content = self.content.strip()
        self.summary = _strip_or_none(self.summary)
        self.topic_key = _strip_or_none(self.topic_key)
        self.source = self.source.strip()
        self.source_kind = self.source_kind.strip()
        self.conflicts_with = list(self.conflicts_with)
        self.tags = list(self.tags)
        self.metadata = dict(self.metadata)
        self.revision = int(self.revision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "memory_type": self.memory_type.value,
            "scope": self.scope.value,
            "topic_key": self.topic_key,
            "content": self.content,
            "summary": self.summary,
            "source": self.source,
            "source_kind": self.source_kind,
            "created_at": self.created_at,
            "last_confirmed_at": self.last_confirmed_at,
            "last_used_at": self.last_used_at,
            "review_after": self.review_after,
            "expires_at": self.expires_at,
            "trust_tier": self.trust_tier.value,
            "salience_tier": self.salience_tier.value,
            "status": self.status.value,
            "supersedes": self.supersedes,
            "conflicts_with": self.conflicts_with,
            "tags": self.tags,
            "metadata": self.metadata,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryRecord":
        return cls(
            record_id=str(payload["record_id"]),
            memory_type=MemoryType(payload["memory_type"]),
            scope=MemoryScope(payload["scope"]),
            topic_key=payload.get("topic_key"),
            content=str(payload["content"]),
            summary=payload.get("summary"),
            source=str(payload["source"]),
            source_kind=str(payload["source_kind"]),
            created_at=payload.get("created_at"),
            last_confirmed_at=payload.get("last_confirmed_at"),
            last_used_at=payload.get("last_used_at"),
            review_after=payload.get("review_after"),
            expires_at=payload.get("expires_at"),
            trust_tier=TrustTier(payload["trust_tier"]),
            salience_tier=SalienceTier(payload["salience_tier"]),
            status=RecordStatus(payload["status"]),
            supersedes=payload.get("supersedes"),
            conflicts_with=list(payload.get("conflicts_with", [])),
            tags=list(payload.get("tags", [])),
            metadata=dict(payload.get("metadata", {})),
            revision=int(payload.get("revision", 1)),
        )


@dataclass
class EpisodeRecord(MemoryRecord):
    task_signature: str = ""
    problem_summary: Optional[str] = None
    approach_summary: Optional[str] = None
    key_actions: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    outcome: Optional[str] = None
    outcome_evidence: Optional[str] = None
    failure_notes: Optional[str] = None
    validation_status: str = "candidate"
    reuse_count: int = 0
    source_session_id: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.task_signature = self.task_signature.strip()
        self.problem_summary = _strip_or_none(self.problem_summary)
        self.approach_summary = _strip_or_none(self.approach_summary)
        self.key_actions = list(self.key_actions)
        self.tools_used = list(self.tools_used)
        self.outcome = _strip_or_none(self.outcome)
        self.outcome_evidence = _strip_or_none(self.outcome_evidence)
        self.failure_notes = _strip_or_none(self.failure_notes)
        self.validation_status = self.validation_status.strip() or "candidate"
        self.reuse_count = int(self.reuse_count)
        self.source_session_id = _strip_or_none(self.source_session_id)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "task_signature": self.task_signature,
                "problem_summary": self.problem_summary,
                "approach_summary": self.approach_summary,
                "key_actions": self.key_actions,
                "tools_used": self.tools_used,
                "outcome": self.outcome,
                "outcome_evidence": self.outcome_evidence,
                "failure_notes": self.failure_notes,
                "validation_status": self.validation_status,
                "reuse_count": self.reuse_count,
                "source_session_id": self.source_session_id,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeRecord":
        return cls(
            record_id=str(payload["record_id"]),
            memory_type=MemoryType(payload.get("memory_type", MemoryType.EPISODIC.value)),
            scope=MemoryScope(payload["scope"]),
            topic_key=payload.get("topic_key"),
            content=str(payload["content"]),
            summary=payload.get("summary"),
            source=str(payload["source"]),
            source_kind=str(payload["source_kind"]),
            created_at=payload.get("created_at"),
            last_confirmed_at=payload.get("last_confirmed_at"),
            last_used_at=payload.get("last_used_at"),
            review_after=payload.get("review_after"),
            expires_at=payload.get("expires_at"),
            trust_tier=TrustTier(payload["trust_tier"]),
            salience_tier=SalienceTier(payload["salience_tier"]),
            status=RecordStatus(payload["status"]),
            supersedes=payload.get("supersedes"),
            conflicts_with=list(payload.get("conflicts_with", [])),
            tags=list(payload.get("tags", [])),
            metadata=dict(payload.get("metadata", {})),
            revision=int(payload.get("revision", 1)),
            task_signature=str(payload.get("task_signature", "")),
            problem_summary=payload.get("problem_summary"),
            approach_summary=payload.get("approach_summary"),
            key_actions=list(payload.get("key_actions", [])),
            tools_used=list(payload.get("tools_used", [])),
            outcome=payload.get("outcome"),
            outcome_evidence=payload.get("outcome_evidence"),
            failure_notes=payload.get("failure_notes"),
            validation_status=str(payload.get("validation_status", "candidate")),
            reuse_count=int(payload.get("reuse_count", 0)),
            source_session_id=payload.get("source_session_id"),
        )


def records_to_sidecar_payload(records: Sequence[MemoryRecord]) -> dict[str, Any]:
    return {
        "version": 1,
        "records": [record.to_dict() for record in records],
    }


def records_from_sidecar_payload(payload: Mapping[str, Any]) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for item in payload.get("records", []):
        if not isinstance(item, Mapping):
            continue
        if item.get("memory_type") == MemoryType.EPISODIC.value and "task_signature" in item:
            records.append(EpisodeRecord.from_dict(item))
        else:
            records.append(MemoryRecord.from_dict(item))
    return records


def normalize_legacy_entry(target: str, content: str, created_at: Optional[str]) -> MemoryRecord:
    normalized_content = content.strip()
    scope = MemoryScope.OPERATOR if target == "user" else MemoryScope.WORKSPACE
    return MemoryRecord(
        record_id=f"legacy-{uuid.uuid4().hex}",
        memory_type=MemoryType.PROFILE,
        scope=scope,
        topic_key=_infer_legacy_topic_key(target=target, content=normalized_content),
        content=normalized_content,
        source="legacy_import",
        source_kind="explicit_user_statement" if target == "user" else "tool_observation",
        created_at=created_at,
        trust_tier=TrustTier.USER_ASSERTED if target == "user" else TrustTier.OBSERVED,
        salience_tier=SalienceTier.MEDIUM,
        status=RecordStatus.ACTIVE,
    )


def _infer_legacy_topic_key(*, target: str, content: str) -> Optional[str]:
    normalized = content.strip().lower()
    if not normalized:
        return None

    if target == "user":
        if any(phrase in normalized for phrase in ("british spelling", "uk spelling", "british english")):
            return "preference:spelling"
        if any(word in normalized for word in ("concise", "detailed", "verbose")) and any(
            word in normalized for word in ("response", "responses", "reply", "replies", "writeup", "writeups")
        ):
            return "preference:response-detail"
        if "prefer" in normalized:
            return f"preference:{_slugify(normalized.replace('user prefers', '').replace('i prefer', '').strip())}"
        return f"preference:{_slugify(normalized)}"

    if any(word in normalized for word in ("deploy", "ship", "release")):
        return "workspace:deploy-command"
    if any(word in normalized for word in ("shell", "bash", "zsh", "fish")):
        return "env:shell"
    if any(word in normalized for word in ("operating system", "ubuntu", "debian", "macos", "linux", "windows")):
        return "env:os"
    if "python" in normalized:
        return "env:python"
    return f"workspace:{_slugify(normalized)}"


def _slugify(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    if not words:
        return "entry"
    return "-".join(words[:4])


def _strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "EpisodeRecord",
    "MemoryRecord",
    "MemoryScope",
    "MemoryType",
    "RecordStatus",
    "SalienceTier",
    "TrustTier",
    "normalize_legacy_entry",
    "records_from_sidecar_payload",
    "records_to_sidecar_payload",
]
