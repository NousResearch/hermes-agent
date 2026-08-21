"""Schemas for the confidence memory provider.

The provider keeps durable memory candidates as atomic statements with a
layer, confidence level, source evidence, TTL, and injection policy.  The
shape mirrors the Notion-Harness-inspired design while remaining local and
profile-scoped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4


class Layer(str, Enum):
    PROFILE = "profile"
    ONGOING_THEME = "ongoing_theme"
    UNRESOLVED_QUESTION = "unresolved_question"


class Confidence(str, Enum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    TENTATIVE = "tentative"


class Status(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class Scope(str, Enum):
    INJECTION = "injection"
    RETRIEVAL_ONLY = "retrieval_only"


class SourceKind(str, Enum):
    USER_STATED = "user_stated"
    USER_CONFIRMED = "user_confirmed"
    DOCUMENT = "document"
    ACTIVITY_PATTERN = "activity_pattern"
    INFERENCE = "inference"


DEFAULT_TTLS = {
    Layer.PROFILE: "180d",
    Layer.ONGOING_THEME: "60d",
    Layer.UNRESOLVED_QUESTION: "14d",
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def parse_ttl(ttl: str) -> timedelta:
    if not ttl or len(ttl) < 2:
        raise ValueError("ttl must look like '14d', '60d', or '180d'")
    unit = ttl[-1]
    value = int(ttl[:-1])
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    raise ValueError("only h/d TTL units are supported")


@dataclass
class MemorySource:
    kind: SourceKind
    observed_at: datetime
    ref: str = ""
    excerpt: str = ""

    def to_json(self) -> dict:
        return {
            "kind": self.kind.value,
            "observedAt": self.observed_at.isoformat(),
            "ref": self.ref,
            "excerpt": self.excerpt,
        }

    @staticmethod
    def from_json(data: dict) -> "MemorySource":
        return MemorySource(
            kind=SourceKind(data["kind"]),
            observed_at=datetime.fromisoformat(data["observedAt"]),
            ref=data.get("ref", ""),
            excerpt=data.get("excerpt", ""),
        )


@dataclass
class MemoryItem:
    statement: str
    layer: Layer
    confidence: Confidence
    sources: list[MemorySource]
    id: str = field(default_factory=lambda: f"m_{uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=now_utc)
    last_reinforced_at: Optional[datetime] = None
    reinforcement_count: int = 0
    ttl: str = ""
    status: Status = Status.ACTIVE
    superseded_by: Optional[str] = None
    scope: Scope = Scope.INJECTION

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("statement is required")
        if not self.sources:
            raise ValueError("at least one source is required")
        if not self.ttl:
            self.ttl = DEFAULT_TTLS[self.layer]
        if self.confidence == Confidence.TENTATIVE:
            # Tentative memories are internal hints only. They must not enter
            # system prompt injection or shared-destination content.
            self.scope = Scope.RETRIEVAL_ONLY
