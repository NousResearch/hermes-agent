"""Data models for the Idea Box subsystem.

All contracts defined in the architecture blueprint (Section 3).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    URL = "url"
    ARTICLE = "article"
    GITHUB_REPO = "github_repo"
    UNKNOWN = "unknown"


class Category(str, Enum):
    TECH = "tech"
    MARKET = "market"
    CONTENT = "content"
    SECURITY = "security"
    DESIGN = "design"
    OPERATIONS = "operations"
    OTHER = "other"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Effort(str, Enum):
    XS = "xs"
    S = "s"
    M = "m"
    L = "l"
    XL = "xl"


class Recommendation(str, Enum):
    PROCEED = "proceed"
    REJECT = "reject"
    AMEND = "amend"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AMENDED = "amended"


class AuditEventType(str, Enum):
    INTAKE = "intake"
    PARSE = "parse"
    TRIAGE = "triage"
    APPROVE = "approve"
    REJECT = "reject"
    AMEND = "amend"
    TASK_CREATED = "task_created"
    ERROR = "error"


@dataclass
class Provenance:
    submitted_by: str       # Discord user ID
    submitted_at: int       # Unix epoch
    channel_id: str
    message_id: str
    guild_id: str


@dataclass
class Source:
    url: Optional[str]
    source_type: SourceType
    title: Optional[str]
    author: Optional[str]
    published_date: Optional[str]
    content_snippet: str
    content_hash: str
    url_fingerprint: str
    raw_text: str
    provenance: Provenance


@dataclass
class ParseResult:
    source: Optional[Source]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DedupResult:
    is_duplicate: bool = False
    existing_task_id: Optional[str] = None
    existing_triage_id: Optional[str] = None


@dataclass
class Risk:
    category: str       # "security" | "legal" | "dependency" | "cost" | "scope" | "unknown"
    severity: str       # "critical" | "high" | "medium" | "low"
    description: str


@dataclass
class Classification:
    category: str
    subcategory: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    specialist: str
    confidence: float = 1.0
    rationale: str = ""


@dataclass
class TriageSummary:
    triage_id: str
    source: Source
    classification: Classification
    confidence: str
    risks: list[Risk]
    effort: str
    recommendation: str
    routing: RoutingDecision
    reasoning: str
    created_at: int


@dataclass
class ApprovalAction:
    triage_id: str
    action: str          # "approve" | "reject" | "amend"
    actor_id: str
    actor_name: str
    timestamp: int
    reason: Optional[str] = None
    kanban_task_id: Optional[str] = None


@dataclass
class ApprovalState:
    triage_id: str
    status: str
    source: Source
    triage_summary: TriageSummary
    created_at: int
    resolved_at: Optional[int] = None
    resolved_by: Optional[str] = None
    resolution_reason: Optional[str] = None
    kanban_task_id: Optional[str] = None


@dataclass
class AuditEvent:
    event_id: str
    event_type: str
    triage_id: str
    timestamp: int
    actor_id: Optional[str] = None
    payload: dict = field(default_factory=dict)


@dataclass
class SourceSubmission:
    raw_text: str
    author_id: str
    channel_id: str
    message_id: str
    guild_id: str
    channel_type: str   # "forum" | "text"
    timestamp: int


def generate_triage_id() -> str:
    """Generate a unique triage ID with t_ prefix."""
    return f"t_{uuid.uuid4().hex[:12]}"


def generate_event_id() -> str:
    """Generate a unique audit event ID."""
    return f"evt_{uuid.uuid4().hex[:16]}"
