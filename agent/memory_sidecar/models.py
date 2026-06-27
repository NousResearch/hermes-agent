from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

OBSERVATION_TYPES: tuple[str, ...] = (
    "decision",
    "change",
    "bug",
    "how_it_works",
    "next_step",
)

PRIVACY_STATES: tuple[str, ...] = (
    "public",
    "redacted",
    "skipped",
)


def validate_observation_type(value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned not in OBSERVATION_TYPES:
        allowed = ", ".join(OBSERVATION_TYPES)
        raise ValueError(f"unknown observation_type '{cleaned}' (allowed: {allowed})")
    return cleaned


def validate_privacy_status(value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned not in PRIVACY_STATES:
        allowed = ", ".join(PRIVACY_STATES)
        raise ValueError(f"unknown privacy_status '{cleaned}' (allowed: {allowed})")
    return cleaned


@dataclass(frozen=True)
class IngestSource:
    source_id: str
    path: str
    last_offset: int = 0
    partial_line: str = ""
    last_event_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise ValueError("source_id must not be empty")
        if not str(self.path).strip():
            raise ValueError("path must not be empty")
        if int(self.last_offset) < 0:
            raise ValueError("last_offset must be >= 0")


@dataclass(frozen=True)
class ObservationFile:
    file_path: str
    change_kind: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.file_path).strip():
            raise ValueError("file_path must not be empty")


@dataclass(frozen=True)
class Observation:
    session_id: str
    event_ts: str
    observation_type: str
    title: str
    summary: str
    detail: str
    concepts: tuple[str, ...] = ()
    files: tuple[ObservationFile, ...] = ()
    privacy_status: str = "public"
    confidence: float = 0.5
    role: Optional[str] = None
    message_id: Optional[str] = None
    created_at: Optional[str] = None
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("session_id must not be empty")
        if not str(self.event_ts).strip():
            raise ValueError("event_ts must not be empty")
        if not str(self.title).strip():
            raise ValueError("title must not be empty")
        if not str(self.summary).strip():
            raise ValueError("summary must not be empty")
        if not str(self.detail).strip():
            raise ValueError("detail must not be empty")
        validate_observation_type(self.observation_type)
        validate_privacy_status(self.privacy_status)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class SessionFact:
    session_id: str
    last_seen_at: str
    user_goal: Optional[str] = None
    latest_summary: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("session_id must not be empty")
        if not str(self.last_seen_at).strip():
            raise ValueError("last_seen_at must not be empty")


@dataclass(frozen=True)
class ContextQuery:
    query: Optional[str] = None
    session_id: Optional[str] = None
    types: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    file_path: Optional[str] = None
    limit: int = 8
    time_bias: str = "recent"

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
        if self.time_bias not in {"recent", "relevant"}:
            raise ValueError("time_bias must be 'recent' or 'relevant'")
        for kind in self.types:
            validate_observation_type(kind)


@dataclass(frozen=True)
class ContextResult:
    observations: tuple[Observation, ...] = ()
    decisions: tuple[Observation, ...] = ()
    changed_files: tuple[str, ...] = ()
    session_fact: Optional[SessionFact] = None
    suggested_follow_ups: tuple[str, ...] = field(default_factory=tuple)
