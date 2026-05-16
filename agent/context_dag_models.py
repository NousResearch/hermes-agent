"""Data models for the DAG-backed context store.

PR1 intentionally keeps these as lightweight persistence models only. Runtime
assembly/compaction behavior is introduced in later PRs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RawSpan:
    """Canonical raw transcript span backed by ``messages.id`` values."""

    start_message_id: int
    end_message_id: int
    source_type: str = "message_span"
    source_id: str = ""
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SummaryNode:
    id: str
    session_id: str
    kind: str
    summary_text: str
    status: str
    source_hash: Optional[str] = None
    summary_hash: Optional[str] = None
    prompt_version: Optional[str] = None
    summary_model: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    token_estimate: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SummarySource:
    summary_id: str
    source_type: str
    source_id: str = ""
    start_message_id: Optional[int] = None
    end_message_id: Optional[int] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    created_at: Optional[float] = None


@dataclass(frozen=True)
class ProjectionItem:
    kind: str
    payload: Dict[str, Any]
    token_estimate: Optional[int] = None


@dataclass(frozen=True)
class Projection:
    session_id: str
    engine_version: str
    status: str
    projection: List[Dict[str, Any]]
    fresh_tail_start_message_id: Optional[int] = None
    latest_raw_message_id: Optional[int] = None
    token_estimate: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


@dataclass(frozen=True)
class Checkpoint:
    session_id: str
    last_ingested_message_id: Optional[int] = None
    last_projection_message_id: Optional[int] = None
    last_anchor_message_id: Optional[int] = None
    anchor_hash: Optional[str] = None
    updated_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MutationLogEntry:
    id: int
    session_id: str
    operation: str
    status: str
    idempotency_key: str
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssemblyBudget:
    max_tokens: int
    fresh_tail_min_tokens: int = 0
    summary_max_tokens: Optional[int] = None
