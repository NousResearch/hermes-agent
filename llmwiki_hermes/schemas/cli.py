"""Schemas used by CLI and provider responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llmwiki_hermes.types import NoteKind


class CommandOutput(BaseModel):
    """Standard CLI payload."""

    ok: bool = True
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class RecallHit(BaseModel):
    """Single recall result."""

    id: str
    title: str
    kind: NoteKind
    path: str
    snippet: str
    score: float
    source_refs: list[str] = Field(default_factory=list)


class RecallResponse(BaseModel):
    """Recall response payload."""

    query: str
    memory_type: str
    results: list[RecallHit]
    recall_block: str
