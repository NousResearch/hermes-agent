"""Structured note schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping

from pydantic import BaseModel, Field, ValidationError, field_validator

from llmwiki_hermes.constants import CURRENT_SCHEMA_VERSION
from llmwiki_hermes.errors import InvalidFrontmatterError
from llmwiki_hermes.types import NoteKind


class NoteFrontmatterBase(BaseModel):
    """Fields shared by all note types."""

    schema_version: int = CURRENT_SCHEMA_VERSION
    id: str
    kind: NoteKind
    title: str
    source_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id", "title")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be empty.")
        return value.strip()

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value < 0:
            raise ValueError("schema_version must be zero or greater.")
        return value


class SourceNoteFrontmatter(NoteFrontmatterBase):
    """Frontmatter for source notes."""

    kind: NoteKind = NoteKind.SOURCE
    source_type: str = "text"
    origin: str = "user_upload"
    ingested_at: datetime
    content_hash: str


class SemanticNoteFrontmatter(NoteFrontmatterBase):
    """Frontmatter for semantic notes."""

    kind: NoteKind = NoteKind.SEMANTIC
    aliases: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    updated_at: date
    confidence: str = "medium"


class EpisodicNoteFrontmatter(NoteFrontmatterBase):
    """Frontmatter for episodic notes."""

    kind: NoteKind = NoteKind.EPISODIC
    date: date
    participants: list[str] = Field(default_factory=list)
    project: str | None = None
    entity_refs: list[str] = Field(default_factory=list)


class NoteDocument(BaseModel):
    """In-memory representation of a Markdown note."""

    frontmatter: dict[str, Any]
    body: str
    path: str


FRONTMATTER_MODELS: dict[str, type[NoteFrontmatterBase]] = {
    NoteKind.SOURCE.value: SourceNoteFrontmatter,
    NoteKind.SEMANTIC.value: SemanticNoteFrontmatter,
    NoteKind.EPISODIC.value: EpisodicNoteFrontmatter,
}


def parse_schema_version(frontmatter: Mapping[str, Any]) -> int | None:
    """Parse a note schema version when present and integer-like."""

    raw_value = frontmatter.get("schema_version")
    if raw_value is None or not str(raw_value).strip():
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def is_unsupported_schema(frontmatter: Mapping[str, Any]) -> bool:
    """Return whether a note uses a schema version newer than the current code supports."""

    parsed = parse_schema_version(frontmatter)
    return parsed is not None and parsed > CURRENT_SCHEMA_VERSION


def validate_note_frontmatter(frontmatter: dict[str, Any]) -> NoteFrontmatterBase:
    """Validate frontmatter based on the note kind."""

    raw_kind = frontmatter.get("kind")
    kind = raw_kind.value if isinstance(raw_kind, NoteKind) else str(raw_kind or "").strip()
    model_cls = FRONTMATTER_MODELS.get(kind)
    if model_cls is None:
        raise InvalidFrontmatterError(f"Unsupported note kind: {raw_kind!r}")
    try:
        return model_cls.model_validate(frontmatter)
    except ValidationError as exc:
        raise InvalidFrontmatterError(str(exc)) from exc
