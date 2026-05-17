from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Literal, Sequence

MemoryType = Literal[
    "profile",
    "project_context",
    "session_summary",
    "artifact_summary",
    "ephemeral",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _slugify_header(header: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", header.lower()).strip("-")
    return slug or "root"


def _iso8601(value: datetime | str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


@dataclass(frozen=True)
class MemorySourceRef:
    source_kind: str
    source_id: str
    source_path: str = ""


@dataclass(frozen=True)
class MemoryChunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    header_path: tuple[str, ...]
    memory_type: MemoryType
    scope: str
    source_kind: str
    source_id: str
    source_path: str
    created_at: str
    updated_at: str
    freshness_hint: str
    confidence: float
    tags: tuple[str, ...] = field(default_factory=tuple)
    canonical: bool = True

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("MemoryChunk.text must be non-empty")
        if not self.source_kind:
            raise ValueError("MemoryChunk.source_kind is required")
        if not self.source_id:
            raise ValueError("MemoryChunk.source_id is required")
        if not self.scope:
            raise ValueError("MemoryChunk.scope is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("MemoryChunk.confidence must be between 0 and 1")

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "memory_type": self.memory_type,
            "scope": self.scope,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "freshness_hint": self.freshness_hint,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "canonical": self.canonical,
            "header_path": list(self.header_path),
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
        }


@dataclass(frozen=True)
class MemoryDocument:
    memory_type: MemoryType
    scope: str
    source: MemorySourceRef
    text: str
    created_at: str | datetime | None = None
    updated_at: str | datetime | None = None
    freshness_hint: str = "durable"
    confidence: float = 1.0
    tags: tuple[str, ...] = field(default_factory=tuple)
    canonical: bool = True
    title: str = ""

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValueError("MemoryDocument.scope is required")
        if not self.text.strip():
            raise ValueError("MemoryDocument.text must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("MemoryDocument.confidence must be between 0 and 1")
        object.__setattr__(self, "created_at", _iso8601(self.created_at))
        object.__setattr__(self, "updated_at", _iso8601(self.updated_at or self.created_at))
        object.__setattr__(self, "text", self.text.strip())

    @property
    def document_id(self) -> str:
        payload = "|".join(
            [
                self.memory_type,
                self.scope,
                self.source.source_kind,
                self.source.source_id,
                self.source.source_path,
                self.title,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def chunk(self, *, target_chars: int = 900, max_chars: int = 1200) -> list[MemoryChunk]:
        if target_chars <= 0 or max_chars <= 0 or target_chars > max_chars:
            raise ValueError("target_chars and max_chars must be positive and target_chars <= max_chars")
        sections = _split_sections(self.text)
        chunks: list[MemoryChunk] = []
        for header_path, section_text in sections:
            parts = _chunk_section_text(section_text, target_chars=target_chars, max_chars=max_chars)
            for part in parts:
                chunk_index = len(chunks)
                stable_id = _stable_chunk_id(
                    document_id=self.document_id,
                    chunk_index=chunk_index,
                    header_path=header_path,
                    text=part,
                )
                chunks.append(
                    MemoryChunk(
                        id=stable_id,
                        document_id=self.document_id,
                        chunk_index=chunk_index,
                        text=part,
                        header_path=header_path,
                        memory_type=self.memory_type,
                        scope=self.scope,
                        source_kind=self.source.source_kind,
                        source_id=self.source.source_id,
                        source_path=self.source.source_path,
                        created_at=self.created_at,
                        updated_at=self.updated_at,
                        freshness_hint=self.freshness_hint,
                        confidence=self.confidence,
                        tags=tuple(self.tags),
                        canonical=self.canonical,
                    )
                )
        return chunks


def _split_sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    sections: list[tuple[tuple[str, ...], str]] = []
    header_stack: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((tuple(header_stack), body))
        buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            header_stack[:] = header_stack[: level - 1]
            header_stack.append(title)
            continue
        buffer.append(line)
    flush()
    if not sections:
        sections.append((tuple(), text.strip()))
    return sections


def _chunk_section_text(text: str, *, target_chars: int, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            if len(current) >= target_chars:
                chunks.append(current)
                current = ""
            continue
        if current:
            chunks.append(current)
            current = ""
        chunks.extend(_split_long_paragraph(paragraph, max_chars=max_chars))
    if current:
        chunks.append(current)
    return chunks


def _split_long_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(sentence) > max_chars:
            split_at = sentence.rfind(" ", 0, max_chars)
            if split_at <= 0:
                split_at = max_chars
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        current = sentence
    if current:
        chunks.append(current)
    return chunks


def _stable_chunk_id(*, document_id: str, chunk_index: int, header_path: Sequence[str], text: str) -> str:
    fingerprint = "|".join(
        [document_id, str(chunk_index), "/".join(_slugify_header(h) for h in header_path), _normalize_text(text)]
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]


__all__ = [
    "MemoryChunk",
    "MemoryDocument",
    "MemorySourceRef",
]
