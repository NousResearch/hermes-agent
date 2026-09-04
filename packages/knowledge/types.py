"""Core data types for the Hermes Knowledge Retrieval subsystem.

Pure data. No I/O, no reasoning, no provider specifics.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _now() -> float:
    return time.time()


@dataclass
class Document:
    """A source document to be indexed."""

    id: str
    title: str
    content: str
    path: str = ""
    source: str = ""            # obsidian | git | mkdocs | markdown | pdf | conversation
    workspace: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    mtime: float = field(default_factory=_now)

    @property
    def checksum(self) -> str:
        h = hashlib.sha256()
        h.update(self.title.encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update(self.content.encode("utf-8", "replace"))
        return h.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checksum"] = self.checksum
        return d


@dataclass
class Citation:
    """Source attribution for a retrieved chunk (Step 8)."""

    title: str
    file: str
    path: str
    score: float
    workspace: str
    chunk_id: str
    document_id: str = ""
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        return f"[{self.title}]({self.path}) · score={self.score:.3f} · chunk={self.chunk_id}"


@dataclass
class Chunk:
    """A retrieved passage plus its attribution."""

    id: str
    text: str
    score: float
    document_id: str = ""
    citation: Optional[Citation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "document_id": self.document_id,
            "citation": self.citation.to_dict() if self.citation else None,
        }


@dataclass
class SearchResult:
    """Result of a semantic search / retrieval call."""

    query: str
    chunks: List[Chunk] = field(default_factory=list)
    answer: str = ""
    provider: str = ""
    workspace: str = "default"
    elapsed_ms: float = 0.0
    cached: bool = False
    confidence: float = 0.0
    error: str = ""

    @property
    def sources(self) -> List[Citation]:
        return [c.citation for c in self.chunks if c.citation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "chunks": [c.to_dict() for c in self.chunks],
            "confidence": round(self.confidence, 4),
            "provider": self.provider,
            "workspace": self.workspace,
            "elapsedTime": round(self.elapsed_ms, 2),
            "cached": self.cached,
            "error": self.error,
        }


@dataclass
class HealthStatus:
    healthy: bool
    provider: str
    detail: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexResult:
    ok: bool
    document_id: str
    action: str = ""     # indexed | updated | deleted | skipped
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SyncReport:
    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["elapsed_ms"] = round(self.elapsed_ms, 2)
        d["counts"] = {
            "added": len(self.added),
            "updated": len(self.updated),
            "deleted": len(self.deleted),
            "unchanged": len(self.unchanged),
            "failed": len(self.failed),
        }
        return d
