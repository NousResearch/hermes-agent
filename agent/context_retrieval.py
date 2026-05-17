from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Protocol, Sequence

from agent.pinecone_memory import PineconeMemoryClient


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    scope: str | None = None
    platform: str | None = None
    min_score: float = 0.35
    max_items: int = 4
    top_k: int | None = None
    source_types: tuple[str, ...] = ()
    volatile_source_types: tuple[str, ...] = ("session_summary", "artifact_summary", "ephemeral")
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class RetrievedMemorySnippet:
    id: str
    text: str
    provenance: str
    source_kind: str
    source_id: str
    source_path: str
    scope: str
    memory_type: str
    score: float
    adjusted_score: float
    canonical: bool
    updated_at: str
    freshness_hint: str
    confidence: float
    stale: bool
    metadata: dict[str, Any]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _freshness_window(hint: str) -> timedelta:
    return {
        "ephemeral": timedelta(hours=6),
        "daily": timedelta(days=2),
        "weekly": timedelta(days=10),
        "monthly": timedelta(days=45),
        "durable": timedelta(days=3650),
    }.get((hint or "").lower(), timedelta(days=30))


class ContextRetriever:
    def __init__(
        self,
        *,
        pinecone: PineconeMemoryClient,
        embed_query: Callable[[str], Sequence[float]] | QueryEmbedder,
        default_max_items: int = 4,
        default_min_score: float = 0.35,
    ) -> None:
        self.pinecone = pinecone
        self.embed_query = embed_query
        self.default_max_items = default_max_items
        self.default_min_score = default_min_score

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedMemorySnippet]:
        if not request.query.strip() or not self.pinecone.is_configured():
            return []

        vector = self._embed(request.query)
        raw_matches = self.pinecone.query(
            vector,
            top_k=request.top_k or max(request.max_items * 3, self.default_max_items * 2, 6),
            filter=self._build_filter(request),
        )
        ranked = self._rank_matches(raw_matches, request)
        ranked.sort(key=lambda item: item.adjusted_score, reverse=True)
        return ranked[: max(1, min(request.max_items, self.default_max_items if request.max_items <= 0 else request.max_items))]

    def _embed(self, query: str) -> Sequence[float]:
        if hasattr(self.embed_query, "embed_query"):
            return getattr(self.embed_query, "embed_query")(query)
        return self.embed_query(query)

    def _build_filter(self, request: RetrievalRequest) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if request.scope:
            clauses.append({"scope": {"$eq": request.scope}})
        if request.platform:
            clauses.append({"tags": {"$in": [request.platform]}})
        if request.source_types:
            clauses.append({"memory_type": {"$in": list(request.source_types)}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _rank_matches(
        self,
        matches: Iterable[dict[str, Any]],
        request: RetrievalRequest,
    ) -> list[RetrievedMemorySnippet]:
        out: list[RetrievedMemorySnippet] = []
        min_score = request.min_score if request.min_score > 0 else self.default_min_score
        max_items = request.max_items if request.max_items > 0 else self.default_max_items
        for match in matches:
            metadata = dict(match.get("metadata") or {})
            text = str(metadata.get("text") or "").strip()
            if not text:
                continue
            score = float(match.get("score") or 0.0)
            if score < min_score:
                continue

            source_kind = str(metadata.get("source_kind") or "")
            source_id = str(metadata.get("source_id") or "")
            source_path = str(metadata.get("source_path") or "")
            memory_type = str(metadata.get("memory_type") or "")
            scope = str(metadata.get("scope") or "")
            updated_at = str(metadata.get("updated_at") or metadata.get("created_at") or "")
            freshness_hint = str(metadata.get("freshness_hint") or "")
            confidence = float(metadata.get("confidence") or 0.0)
            canonical = bool(metadata.get("canonical", False))
            header_path = metadata.get("header_path") or []
            header_suffix = " / ".join(str(part) for part in header_path if part)
            provenance = source_path or source_id or source_kind or "unknown"
            if header_suffix:
                provenance = f"{provenance} ({header_suffix})"

            adjusted = score
            stale = False
            updated_dt = _parse_dt(updated_at)
            if updated_dt is not None:
                age = max(request.now - updated_dt, timedelta())
                window = _freshness_window(freshness_hint)
                if age > window:
                    stale = True
                    if memory_type in request.volatile_source_types:
                        continue
                    adjusted -= 0.30
                elif age > window / 2:
                    adjusted -= 0.08

            if canonical:
                adjusted += 0.12
            else:
                adjusted -= 0.05
            if source_kind == "file":
                adjusted += 0.08
            elif source_kind in {"session_summary", "artifact_summary"}:
                adjusted -= 0.06
            if memory_type == "profile":
                adjusted += 0.20
            elif memory_type in {"session_summary", "artifact_summary", "ephemeral"}:
                adjusted -= 0.04
            adjusted += max(min(confidence, 1.0), 0.0) * 0.05

            if adjusted < min_score:
                continue
            out.append(
                RetrievedMemorySnippet(
                    id=str(match.get("id") or ""),
                    text=text,
                    provenance=provenance,
                    source_kind=source_kind,
                    source_id=source_id,
                    source_path=source_path,
                    scope=scope,
                    memory_type=memory_type,
                    score=score,
                    adjusted_score=adjusted,
                    canonical=canonical,
                    updated_at=updated_at,
                    freshness_hint=freshness_hint,
                    confidence=confidence,
                    stale=stale,
                    metadata=metadata,
                )
            )
        out.sort(key=lambda item: item.adjusted_score, reverse=True)
        return out[:max_items]


__all__ = [
    "ContextRetriever",
    "RetrievedMemorySnippet",
    "RetrievalRequest",
]
