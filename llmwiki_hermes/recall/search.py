"""Recall service built on the SQLite sidecar index."""

from __future__ import annotations

import json
import logging
import re

from llmwiki_hermes.recall.format import format_recall_block
from llmwiki_hermes.recall.rank import score_row
from llmwiki_hermes.schemas.cli import CommandOutput, RecallHit, RecallResponse
from llmwiki_hermes.settings import WikiSettings
from llmwiki_hermes.storage.sqlite_index import IndexService
from llmwiki_hermes.storage.vault import VaultService

logger = logging.getLogger(__name__)

ENGLISH_STOPWORDS = {
    "a",
    "an",
    "are",
    "be",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "please",
    "tell",
    "the",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "you",
}


def normalize_query(query: str) -> str:
    """Normalize natural-language input into an FTS-safe query."""

    tokens = re.findall(r"\w+(?:[-']\w+)*", query, flags=re.UNICODE)
    filtered = [
        token for token in tokens if not (token.isascii() and token.lower() in ENGLISH_STOPWORDS)
    ]
    normalized = filtered or tokens
    return " ".join(normalized).strip()


class RecallService:
    """Search, rank, and format knowledge recall."""

    def __init__(self, settings: WikiSettings) -> None:
        self.settings = settings
        self.vault_service = VaultService(settings.vault_path)
        self.index_service = IndexService(self.vault_service)

    @classmethod
    def from_settings(cls, settings: WikiSettings) -> "RecallService":
        return cls(settings)

    def recall(self, query: str, memory_type: str, top_k: int) -> RecallResponse:
        """Search and rank notes."""

        normalized = normalize_query(query)
        if not normalized:
            logger.warning("Recall query normalized to empty input.")
            return RecallResponse(
                query="",
                memory_type=memory_type,
                results=[],
                recall_block=format_recall_block([]),
            )
        kind_filter = None if memory_type == "auto" else memory_type
        try:
            rows = self.index_service.search(query=normalized, kind=kind_filter, top_k=top_k)
        except Exception:
            logger.exception(
                "Recall search failed for query %r with memory_type=%s.",
                normalized,
                memory_type,
            )
            raise

        best_by_note: dict[str, RecallHit] = {}
        for row in rows:
            hit = RecallHit(
                id=row["id"],
                title=row["title"],
                kind=row["kind"],
                path=row["path"],
                snippet=(
                    (row.get("snippet") or row.get("heading") or "").strip()[:280] or row["title"]
                ),
                score=score_row(row, normalized, memory_type),
                source_refs=json.loads(row["source_refs_json"] or "[]"),
            )
            existing = best_by_note.get(hit.id)
            if existing is None or hit.score > existing.score:
                best_by_note[hit.id] = hit
        ranked = sorted(best_by_note.values(), key=lambda item: item.score, reverse=True)[:top_k]
        logger.debug(
            "Recall returned %s note(s) for query %r with memory_type=%s.",
            len(ranked),
            normalized,
            memory_type,
        )
        return RecallResponse(
            query=normalized,
            memory_type=memory_type,
            results=ranked,
            recall_block=format_recall_block(ranked),
        )

    def recall_cli(self, query: str, memory_type: str, top_k: int) -> CommandOutput:
        response = self.recall(query=query, memory_type=memory_type, top_k=top_k)
        return CommandOutput(
            message=response.recall_block,
            data=response.model_dump(mode="json"),
        )
