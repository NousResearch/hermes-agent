"""Provenance for memory search results.

EVERY search result MUST carry enough provenance to answer: *where did this
come from, via which layer, by what method, when, and why was it returned?*
This makes the index a transparent, auditable cache rather than an opaque
black box. No result leaves the Router or the index without a full
:class:`SearchResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class SearchResult:
    """A single memory search hit with full provenance.

    Attributes:
        source_file: path to the originating markdown (absolute, or relative
            to HERMES_HOME). Never empty.
        memory_layer: which layer produced it, e.g. ``"L1-identity"``,
            ``"L5-index"``. Never empty.
        retrieval_method: how it was found, e.g. ``"fts5"``, ``"sqlite-like"``,
            ``"direct-file"``. Never empty.
        content: the raw matched chunk/text. NOT a summary.
        timestamp: file mtime (ISO-8601 string) or None.
        snippet: optional short excerpt (<=200 chars).
        score: optional relevance score.
        intent: the intent that produced this result.
        capability: the capability name that served it.
        extra: free-form metadata dict.
    """

    source_file: str
    memory_layer: str
    retrieval_method: str
    content: str = ""
    timestamp: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    intent: Optional[str] = None
    capability: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Strict provenance invariants — every result must answer the
        # where/which/how questions, even when a search returns nothing useful.
        if not self.source_file:
            raise ValueError("SearchResult.source_file must not be empty")
        if not self.memory_layer:
            raise ValueError("SearchResult.memory_layer must not be empty")
        if not self.retrieval_method:
            raise ValueError("SearchResult.retrieval_method must not be empty")
        if self.snippet is None and self.content:
            snippet = self.content[:200]
            self.snippet = snippet


def format_result(r: SearchResult) -> dict[str, Any]:
    """Serialize a :class:`SearchResult` to a plain JSON-friendly dict."""
    return asdict(r)


def format_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Serialize a list of results.

    Guarantees provenance fields are present on every item.
    """
    out: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, SearchResult):
            raise TypeError(f"Expected SearchResult, got {type(r).__name__}")
        out.append(format_result(r))
    return out
