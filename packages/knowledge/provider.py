"""KnowledgeProvider — the only contract Hermes depends on (Step 2).

Every retrieval backend (AnythingLLM, Qdrant, pgvector, Chroma, Weaviate,
local) implements this interface. Hermes' reasoning engine never imports a
concrete provider; it talks to KnowledgeService, which talks to this ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .types import Document, HealthStatus, IndexResult, SearchResult


class KnowledgeProvider(ABC):
    """Abstract retrieval backend."""

    name: str = "abstract"

    # -- read ----------------------------------------------------------
    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 5,
        workspace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Semantic search: return scored chunks with citations, no answer."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 5,
        workspace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Retrieval-augmented answer: chunks + a provider-synthesised answer."""

    # -- write ---------------------------------------------------------
    @abstractmethod
    def index(self, document: Document) -> IndexResult: ...

    @abstractmethod
    def update(self, document: Document) -> IndexResult: ...

    @abstractmethod
    def delete(self, document_id: str, workspace: Optional[str] = None) -> IndexResult: ...

    # -- ops -----------------------------------------------------------
    @abstractmethod
    def health(self) -> HealthStatus: ...

    # optional capability, default derived from search
    def find_similar(
        self, document_id: str, limit: int = 5, workspace: Optional[str] = None
    ) -> SearchResult:
        return self.search(document_id, limit=limit, workspace=workspace)

    def list_documents(self, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
        return []
