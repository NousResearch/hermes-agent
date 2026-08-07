"""packages.knowledge — Hermes Knowledge Retrieval (RAG) subsystem.

Layering (Hermes never imports a concrete provider):

    Hermes reasoning loop / knowledge_search tool
                    |
              KnowledgeService          <- policy: cache, retry, rerank, merge
                    |
             KnowledgeProvider (ABC)    <- the only contract
                    |
    LocalProvider | AnythingLLMProvider | Qdrant | Weaviate | Chroma | pgvector

This module performs retrieval, indexing, sync, search, embeddings and
citation assembly. It performs no reasoning.
"""
from .cache import TTLCache
from .config import KnowledgeConfig
from .provider import KnowledgeProvider
from .providers import (
    AnythingLLMProvider,
    ChromaProvider,
    LocalProvider,
    PgVectorProvider,
    QdrantProvider,
    WeaviateProvider,
    build_provider,
    register_provider,
)
from .service import KnowledgeService, get_knowledge_service
from .sync import DocumentSynchronizer, walk_source, read_document
from .worker import KnowledgeSyncWorker, build_worker_from_config, start_health_server
from .types import (
    Chunk,
    Citation,
    Document,
    HealthStatus,
    IndexResult,
    SearchResult,
    SyncReport,
)

__version__ = "1.0.0"

__all__ = [
    "KnowledgeProvider", "KnowledgeService", "get_knowledge_service",
    "KnowledgeConfig", "TTLCache", "DocumentSynchronizer",
    "walk_source", "read_document",
    "KnowledgeSyncWorker", "build_worker_from_config", "start_health_server",
    "build_provider", "register_provider",
    "LocalProvider", "AnythingLLMProvider", "QdrantProvider",
    "WeaviateProvider", "ChromaProvider", "PgVectorProvider",
    "Document", "Chunk", "Citation", "SearchResult", "HealthStatus",
    "IndexResult", "SyncReport",
]
