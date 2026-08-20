"""Provider registry — the single place a new backend is wired in (Step 9)."""
from __future__ import annotations

from typing import Any, Callable, Dict

from ..provider import KnowledgeProvider
from .anythingllm_provider import AnythingLLMProvider, AnythingLLMError
from .future_providers import (
    ChromaProvider,
    PgVectorProvider,
    QdrantProvider,
    WeaviateProvider,
)
from .local_provider import LocalProvider

PROVIDER_REGISTRY: Dict[str, Callable[..., KnowledgeProvider]] = {
    "local": LocalProvider,
    "anythingllm": AnythingLLMProvider,
    "qdrant": QdrantProvider,
    "weaviate": WeaviateProvider,
    "chroma": ChromaProvider,
    "pgvector": PgVectorProvider,
}


def register_provider(name: str, factory: Callable[..., KnowledgeProvider]) -> None:
    """Third-party/plugin backends register here — no Hermes changes needed."""
    PROVIDER_REGISTRY[name] = factory


def build_provider(name: str, **kwargs: Any) -> KnowledgeProvider:
    try:
        factory = PROVIDER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown knowledge provider {name!r}; "
            f"available: {sorted(PROVIDER_REGISTRY)}"
        ) from None
    return factory(**kwargs)


__all__ = [
    "PROVIDER_REGISTRY", "register_provider", "build_provider",
    "LocalProvider", "AnythingLLMProvider", "AnythingLLMError",
    "QdrantProvider", "WeaviateProvider", "ChromaProvider", "PgVectorProvider",
]
