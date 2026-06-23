"""Embedding provider abstraction — Strategy pattern for vector generation.

Provides a common interface for generating text embeddings with swappable
backends: Ollama (local, free) and OpenAI (API).

Usage:
    from agent.embedding_provider import resolve_embedding_provider

    config = {"provider": "ollama", "model": "nomic-embed-text", ...}
    provider = resolve_embedding_provider(config)
    vector = provider.generate_embedding("some text")
"""

from abc import ABC, abstractmethod
from typing import Optional


class EmbeddingProvider(ABC):
    """Abstract base for embedding generation strategies."""

    @abstractmethod
    def generate_embedding(self, text: str) -> "list[float]":
        """Generate a vector embedding for the given text."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the generated vectors."""


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings via Ollama's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        dimensions: int = 768,
    ):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._dimensions = dimensions
        self._client = None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_client(self):
        """Lazy-init the OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    def generate_embedding(self, text: str) -> "list[float]":
        if not text or not text.strip():
            return [0.0] * self._dimensions

        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=text,
        )
        return list(response.data[0].embedding)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings via OpenAI's embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dimensions: int = 1536,
    ):
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._client = None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_client(self):
        """Lazy-init the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def generate_embedding(self, text: str) -> "list[float]":
        if not text or not text.strip():
            return [0.0] * self._dimensions

        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=text,
        )
        return list(response.data[0].embedding)


def resolve_embedding_provider(config: Optional[dict]) -> Optional[EmbeddingProvider]:
    """Factory: select the right embedding strategy from config.

    Args:
        config: Dict with keys provider, model, base_url, api_key, dimensions.
                None or empty dict returns None.

    Returns:
        An EmbeddingProvider instance, or None if the provider is unrecognized
        or config is empty.
    """
    if not config:
        return None

    provider_type = config.get("provider", "").lower()
    model = config.get("model", "")
    base_url = config.get("base_url", "")
    api_key = config.get("api_key", "")
    dimensions = config.get("dimensions", 768)

    if provider_type == "ollama":
        return OllamaEmbeddingProvider(
            model=model or "nomic-embed-text",
            base_url=base_url or "http://localhost:11434/v1",
            api_key=api_key or "ollama",
            dimensions=dimensions or 768,
        )
    elif provider_type == "openai":
        return OpenAIEmbeddingProvider(
            model=model or "text-embedding-3-small",
            api_key=api_key,
            dimensions=dimensions or 1536,
        )

    return None
