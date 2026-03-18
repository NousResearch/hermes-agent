"""
Embedding provider for cognitive memory.

API-first approach using litellm (already a project dependency).
Supports any provider litellm supports: OpenAI, Cohere, Azure, etc.
No extra dependencies required.
"""

import logging
import math
from typing import List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Default embedding model (OpenAI, cheap and fast)
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings produced."""
        ...

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single call."""
        ...


class LiteLLMEmbedder:
    """
    Embedding provider using litellm.

    Uses the same API key infrastructure the agent already has configured.
    Supports OpenAI, Cohere, Azure, and any litellm-compatible provider.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._dimensions = dimensions or DEFAULT_DIMENSIONS
        self._initialized = False

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _ensure_init(self):
        if not self._initialized:
            try:
                import litellm
                self._litellm = litellm
                self._initialized = True
            except ImportError:
                raise RuntimeError("litellm is required for embeddings but not installed")

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string using litellm."""
        self._ensure_init()
        try:
            kwargs = {
                "model": self._model,
                "input": [text],
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base

            response = self._litellm.embedding(**kwargs)
            embedding = response.data[0]["embedding"]

            # Update dimensions from actual response
            if len(embedding) != self._dimensions:
                self._dimensions = len(embedding)

            return embedding

        except Exception as e:
            logger.warning("Embedding failed for text (len=%d): %s", len(text), e)
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single API call."""
        if not texts:
            return []

        self._ensure_init()
        try:
            kwargs = {
                "model": self._model,
                "input": texts,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base

            response = self._litellm.embedding(**kwargs)

            embeddings = [item["embedding"] for item in response.data]

            if embeddings and len(embeddings[0]) != self._dimensions:
                self._dimensions = len(embeddings[0])

            return embeddings

        except Exception as e:
            logger.warning("Batch embedding failed for %d texts: %s", len(texts), e)
            raise


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0, where 1.0 means identical.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def get_embedder(config: Optional[dict] = None) -> LiteLLMEmbedder:
    """
    Factory function to create an embedder from config.

    Config format (from config.yaml):
        memory:
          embedding:
            provider: "litellm"
            model: "text-embedding-3-small"
            api_key: null  # uses env var if not set
            api_base: null
    """
    if config is None:
        config = {}

    embedding_config = config.get("embedding", {})

    return LiteLLMEmbedder(
        model=embedding_config.get("model", DEFAULT_EMBEDDING_MODEL),
        api_key=embedding_config.get("api_key"),
        api_base=embedding_config.get("api_base"),
        dimensions=embedding_config.get("dimensions"),
    )
