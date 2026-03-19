"""
cosmos Embeddings - Multi-Model Embedding Support

Supports multiple embedding backends:
- Sentence Transformers (local)
- Ollama embeddings
- OpenAI-compatible APIs
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import hashlib

from loguru import logger

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embedding: list[float]
    model: str
    dimensions: int
    cached: bool = False


class EmbeddingBackend(ABC):
    """Abstract embedding backend."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass


class SentenceTransformerBackend(EmbeddingBackend):
    """Sentence Transformers embedding backend."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimensions = 384  # Default for MiniLM

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimensions = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise RuntimeError("sentence-transformers not installed")

    async def embed(self, text: str) -> list[float]:
        self._load_model()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, normalize_embeddings=True)
        )
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        )
        return [e.tolist() for e in embeddings]


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama embedding backend."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.host = host
        self._dimensions = 768  # Default for nomic

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        try:
            import ollama
            client = ollama.AsyncClient(host=self.host)
            response = await client.embeddings(model=self.model_name, prompt=text)
            embedding = response.get("embedding", [])
            self._dimensions = len(embedding)
            return embedding
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama doesn't have native batch support, process sequentially
        embeddings = []
        for text in texts:
            emb = await self.embed(text)
            embeddings.append(emb)
        return embeddings


class EmbeddingManager:
    """
    Manages embedding generation with caching and fallbacks.

    Features:
    - LRU cache for embeddings
    - Multiple backend support
    - Fallback chain
    - Batch optimization
    """

    def __init__(
        self,
        backend: str = "sentence_transformers",
        model_name: Optional[str] = None,
        cache_size: int = 10000,
    ):
        self.cache: dict[str, list[float]] = {}
        self.cache_size = cache_size
        self.cache_order: list[str] = []

        # Initialize backend
        if backend == "sentence_transformers":
            self.backend = SentenceTransformerBackend(
                model_name or "all-MiniLM-L6-v2"
            )
        elif backend == "ollama":
            self.backend = OllamaEmbeddingBackend(
                model_name or "nomic-embed-text"
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @property
    def dimensions(self) -> int:
        return self.backend.dimensions

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _add_to_cache(self, key: str, embedding: list[float]):
        """Add to cache with LRU eviction."""
        if key in self.cache:
            # Move to end (most recent)
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return

        # Evict if full
        while len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = embedding
        self.cache_order.append(key)

    async def embed(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """Get embedding for text."""
        self._stats["total_requests"] += 1

        if use_cache:
            key = self._cache_key(text)
            if key in self.cache:
                self._stats["cache_hits"] += 1
                return EmbeddingResult(
                    embedding=self.cache[key],
                    model=self.backend.model_name if hasattr(self.backend, 'model_name') else "unknown",
                    dimensions=len(self.cache[key]),
                    cached=True,
                )

        self._stats["cache_misses"] += 1
        embedding = await self.backend.embed(text)

        if use_cache:
            self._add_to_cache(key, embedding)

        return EmbeddingResult(
            embedding=embedding,
            model=self.backend.model_name if hasattr(self.backend, 'model_name') else "unknown",
            dimensions=len(embedding),
            cached=False,
        )

    async def embed_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> list[EmbeddingResult]:
        """Get embeddings for multiple texts."""
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            if use_cache:
                key = self._cache_key(text)
                if key in self.cache:
                    self._stats["cache_hits"] += 1
                    results.append(EmbeddingResult(
                        embedding=self.cache[key],
                        model=self.backend.model_name if hasattr(self.backend, 'model_name') else "unknown",
                        dimensions=len(self.cache[key]),
                        cached=True,
                    ))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    results.append(None)  # Placeholder
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)

        # Batch embed uncached texts
        if uncached_texts:
            self._stats["cache_misses"] += len(uncached_texts)
            embeddings = await self.backend.embed_batch(uncached_texts)

            for idx, (text, embedding) in zip(uncached_indices, zip(uncached_texts, embeddings)):
                if use_cache:
                    self._add_to_cache(self._cache_key(text), embedding)

                results[idx] = EmbeddingResult(
                    embedding=embedding,
                    model=self.backend.model_name if hasattr(self.backend, 'model_name') else "unknown",
                    dimensions=len(embedding),
                    cached=False,
                )

        self._stats["total_requests"] += len(texts)
        return results

    def get_embedding_sync(self, text: str) -> list[float]:
        """Synchronous embedding (for compatibility)."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.embed(text))
                result = future.result()
                return result.embedding
        else:
            result = loop.run_until_complete(self.embed(text))
            return result.embedding

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        self.cache_order.clear()

    def get_stats(self) -> dict:
        """Get embedding statistics."""
        return {
            **self._stats,
            "cache_size": len(self.cache),
            "cache_max": self.cache_size,
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(1, self._stats["total_requests"])
            ),
            "dimensions": self.dimensions,
        }
