"""Embedding providers for semantic vault search.

Priority:
  1. Local sentence-transformers (all-MiniLM-L6-v2, 80 MB, no API cost)
  2. OpenAI text-embedding-3-small  (if OPENAI_API_KEY is set)
  3. None → BM25-only mode (always works, zero deps)

The provider is selected once at index build time and cached.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM_MINILM = 384
EMBED_DIM_OPENAI = 1536


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim)."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# sentence-transformers (local)
# ---------------------------------------------------------------------------

_ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_st_instance: Optional["_SentenceTransformerProvider"] = None


class _SentenceTransformerProvider(EmbeddingProvider):
    dim = EMBED_DIM_MINILM

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(_ST_MODEL_NAME)

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# OpenAI embeddings
# ---------------------------------------------------------------------------

_OPENAI_MODEL = "text-embedding-3-small"


class _OpenAIEmbeddingProvider(EmbeddingProvider):
    dim = EMBED_DIM_OPENAI

    def __init__(self, api_key: str) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key)

    def encode(self, texts: List[str]) -> np.ndarray:
        # OpenAI batch limit is 2048 items; chunk if needed
        results: list[list[float]] = []
        for i in range(0, len(texts), 512):
            batch = texts[i : i + 512]
            resp = self._client.embeddings.create(model=_OPENAI_MODEL, input=batch)
            results.extend([d.embedding for d in sorted(resp.data, key=lambda x: x.index)])
        arr = np.array(results, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider_cache: Optional[EmbeddingProvider] = None


def get_embedding_provider(*, force_reload: bool = False) -> Optional[EmbeddingProvider]:
    """Return the best available embedding provider, or None if none are available."""
    global _provider_cache
    if _provider_cache is not None and not force_reload:
        return _provider_cache

    # 1. Try sentence-transformers (local, no cost)
    try:
        import sentence_transformers  # noqa: F401
        _provider_cache = _SentenceTransformerProvider()
        logger.info("obsidian-memory: using local sentence-transformers embeddings")
        return _provider_cache
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("obsidian-memory: sentence-transformers failed to load: %s", exc)

    # 2. Try OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            _provider_cache = _OpenAIEmbeddingProvider(api_key)
            logger.info("obsidian-memory: using OpenAI text-embedding-3-small")
            return _provider_cache
        except Exception as exc:
            logger.warning("obsidian-memory: OpenAI embeddings failed: %s", exc)

    logger.info("obsidian-memory: no embedding provider available — using BM25 only")
    return None


def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return top-k (index, score) pairs by cosine similarity.

    Both query_vec and matrix rows should be L2-normalised (dot product == cosine).
    """
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query_vec  # shape (N,)
    k = min(k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(int(i), float(scores[i])) for i in top_idx]


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]
