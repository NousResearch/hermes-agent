#!/usr/bin/env python3
"""
Embedding Infrastructure — Configurable local (fastembed) + API (OpenAI) embedders.

Provides a shared embedding capability for cognitive memory recall (#509),
semantic codebase search (#489), and future similarity-based operations.

Usage:
    embedder = get_embedder(config)
    vector = embedder.embed_text("some text")
    vectors = embedder.embed_texts(["text1", "text2"])

Config (config.yaml):
    embeddings:
      provider: "local"           # "local" or "openai"
      model: "all-MiniLM-L6-v2"  # for local
      # model: "text-embedding-3-small"  # for openai
"""
from __future__ import annotations

import logging
import math
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    def embed_text(self, text: str) -> list[float]: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...


# ---------------------------------------------------------------------------
# Local embedder (fastembed)
# ---------------------------------------------------------------------------

class FastEmbedEmbedder:
    """Local embeddings via fastembed (all-MiniLM-L6-v2, 384 dims).

    ~100MB model downloaded on first use to ~/.cache/fastembed/.
    No API key needed, private, fast (~5ms per embed).
    Requires: pip install fastembed
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self._model = None  # Lazy initialization

    def _load(self):
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is not installed. "
                "Install it with: pip install fastembed\n"
                "Or: pip install 'hermes-agent[embeddings]'"
            )
        logger.info("Loading fastembed model '%s' (first use may download ~100MB)...", self.model_name)
        self._model = TextEmbedding(model_name=self.model_name)
        logger.info("fastembed model loaded.")

    def embed_text(self, text: str) -> list[float]:
        self._load()
        results = list(self._model.embed([text]))
        return results[0].tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._load()
        results = list(self._model.embed(texts))
        return [r.tolist() for r in results]

    @property
    def dimensions(self) -> int:
        return 384  # all-MiniLM-L6-v2 fixed dims


# ---------------------------------------------------------------------------
# OpenAI embedder
# ---------------------------------------------------------------------------

class OpenAIEmbedder:
    """API embeddings via OpenAI (text-embedding-3-small, 1536 dims).

    Uses existing OpenAI client from config.
    Higher quality but costs ~$0.02/1M tokens.
    Requires: openai (already a dependency)
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = None, base_url: str = None):
        self.model_name = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # Lazy initialization

    def _load(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is not installed.")
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def embed_text(self, text: str) -> list[float]:
        self._load()
        response = self._client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._load()
        response = self._client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        return self._DIMENSIONS.get(self.model_name, 1536)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(config: dict) -> Embedder:
    """Factory: returns configured embedder based on config dict.

    Args:
        config: Full config dict. Reads from config["embeddings"] section.

    Returns:
        An Embedder instance.

    Raises:
        ValueError: If provider is unknown.
        ImportError: If required package is not installed.
    """
    emb_config = config.get("embeddings", {})
    provider = emb_config.get("provider", "local")
    model = emb_config.get("model")

    if provider == "local":
        effective_model = model or FastEmbedEmbedder.DEFAULT_MODEL
        return FastEmbedEmbedder(model=effective_model)

    elif provider == "openai":
        effective_model = model or OpenAIEmbedder.DEFAULT_MODEL
        api_key = emb_config.get("api_key")
        base_url = emb_config.get("base_url")
        return OpenAIEmbedder(model=effective_model, api_key=api_key, base_url=base_url)

    else:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            "Supported providers: 'local', 'openai'"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1]. Higher = more similar.
    Returns 0.0 if either vector has zero magnitude.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match: {len(a)} != {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Compute NxN pairwise cosine similarity matrix.

    Useful for deduplication: if matrix[i][j] >= 0.98, items i and j are near-duplicates.

    Returns:
        NxN matrix where matrix[i][j] = cosine_similarity(vectors[i], vectors[j])
    """
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix
