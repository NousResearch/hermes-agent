"""Embedding capture backends for kv-memory.

Pluggable architecture: each backend implements a simple interface.
The active backend is selected via config (embedding_backend key).

Backends:
  - SentenceTransformersBackend: CPU-friendly, 384-dim, always available.
    Uses all-MiniLM-L6-v2 by default (~80MB download).
  - APIEmbeddingBackend: Uses cloud provider APIs (OpenAI, etc.).
  - vLLMHiddenStateBackend: Future backend for local vLLM inference.
    Reports unavailable until vLLM exposes hidden states via its HTTP API.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Backend interface
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingBackend(ABC):
    """Abstract interface for embedding generation backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can generate embeddings right now."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Generate a float32 embedding vector for the given text."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the generated embeddings."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend name for logging."""


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: Sentence-Transformers (default, always available)
# ═══════════════════════════════════════════════════════════════════════════════

class SentenceTransformersBackend(EmbeddingBackend):
    """Embedding generation via sentence-transformers.

    Uses all-MiniLM-L6-v2 (384-dim) by default — small (~80MB), fast,
    and runs on CPU. This is the universal fallback that works everywhere.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = ""):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    @property
    def backend_name(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    @property
    def embedding_dim(self) -> int:
        if self._model is not None:
            meth = getattr(self._model, "get_embedding_dimension", None)
            if meth is None:
                meth = getattr(self._model, "get_sentence_embedding_dimension", None)
            if meth is not None:
                return meth()
        return 384  # default for all-MiniLM-L6-v2

    def is_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            logger.debug("sentence-transformers not installed")
            return False

    def encode(self, text: str) -> np.ndarray:
        if self._model is None:
            self._load_model()
        embedding = self._model.encode([text], show_progress_bar=False,
                                        convert_to_numpy=True)
        return embedding[0].astype(np.float32)

    def _load_model(self):
        import sentence_transformers
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = sentence_transformers.SentenceTransformer(
            self._model_name, device="cpu"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: vLLM Hidden State Capture (future)
# ═══════════════════════════════════════════════════════════════════════════════

class vLLMHiddenStateBackend(EmbeddingBackend):
    """Capture hidden states from a locally-running vLLM inference server.

    This backend requires vLLM to expose hidden states via its HTTP API
    or a custom endpoint. As of vLLM 0.24, this is not yet supported.
    When available, it will enable true model-native embeddings —
    capturing the inference model's own internal representations rather
    than using a separate embedding model.

    Until then, this backend reports unavailable and the system falls
    back to sentence-transformers.
    """

    def __init__(self, model_name: str = ""):
        self._model_name = model_name

    @property
    def backend_name(self) -> str:
        return f"vllm-hidden-state/{self._model_name or 'auto'}"

    @property
    def embedding_dim(self) -> int:
        return 0  # unknown until model is loaded

    def is_available(self) -> bool:
        return False  # not yet available (requires vLLM API support)

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError(
            "vLLM hidden-state capture is not available in Phase 1. "
            "Set embedding_backend='sentence-transformers' in config."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: API-based embeddings (e.g., OpenAI)
# ═══════════════════════════════════════════════════════════════════════════════

class APIEmbeddingBackend(EmbeddingBackend):
    """Embedding generation via cloud provider APIs (OpenAI, etc.).

    Uses the provider configured in Hermes. Requires API credentials.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self._model_name = model_name
        self._dim = 1536  # default for text-embedding-3-small

    @property
    def backend_name(self) -> str:
        return f"api/{self._model_name}"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def is_available(self) -> bool:
        try:
            from openai import OpenAI  # noqa: F401
            return True
        except ImportError:
            return False

    def encode(self, text: str) -> np.ndarray:
        try:
            from openai import OpenAI
            import os

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            response = client.embeddings.create(
                model=self._model_name,
                input=text,
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error("API embedding failed: %s", e)
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# Backend factory
# ═══════════════════════════════════════════════════════════════════════════════

_BACKENDS: dict[str, type[EmbeddingBackend]] = {
    "sentence-transformers": SentenceTransformersBackend,
    "vllm": vLLMHiddenStateBackend,
    "api": APIEmbeddingBackend,
}


def create_embedding_backend(
    backend_name: str = "auto",
    model_name: str = "",
) -> EmbeddingBackend:
    """Create an embedding backend, falling back through available options.

    Resolution order for "auto":
      1. vllm (if available — requires local GPU server)
      2. sentence-transformers (always available if installed)
      3. api (requires credentials)

    If a specific backend is requested but unavailable, falls back
    to the next available option with a warning.
    """
    if backend_name == "auto":
        # Try vLLM first (most accurate), then sentence-transformers, then API
        fallback_order = ["vllm", "sentence-transformers", "api"]
    else:
        fallback_order = [backend_name] + [
            b for b in ["vllm", "sentence-transformers", "api"]
            if b != backend_name
        ]

    for name in fallback_order:
        if name not in _BACKENDS:
            continue
        try:
            backend_cls = _BACKENDS[name]
            backend = backend_cls(model_name=model_name)
            if backend.is_available():
                if name != fallback_order[0]:
                    logger.info(
                        "Embedding backend '%s' unavailable, "
                        "falling back to '%s'", fallback_order[0], name
                    )
                return backend
        except Exception as e:
            logger.debug("Backend '%s' init failed: %s", name, e)
            continue

    raise RuntimeError(
        "No embedding backend available. Install sentence-transformers: "
        "pip install sentence-transformers"
    )
