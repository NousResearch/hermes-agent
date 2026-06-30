"""Embedding-based semantic similarity index for memory entries.

Uses ONNX Runtime with a quantized MiniLM model to produce 384-dim
embeddings. Index is stored as a numpy array on disk for fast cosine
similarity search.

Graceful degradation: if onnxruntime is unavailable, falls back to
empty results (Phase 2 heuristics still work via context-tag + FTS5).
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Model info
EMBEDDING_DIM = 384
MODEL_DIR = "/root/.cache/onnx/all-MiniLM-L6-v2"
MODEL_FILE = os.path.join(MODEL_DIR, "onnx", "model_quantized.onnx")

# Cache file for precomputed embeddings
EMBEDDINGS_CACHE_FILE = "embeddings_cache.pkl"

# How often to re-embed (seconds). Default: rebuild if entries changed.
# Set to 0 to always re-embed on search.
REEMBED_INTERVAL = 300  # 5 min


class EmbeddingIndex:
    """Embedding-based semantic similarity index for memory entries.

    Thread-safe. Embeddings are computed lazily and cached on disk.
    """

    def __init__(self, cache_dir: str | Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Model state (lazy loaded)
        self._session = None
        self._tokenizer = None

        # Cached embeddings
        self._cached_entries: List[str] = []
        self._cached_embeddings: Optional[np.ndarray] = None
        self._cache_mtime: float = 0.0

        # ONNX input names (determined during init)
        self._input_names: Optional[List[str]] = None
        self._output_names: Optional[List[str]] = None

        self._available = self._probe_model()

    def _probe_model(self) -> bool:
        """Check if the ONNX model exists and can be loaded."""
        if not os.path.exists(MODEL_FILE):
            logger.warning("Embedding model not found at %s — embeddings disabled", MODEL_FILE)
            return False

        try:
            import onnxruntime as ort  # noqa: F401
            return True
        except ImportError:
            logger.warning("onnxruntime not installed — embeddings disabled")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_model(self) -> bool:
        """Lazy-load the ONNX session and tokenizer."""
        if self._session is not None:
            return True
        if not self._available:
            return False

        try:
            import onnxruntime as ort

            # ONNX session
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 2  # limit CPU usage
            self._session = ort.InferenceSession(
                MODEL_FILE, sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]

            # Tokenizer from transformers
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

            logger.debug("ONNX embedding model loaded: %s", MODEL_FILE)
            return True
        except Exception as e:
            logger.warning("Failed to load embedding model: %s", e)
            self._available = False
            return False

    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean pooling — average token embeddings weighted by attention mask."""
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(model_output.dtype)
        sum_embeddings = np.sum(model_output * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def embed_text(self, text: str) -> np.ndarray:
        """Produce a single 384-dim embedding vector for a text string.

        Returns:
            numpy array of shape (384,) or zeros if model unavailable.
        """
        if not self._ensure_model() or self._tokenizer is None or self._session is None:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        try:
            # Tokenize
            inputs = self._tokenizer(
                text, padding=True, truncation=True, max_length=256,
                return_tensors="np",
            )

            # ONNX inference
            ort_inputs = {
                self._input_names[0]: inputs["input_ids"],
                self._input_names[1]: inputs["attention_mask"],
            }
            if len(self._input_names) > 2:
                ort_inputs[self._input_names[2]] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

            outputs = self._session.run(self._output_names, ort_inputs)

            # Mean pooling
            embedding = self._mean_pooling(outputs[0], inputs["attention_mask"])
            # Normalize to unit length
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = embedding / np.clip(norm, a_min=1e-9, a_max=None)

            return embedding[0].astype(np.float32)

        except Exception as e:
            logger.debug("Embedding failed: %s", e)
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts in a single ONNX inference call.

        Returns:
            numpy array of shape (N, 384), or zeros if model unavailable.
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        if not self._ensure_model() or self._tokenizer is None or self._session is None:
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

        try:
            # Tokenize batch
            inputs = self._tokenizer(
                texts, padding=True, truncation=True, max_length=256,
                return_tensors="np",
            )

            ort_inputs = {
                self._input_names[0]: inputs["input_ids"],
                self._input_names[1]: inputs["attention_mask"],
            }
            if len(self._input_names) > 2:
                ort_inputs[self._input_names[2]] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

            outputs = self._session.run(self._output_names, ort_inputs)

            embeddings = self._mean_pooling(outputs[0], inputs["attention_mask"])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.debug("Batch embedding failed: %s", e)
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    def rebuild(self, entries: List[str]) -> int:
        """Compute and cache embeddings for all entries.

        Args:
            entries: Raw memory entry strings.

        Returns:
            Number of entries embedded.
        """
        with self._lock:
            if not self._available:
                return 0

            if not entries:
                self._cached_entries = []
                self._cached_embeddings = None
                self._save_cache()
                return 0

            # Strip markup for embedding (we want semantic content, not tags)
            from agent.memory.scored_memory import strip_memory_markup
            clean_texts = [strip_memory_markup(e) for e in entries]

            logger.debug("Embedding %d entries...", len(clean_texts))
            embeddings = self.embed_batch(clean_texts)

            self._cached_entries = list(entries)
            self._cached_embeddings = embeddings
            self._cache_mtime = time.time()
            self._save_cache()

            return len(entries)

    def search(
        self,
        query: str,
        entries: List[str],
        limit: int = 20,
    ) -> List[Dict]:
        """Search memory entries by semantic similarity.

        Combines the provided entries list with cached embeddings. If
        entries have changed (different count/content), rebuilds cache.

        Args:
            query: Search query text.
            entries: Current memory entries (to check cache freshness).
            limit: Max results.

        Returns:
            List of dicts with keys: content, score (cosine sim 0-1).
        """
        if not self._available or not entries or not query:
            return []

        with self._lock:
            # Check if cache needs rebuilding
            if not self._is_cache_fresh(entries):
                self.rebuild(entries)

            if self._cached_embeddings is None or len(self._cached_embeddings) == 0:
                return []

            # Embed query
            query_vec = self.embed_text(query)
            if np.allclose(query_vec, 0):
                return []

            # Compute cosine similarities
            from agent.memory.scored_memory import strip_memory_markup

            # self._cached_embeddings shape: (N, 384)
            # query_vec shape: (384,)
            sims = np.dot(self._cached_embeddings, query_vec)
            # Bound to [0, 1] from [-1, 1]
            sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)

            # Top-K indices
            top_k = min(limit, len(sims))
            if top_k == 0:
                return []

            indices = np.argsort(sims)[-top_k:][::-1]

            results = []
            for idx in indices:
                if idx < len(entries):
                    clean = strip_memory_markup(entries[idx])
                    results.append({
                        "content": clean,
                        "entry_index": int(idx),
                        "score": float(sims[idx]),
                    })

            return results

    def _is_cache_fresh(self, entries: List[str]) -> bool:
        """Check if cached embeddings match the current entry list."""
        if self._cached_embeddings is None:
            return False
        if len(self._cached_entries) != len(entries):
            return False
        # Quick check: compare first/last entry fingerprints
        if self._cached_entries and entries:
            if (hash(self._cached_entries[0]) != hash(entries[0]) or
                    hash(self._cached_entries[-1]) != hash(entries[-1])):
                return False
        return True

    def _save_cache(self) -> None:
        """Persist embeddings to disk for faster cold starts."""
        try:
            cache_path = self._cache_dir / EMBEDDINGS_CACHE_FILE
            data = {
                "entries": self._cached_entries,
                "embeddings": self._cached_embeddings,
                "mtime": self._cache_mtime,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.debug("Failed to save embedding cache: %s", e)

    def _load_cache(self, entries: List[str]) -> bool:
        """Load embeddings from disk cache if it matches the entry list."""
        try:
            cache_path = self._cache_dir / EMBEDDINGS_CACHE_FILE
            if not cache_path.exists():
                return False
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                return False
            cached_entries = data.get("entries", [])
            if cached_entries == entries:  # exact list match
                self._cached_entries = cached_entries
                self._cached_embeddings = data.get("embeddings")
                self._cache_mtime = data.get("mtime", 0.0)
                logger.debug("Loaded embedding cache: %d entries", len(cached_entries))
                return True
            return False
        except Exception as e:
            logger.debug("Failed to load embedding cache: %s", e)
            return False

    def close(self) -> None:
        """Release model resources."""
        self._session = None
        self._tokenizer = None


def default_embedding_cache_dir() -> Path:
    """Return the default cache directory under HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "memories"
