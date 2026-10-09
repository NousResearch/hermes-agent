"""Opt-in semantic tier for Tameru — embeddings + optional cross-encoder.

Contract (v0.8.0):
  - NEVER imported by the core engine. The engine accepts a tier object
    duck-typed on two methods; this module is one possible implementation.
  - Zero deps by default: if sentence-transformers is missing, `available`
    is False and every method raises SemanticUnavailable — callers must
    fall back to lexical.
  - Local-only: models load from disk/HF cache. No network at query time
    beyond the one-time HF download the user opted into.
  - Deterministic given a pinned model: encode() is inference-only, no RNG.
"""
from __future__ import annotations

from typing import Any, Sequence

DEFAULT_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SemanticUnavailable(RuntimeError):
    """Raised when the tier is used without its optional dependencies."""


class SemanticTier:
    """Bi-encoder block/query scorer with optional CE rerank."""

    def __init__(
        self,
        bi_model: str = DEFAULT_BI_ENCODER,
        cross_model: str | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
        except ImportError as e:  # pragma: no cover - env-dependent
            raise SemanticUnavailable(
                "sentence-transformers not installed; pip install tameru[semantic]"
            ) from e
        self._bi = SentenceTransformer(bi_model, device=device)
        self._ce = CrossEncoder(cross_model, device=device) if cross_model else None
        self.available = True

    # -- bi-encoder -----------------------------------------------------
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._bi.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def similarity(self, query_vec: Sequence[float], vecs: Sequence[Sequence[float]]) -> list[float]:
        return [sum(a * b for a, b in zip(query_vec, v)) for v in vecs]

    def score_against_query(self, query: str, texts: Sequence[str]) -> tuple[list[float], list[float]]:
        """Returns (query_vector, per-text cosine similarities)."""
        qv = self.encode([query])[0]
        tv = self.encode(list(texts))
        return qv, self.similarity(qv, tv)

    # -- cross-encoder ---------------------------------------------------
    def cross_scores(self, query: str, texts: Sequence[str]) -> list[float] | None:
        if self._ce is None:
            return None
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self._ce.predict(pairs)]


class _DisabledTier:
    """Null-object fallback: every call reports unavailable."""

    available = False

    def __getattr__(self, name):  # noqa: D105
        def _raise(*a: Any, **k: Any):
            raise SemanticUnavailable("semantic tier disabled")

        return _raise


def resolve_tier(tier: Any = None) -> Any:
    """Accept a live SemanticTier / None / dict config; always returns a usable object."""
    if tier is None:
        return _DisabledTier()
    if isinstance(tier, dict):
        return SemanticTier(**tier)
    return tier
