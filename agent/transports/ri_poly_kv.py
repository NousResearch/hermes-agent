"""RecursiveIntell poly-kv vector operations for Hermes.

Provides Rust-backed KV-cache pooling, shape validation, and
in-process vector scoring via the poly-kv crate (alpha).

Usage::

    from agent.transports.ri_poly_kv import (
        RiPolyKvScorer, validate_shape, native_available,
    )

    scorer = RiPolyKvScorer()
    if scorer.enabled:
        scores = scorer.cosine_batch(query_vec, candidates)
        top = scorer.topk_compressed(query_vec, candidates, k=5)
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

_NATIVE_AVAILABLE = False
try:
    import poly_kv._native as _pkn

    _NATIVE_AVAILABLE = True
except ImportError:
    _pkn = None  # type: ignore[assignment]
    logger.debug("poly-kv native extension not available")


def native_available() -> bool:
    """Check if the poly-kv native extension is installed."""
    return _NATIVE_AVAILABLE


def validate_shape(shape_json: str) -> str:
    """Validate a KV-cache shape specification. Returns JSON result string."""
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("poly-kv native extension not installed")
    return _pkn.validate_shape_json(shape_json)


def build_synthetic_pool(shape_json: str) -> str:
    """Build a synthetic KV-cache pool. Returns JSON receipt string."""
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("poly-kv native extension not installed")
    return _pkn.build_synthetic_pool_receipts_json(shape_json)


# ── Phase 3: RiPolyKvScorer — in-process vector scoring ──────────


class RiPolyKvScorer:
    """In-process vector scorer gated by ``HERMES_RI_POLY_KV=1``.

    Accelerates embedding similarity and compressed-domain retrieval
    without MCP roundtrips.  Falls through silently on any error so
    the caller can route back to the MCP semantic-memory path.

    Cosine similarity is computed in pure Python (standard dot-product
    math) — the Rust crate is used for shape validation and compressed
    pool operations where the alpha poly-kv native methods are stable.
    """

    @property
    def enabled(self) -> bool:
        """True by default when the native extension is importable.
        Set HERMES_RI_POLY_KV=0 to disable."""
        if os.environ.get("HERMES_RI_POLY_KV") == "0":
            return False
        return _NATIVE_AVAILABLE

    # ── cosine similarity (pure Python, fast enough for small batches) ─

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _norm(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def cosine_batch(
        self, query_vec: list[float], candidates: list[list[float]]
    ) -> list[float] | None:
        """Compute cosine similarity between ``query_vec`` and each candidate.

        Returns ``None`` when disabled or on error (caller falls back to MCP).
        """
        if not self.enabled:
            return None
        try:
            q_norm = self._norm(query_vec)
            if q_norm == 0.0:
                return [0.0] * len(candidates)
            scores: list[float] = []
            for c in candidates:
                c_norm = self._norm(c)
                if c_norm == 0.0:
                    scores.append(0.0)
                else:
                    scores.append(self._dot(query_vec, c) / (q_norm * c_norm))
            return scores
        except Exception as exc:
            logger.debug("ri_poly_kv: cosine_batch failed: %s", exc)
            return None

    def cosine(
        self, query_vec: list[float], candidate_vec: list[float]
    ) -> float | None:
        """Compute cosine similarity between two vectors.

        Returns ``None`` when disabled or on error.
        """
        if not self.enabled:
            return None
        try:
            q_norm = self._norm(query_vec)
            c_norm = self._norm(candidate_vec)
            if q_norm == 0.0 or c_norm == 0.0:
                return 0.0
            return self._dot(query_vec, candidate_vec) / (q_norm * c_norm)
        except Exception as exc:
            logger.debug("ri_poly_kv: cosine failed: %s", exc)
            return None

    def topk_compressed(
        self,
        query_vec: list[float],
        candidates: list[list[float]],
        *,
        k: int = 10,
    ) -> list[tuple[int, float]] | None:
        """Top-k cosine scoring, returning ``[(candidate_index, score), ...]``.

        Uses a threshold-sorted pass (same math as cosine_batch) that is
        efficient for small to medium candidate sets.  For large-scale
        compressed retrieval the MCP server's FibQuant scoring is the
        right tool; this method accelerates the common local-embedding
        path.

        Returns ``None`` when disabled or on error.
        """
        scores = self.cosine_batch(query_vec, candidates)
        if scores is None:
            return None
        try:
            indexed = [(i, s) for i, s in enumerate(scores)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            return indexed[:k]
        except Exception as exc:
            logger.debug("ri_poly_kv: topk_compressed failed: %s", exc)
            return None

    # ── shape validation (delegates to native when available) ─────

    def validate(self, shape_spec: dict[str, Any]) -> dict[str, Any] | None:
        """Validate a shape spec through the native poly-kv validator.

        Returns the parsed validation result or ``None`` on error.
        """
        if not self.enabled:
            return None
        try:
            raw = _pkn.validate_shape_json(json.dumps(shape_spec))
            return json.loads(raw)
        except Exception as exc:
            logger.debug("ri_poly_kv: validate failed: %s", exc)
            return None

    def build_synthetic_pool_receipt(
        self, shape_spec: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build a synthetic pool and return the receipt dict, or ``None``."""
        if not self.enabled:
            return None
        try:
            raw = _pkn.build_synthetic_pool_receipts_json(json.dumps(shape_spec))
            return json.loads(raw)
        except Exception as exc:
            logger.debug("ri_poly_kv: build_synthetic_pool failed: %s", exc)
            return None

    def health(self) -> dict[str, Any]:
        """Quick health check."""
        return {
            "enabled": self.enabled,
            "native_available": _NATIVE_AVAILABLE,
            "gate_env": os.environ.get("HERMES_RI_POLY_KV"),
        }
