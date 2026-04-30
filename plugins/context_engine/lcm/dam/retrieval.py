"""DAM Retrieval Engine — semantic search over the LCM conversation store.

Integrates DenseAssociativeMemory and MessageEncoder with the LcmEngine's
ImmutableStore to provide associative retrieval over conversation history.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from plugins.context_engine.lcm.dam.network import DenseAssociativeMemory
from plugins.context_engine.lcm.dam.encoder import MessageEncoder


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class DAMRetriever:
    """Semantic retrieval engine backed by a DenseAssociativeMemory.

    Keeps a pattern cache mapping msg_id -> encoded vector so that cosine
    ranking after DAM recall is fast and doesn't re-encode every message.

    Parameters
    ----------
    net:
        The DenseAssociativeMemory network to train and query.
    enc:
        The MessageEncoder used to vectorize messages and query strings.
    storage_dir:
        Directory for any optional state persistence (not required for core ops).
    """

    def __init__(
        self,
        net: DenseAssociativeMemory,
        enc: MessageEncoder,
        storage_dir: Path | None = None,
    ) -> None:
        self.net = net
        self.enc = enc
        self.storage_dir = storage_dir

        # msg_id -> unit vector
        self._pattern_cache: dict[int, np.ndarray] = {}
        # Tracks how many store messages have been indexed so far
        self._last_indexed_id: int = 0

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync_from_store(self, store: Any) -> int:
        """Index any new messages from an explicit store reference.

        Avoids the circular dependency introduced by ``sync_with_store()``
        (which calls ``get_engine()``).  The engine passes ``self.store``
        directly so there is no global state involved.

        Returns the count of newly indexed messages.
        """
        total = len(store)
        new_messages: list[np.ndarray] = []
        new_ids: list[int] = []

        for msg_id in range(self._last_indexed_id, total):
            msg = store.get(msg_id)
            if msg is None:
                continue
            vec = self.enc.encode(msg)
            self._pattern_cache[msg_id] = vec
            new_messages.append(vec)
            new_ids.append(msg_id)

        if new_messages:
            patterns = np.stack(new_messages, axis=0)
            self.net.learn(patterns)
            self._last_indexed_id = total

        return len(new_ids)

    def sync_with_store(self) -> int:
        """Index any new messages from the active LcmEngine's store.

        Reads messages from ``_last_indexed_id`` up to the current store
        length, encodes them, and trains the network on the new batch.

        Returns the count of newly indexed messages.
        """
        try:
            from plugins.context_engine.lcm.tools import get_engine
        except ImportError:
            return 0

        engine = get_engine()
        if engine is None:
            return 0

        return self.sync_from_store(engine.store)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[tuple[int, float]]:
        """Encode query string, run DAM recall, rank cache by cosine similarity.

        Returns a list of (msg_id, score) sorted by descending score.
        An empty query returns an empty list.
        """
        if not query.strip():
            return []

        if not self._pattern_cache:
            return []

        q_vec = self.enc.encode_text(query)
        recalled, _ = self.net.recall(q_vec)

        scores = [
            (msg_id, _cosine_similarity(recalled, vec))
            for msg_id, vec in self._pattern_cache.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    # ------------------------------------------------------------------
    # Recall similar
    # ------------------------------------------------------------------

    def recall_similar(self, msg_id: int, limit: int = 5) -> list[tuple[int, float]]:
        """Find messages similar to the one at msg_id using DAM recall.

        Looks up the cached vector for msg_id, runs recall, then ranks
        all other cached messages by cosine similarity to the recalled vector.

        Returns a list of (msg_id, score) excluding the query message itself,
        sorted by descending score.
        """
        if msg_id not in self._pattern_cache:
            return []

        anchor_vec = self._pattern_cache[msg_id]
        recalled, _ = self.net.recall(anchor_vec)

        scores = [
            (mid, _cosine_similarity(recalled, vec))
            for mid, vec in self._pattern_cache.items()
            if mid != msg_id
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(
        self,
        queries: list[str],
        operation: str = "AND",
        limit: int = 10,
    ) -> list[tuple[int, float]]:
        """Multi-term composition using hidden-layer activation algebra.

        Each query is encoded and projected to binary hidden activations.
        Activations are combined:
          - AND: element-wise min (intersection — all terms must fire)
          - OR:  element-wise max (union — any term may fire)

        The composed activation is used to reconstruct a visible vector which
        is then ranked against the cache by cosine similarity.

        Returns a list of (msg_id, score) sorted by descending score.
        """
        if not queries or not self._pattern_cache:
            return []

        hidden_acts: list[np.ndarray] = []
        for q in queries:
            if not q.strip():
                continue
            q_vec = self.enc.encode_text(q)
            h = self.net.get_hidden_activations(q_vec)
            hidden_acts.append(h)

        if not hidden_acts:
            return []

        op = operation.upper()
        if op == "AND":
            composed = hidden_acts[0]
            for h in hidden_acts[1:]:
                composed = np.minimum(composed, h)
        elif op == "OR":
            composed = hidden_acts[0]
            for h in hidden_acts[1:]:
                composed = np.maximum(composed, h)
        elif op == "NOT":
            # NOT is defined only for a single query (invert its binary activations).
            # Multi-query NOT is semantically ambiguous — reject to avoid silent wrong results.
            if len(hidden_acts) != 1:
                logger.warning(
                    "compose(operation='NOT') requires exactly 1 query; "
                    "got %d — returning empty results",
                    len(hidden_acts),
                )
                return []
            # Hidden activations are binary (0.0 / 1.0 float32); inversion is exact.
            composed = 1.0 - hidden_acts[0]
        else:
            logger.warning("compose(): unknown operation %r — returning empty results", operation)
            return []

        # Reconstruct a visible vector from the composed hidden state
        # Use the network's internal reconstruct path directly
        recon = self.net._reconstruct(composed)

        scores = [
            (msg_id, _cosine_similarity(recon, vec))
            for msg_id, vec in self._pattern_cache.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a dictionary of network and retriever statistics."""
        return {
            "nv": self.net.nv,
            "nh": self.net.nh,
            "patterns_stored": len(self._pattern_cache),
            "n_patterns_trained": self.net.n_patterns_trained,
            "last_indexed_id": self._last_indexed_id,
        }
