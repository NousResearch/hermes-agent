"""Hybrid search: BM25 full-text + semantic cosine similarity with RRF fusion.

BM25 is always available (pure Python). Semantic search activates when an
embedding provider is available. Reciprocal Rank Fusion (RRF) merges the
two result lists into a single ranked output.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from plugins.memory.obsidian.embeddings import EmbeddingProvider, cosine_top_k

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Index:
    """BM25Okapi over a list of text chunks."""

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self._corpus: List[List[str]] = []
        self._ids: List[int] = []  # maps corpus index → chunk id
        self._df: dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0

    def build(self, chunks: List[Tuple[int, str]]) -> None:
        """chunks: list of (chunk_id, text)."""
        self._corpus = []
        self._ids = []
        self._df = defaultdict(int)
        for cid, text in chunks:
            tokens = _tokenize(text)
            self._corpus.append(tokens)
            self._ids.append(cid)
            for tok in set(tokens):
                self._df[tok] += 1
        total_len = sum(len(doc) for doc in self._corpus)
        self._avgdl = total_len / max(len(self._corpus), 1)

    def _idf(self, term: str) -> float:
        n = len(self._corpus)
        df = self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """Return top-k (chunk_id, score) pairs."""
        if not self._corpus:
            return []
        query_terms = _tokenize(query)
        scores: List[float] = []
        for doc in self._corpus:
            score = 0.0
            doc_len = len(doc)
            tf_map: dict[str, int] = defaultdict(int)
            for tok in doc:
                tf_map[tok] += 1
            for term in query_terms:
                tf = tf_map.get(term, 0)
                idf = self._idf(term)
                num = tf * (self.K1 + 1)
                den = tf + self.K1 * (1 - self.B + self.B * doc_len / max(self._avgdl, 1))
                score += idf * (num / den)
            scores.append(score)
        k = min(k, len(scores))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], scores[i]) for i in top_idx if scores[i] > 0]


# ---------------------------------------------------------------------------
# SemanticIndex
# ---------------------------------------------------------------------------

class SemanticIndex:
    """Cosine-similarity search over pre-computed embeddings."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._matrix: Optional[np.ndarray] = None   # shape (N, dim)
        self._ids: List[int] = []

    def build(self, chunks: List[Tuple[int, str]]) -> None:
        if not chunks:
            self._matrix = np.zeros((0, self._provider.dim), dtype=np.float32)
            self._ids = []
            return
        ids, texts = zip(*chunks)
        self._ids = list(ids)
        self._matrix = self._provider.encode(list(texts))

    def update_chunk(self, chunk_id: int, text: str) -> None:
        vec = self._provider.encode_one(text)
        if chunk_id in self._ids:
            idx = self._ids.index(chunk_id)
            if self._matrix is not None:
                self._matrix[idx] = vec
        else:
            self._ids.append(chunk_id)
            new_row = vec.reshape(1, -1)
            self._matrix = (
                new_row if self._matrix is None or self._matrix.shape[0] == 0
                else np.vstack([self._matrix, new_row])
            )

    def remove_chunks(self, chunk_ids: List[int]) -> None:
        keep = [i for i, cid in enumerate(self._ids) if cid not in set(chunk_ids)]
        self._ids = [self._ids[i] for i in keep]
        if self._matrix is not None and self._matrix.shape[0] > 0:
            self._matrix = self._matrix[keep]

    def search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q_vec = self._provider.encode_one(query)
        return cosine_top_k(q_vec, self._matrix, k)


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    *ranked_lists: List[Tuple[int, float]],
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    rrf: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (cid, _score) in enumerate(ranked):
            rrf[cid] += 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# HybridSearch
# ---------------------------------------------------------------------------

class HybridSearch:
    """BM25 + semantic RRF hybrid search over note chunks."""

    def __init__(
        self,
        bm25: BM25Index,
        semantic: Optional[SemanticIndex] = None,
        *,
        bm25_k: int = 20,
        sem_k: int = 20,
        top_n: int = 8,
    ) -> None:
        self._bm25 = bm25
        self._semantic = semantic
        self._bm25_k = bm25_k
        self._sem_k = sem_k
        self._top_n = top_n

    def search(self, query: str) -> List[Tuple[int, float]]:
        """Return (chunk_id, rrf_score) pairs, best first."""
        bm25_results = self._bm25.search(query, k=self._bm25_k)
        if self._semantic is not None:
            sem_results = self._semantic.search(query, k=self._sem_k)
            return _rrf_fuse(bm25_results, sem_results, top_n=self._top_n)
        return bm25_results[: self._top_n]
