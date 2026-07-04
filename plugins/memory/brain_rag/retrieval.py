"""Hybrid BM25 + vector retrieval with RRF fusion and MMR reranking."""

from __future__ import annotations

from typing import Dict, List, Set

from .embeddings import cosine_similarity, hash_embed, tfidf_embed
from .store import BrainRAGStore


class BrainRAGRetriever:
    """Advanced hybrid retriever combining lexical and semantic signals."""

    def __init__(
        self,
        store: BrainRAGStore,
        *,
        bm25_weight: float = 0.45,
        vector_weight: float = 0.55,
        rrf_k: int = 60,
    ):
        self.store = store
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

    def search(self, query: str, *, limit: int = 8) -> List[Dict]:
        """Hybrid search across document chunks and explicit memories."""
        query = (query or "").strip()
        if not query:
            return []

        fts_chunks = self.store.fts_search_chunks(query, limit=limit * 3)
        fts_memories = self.store.fts_search_memories(query, limit=limit * 2)

        # Vector scan over recent corpus (local scale — no external vector DB)
        query_vec = self._query_vector(query)
        vector_hits = self._vector_scan(query_vec, limit=limit * 3)

        fused = self._reciprocal_rank_fusion(
            [
                ("bm25_chunk", fts_chunks),
                ("bm25_memory", fts_memories),
                ("vector", vector_hits),
            ]
        )

        # MMR diversity rerank
        ranked = self._mmr_rerank(query_vec, fused, limit=limit)
        return ranked

    def _query_vector(self, query: str) -> List[float]:
        idf = self.store._idf
        if idf:
            return tfidf_embed(query, idf)
        return hash_embed(query)

    def _vector_scan(self, query_vec: List[float], limit: int) -> List[Dict]:
        candidates = self.store.all_chunks(limit=300) + self.store.all_memories(limit=100)
        scored: List[Dict] = []
        for hit in candidates:
            vec = hit.get("vector") or []
            sim = cosine_similarity(query_vec, vec)
            if sim <= 0.01:
                continue
            entry = dict(hit)
            entry["vector_score"] = sim
            scored.append(entry)
        scored.sort(key=lambda x: x["vector_score"], reverse=True)
        return scored[:limit]

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[tuple],
    ) -> List[Dict]:
        """RRF fusion across multiple retrieval strategies."""
        scores: Dict[str, float] = {}
        items: Dict[str, Dict] = {}
        for _name, hits in ranked_lists:
            for rank, hit in enumerate(hits, start=1):
                key = f"{hit.get('kind', 'chunk')}:{hit['id']}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                if key not in items:
                    items[key] = dict(hit)
                else:
                    items[key].update({k: v for k, v in hit.items() if v is not None})
        fused = []
        for key, score in scores.items():
            entry = dict(items[key])
            entry["rrf_score"] = score
            fused.append(entry)
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    def _mmr_rerank(
        self,
        query_vec: List[float],
        candidates: List[Dict],
        limit: int,
        lambda_mult: float = 0.7,
    ) -> List[Dict]:
        """Maximal Marginal Relevance for diverse top-k results."""
        if not candidates:
            return []
        selected: List[Dict] = []
        selected_vecs: List[List[float]] = []
        pool = list(candidates)
        used: Set[str] = set()

        while pool and len(selected) < limit:
            best_idx = 0
            best_score = -1.0
            for i, cand in enumerate(pool):
                key = f"{cand.get('kind', 'chunk')}:{cand['id']}"
                if key in used:
                    continue
                vec = cand.get("vector") or []
                relevance = cosine_similarity(query_vec, vec) if vec else cand.get("rrf_score", 0.0)
                diversity = 0.0
                if selected_vecs and vec:
                    diversity = max(cosine_similarity(vec, sv) for sv in selected_vecs)
                mmr = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            chosen = pool.pop(best_idx)
            key = f"{chosen.get('kind', 'chunk')}:{chosen['id']}"
            used.add(key)
            chosen["score"] = round(best_score, 4)
            selected.append(chosen)
            if chosen.get("vector"):
                selected_vecs.append(chosen["vector"])
        return selected

    def format_context(self, hits: List[Dict], *, max_chars: int = 3000) -> str:
        """Format retrieval hits for model context injection."""
        if not hits:
            return ""
        lines: List[str] = []
        total = 0
        for i, hit in enumerate(hits, 1):
            kind = hit.get("kind", "chunk")
            source = hit.get("source") or hit.get("category") or "knowledge"
            snippet = (hit.get("content") or "").strip()
            if not snippet:
                continue
            line = f"[{i}] ({kind}/{source}) {snippet}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)
