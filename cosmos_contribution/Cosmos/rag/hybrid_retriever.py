"""
cosmos Hybrid Retriever - Semantic + Keyword Search

Novel Approaches:
1. Reciprocal Rank Fusion - Combine multiple ranking signals
2. Query Expansion - Augment queries with related terms
3. Adaptive Weighting - Learn optimal weights from feedback
4. Contextual Reranking - Use LLM for final ranking
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from collections import defaultdict

from loguru import logger

try:
    import numpy as np
except ImportError:
    np = None

from Cosmos.rag.embeddings import EmbeddingManager


@dataclass
class Document:
    """A document for retrieval."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    bm25_tokens: Optional[list[str]] = None


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    document: Document
    score: float
    search_type: str  # "semantic", "keyword", "hybrid"
    debug_info: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Hybrid retrieval combining semantic and keyword search.

    Features:
    - FAISS for semantic search
    - BM25 for keyword search
    - Reciprocal Rank Fusion for combining
    - Optional LLM reranking
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        semantic_weight: float = 0.7,
        use_reranking: bool = False,
    ):
        self.embedding_manager = embedding_manager
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self.use_reranking = use_reranking

        # Document storage
        self.documents: dict[str, Document] = {}
        self.embeddings: Optional[Any] = None  # numpy array
        self.id_to_index: dict[str, int] = {}
        self.index_to_id: dict[int, str] = {}

        # BM25 index
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []

        # FAISS index
        self._faiss_index = None

        # Statistics for adaptive weighting
        self._feedback_history: list[dict] = []

        # Reranking LLM (set externally)
        self.rerank_llm = None

    async def add_document(self, doc: Document):
        """Add a document to the retriever."""
        # Get embedding if not present
        if doc.embedding is None:
            result = await self.embedding_manager.embed(doc.content)
            doc.embedding = result.embedding

        # Tokenize for BM25
        if doc.bm25_tokens is None:
            doc.bm25_tokens = self._tokenize(doc.content)

        self.documents[doc.id] = doc

        # Update indices
        self._add_to_semantic_index(doc)
        self._add_to_keyword_index(doc)

    async def add_documents(self, docs: list[Document]):
        """Add multiple documents."""
        # Batch embed
        texts_to_embed = [d.content for d in docs if d.embedding is None]
        if texts_to_embed:
            results = await self.embedding_manager.embed_batch(texts_to_embed)
            embed_idx = 0
            for doc in docs:
                if doc.embedding is None:
                    doc.embedding = results[embed_idx].embedding
                    embed_idx += 1

        for doc in docs:
            if doc.bm25_tokens is None:
                doc.bm25_tokens = self._tokenize(doc.content)
            self.documents[doc.id] = doc
            self._add_to_semantic_index(doc)
            self._add_to_keyword_index(doc)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        import re
        # Simple tokenization
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove stopwords (simple list)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                     'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                     'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same',
                     'so', 'than', 'too', 'very', 'just', 'but', 'and', 'or', 'if', 'because',
                     'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                     'against', 'between', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def _add_to_semantic_index(self, doc: Document):
        """Add document to semantic index."""
        if doc.embedding is None:
            return

        index = len(self.id_to_index)
        self.id_to_index[doc.id] = index
        self.index_to_id[index] = doc.id

        vec = np.array([doc.embedding], dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = vec
        else:
            self.embeddings = np.vstack([self.embeddings, vec])

        # Update FAISS index
        try:
            import faiss
            if self._faiss_index is None:
                dim = len(doc.embedding)
                self._faiss_index = faiss.IndexFlatIP(dim)

            faiss.normalize_L2(vec)
            self._faiss_index.add(vec)
        except ImportError:
            pass  # Use numpy fallback

    def _add_to_keyword_index(self, doc: Document):
        """Add document to BM25 index."""
        if doc.bm25_tokens:
            self._corpus_tokens.append(doc.bm25_tokens)
            self._bm25 = None  # Invalidate, rebuild on search

    def _build_bm25(self):
        """Build BM25 index."""
        if not self._corpus_tokens:
            return

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus_tokens)
        except ImportError:
            logger.warning("rank_bm25 not installed")

    async def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: Optional[float] = None,
        filter_fn: Optional[Callable] = None,
        expand_query: bool = True,
    ) -> list[RetrievalResult]:
        """
        Hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Override default weight
            filter_fn: Optional filter function
            expand_query: Whether to expand query with synonyms

        Returns:
            List of RetrievalResult
        """
        if not self.documents:
            return []

        sem_weight = semantic_weight if semantic_weight is not None else self.semantic_weight
        kw_weight = 1.0 - sem_weight

        # Optionally expand query
        expanded_query = query
        if expand_query:
            expanded_query = self._expand_query(query)

        results = []

        # Semantic search
        if sem_weight > 0:
            semantic_results = await self._semantic_search(expanded_query, top_k * 2)
            for doc_id, score in semantic_results:
                results.append((doc_id, score * sem_weight, "semantic"))

        # Keyword search
        if kw_weight > 0:
            keyword_results = self._keyword_search(expanded_query, top_k * 2)
            for doc_id, score in keyword_results:
                results.append((doc_id, score * kw_weight, "keyword"))

        # Combine using RRF
        combined = self._reciprocal_rank_fusion(results)

        # Apply filter
        filtered = []
        for doc_id, score, search_type in combined:
            if doc_id not in self.documents:
                continue

            doc = self.documents[doc_id]

            if filter_fn and not filter_fn(doc):
                continue

            filtered.append(RetrievalResult(
                document=doc,
                score=score,
                search_type=search_type,
            ))

        filtered = filtered[:top_k]

        # Optional reranking
        if self.use_reranking and self.rerank_llm and filtered:
            filtered = await self._rerank(query, filtered)

        return filtered

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Semantic search using embeddings."""
        result = await self.embedding_manager.embed(query)
        query_vec = np.array([result.embedding], dtype=np.float32)

        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            import faiss
            faiss.normalize_L2(query_vec)
            scores, indices = self._faiss_index.search(
                query_vec, min(top_k, self._faiss_index.ntotal)
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self.index_to_id:
                    doc_id = self.index_to_id[idx]
                    results.append((doc_id, float(score)))
            return results

        elif self.embeddings is not None:
            # Numpy fallback
            similarities = np.dot(self.embeddings, query_vec.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if idx in self.index_to_id:
                    doc_id = self.index_to_id[idx]
                    results.append((doc_id, float(similarities[idx])))
            return results

        return []

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Keyword search using BM25."""
        if self._bm25 is None:
            self._build_bm25()

        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Normalize scores
        max_score = max(scores) if max(scores) > 0 else 1
        scores = [s / max_score for s in scores]

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        doc_ids = list(self.documents.keys())
        for idx in top_indices:
            if idx < len(doc_ids) and scores[idx] > 0:
                results.append((doc_ids[idx], float(scores[idx])))

        return results

    def _reciprocal_rank_fusion(
        self,
        results: list[tuple[str, float, str]],
        k: int = 60,
    ) -> list[tuple[str, float, str]]:
        """Combine results using Reciprocal Rank Fusion."""
        # Group by source
        semantic_results = sorted(
            [(id, s, t) for id, s, t in results if t == "semantic"],
            key=lambda x: x[1], reverse=True
        )
        keyword_results = sorted(
            [(id, s, t) for id, s, t in results if t == "keyword"],
            key=lambda x: x[1], reverse=True
        )

        # Compute RRF scores
        rrf_scores: dict[str, tuple[float, str]] = {}

        for rank, (doc_id, _, _) in enumerate(semantic_results):
            score = 1.0 / (k + rank + 1)
            if doc_id in rrf_scores:
                rrf_scores[doc_id] = (rrf_scores[doc_id][0] + score, "hybrid")
            else:
                rrf_scores[doc_id] = (score, "semantic")

        for rank, (doc_id, _, _) in enumerate(keyword_results):
            score = 1.0 / (k + rank + 1)
            if doc_id in rrf_scores:
                rrf_scores[doc_id] = (rrf_scores[doc_id][0] + score, "hybrid")
            else:
                rrf_scores[doc_id] = (score, "keyword")

        # Sort by combined score
        combined = [
            (doc_id, score, search_type)
            for doc_id, (score, search_type) in rrf_scores.items()
        ]
        return sorted(combined, key=lambda x: x[1], reverse=True)

    def _expand_query(self, query: str) -> str:
        """Expand query with related terms."""
        # Simple synonym expansion
        synonyms = {
            "error": "bug issue problem",
            "fix": "solve resolve repair",
            "create": "make build generate",
            "fast": "quick rapid speedy",
            "slow": "sluggish delayed",
            "help": "assist support",
            "find": "search locate discover",
            "show": "display present",
        }

        words = query.lower().split()
        expanded_words = []

        for word in words:
            expanded_words.append(word)
            if word in synonyms:
                expanded_words.extend(synonyms[word].split())

        return " ".join(expanded_words)

    async def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Rerank results using LLM."""
        if not self.rerank_llm:
            return results

        # Build reranking prompt
        docs_text = "\n".join([
            f"[{i}] {r.document.content[:200]}..."
            for i, r in enumerate(results)
        ])

        prompt = f"""Rank these documents by relevance to the query.
Query: {query}

Documents:
{docs_text}

Return the indices in order of relevance (most relevant first), as comma-separated numbers."""

        try:
            response = await self.rerank_llm.generate(prompt)
            # Parse response to get order
            import re
            indices = [int(i) for i in re.findall(r'\d+', response.text)]

            # Reorder results
            reranked = []
            seen = set()
            for idx in indices:
                if 0 <= idx < len(results) and idx not in seen:
                    seen.add(idx)
                    result = results[idx]
                    result.debug_info["rerank_position"] = len(reranked)
                    reranked.append(result)

            # Add any not mentioned
            for i, result in enumerate(results):
                if i not in seen:
                    reranked.append(result)

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    def add_feedback(self, query: str, doc_id: str, relevant: bool):
        """Add relevance feedback for adaptive weighting."""
        self._feedback_history.append({
            "query": query,
            "doc_id": doc_id,
            "relevant": relevant,
        })

        # Periodically adapt weights
        if len(self._feedback_history) >= 50:
            self._adapt_weights()

    def _adapt_weights(self):
        """Adapt semantic/keyword weights based on feedback."""
        # Simple adaptation: if keyword-found docs are often relevant, increase keyword weight
        # This is a placeholder for more sophisticated learning
        pass

    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "document_count": len(self.documents),
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "has_faiss": self._faiss_index is not None,
            "has_bm25": self._bm25 is not None,
            "feedback_count": len(self._feedback_history),
        }
