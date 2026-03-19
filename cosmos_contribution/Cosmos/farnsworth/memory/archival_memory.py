"""
Farnsworth Archival Memory - Long-Term Vector Storage

Novel Approaches:
1. Hierarchical Indexing - Multi-level HNSW for scale
2. Temporal Weighting - Time-aware similarity scoring
3. Concept Clustering - Automatic topic organization
4. Hybrid Search - Dense + sparse retrieval fusion
"""

import asyncio
import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
import numpy as np

from loguru import logger


@dataclass
class ArchivalEntry:
    """An entry in archival memory."""
    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    cluster_id: Optional[int] = None

    # Retrieval statistics
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None
    relevance_feedback: float = 0.0  # Accumulated user feedback

    def to_dict(self) -> dict:
        """Serialize entry."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "cluster_id": self.cluster_id,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved.isoformat() if self.last_retrieved else None,
            "relevance_feedback": self.relevance_feedback,
        }


@dataclass
class SearchResult:
    """Result from archival search."""
    entry: ArchivalEntry
    score: float
    search_type: str  # "semantic", "keyword", "hybrid"
    debug_info: dict = field(default_factory=dict)


# =============================================================================
# HYBRID RETRIEVAL CONFIGURATION
# =============================================================================

@dataclass
class HybridWeights:
    """Configurable weights for hybrid retrieval scoring."""
    semantic: float = 0.4
    keyword: float = 0.2
    temporal: float = 0.2
    graph: float = 0.1
    attention: float = 0.1

    def normalize(self) -> "HybridWeights":
        """Ensure weights sum to 1.0."""
        total = self.semantic + self.keyword + self.temporal + self.graph + self.attention
        if total == 0:
            return HybridWeights()
        return HybridWeights(
            semantic=self.semantic / total,
            keyword=self.keyword / total,
            temporal=self.temporal / total,
            graph=self.graph / total,
            attention=self.attention / total,
        )


class ArchivalMemory:
    """
    Long-term memory storage with vector search capabilities.

    Features:
    - FAISS for efficient similarity search
    - ChromaDB integration for metadata filtering
    - Hybrid semantic + keyword retrieval
    - Automatic clustering for organization
    """

    def __init__(
        self,
        data_dir: str = "./data/archival",
        embedding_dim: int = 384,
        use_gpu: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu

        # Storage
        self.entries: dict[str, ArchivalEntry] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.id_to_index: dict[str, int] = {}
        self.index_to_id: dict[int, str] = {}

        # FAISS index
        self._faiss_index = None

        # Clustering
        self.cluster_centroids: Optional[np.ndarray] = None
        self.num_clusters: int = 0

        # BM25 for keyword search
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []

        # Embedding function (set externally)
        self.embed_fn: Optional[Callable] = None

        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize FAISS index and load existing data."""
        if self._initialized:
            return

        try:
            import faiss

            # Create FAISS index
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self._faiss_index = faiss.GpuIndexFlatIP(res, self.embedding_dim)
                    logger.info("Using GPU FAISS index")
                except Exception:
                    self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                    logger.info("GPU not available, using CPU FAISS index")
            else:
                self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            # Load existing data
            await self._load_from_disk()

            self._initialized = True
            logger.info(f"Archival memory initialized with {len(self.entries)} entries")

        except ImportError:
            logger.warning("FAISS not installed, using numpy fallback")
            self._faiss_index = None
            self._initialized = True

    async def store(
        self,
        content: str,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        embedding: Optional[list[float]] = None,
    ) -> str:
        """
        Store content in archival memory.

        Returns the entry ID.
        """
        async with self._lock:
            # Generate ID
            entry_id = hashlib.sha256(
                f"{content[:100]}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

            # Get embedding if not provided
            if embedding is None and self.embed_fn:
                embedding = await self._get_embedding(content)

            entry = ArchivalEntry(
                id=entry_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                tags=tags or [],
            )

            self.entries[entry_id] = entry

            # Update index
            if embedding:
                self._add_to_index(entry_id, embedding)

            # Update BM25 corpus
            self._update_bm25_corpus(content)

            # Persist
            await self._save_entry(entry)

            return entry_id

    async def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        filter_tags: Optional[list[str]] = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Hybrid search combining semantic and keyword retrieval.

        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic vs keyword (0-1)
            filter_tags: Only return entries with these tags
            min_score: Minimum score threshold

        Returns:
            List of SearchResult objects
        """
        async with self._lock:
            if not self.entries:
                return []

            results = []

            # Semantic search
            if semantic_weight > 0 and self.embed_fn:
                semantic_results = await self._semantic_search(query, top_k * 2)
                results.extend([
                    (r[0], r[1] * semantic_weight, "semantic")
                    for r in semantic_results
                ])

            # Keyword search
            if semantic_weight < 1:
                keyword_results = self._keyword_search(query, top_k * 2)
                results.extend([
                    (r[0], r[1] * (1 - semantic_weight), "keyword")
                    for r in keyword_results
                ])

            # Combine and deduplicate using reciprocal rank fusion
            combined = self._reciprocal_rank_fusion(results)

            # Apply filters
            filtered = []
            for entry_id, score, search_type in combined:
                if entry_id not in self.entries:
                    continue

                entry = self.entries[entry_id]

                # Tag filter
                if filter_tags and not any(t in entry.tags for t in filter_tags):
                    continue

                # Score filter
                if score < min_score:
                    continue

                # Apply temporal weighting (newer = slightly higher)
                age_days = (datetime.now() - entry.created_at).days
                temporal_boost = 1.0 / (1.0 + age_days / 365)
                adjusted_score = score * (0.9 + 0.1 * temporal_boost)

                # Apply relevance feedback
                feedback_boost = entry.relevance_feedback * 0.1
                adjusted_score += feedback_boost

                filtered.append(SearchResult(
                    entry=entry,
                    score=adjusted_score,
                    search_type=search_type,
                    debug_info={
                        "raw_score": score,
                        "temporal_boost": temporal_boost,
                        "feedback_boost": feedback_boost,
                    }
                ))

            # Sort by score and limit
            filtered.sort(key=lambda x: x.score, reverse=True)
            filtered = filtered[:top_k]

            # Update retrieval stats
            for result in filtered:
                result.entry.retrieval_count += 1
                result.entry.last_retrieved = datetime.now()

            return filtered

    async def _semantic_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Perform semantic search using embeddings."""
        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)

        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            import faiss
            # Normalize for cosine similarity
            faiss.normalize_L2(query_vec)
            scores, indices = self._faiss_index.search(query_vec, min(top_k, self._faiss_index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self.index_to_id:
                    entry_id = self.index_to_id[idx]
                    results.append((entry_id, float(score)))
            return results

        elif self.embeddings is not None:
            # Numpy fallback
            similarities = np.dot(self.embeddings, query_vec.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if idx in self.index_to_id:
                    entry_id = self.index_to_id[idx]
                    results.append((entry_id, float(similarities[idx])))
            return results

        return []

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Perform keyword search using BM25."""
        if self._bm25 is None:
            self._build_bm25()

        if self._bm25 is None:
            return []

        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Get top results
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        entry_ids = list(self.entries.keys())
        for idx in top_indices:
            if idx < len(entry_ids) and scores[idx] > 0:
                results.append((entry_ids[idx], float(scores[idx])))

        return results

    def _reciprocal_rank_fusion(
        self,
        results: list[tuple[str, float, str]],
        k: int = 60,
    ) -> list[tuple[str, float, str]]:
        """Combine results using reciprocal rank fusion."""
        # Group by entry_id
        entry_scores: dict[str, tuple[float, str]] = {}

        # Sort results by score within each search type
        semantic_results = sorted(
            [(id, s, t) for id, s, t in results if t == "semantic"],
            key=lambda x: x[1],
            reverse=True,
        )
        keyword_results = sorted(
            [(id, s, t) for id, s, t in results if t == "keyword"],
            key=lambda x: x[1],
            reverse=True,
        )

        # Compute RRF scores
        for rank, (entry_id, _, search_type) in enumerate(semantic_results):
            rrf_score = 1.0 / (k + rank + 1)
            if entry_id in entry_scores:
                entry_scores[entry_id] = (
                    entry_scores[entry_id][0] + rrf_score,
                    "hybrid",
                )
            else:
                entry_scores[entry_id] = (rrf_score, search_type)

        for rank, (entry_id, _, search_type) in enumerate(keyword_results):
            rrf_score = 1.0 / (k + rank + 1)
            if entry_id in entry_scores:
                entry_scores[entry_id] = (
                    entry_scores[entry_id][0] + rrf_score,
                    "hybrid",
                )
            else:
                entry_scores[entry_id] = (rrf_score, search_type)

        # Convert back to list
        combined = [
            (entry_id, score, search_type)
            for entry_id, (score, search_type) in entry_scores.items()
        ]

        return sorted(combined, key=lambda x: x[1], reverse=True)

    def _add_to_index(self, entry_id: str, embedding: list[float]):
        """Add embedding to FAISS index."""
        index = len(self.id_to_index)
        self.id_to_index[entry_id] = index
        self.index_to_id[index] = entry_id

        vec = np.array([embedding], dtype=np.float32)

        if self._faiss_index is not None:
            import faiss
            faiss.normalize_L2(vec)
            self._faiss_index.add(vec)

        # Also maintain numpy array for fallback
        if self.embeddings is None:
            self.embeddings = vec
        else:
            self.embeddings = np.vstack([self.embeddings, vec])

    def _update_bm25_corpus(self, content: str):
        """Update BM25 corpus with new document."""
        tokens = content.lower().split()
        self._corpus_tokens.append(tokens)
        self._bm25 = None  # Invalidate, rebuild on next search

    def _build_bm25(self):
        """Build BM25 index from corpus."""
        if not self._corpus_tokens:
            # Build from entries
            self._corpus_tokens = [
                entry.content.lower().split()
                for entry in self.entries.values()
            ]

        if self._corpus_tokens:
            try:
                from rank_bm25 import BM25Okapi
                self._bm25 = BM25Okapi(self._corpus_tokens)
            except ImportError:
                logger.warning("rank_bm25 not installed, keyword search disabled")

    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text."""
        if self.embed_fn is None:
            return None

        try:
            if asyncio.iscoroutinefunction(self.embed_fn):
                return await self.embed_fn(text)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.embed_fn, text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    async def _save_entry(self, entry: ArchivalEntry):
        """Save entry to disk."""
        entry_file = self.data_dir / f"{entry.id}.json"
        data = entry.to_dict()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: entry_file.write_text(json.dumps(data), encoding='utf-8')
        )

        # Save embedding separately (binary)
        if entry.embedding:
            emb_file = self.data_dir / f"{entry.id}.emb"
            await loop.run_in_executor(
                None,
                lambda: pickle.dump(entry.embedding, emb_file.open('wb'))
            )

    async def _load_from_disk(self):
        """Load all entries from disk."""
        for entry_file in self.data_dir.glob("*.json"):
            try:
                data = json.loads(entry_file.read_text(encoding='utf-8'))
                entry = ArchivalEntry(
                    id=data["id"],
                    content=data["content"],
                    metadata=data.get("metadata", {}),
                    created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
                    tags=data.get("tags", []),
                    cluster_id=data.get("cluster_id"),
                    retrieval_count=data.get("retrieval_count", 0),
                    relevance_feedback=data.get("relevance_feedback", 0.0),
                )

                # Load embedding
                emb_file = self.data_dir / f"{entry.id}.emb"
                if emb_file.exists():
                    entry.embedding = pickle.load(emb_file.open('rb'))

                self.entries[entry.id] = entry

                if entry.embedding:
                    self._add_to_index(entry.id, entry.embedding)

            except Exception as e:
                logger.error(f"Failed to load entry {entry_file}: {e}")

    # =========================================================================
    # CROSS-ATTENTION HYBRID RETRIEVAL (AGI Upgrade 2)
    # =========================================================================

    async def hybrid_recall(
        self,
        query: str,
        top_k: int = 10,
        oversample_factor: int = 3,
        use_attention: bool = True,
        weights: Optional[HybridWeights] = None,
        knowledge_graph=None,  # Optional KnowledgeGraph for context enrichment
    ) -> list[SearchResult]:
        """
        Enhanced hybrid retrieval combining semantic, keyword, temporal, and graph signals.

        Pipeline:
        1. Oversample candidates from FAISS (semantic) + BM25 (keyword)
        2. Apply temporal decay scores
        3. Fetch graph neighbors for context enrichment
        4. Cross-attention reranking
        5. Return top_k

        Args:
            query: Search query
            top_k: Number of results to return
            oversample_factor: Oversample ratio for initial retrieval
            use_attention: Whether to apply cross-attention reranking
            weights: Custom weights for score combination
            knowledge_graph: Optional KnowledgeGraph for context enrichment

        Returns:
            List of SearchResult objects with enhanced scores
        """
        async with self._lock:
            if not self.entries:
                return []

            weights = (weights or HybridWeights()).normalize()
            candidate_k = top_k * oversample_factor

            # Step 1: Oversample candidates
            candidates = []  # List of (entry_id, score, source)

            # Semantic search
            if self.embed_fn:
                semantic_results = await self._semantic_search(query, candidate_k)
                for entry_id, score in semantic_results:
                    candidates.append((entry_id, score * weights.semantic, "semantic"))

            # Keyword search
            keyword_results = self._keyword_search(query, candidate_k)
            for entry_id, score in keyword_results:
                candidates.append((entry_id, score * weights.keyword, "keyword"))

            # Combine using RRF
            combined = self._reciprocal_rank_fusion(candidates)

            # Step 2: Apply temporal decay
            temporal_scored = []
            for entry_id, score, search_type in combined[:candidate_k]:
                if entry_id not in self.entries:
                    continue

                entry = self.entries[entry_id]
                age_days = (datetime.now() - entry.created_at).days
                temporal_boost = 1.0 / (1.0 + age_days / 365)
                temporal_score = score + temporal_boost * weights.temporal

                temporal_scored.append((entry_id, temporal_score, search_type))

            # Step 3: Fetch graph context if available
            graph_context = []
            if knowledge_graph:
                candidate_ids = [c[0] for c in temporal_scored[:top_k * 2]]
                graph_context = await self._fetch_graph_context(
                    query, candidate_ids, knowledge_graph, max_hops=1
                )

            # Step 4: Cross-attention reranking
            if use_attention and self.embed_fn:
                query_embedding = await self._get_embedding(query)
                if query_embedding:
                    reranked = self._cross_attention_rerank(
                        query_embedding,
                        temporal_scored,
                        graph_context,
                        weights,
                    )
                    temporal_scored = reranked

            # Step 5: Build final results
            results = []
            for entry_id, final_score, search_type in temporal_scored[:top_k]:
                if entry_id not in self.entries:
                    continue

                entry = self.entries[entry_id]
                entry.retrieval_count += 1
                entry.last_retrieved = datetime.now()

                results.append(SearchResult(
                    entry=entry,
                    score=final_score,
                    search_type="hybrid_attention" if use_attention else search_type,
                    debug_info={
                        "weights": {
                            "semantic": weights.semantic,
                            "keyword": weights.keyword,
                            "temporal": weights.temporal,
                            "graph": weights.graph,
                            "attention": weights.attention,
                        },
                        "graph_context_count": len(graph_context),
                    }
                ))

            return results

    def _cross_attention_rerank(
        self,
        query_embedding: list[float],
        candidates: list[tuple[str, float, str]],
        graph_context: Optional[list] = None,
        weights: Optional[HybridWeights] = None,
    ) -> list[tuple[str, float, str]]:
        """
        Rerank candidates using cross-attention mechanism.

        Attention: softmax(query @ candidates.T / sqrt(dim))
        Final score = original + attention_weight + graph_boost

        Args:
            query_embedding: Query vector
            candidates: List of (entry_id, score, source)
            graph_context: Related entities from knowledge graph
            weights: Scoring weights

        Returns:
            Reranked list of (entry_id, final_score, source)
        """
        weights = weights or HybridWeights()

        if not candidates:
            return []

        query_vec = np.array(query_embedding)
        dim = len(query_embedding)

        # Build candidate embedding matrix
        candidate_embeddings = []
        valid_candidates = []

        for entry_id, score, source in candidates:
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                if entry.embedding:
                    candidate_embeddings.append(entry.embedding)
                    valid_candidates.append((entry_id, score, source))

        if not candidate_embeddings:
            return candidates

        # Compute attention scores
        candidate_matrix = np.array(candidate_embeddings)  # Shape: (n_candidates, dim)

        # Attention: softmax(Q @ K^T / sqrt(d))
        attention_scores = np.dot(candidate_matrix, query_vec) / np.sqrt(dim)

        # Softmax normalization
        attention_weights = np.exp(attention_scores - np.max(attention_scores))
        attention_weights = attention_weights / attention_weights.sum()

        # Graph context boost
        graph_boost = {}
        if graph_context:
            graph_entity_ids = {e.id for e in graph_context if hasattr(e, 'id')}
            for entry_id, _, _ in valid_candidates:
                if entry_id in self.entries:
                    entry = self.entries[entry_id]
                    # Check if entry content mentions graph entities
                    content_lower = entry.content.lower()
                    boost = sum(
                        1 for e in graph_context
                        if hasattr(e, 'name') and e.name.lower() in content_lower
                    )
                    graph_boost[entry_id] = min(1.0, boost * 0.2)

        # Compute final scores
        reranked = []
        for i, (entry_id, original_score, source) in enumerate(valid_candidates):
            attention_contribution = attention_weights[i] * weights.attention
            graph_contribution = graph_boost.get(entry_id, 0) * weights.graph

            final_score = original_score + attention_contribution + graph_contribution
            reranked.append((entry_id, final_score, source))

        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    async def _fetch_graph_context(
        self,
        query: str,
        candidate_ids: list[str],
        knowledge_graph,
        max_hops: int = 1,
    ) -> list:
        """
        Fetch related entities from knowledge graph for context enrichment.

        Args:
            query: Search query
            candidate_ids: IDs of candidate entries
            knowledge_graph: KnowledgeGraph instance
            max_hops: Maximum graph traversal depth

        Returns:
            List of related Entity objects
        """
        if not knowledge_graph:
            return []

        try:
            # Query the knowledge graph
            graph_result = await knowledge_graph.query(query, max_hops=max_hops)
            return graph_result.entities if hasattr(graph_result, 'entities') else []
        except Exception as e:
            logger.debug(f"Graph context fetch failed: {e}")
            return []

    async def add_feedback(self, entry_id: str, feedback: float):
        """Add relevance feedback to an entry (-1 to 1)."""
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            entry.relevance_feedback = max(-1, min(1,
                entry.relevance_feedback * 0.9 + feedback * 0.1
            ))
            await self._save_entry(entry)

    async def get_entry(self, entry_id: str) -> Optional[ArchivalEntry]:
        """Get a specific entry by ID."""
        return self.entries.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        if entry_id not in self.entries:
            return False

        del self.entries[entry_id]

        # Remove files
        entry_file = self.data_dir / f"{entry_id}.json"
        emb_file = self.data_dir / f"{entry_id}.emb"

        if entry_file.exists():
            entry_file.unlink()
        if emb_file.exists():
            emb_file.unlink()

        # Note: FAISS index rebuild would be needed for proper removal
        # For now, entries are effectively orphaned in the index

        return True

    def get_stats(self) -> dict:
        """Get archival memory statistics."""
        return {
            "total_entries": len(self.entries),
            "indexed_entries": len(self.id_to_index),
            "total_retrievals": sum(e.retrieval_count for e in self.entries.values()),
            "has_faiss": self._faiss_index is not None,
            "has_bm25": self._bm25 is not None,
            "storage_path": str(self.data_dir),
        }

    def set_huggingface_embeddings(self, model: str = "all-MiniLM-L6-v2"):
        """
        Configure archival memory to use HuggingFace embeddings.

        Uses local sentence-transformers for fast, private embedding generation.

        Args:
            model: HuggingFace embedding model ID (default: all-MiniLM-L6-v2)
        """
        try:
            from farnsworth.integration.external.huggingface import get_huggingface_provider

            hf = get_huggingface_provider()
            if hf is None:
                logger.warning("HuggingFace provider not available for embeddings")
                return False

            async def hf_embed(text: str) -> list[float]:
                result = await hf.embed(text, model=model, prefer_local=True)
                embeddings = result.get("embeddings")
                if embeddings:
                    # Update embedding dim if needed
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        self.embedding_dim = len(embeddings)
                    return embeddings
                return None

            self.embed_fn = hf_embed
            logger.info(f"Archival memory using HuggingFace embeddings: {model}")
            return True

        except ImportError:
            logger.warning("HuggingFace integration not available")
            return False


def create_archival_with_huggingface(
    data_dir: str = "./data/archival",
    embedding_model: str = "all-MiniLM-L6-v2",
    use_gpu: bool = True
) -> ArchivalMemory:
    """
    Create an archival memory instance with HuggingFace embeddings.

    This provides:
    - Local sentence-transformer embeddings (no API needed)
    - GPU acceleration when available
    - Semantic search over long-term memory

    Args:
        data_dir: Directory for storing memory files
        embedding_model: HuggingFace model for embeddings
        use_gpu: Whether to use GPU for FAISS

    Returns:
        ArchivalMemory instance configured with HuggingFace
    """
    memory = ArchivalMemory(data_dir=data_dir, use_gpu=use_gpu)
    memory.set_huggingface_embeddings(embedding_model)
    return memory
