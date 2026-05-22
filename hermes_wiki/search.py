from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import chunk_markdown, parse_frontmatter

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize_for_sparse_search(text: str) -> list[str]:
    """Tokenize text for deterministic lexical/sparse-style matching.

    The tokenizer intentionally keeps underscores inside tokens for config keys
    while splitting paths, hyphenated model names, and dotted versions into
    searchable literal parts.
    """

    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _result_key(result: WikiSearchResult) -> tuple[str, int]:
    return (result.page_path, result.chunk_index)


def reciprocal_rank_fusion(
    ranked_lists: list[list[WikiSearchResult]],
    *,
    limit: int,
    k: int = 60,
) -> list[WikiSearchResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion."""

    fused_scores: dict[tuple[str, int], float] = {}
    representatives: dict[tuple[str, int], WikiSearchResult] = {}
    best_rank: dict[tuple[str, int], int] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            key = _result_key(result)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            representatives.setdefault(key, result)
            best_rank[key] = min(best_rank.get(key, rank), rank)

    ordered_keys = sorted(
        fused_scores,
        key=lambda key: (-fused_scores[key], best_rank[key], representatives[key].page_path, representatives[key].chunk_index),
    )
    fused: list[WikiSearchResult] = []
    for key in ordered_keys[: max(1, int(limit or 1))]:
        base = representatives[key]
        fused.append(
            WikiSearchResult(
                page_path=base.page_path,
                title=base.title,
                page_type=base.page_type,
                chunk_index=base.chunk_index,
                text=base.text,
                score=fused_scores[key],
                tags=base.tags,
            )
        )
    return fused


class _VectorCoreAsyncBridge:
    """Dedicated event-loop thread for vector-core async clients used synchronously."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="llm-wiki-vector-core", daemon=True)
        self._thread.start()
        self._closed = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        if self._closed:
            raise RuntimeError("vector-core async bridge is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()


def _run_vector_core_async(coro):
    """Run a vector-core coroutine for tests/one-shot utility code."""

    bridge = _VectorCoreAsyncBridge()
    try:
        return bridge.run(coro)
    finally:
        bridge.close()


class WikiSearchDependencyError(RuntimeError):
    """Raised when optional LLM Wiki search dependencies are missing."""


class EmbeddingCache:
    """Hermes adapter around vector-core's persistent embedding cache.

    vector-core keys cache rows by a caller-provided content hash. Hermes folds
    model and dimension into that hash so cache rows cannot leak across
    embedding model migrations while still using vector-core's SQLite/WAL,
    JSON serialization, corruption handling, stats, and LRU behavior.
    """

    def __init__(self, cache_path: Path | str, *, max_entries: int = 100000):
        try:
            from vector_core.embeddings import EmbeddingCache as VectorCoreEmbeddingCache
        except Exception as exc:  # pragma: no cover - dependency guard
            raise WikiSearchDependencyError(
                "vector-core is required for LLM Wiki embeddings; install hermes-agent[llm-wiki]"
            ) from exc

        self._vector_cache = VectorCoreEmbeddingCache(
            Path(cache_path).expanduser(),
            max_entries=max(1, int(max_entries or 1)),
        )
        self._ensure_vector_core_schema()

    @property
    def cache_path(self) -> Path:
        return Path(self._vector_cache.cache_path)

    @staticmethod
    def key_for(text: str, *, model: str, dim: int) -> str:
        payload = f"{model}\0{int(dim)}\0{text or ''}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_vector_core_schema(self) -> None:
        """Drop the old Hermes-local cache schema if this profile already created it."""

        conn = self._vector_cache._get_conn()
        rows = conn.execute("PRAGMA table_info(embeddings)").fetchall()
        columns = {row[1] for row in rows}
        if rows and "content_hash" not in columns:
            conn.execute("DROP TABLE embeddings")
            conn.commit()
            self._vector_cache._init_db()

    def get(self, cache_key: str) -> list[float] | None:
        return self._vector_cache.get(cache_key)

    def set(self, cache_key: str, embedding: list[float], *, model: str, dim: int) -> None:
        # The namespaced cache_key already includes model+dim. model is still
        # passed through for vector-core metadata and operator inspection.
        self._vector_cache.set(cache_key, embedding, model=model)

    def stats(self) -> dict[str, float | int]:
        return self._vector_cache.stats()

    def close(self) -> None:
        self._vector_cache.close()


class _EmbeddingClient:
    """Synchronous wrapper around vector-core's OpenAI-compatible embedding client."""

    def __init__(self, config: WikiConfig, *, cache_enabled: bool = True):
        try:
            from vector_core.embeddings import EmbeddingClient as VectorCoreEmbeddingClient
        except Exception as exc:  # pragma: no cover - dependency guard
            raise WikiSearchDependencyError(
                "vector-core is required for LLM Wiki embeddings; install hermes-agent[llm-wiki]"
            ) from exc

        base_url = config.embedding_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        self._client = VectorCoreEmbeddingClient(
            base_url=base_url,
            model=config.embedding_model,
            dim=config.embedding_dim,
            timeout=60.0,
        )
        self._bridge = _VectorCoreAsyncBridge()
        self._model = config.embedding_model
        self._dim = config.embedding_dim
        self._cache: EmbeddingCache | None = None
        if cache_enabled:
            cache_path = config.embedding_cache_path or (config.wiki_path / ".cache" / "embeddings.sqlite3")
            self._cache = EmbeddingCache(cache_path, max_entries=config.embedding_cache_max_entries)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        missing_positions: dict[str, list[int]] = {}

        for position, text in enumerate(texts):
            cache_key = EmbeddingCache.key_for(text, model=self._model, dim=self._dim)
            if self._cache is not None:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    results[position] = cached
                    continue
            if cache_key not in missing_positions:
                missing_texts.append(text)
                missing_keys.append(cache_key)
                missing_positions[cache_key] = []
            missing_positions[cache_key].append(position)

        if missing_texts:
            vectors = [self._normalize_dim(list(vector)) for vector in self._bridge.run(self._client.embed_batch(missing_texts))]
            for cache_key, vector in zip(missing_keys, vectors):
                if self._cache is not None:
                    self._cache.set(cache_key, vector, model=self._model, dim=self._dim)
                for position in missing_positions[cache_key]:
                    results[position] = vector

        return [vector if vector is not None else [] for vector in results]

    def embed_single(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def _normalize_dim(self, vector: list[float]) -> list[float]:
        if len(vector) == self._dim:
            return vector
        if len(vector) > self._dim:
            return vector[: self._dim]
        return vector + [0.0] * (self._dim - len(vector))

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
        close = getattr(self._client, "close", None)
        if close:
            self._bridge.run(close())
        self._bridge.close()


class WikiSearch:
    """Semantic search over wiki pages using Qdrant and OpenAI-compatible embeddings."""

    def __init__(self, config: WikiConfig, *, ensure_collection: bool = True, read_only: bool = False):
        self.config = config
        self.read_only = read_only
        try:
            from qdrant_client import QdrantClient, models
        except Exception as exc:  # pragma: no cover - dependency guard
            raise WikiSearchDependencyError(
                "qdrant-client is required for LLM Wiki vector search; install hermes-agent[llm-wiki]"
            ) from exc

        self._models = models
        self._client = QdrantClient(url=config.qdrant_url)
        self._embedder = _EmbeddingClient(config, cache_enabled=not read_only)
        if ensure_collection:
            self._ensure_collection()

    def _ensure_collection(self) -> None:
        models = self._models
        try:
            self._client.get_collection(self.config.collection_name)
        except Exception:
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.config.embedding_dim,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

        for field in ["page_path", "page_type", "tags"]:
            try:
                self._client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Existing indexes and older Qdrant versions are both safe to ignore.
                pass

    def index_page(self, page_path: Path) -> int:
        """Index a single wiki page. Returns number of chunks indexed."""
        if not page_path.exists():
            return 0

        text = page_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if not body.strip():
            return 0

        rel_path = str(page_path.relative_to(self.config.wiki_path))

        title = fm.get("title", page_path.stem)
        page_type = fm.get("type", "unknown")
        tags = fm.get("tags", [])
        updated = str(fm.get("updated", ""))

        chunks = chunk_markdown(body, max_tokens=self.config.chunk_max_tokens)
        if not chunks:
            return 0

        vectors = self._embedder.embed_batch(chunks)
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(self._point(
                point_id=self._chunk_id(rel_path, i),
                vector=vector,
                payload={
                    "page_path": rel_path,
                    "page_type": page_type,
                    "title": title,
                    "tags": tags if isinstance(tags, list) else [tags],
                    "updated": updated,
                    "chunk_index": i,
                    "text": chunk[:2000],
                },
            ))

        self._client.upsert(self.config.collection_name, points=points)
        self._delete_stale_page_chunks(rel_path, keep_chunks=len(points))
        logger.info("Indexed %d chunks from %s", len(chunks), rel_path)
        return len(chunks)

    def index_source(self, source_path: Path) -> int:
        """Index a raw source file. Returns number of chunks indexed."""
        if not source_path.exists():
            return 0

        text = source_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if not body.strip():
            return 0

        rel_path = str(source_path.relative_to(self.config.wiki_path))

        chunks = chunk_markdown(body, max_tokens=self.config.chunk_max_tokens)
        if not chunks:
            return 0

        vectors = self._embedder.embed_batch(chunks)
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(self._point(
                point_id=self._chunk_id(rel_path, i),
                vector=vector,
                payload={
                    "page_path": rel_path,
                    "page_type": "source",
                    "title": source_path.stem,
                    "tags": [],
                    "updated": str(fm.get("ingested", "")),
                    "chunk_index": i,
                    "text": chunk[:2000],
                },
            ))

        self._client.upsert(self.config.collection_name, points=points)
        self._delete_stale_page_chunks(rel_path, keep_chunks=len(points))
        logger.info("Indexed %d source chunks from %s", len(chunks), rel_path)
        return len(chunks)

    def _point(self, point_id: str, vector: list[float], payload: dict[str, Any]):
        models = self._models
        return models.PointStruct(
            id=point_id,
            vector={"dense": vector, "sparse": models.SparseVector(indices=[], values=[])},
            payload=payload,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        page_type: str | None = None,
        tags: list[str] | None = None,
        exclude_sources: bool = False,
        search_mode: str = "dense",
    ) -> list[WikiSearchResult]:
        """Search across the wiki.

        Modes:
        - dense: semantic vector search (default for backward compatibility)
        - sparse: deterministic lexical search over indexed payload text
        - hybrid: dense + sparse fused with Reciprocal Rank Fusion
        """

        mode = (search_mode or "dense").strip().lower()
        if mode == "dense":
            return self._dense_search(
                query,
                limit=limit,
                page_type=page_type,
                tags=tags,
                exclude_sources=exclude_sources,
            )
        if mode == "sparse":
            return self.sparse_search(
                query,
                limit=limit,
                page_type=page_type,
                tags=tags,
                exclude_sources=exclude_sources,
            )
        if mode == "hybrid":
            dense_results = self._dense_search(
                query,
                limit=max(limit * 2, limit),
                page_type=page_type,
                tags=tags,
                exclude_sources=exclude_sources,
            )
            sparse_results = self.sparse_search(
                query,
                limit=max(limit * 2, limit),
                page_type=page_type,
                tags=tags,
                exclude_sources=exclude_sources,
            )
            return reciprocal_rank_fusion([dense_results, sparse_results], limit=limit)
        raise ValueError("search_mode must be one of: dense, sparse, hybrid")

    def _dense_search(
        self,
        query: str,
        *,
        limit: int,
        page_type: str | None = None,
        tags: list[str] | None = None,
        exclude_sources: bool = False,
    ) -> list[WikiSearchResult]:
        """Semantic search across the wiki."""
        query_vector = self._embedder.embed_single(query)
        models = self._models

        filter_conditions = []
        if page_type:
            filter_conditions.append(models.FieldCondition(key="page_type", match=models.MatchValue(value=page_type)))
        if tags:
            for tag in tags:
                filter_conditions.append(models.FieldCondition(key="tags", match=models.MatchValue(value=tag)))

        response = self._client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            using="dense",
            query_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
            limit=limit * 2 if exclude_sources else limit,
            with_payload=True,
        )
        hits = response.points

        results = []
        for h in hits:
            payload = h.payload or {}
            if exclude_sources and payload.get("page_type") == "source":
                continue
            results.append(WikiSearchResult(
                page_path=payload.get("page_path", ""),
                title=payload.get("title", ""),
                page_type=payload.get("page_type", ""),
                chunk_index=payload.get("chunk_index", 0),
                text=payload.get("text", ""),
                score=h.score,
                tags=payload.get("tags", []),
            ))

        return results[:limit]

    def sparse_search(
        self,
        query: str,
        limit: int = 10,
        page_type: str | None = None,
        tags: list[str] | None = None,
        exclude_sources: bool = False,
    ) -> list[WikiSearchResult]:
        """Deterministic lexical search over indexed payloads.

        This is a local sparse-style ranker over Qdrant payload text. It improves
        recall for literal names, IDs, commands, config keys, and file paths
        without adding another dependency or mutating the vector collection.
        """

        query_terms = Counter(tokenize_for_sparse_search(query))
        if not query_terms:
            return []

        results: list[WikiSearchResult] = []
        offset = None
        while True:
            try:
                points, next_offset = self._client.scroll(
                    collection_name=self.config.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("Could not run sparse wiki search: %s", exc)
                return results[:limit]

            for point in points:
                payload = getattr(point, "payload", None) or {}
                if not self._payload_matches_filters(
                    payload,
                    page_type=page_type,
                    tags=tags,
                    exclude_sources=exclude_sources,
                ):
                    continue
                score = self._lexical_score(query_terms, payload)
                if score <= 0:
                    continue
                results.append(
                    WikiSearchResult(
                        page_path=str(payload.get("page_path", "")),
                        title=str(payload.get("title", "")),
                        page_type=str(payload.get("page_type", "")),
                        chunk_index=int(payload.get("chunk_index", 0) or 0),
                        text=str(payload.get("text", "")),
                        score=score,
                        tags=list(payload.get("tags", []) or []),
                    )
                )

            if next_offset is None:
                break
            offset = next_offset

        results.sort(key=lambda item: (-float(item.score or 0.0), item.page_path, item.chunk_index))
        return results[: max(1, int(limit or 1))]

    def _payload_matches_filters(
        self,
        payload: dict[str, Any],
        *,
        page_type: str | None,
        tags: list[str] | None,
        exclude_sources: bool,
    ) -> bool:
        if exclude_sources and payload.get("page_type") == "source":
            return False
        if page_type and payload.get("page_type") != page_type:
            return False
        if tags:
            payload_tags = set(payload.get("tags", []) or [])
            if not all(tag in payload_tags for tag in tags):
                return False
        return bool(payload.get("page_path"))

    def _lexical_score(self, query_terms: Counter[str], payload: dict[str, Any]) -> float:
        tags = payload.get("tags", []) or []
        weighted_text = " ".join(
            [
                str(payload.get("title", "")),
                str(payload.get("title", "")),
                str(payload.get("page_path", "")),
                " ".join(str(tag) for tag in tags),
                " ".join(str(tag) for tag in tags),
                str(payload.get("text", "")),
            ]
        )
        doc_terms = Counter(tokenize_for_sparse_search(weighted_text))
        score = 0.0
        for term, query_count in query_terms.items():
            count = doc_terms.get(term, 0)
            if count:
                score += min(count, 5) * query_count
        return score / max(1, sum(query_terms.values()))

    def reindex_all(self) -> dict[str, int]:
        """Re-index all wiki pages and sources without dropping the collection first."""
        self._ensure_collection()
        old_paths = self._existing_indexed_paths()
        seen_paths: set[str] = set()

        counts = {"pages": 0, "sources": 0, "chunks": 0}

        for subdir in ["entities", "concepts", "comparisons", "queries"]:
            dir_path = self.config.wiki_path / subdir
            if dir_path.exists():
                for md_file in sorted(dir_path.glob("*.md")):
                    n = self.index_page(md_file)
                    counts["pages"] += 1
                    counts["chunks"] += n
                    seen_paths.add(str(md_file.relative_to(self.config.wiki_path)))

        for source_subdir in ["articles", "papers", "transcripts"]:
            dir_path = self.config.raw_dir / source_subdir
            if dir_path.exists():
                for md_file in sorted(dir_path.glob("*.md")):
                    n = self.index_source(md_file)
                    counts["sources"] += 1
                    counts["chunks"] += n
                    seen_paths.add(str(md_file.relative_to(self.config.wiki_path)))

        for stale_path in sorted(old_paths - seen_paths):
            self._delete_page_chunks(stale_path)

        logger.info(
            "Full reindex: %d pages, %d sources, %d chunks",
            counts["pages"], counts["sources"], counts["chunks"],
        )
        return counts

    def _existing_indexed_paths(self) -> set[str]:
        """Return page/source paths already present in the vector collection."""
        paths: set[str] = set()
        offset = None
        while True:
            try:
                points, next_offset = self._client.scroll(
                    collection_name=self.config.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("Could not list existing wiki index paths: %s", exc)
                return paths
            for point in points:
                payload = getattr(point, "payload", None) or {}
                page_path = payload.get("page_path")
                if isinstance(page_path, str) and page_path:
                    paths.add(page_path)
            if next_offset is None:
                return paths
            offset = next_offset

    def collection_stats(self) -> dict:
        """Get stats about the vector collection."""
        try:
            info = self._client.get_collection(collection_name=self.config.collection_name)
            return {"collection": self.config.collection_name, "points": info.points_count}
        except Exception:
            return {"collection": self.config.collection_name, "points": 0}

    def _delete_page_chunks(self, rel_path: str) -> None:
        models = self._models
        try:
            self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="page_path", match=models.MatchValue(value=rel_path))]
                    )
                ),
            )
        except Exception as e:
            logger.warning("Could not delete chunks for %s: %s", rel_path, e)

    def _delete_stale_page_chunks(self, rel_path: str, keep_chunks: int) -> None:
        """Delete old chunks beyond the freshly upserted chunk range for a page."""
        models = self._models
        try:
            self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="page_path", match=models.MatchValue(value=rel_path)),
                            models.FieldCondition(key="chunk_index", range=models.Range(gte=keep_chunks)),
                        ]
                    )
                ),
            )
        except Exception as e:
            logger.warning("Could not delete stale chunks for %s: %s", rel_path, e)

    def _chunk_id(self, page_path: str, chunk_index: int) -> str:
        """Generate a deterministic UUID-format ID for a chunk."""
        raw = f"{page_path}::{chunk_index}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    def close(self) -> None:
        self._embedder.close()
        close = getattr(self._client, "close", None)
        if close:
            close()


class WikiSearchResult:
    """A search result from the wiki vector store."""

    __slots__ = ("page_path", "title", "page_type", "chunk_index", "text", "score", "tags")

    def __init__(
        self,
        page_path: str,
        title: str,
        page_type: str,
        chunk_index: int,
        text: str,
        score: float,
        tags: list[str],
    ):
        self.page_path = page_path
        self.title = title
        self.page_type = page_type
        self.chunk_index = chunk_index
        self.text = text
        self.score = score
        self.tags = tags

    def __repr__(self) -> str:
        return f"WikiSearchResult({self.title!r}, score={self.score:.3f}, type={self.page_type})"
