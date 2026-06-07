"""Hippocampal Index -- sparse index storage for memory pattern completion.

The hippocampus stores INDICES, not memories. Each memory is represented as a
sparse set of concept-node pointers in the hippocampal index. When a partial
cue arrives, the hippocampus pattern-completes by matching concept overlap
against stored indices and reranking with embedding similarity.

This module provides optional integration with the MemoryPipeline: every fact
passing the salience gate can be indexed here for fast associative retrieval.

Scientific basis:
  Teyler & Rudy (2007) -- The hippocampal indexing theory and episodic memory.
    Hippocampus, 17(12), 1150-1162.
  McClelland, McNaughton & O'Reilly (1995) -- Why there are complementary
    learning systems in the hippocampus and neocortex.

All methods are best-effort: exceptions caught and logged, never blocking.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import struct
import threading
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_HIPPOCAMPAL_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS hippocampal_index ("
    "    index_id     INTEGER PRIMARY KEY AUTOINCREMENT,"
    "    memory_ref   TEXT NOT NULL,"
    "    concept_nodes TEXT NOT NULL,"
    "    embedding    BLOB,"
    "    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
    "    UNIQUE(memory_ref)"
    ");"
    ""
    "CREATE TABLE IF NOT EXISTS concept_nodes ("
    "    concept_id   INTEGER PRIMARY KEY AUTOINCREMENT,"
    "    concept      TEXT NOT NULL UNIQUE,"
    "    frequency    INTEGER DEFAULT 1,"
    "    first_seen   TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ");"
    ""
    "CREATE INDEX IF NOT EXISTS idx_hipp_ref ON hippocampal_index(memory_ref);"
    "CREATE INDEX IF NOT EXISTS idx_concept_name ON concept_nodes(concept);"
)

# Entity extraction patterns (mirrors store.py)
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)


def _extract_concepts(text: str, entities: Sequence[str] | None = None) -> list[str]:
    """Extract concept labels from text and merge with explicit entities.

    Concept sources:
      1. Explicit entity list (highest priority, already resolved)
      2. Capitalized multi-word phrases
      3. Double-quoted terms
      4. Single-quoted terms
      5. AKA patterns

    Returns a deduplicated, lowercased list preserving first-seen order.
    """
    seen: set[str] = set()
    concepts: list[str] = []

    def _add(name: str) -> None:
        key = name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            concepts.append(key)

    # Explicit entities first
    if entities:
        for ent in entities:
            _add(ent)

    # Regex extraction from text
    for m in _RE_CAPITALIZED.finditer(text):
        _add(m.group(1))
    for m in _RE_DOUBLE_QUOTE.finditer(text):
        _add(m.group(1))
    for m in _RE_SINGLE_QUOTE.finditer(text):
        _add(m.group(1))
    for m in _RE_AKA.finditer(text):
        _add(m.group(1))
        _add(m.group(2))

    # Single capitalized words (proper nouns not caught by multi-word pattern)
    for word in re.findall(r'\b([A-Z][a-z]{2,})\b', text):
        _add(word)

    return concepts


def _embedding_to_bytes(vec: Sequence[float]) -> bytes:
    """Pack a float32 embedding into a compact BLOB."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _bytes_to_embedding(data: bytes) -> list[float]:
    """Unpack a BLOB back into a float32 list."""
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors. Assumes pre-normalized inputs."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# HippocampalIndex
# ---------------------------------------------------------------------------


class HippocampalIndex:
    """Sparse hippocampal index for associative memory retrieval.

    Stores concept-node indices pointing to memory references (fact_ids or
    content hashes). Does NOT store the memories themselves -- those live in
    the neocortical facts table.

    Usage::

        index = HippocampalIndex(conn, lock, embedding_fn=model.encode)
        index.init_tables()

        # Index a memory
        index.index_memory("fact:42", "Alice joined the backend team",
                           entities=["Alice"], embedding=[0.1, 0.2, ...])

        # Pattern-complete from a partial cue
        results = index.pattern_complete("Alice backend")
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: threading.RLock,
        embedding_fn: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        """
        Args:
            conn: Shared SQLite connection (same DB as MemoryStore).
            lock: Shared threading lock.
            embedding_fn: Optional callable that takes a text string and returns
                a float vector (list or ndarray). If None, embedding-based
                reranking is skipped and pattern_complete relies on concept
                overlap alone.
        """
        self._conn = conn
        self._lock = lock
        self._embedding_fn = embedding_fn

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_tables(self) -> None:
        """Create hippocampal tables if they don't exist."""
        try:
            with self._lock:
                self._conn.executescript(_HIPPOCAMPAL_SCHEMA)
                self._conn.commit()
        except Exception as e:
            logger.debug("HippocampalIndex init_tables failed: %s", e)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_memory(
        self,
        memory_ref: str,
        content: str,
        entities: Sequence[str] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> bool:
        """Store a sparse hippocampal index entry for a memory.

        Extracts concepts from content + entities, upserts into
        hippocampal_index, and updates concept_nodes frequency counts.

        Args:
            memory_ref: Unique reference to the memory (e.g. "fact:42").
            content: The memory text content.
            entities: Optional pre-extracted entity names.
            embedding: Optional pre-computed embedding vector.

        Returns:
            True if the index was stored, False on failure.
        """
        try:
            concepts = _extract_concepts(content, entities)
            if not concepts:
                logger.debug("No concepts extracted for %s", memory_ref)
                return False

            concept_pipe = "|".join(concepts)

            # Serialize embedding if provided
            emb_blob: bytes | None = None
            if embedding is not None:
                try:
                    emb_blob = _embedding_to_bytes(embedding)
                except Exception:
                    emb_blob = None

            with self._lock:
                # Upsert the index entry (UNIQUE on memory_ref)
                self._conn.execute(
                    """
                    INSERT INTO hippocampal_index
                        (memory_ref, concept_nodes, embedding)
                    VALUES (?, ?, ?)
                    ON CONFLICT(memory_ref) DO UPDATE SET
                        concept_nodes = excluded.concept_nodes,
                        embedding = COALESCE(excluded.embedding, hippocampal_index.embedding)
                    """,
                    (memory_ref, concept_pipe, emb_blob),
                )

                # Update concept_nodes frequency counts
                for concept in concepts:
                    self._conn.execute(
                        """
                        INSERT INTO concept_nodes (concept, frequency)
                        VALUES (?, 1)
                        ON CONFLICT(concept) DO UPDATE SET
                            frequency = frequency + 1
                        """,
                        (concept,),
                    )

                self._conn.commit()
            return True

        except Exception as e:
            logger.debug("HippocampalIndex.index_memory failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Pattern completion
    # ------------------------------------------------------------------

    def pattern_complete(
        self,
        partial_cue: str,
        limit: int = 10,
    ) -> list[dict]:
        """Pattern-complete from a partial cue -- retrieve matching memories.

        Pipeline:
          1. Extract concepts from the partial cue.
          2. Find all index entries whose concept_nodes overlap the cue concepts.
          3. Score each entry by concept overlap (Jaccard) + embedding similarity.
          4. Return top-LIMIT results sorted by combined score.

        Returns:
            List of dicts with keys: memory_ref, concepts, overlap, similarity, score.
            Empty list on failure or no matches.
        """
        try:
            cue_concepts = _extract_concepts(partial_cue)
            if not cue_concepts:
                return []

            cue_set = set(cue_concepts)
            cue_emb: Sequence[float] | None = None

            # Pre-compute cue embedding if embedding_fn is available
            if self._embedding_fn is not None:
                try:
                    cue_emb = self._embedding_fn(partial_cue)
                except Exception:
                    cue_emb = None

            with self._lock:
                # Fetch all index entries.
                # For large stores, this could be optimized with an inverted index
                # table, but for typical sizes (<10K entries) a full scan is fine.
                rows = self._conn.execute(
                    "SELECT index_id, memory_ref, concept_nodes, embedding "
                    "FROM hippocampal_index"
                ).fetchall()

            if not rows:
                return []

            scored: list[dict] = []
            for row in rows:
                entry_concepts = set(
                    c.strip() for c in row["concept_nodes"].split("|") if c.strip()
                )
                if not entry_concepts:
                    continue

                # Jaccard overlap
                intersection = len(cue_set & entry_concepts)
                union = len(cue_set | entry_concepts)
                overlap = intersection / union if union > 0 else 0.0

                if overlap <= 0.0:
                    continue

                # Embedding similarity (optional)
                similarity = 0.0
                if cue_emb is not None and row["embedding"] is not None:
                    try:
                        entry_emb = _bytes_to_embedding(row["embedding"])
                        similarity = (_cosine_similarity(cue_emb, entry_emb) + 1.0) / 2.0
                    except Exception:
                        similarity = 0.0

                # Combined score: weighted blend of overlap and embedding sim.
                # When embedding is available, give it 40% weight (matching
                # the retrieval.py convention). Otherwise, pure overlap.
                if cue_emb is not None and row["embedding"] is not None:
                    score = 0.6 * overlap + 0.4 * similarity
                else:
                    score = overlap

                scored.append({
                    "memory_ref": row["memory_ref"],
                    "concepts": sorted(entry_concepts),
                    "overlap": round(overlap, 4),
                    "similarity": round(similarity, 4),
                    "score": round(score, 4),
                })

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]

        except Exception as e:
            logger.debug("HippocampalIndex.pattern_complete failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_index_stats(self) -> dict:
        """Return statistics about the hippocampal index.

        Returns:
            Dict with keys:
              - indexed_memories: count of entries in hippocampal_index
              - unique_concepts: count of distinct concept nodes
              - avg_index_size: average number of concepts per index entry
              - total_concept_refs: sum of all concept frequencies
        """
        try:
            with self._lock:
                mem_count = self._conn.execute(
                    "SELECT COUNT(*) FROM hippocampal_index"
                ).fetchone()[0]

                concept_count = self._conn.execute(
                    "SELECT COUNT(*) FROM concept_nodes"
                ).fetchone()[0]

                # Average number of concepts per index entry
                avg_row = self._conn.execute(
                    "SELECT AVG(concept_count) FROM ("
                    "  SELECT LENGTH(concept_nodes) - LENGTH(REPLACE(concept_nodes, '|', '')) + 1"
                    "    AS concept_count"
                    "  FROM hippocampal_index"
                    "  WHERE concept_nodes != ''"
                    ")"
                ).fetchone()
                avg_size = round(avg_row[0], 2) if avg_row and avg_row[0] else 0.0

                total_refs = self._conn.execute(
                    "SELECT COALESCE(SUM(frequency), 0) FROM concept_nodes"
                ).fetchone()[0]

            return {
                "indexed_memories": mem_count,
                "unique_concepts": concept_count,
                "avg_index_size": avg_size,
                "total_concept_refs": total_refs,
            }

        except Exception as e:
            logger.debug("HippocampalIndex.get_index_stats failed: %s", e)
            return {
                "indexed_memories": 0,
                "unique_concepts": 0,
                "avg_index_size": 0.0,
                "total_concept_refs": 0,
            }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def remove_index(self, memory_ref: str) -> bool:
        """Remove an index entry by memory_ref. Returns True if found."""
        try:
            with self._lock:
                cur = self._conn.execute(
                    "DELETE FROM hippocampal_index WHERE memory_ref = ?",
                    (memory_ref,),
                )
                self._conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            logger.debug("HippocampalIndex.remove_index failed: %s", e)
            return False

    def get_frequent_concepts(self, limit: int = 20) -> list[dict]:
        """Return the most frequently indexed concepts.

        Useful for understanding what the memory system considers important.
        """
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT concept, frequency, first_seen "
                    "FROM concept_nodes ORDER BY frequency DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("HippocampalIndex.get_frequent_concepts failed: %s", e)
            return []
