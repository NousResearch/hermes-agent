"""
Hypergraph memory extension for holographic store.

A hypergraph extends a regular graph by allowing edges (hyperedges) to connect
multiple nodes, not just two. This enables more expressive relationships.

Example:
  Regular graph:    A -- B -- C    (edges connect exactly 2 nodes)
  Hypergraph:       (A, B) --> C   (hyperedge connects A,B to C)

This module provides:
  - HyperEdge: typed N-ary relationships between entities
  - Multi-hop traversal: find paths through hyperedges
  - N-ary queries: "What depends on A and B together?"
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------------------------------------------------------------
# Schema for hypergraph tables
# -----------------------------------------------------------------------------

_HYPERGRAPH_SCHEMA = """
-- Hyperedges: typed N-ary relationships between entities
CREATE TABLE IF NOT EXISTS hyper_edges (
    edge_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    relation_type TEXT NOT NULL,          -- e.g., 'depends_on', 'part_of', 'uses'
    head_entity   TEXT NOT NULL,          -- The primary entity (subject)
    tail_entities TEXT NOT NULL,           -- JSON array of related entities (objects)
    weight        REAL DEFAULT 1.0,       -- Edge strength
    source_fact   INTEGER REFERENCES facts(fact_id) ON DELETE SET NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hyper_edges_relation ON hyper_edges(relation_type);
CREATE INDEX IF NOT EXISTS idx_hyper_edges_head     ON hyper_edges(head_entity);

-- Entity co-occurrence for implicit hyperedge discovery
CREATE TABLE IF NOT EXISTS entity_cooccurrence (
    entity_1      TEXT NOT NULL,
    entity_2      TEXT NOT NULL,
    relation_type TEXT NOT NULL DEFAULT 'co_occurs',
    co_count      INTEGER DEFAULT 1,
    last_seen     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_1, entity_2, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_cooc_entity ON entity_cooccurrence(entity_1, entity_2);
"""


# -----------------------------------------------------------------------------
# Common relation patterns (for extraction)
# -----------------------------------------------------------------------------

_RELATION_PATTERNS = {
    "depends_on": [
        r"(\w+)\s+depends\s+on\s+([\w\s,]+)",
        r"(\w+)\s+requires\s+([\w\s,]+)",
        r"(\w+)\s+needs\s+([\w\s,]+)",
    ],
    "part_of": [
        r"(\w+)\s+is\s+(?:a|part of|an?)\s+([\w\s]+)",
        r"(\w+)\s+belongs\s+to\s+([\w\s]+)",
    ],
    "uses": [
        r"(\w+)\s+uses?\s+([\w\s,]+)",
        r"(\w+)\s+via\s+([\w\s,]+)",
    ],
    "related_to": [
        r"(\w+)\s+relates?\s+to\s+([\w\s,]+)",
        r"(\w+)\s+connected\s+to\s+([\w\s,]+)",
    ],
}


# -----------------------------------------------------------------------------
# HyperGraph class
# -----------------------------------------------------------------------------

class HyperGraph:
    """Hypergraph extension for memory store.

    Supports:
      - Adding typed N-ary hyperedges
      - Multi-hop traversal (A → B → C)
      - N-ary queries (what connects to A, B, and C?)
      - Implicit hyperedge discovery via entity co-occurrence
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        """Initialize with an existing database connection.

        Args:
            conn: SQLite connection to the memory store database
            lock: Thread lock for synchronization
        """
        self._conn = conn
        self._lock = lock
        self._init_schema()

    def _init_schema(self) -> None:
        """Create hypergraph tables if they don't exist."""
        with self._lock:
            self._conn.executescript(_HYPERGRAPH_SCHEMA)
            self._conn.commit()

    # -------------------------------------------------------------------------
    # Core hyperedge operations
    # -------------------------------------------------------------------------

    def add_hyperedge(
        self,
        relation_type: str,
        head_entity: str,
        tail_entities: List[str],
        weight: float = 1.0,
        source_fact_id: Optional[int] = None,
    ) -> int:
        """Add a hyperedge to the graph.

        Args:
            relation_type: Type of relationship (e.g., 'depends_on', 'uses')
            head_entity: The primary entity (subject)
            tail_entities: List of related entities (objects)
            weight: Edge strength (0.0 to 1.0)
            source_fact_id: Optional fact ID this was extracted from

        Returns:
            The new edge_id
        """
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO hyper_edges
                    (relation_type, head_entity, tail_entities, weight, source_fact)
                VALUES (?, ?, ?, ?, ?)
                """,
                (relation_type, head_entity, json.dumps(tail_entities), weight, source_fact_id),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_hyperedges(
        self,
        entity: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get hyperedges, optionally filtered by entity or relation type.

        Args:
            entity: Filter by entities involved (both head and tail)
            relation_type: Filter by relation type
            limit: Maximum number of results

        Returns:
            List of hyperedge dicts with keys: edge_id, relation_type,
            head_entity, tail_entities, weight
        """
        with self._lock:
            conditions = []
            params: List[Any] = []

            if entity:
                # Match if entity is in head OR any tail
                conditions.append(
                    "(head_entity = ? OR tail_entities LIKE ?)"
                )
                params.extend([entity, f'%"{entity}"%'])

            if relation_type:
                conditions.append("relation_type = ?")
                params.append(relation_type)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            rows = self._conn.execute(
                f"""
                SELECT edge_id, relation_type, head_entity, tail_entities, weight, source_fact
                FROM hyper_edges
                {where}
                ORDER BY weight DESC, edge_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

            results = []
            for row in rows:
                results.append({
                    "edge_id": row["edge_id"],
                    "relation_type": row["relation_type"],
                    "head_entity": row["head_entity"],
                    "tail_entities": json.loads(row["tail_entities"]),
                    "weight": row["weight"],
                    "source_fact": row["source_fact"],
                })
            return results

    def traverse(
        self,
        start_entity: str,
        hops: int = 2,
        relation_filter: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Traverse hyperedges from a starting entity.

        Args:
            start_entity: Entity to start traversal from
            hops: Number of hyperedge traversals (1 = direct neighbors, 2 = neighbors of neighbors)
            relation_filter: Only follow these relation types (None = all)
            limit: Maximum paths to return

        Returns:
            List of paths, each as {
                'path': [entity, ...],
                'score': float
            }
        """
        with self._lock:
            if relation_filter:
                rel_clause = f"AND relation_type IN ({','.join('?' * len(relation_filter))})"
                rel_params = relation_filter
            else:
                rel_clause = ""
                rel_params = []

            # BFS-style traversal with path tracking
            # Queue entries: (current_entity, full_path, accumulated_score)
            queue: List[Tuple[str, List[str], float]] = [(start_entity, [start_entity], 1.0)]
            results: List[Tuple[List[str], float]] = []
            visited: Set[str] = {start_entity}

            while queue and len(results) < limit * 2:
                current, path, score = queue.pop(0)

                # If path already has desired length, record and continue
                if len(path) > hops + 1:
                    results.append((path, score))
                    continue

                # Find hyperedges where current entity is the head
                params = [current] + rel_params
                edges = self._conn.execute(
                    f"""
                    SELECT head_entity, tail_entities, relation_type, weight
                    FROM hyper_edges
                    WHERE head_entity = ?
                    {rel_clause}
                    """,
                    params,
                ).fetchall()

                for row in edges:
                    head = row["head_entity"]
                    tails: List[str] = json.loads(row["tail_entities"])
                    weight = row["weight"]
                    new_score = score * weight * 0.8  # discount per hop

                    for tail in tails:
                        if tail in visited:
                            continue
                        new_path = path + [tail]
                        results.append((new_path, new_score))
                        if len(new_path) <= hops + 1:
                            queue.append((tail, new_path, new_score))
                        visited.add(tail)

            # Sort by score and return top-K
            results.sort(key=lambda x: -x[1])
            return [
                {"path": path, "score": round(score, 3)}
                for path, score in results[:limit]
            ]

    def query_n_ary(
        self,
        entities: List[str],
        relation_type: Optional[str] = None,
        mode: str = "connected_by_all",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Query hyperedges connecting multiple entities.

        Args:
            entities: List of entities that must be connected
            relation_type: Optional filter by relation type
            mode: 'connected_by_all' (all entities in same edge) or
                  'connected_by_any' (any entity in edge)
            limit: Maximum results

        Returns:
            List of hyperedges satisfying the query
        """
        with self._lock:
            if not entities:
                return []

            results = []
            for edge in self.get_hyperedges(relation_type=relation_type, limit=100):
                edge_entities = {edge["head_entity"]} | set(edge["tail_entities"])

                if mode == "connected_by_all":
                    # All query entities must be in this hyperedge
                    if all(e in edge_entities for e in entities):
                        results.append(edge)
                else:
                    # Any query entity in this hyperedge
                    if any(e in edge_entities for e in entities):
                        results.append(edge)

                if len(results) >= limit:
                    break

            return results

    # -------------------------------------------------------------------------
    # Entity co-occurrence (for implicit hyperedge discovery)
    # -------------------------------------------------------------------------

    def record_cooccurrence(
        self,
        entity_1: str,
        entity_2: str,
        relation_type: str = "co_occurs",
    ) -> None:
        """Record that two entities appeared together in a context.

        This is used for implicit hyperedge discovery - entities that
        frequently co-occur may have a latent relationship.
        """
        with self._lock:
            # Ensure consistent ordering (alphabetically smaller first)
            e1, e2 = sorted([entity_1, entity_2])
            self._conn.execute(
                """
                INSERT INTO entity_cooccurrence
                    (entity_1, entity_2, relation_type, co_count, last_seen)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(entity_1, entity_2, relation_type)
                DO UPDATE SET co_count = co_count + 1, last_seen = CURRENT_TIMESTAMP
                """,
                (e1, e2, relation_type),
            )
            self._conn.commit()

    def get_strong_cooccurrences(
        self,
        entity: str,
        min_count: int = 3,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get entities that frequently co-occur with the given entity.

        These strong co-occurrences suggest implicit relationships
        that could be promoted to explicit hyperedges.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    CASE WHEN entity_1 = ? THEN entity_2 ELSE entity_1 END AS co_entity,
                    relation_type,
                    co_count
                FROM entity_cooccurrence
                WHERE (entity_1 = ? OR entity_2 = ?) AND co_count >= ?
                ORDER BY co_count DESC
                LIMIT ?
                """,
                (entity, entity, entity, min_count, limit),
            ).fetchall()

            return [
                {
                    "entity": row["co_entity"],
                    "relation_type": row["relation_type"],
                    "co_count": row["co_count"],
                }
                for row in rows
            ]

    # -------------------------------------------------------------------------
    # Relationship extraction from text
    # -------------------------------------------------------------------------

    def extract_and_add_relations(
        self,
        text: str,
        fact_id: Optional[int] = None,
    ) -> List[int]:
        """Extract typed relationships from text and add as hyperedges.

        Args:
            text: Text to extract relationships from
            fact_id: Optional fact ID this was extracted from

        Returns:
            List of new edge_ids
        """
        edge_ids = []

        for rel_type, patterns in _RELATION_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    head = match.group(1).strip()
                    tail_str = match.group(2).strip()

                    # Parse tail entities (split by comma or 'and')
                    tails = [
                        t.strip()
                        for t in re.split(r"[,(?:\s+and\s+)]", tail_str)
                        if t.strip()
                    ]

                    if head and tails:
                        edge_id = self.add_hyperedge(
                            relation_type=rel_type,
                            head_entity=head,
                            tail_entities=tails,
                            source_fact_id=fact_id,
                        )
                        edge_ids.append(edge_id)

                        # Record co-occurrences for implicit discovery
                        for tail in tails:
                            self.record_cooccurrence(head, tail, rel_type)

        return edge_ids

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Get hypergraph statistics."""
        with self._lock:
            edge_count = self._conn.execute(
                "SELECT COUNT(*) FROM hyper_edges"
            ).fetchone()[0]

            rel_types = self._conn.execute(
                "SELECT COUNT(DISTINCT relation_type) FROM hyper_edges"
            ).fetchone()[0]

            cooc_count = self._conn.execute(
                "SELECT COUNT(*) FROM entity_cooccurrence"
            ).fetchone()[0]

            return {
                "total_hyperedges": edge_count,
                "unique_relation_types": rel_types,
                "cooccurrence_pairs": cooc_count,
            }

    def suggest_hyperedges_from_cooccurrence(
        self,
        min_count: int = 5,
        relation_type: str = "related_to",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Suggest new hyperedges based on strong co-occurrences.

        Entities that frequently co-occur but don't have an explicit
        hyperedge may benefit from one being created.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT entity_1, entity_2, co_count
                FROM entity_cooccurrence
                WHERE co_count >= ?
                AND relation_type = 'co_occurs'
                AND NOT EXISTS (
                    SELECT 1 FROM hyper_edges
                    WHERE (head_entity = entity_1 AND tail_entities LIKE '%' || entity_2 || '%')
                       OR (head_entity = entity_2 AND tail_entities LIKE '%' || entity_1 || '%')
                )
                ORDER BY co_count DESC
                LIMIT ?
                """,
                (min_count, limit),
            ).fetchall()

            suggestions = []
            for row in rows:
                # Determine direction based on which entity is "larger" (more connections)
                e1_count = self._conn.execute(
                    "SELECT COUNT(*) FROM hyper_edges WHERE head_entity = ?",
                    (row["entity_1"],)
                ).fetchone()[0]
                e2_count = self._conn.execute(
                    "SELECT COUNT(*) FROM hyper_edges WHERE head_entity = ?",
                    (row["entity_2"],)
                ).fetchone()[0]

                if e1_count >= e2_count:
                    head, tail = row["entity_1"], row["entity_2"]
                else:
                    head, tail = row["entity_2"], row["entity_1"]

                suggestions.append({
                    "suggested_head": head,
                    "suggested_tail": tail,
                    "relation_type": relation_type,
                    "co_count": row["co_count"],
                    "confidence": min(1.0, row["co_count"] / 20.0),
                })

            return suggestions
