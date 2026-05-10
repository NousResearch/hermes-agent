"""
Knowledge Graph — Semantic Memory Backend

SQLite-backed knowledge graph for semantic memory (Tulving's semantic
memory layer).  Stores entities (nodes) and their relationships (edges).

No external graph DB required — uses two SQLite tables (kg_nodes, kg_edges)
which are created lazily inside the existing Hermes sessions database.

Node types:  Person, Concept, Task, File, Tool, Preference, Fact
Edge types:  relates_to, has_property, caused_by, requires, contradicts, ...
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class KGNode:
    id: str
    entity_type: str          # Person / Concept / Task / File / Preference / Fact
    label: str
    properties: dict = field(default_factory=dict)
    importance: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_row(self) -> tuple:
        return (
            self.id, self.entity_type, self.label,
            json.dumps(self.properties, ensure_ascii=False),
            self.importance, self.created_at, self.last_updated,
        )


@dataclass
class KGEdge:
    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_row(self) -> tuple:
        return (self.id, self.source_id, self.target_id,
                self.relation_type, self.weight, self.created_at)


# ── Knowledge Graph ───────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kg_nodes (
    id           TEXT PRIMARY KEY,
    entity_type  TEXT NOT NULL,
    label        TEXT NOT NULL,
    properties   TEXT DEFAULT '{}',
    importance   REAL DEFAULT 0.5,
    created_at   TEXT,
    last_updated TEXT
);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_label ON kg_nodes(label);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_type  ON kg_nodes(entity_type);

CREATE TABLE IF NOT EXISTS kg_edges (
    id            TEXT PRIMARY KEY,
    source_id     TEXT REFERENCES kg_nodes(id),
    target_id     TEXT REFERENCES kg_nodes(id),
    relation_type TEXT NOT NULL,
    weight        REAL DEFAULT 1.0,
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(target_id);
"""


class KnowledgeGraph:
    """
    SQLite-backed semantic knowledge graph.

    Typical usage:
        kg = KnowledgeGraph(db_path)
        node_id = kg.upsert_node("Concept", "Python", {"version": "3.12"})
        kg.add_edge(node_id, other_id, "related_to")
        results = kg.search_nodes("Python")
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    # ── Node operations ───────────────────────────────────────────────────

    def upsert_node(
        self,
        entity_type: str,
        label: str,
        properties: Optional[dict] = None,
        importance: float = 0.5,
    ) -> str:
        """Insert or update a node; returns node ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM kg_nodes WHERE label = ? AND entity_type = ?",
                (label, entity_type),
            ).fetchone()

            if row:
                node_id = row["id"]
                conn.execute(
                    "UPDATE kg_nodes SET properties=?, importance=?, last_updated=? WHERE id=?",
                    (
                        json.dumps(properties or {}, ensure_ascii=False),
                        importance,
                        datetime.now().isoformat(),
                        node_id,
                    ),
                )
                return node_id

            node_id = str(uuid.uuid4())
            node = KGNode(
                id=node_id, entity_type=entity_type, label=label,
                properties=properties or {}, importance=importance,
            )
            conn.execute(
                "INSERT INTO kg_nodes VALUES (?,?,?,?,?,?,?)",
                node.to_row(),
            )
            return node_id

    def get_node(self, node_id: str) -> Optional[KGNode]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM kg_nodes WHERE id = ?", (node_id,)
            ).fetchone()
        if not row:
            return None
        return KGNode(
            id=row["id"], entity_type=row["entity_type"], label=row["label"],
            properties=json.loads(row["properties"]),
            importance=row["importance"],
            created_at=row["created_at"], last_updated=row["last_updated"],
        )

    def search_nodes(self, query: str, limit: int = 10) -> list[KGNode]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM kg_nodes WHERE label LIKE ? ORDER BY importance DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [
            KGNode(
                id=r["id"], entity_type=r["entity_type"], label=r["label"],
                properties=json.loads(r["properties"]),
                importance=r["importance"],
                created_at=r["created_at"], last_updated=r["last_updated"],
            )
            for r in rows
        ]

    # ── Edge operations ───────────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> str:
        edge_id = str(uuid.uuid4())
        edge = KGEdge(
            id=edge_id, source_id=source_id, target_id=target_id,
            relation_type=relation_type, weight=weight,
        )
        with self._conn() as conn:
            exists = conn.execute(
                "SELECT id FROM kg_edges WHERE source_id=? AND target_id=? AND relation_type=?",
                (source_id, target_id, relation_type),
            ).fetchone()
            if not exists:
                conn.execute("INSERT INTO kg_edges VALUES (?,?,?,?,?,?)", edge.to_row())
                return edge_id
            return exists["id"]

    def get_neighbours(
        self,
        node_id: str,
        relation_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[tuple]:
        with self._conn() as conn:
            if relation_type:
                rows = conn.execute(
                    """SELECT e.*, n.* FROM kg_edges e
                       JOIN kg_nodes n ON n.id = e.target_id
                       WHERE e.source_id=? AND e.relation_type=?
                       ORDER BY e.weight DESC LIMIT ?""",
                    (node_id, relation_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT e.*, n.* FROM kg_edges e
                       JOIN kg_nodes n ON n.id = e.target_id
                       WHERE e.source_id=?
                       ORDER BY e.weight DESC LIMIT ?""",
                    (node_id, limit),
                ).fetchall()

        results = []
        for r in rows:
            edge = KGEdge(
                id=r["id"], source_id=r["source_id"], target_id=r["target_id"],
                relation_type=r["relation_type"], weight=r["weight"],
                created_at=r["created_at"],
            )
            node = KGNode(
                id=r["id"], entity_type=r["entity_type"], label=r["label"],
                properties=json.loads(r["properties"]),
                importance=r["importance"],
                created_at=r["created_at"], last_updated=r["last_updated"],
            )
            results.append((edge, node))
        return results

    def stats(self) -> dict:
        with self._conn() as conn:
            n_nodes = conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
            n_edges = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        return {"nodes": n_nodes, "edges": n_edges}

    def connection_density(self, node_id: str) -> int:
        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM kg_edges WHERE source_id=? OR target_id=?",
                (node_id, node_id),
            ).fetchone()[0]
        return count
