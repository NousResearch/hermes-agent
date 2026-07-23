"""Knowledge Graph — entity-relationship memory for Hermes.

Stores (entity, relation, entity) triples with metadata.
Supports path queries (BFS), dependency tracing, and graph reasoning.
Uses SQLite as the backing store — lightweight, zero-config.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE(subject_id, predicate, object_id)
);
CREATE INDEX IF NOT EXISTS idx_rel_subj ON relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_rel_obj ON relations(object_id);
CREATE INDEX IF NOT EXISTS idx_rel_pred ON relations(predicate);
"""


class KnowledgeGraph:
    """Thread-safe entity-relationship graph backed by SQLite."""

    def __init__(self):
        db_path = get_hermes_home() / "knowledge_graph.db"
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()

    def _get_conn(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def upsert_entity(self, name, etype="", metadata=None):
        name = name.strip().lower()
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
                if row:
                    conn.execute("UPDATE entities SET type=?, metadata=? WHERE id=?",
                                 (etype, json.dumps(metadata or {}), row[0]))
                    return row[0]
                cur = conn.execute("INSERT INTO entities(name,type,metadata) VALUES(?,?,?)",
                                   (name, etype, json.dumps(metadata or {})))
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def add_relation(self, subject, predicate, obj, weight=1.0, metadata=None):
        subject = subject.strip().lower()
        obj = obj.strip().lower()
        with self._lock:
            conn = self._get_conn()
            try:
                sid = self._ensure_entity(conn, subject)
                oid = self._ensure_entity(conn, obj)
                conn.execute("""INSERT OR REPLACE INTO relations
                    (subject_id,predicate,object_id,weight,metadata)
                    VALUES(?,?,?,?,?)""",
                             (sid, predicate, oid, weight, json.dumps(metadata or {})))
                conn.commit()
                return True
            finally:
                conn.close()

    def _ensure_entity(self, conn, name):
        row = conn.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
        if row:
            return row[0]
        cur = conn.execute("INSERT INTO entities(name) VALUES(?)", (name,))
        return cur.lastrowid

    def query_relations(self, subject=None, predicate=None, obj=None, limit=20):
        conn = self._get_conn()
        try:
            wheres, params = [], []
            joins = ""
            if subject:
                joins += " JOIN entities se ON r.subject_id=se.id"
                wheres.append("se.name=?"); params.append(subject.strip().lower())
            if predicate:
                wheres.append("r.predicate=?"); params.append(predicate)
            if obj:
                joins += " JOIN entities oe ON r.object_id=oe.id"
                wheres.append("oe.name=?"); params.append(obj.strip().lower())
            if "se" not in joins:
                joins += " JOIN entities se ON r.subject_id=se.id"
            if "oe" not in joins:
                joins += " JOIN entities oe ON r.object_id=oe.id"
            where = "WHERE " + " AND ".join(wheres) if wheres else ""
            rows = conn.execute(
                f"SELECT se.name AS subject, r.predicate, oe.name AS object,"
                f" r.weight, r.metadata FROM relations r{joins} {where}"
                f" ORDER BY r.created_at DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def find_path(self, from_entity, to_entity, max_depth=5):
        from_e = from_entity.strip().lower()
        to_e = to_entity.strip().lower()
        conn = self._get_conn()
        try:
            fr = conn.execute("SELECT id FROM entities WHERE name=?", (from_e,)).fetchone()
            tr = conn.execute("SELECT id FROM entities WHERE name=?", (to_e,)).fetchone()
            if not fr or not tr:
                return None
            start_id, target_id = fr[0], tr[0]
            visited = {start_id}
            queue = [(start_id, [start_id])]
            while queue:
                current, path = queue.pop(0)
                if len(path) > max_depth + 1:
                    break
                for row in conn.execute(
                    "SELECT object_id, predicate FROM relations WHERE subject_id=?", (current,)
                ).fetchall():
                    nid = row[0]
                    if nid == target_id:
                        path_ids = path + [nid]
                        names = []
                        for pid in path_ids:
                            er = conn.execute("SELECT name FROM entities WHERE id=?", (pid,)).fetchone()
                            names.append(er[0] if er else str(pid))
                        return {"found": True, "depth": len(names)-1, "path": names}
                    if nid not in visited:
                        visited.add(nid)
                        queue.append((nid, path + [nid]))
            return {"found": False, "depth": max_depth}
        finally:
            conn.close()

    def stats(self):
        conn = self._get_conn()
        try:
            ec = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            rc = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            return {"entities": ec, "relations": rc}
        finally:
            conn.close()


_kg = None


def get_kg():
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg


def register(ctx):
    kg = get_kg()

    def _upsert(args, **kw):
        triples = args.get("triples", [])
        results = []
        for t in triples:
            s = t.get("subject") or t.get("s")
            p = t.get("predicate") or t.get("p")
            o = t.get("object") or t.get("o")
            if s and p and o:
                kg.add_relation(s, p, o, weight=t.get("weight", 1.0))
                results.append({"subject": s, "predicate": p, "object": o})
        return json.dumps({"ok": True, "results": results}, ensure_ascii=False)

    def _query(args, **kw):
        rows = kg.query_relations(
            subject=args.get("subject"), predicate=args.get("predicate"),
            obj=args.get("object"), limit=args.get("limit", 20))
        return json.dumps({"ok": True, "count": len(rows), "relations": rows}, ensure_ascii=False)

    def _path(args, **kw):
        r = kg.find_path(args["from"], args["to"], max_depth=args.get("max_depth", 5))
        return json.dumps(r or {"found": False}, ensure_ascii=False)

    ctx.register_tool("knowledge_graph_upsert", schema={
        "name": "knowledge_graph_upsert",
        "description": "Add (subject, predicate, object) triples to the knowledge graph. Build entity relationship maps.",
        "parameters": {"type": "object", "properties": {
            "triples": {"type": "array", "items": {"type": "object", "properties": {
                "subject": {"type": "string"}, "predicate": {"type": "string"},
                "object": {"type": "string"}, "weight": {"type": "number"}},
                "required": ["subject", "predicate", "object"]}}},
            "required": ["triples"]},
    }, handler=_upsert)

    ctx.register_tool("knowledge_graph_query", schema={
        "name": "knowledge_graph_query",
        "description": "Query the knowledge graph. Filter by subject, predicate, or object.",
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string"}, "predicate": {"type": "string"},
            "object": {"type": "string"}, "limit": {"type": "integer"}}},
    }, handler=_query)

    ctx.register_tool("knowledge_graph_path", schema={
        "name": "knowledge_graph_path",
        "description": "BFS shortest path between two entities in the knowledge graph.",
        "parameters": {"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"},
            "max_depth": {"type": "integer"}},
            "required": ["from", "to"]},
    }, handler=_path)

    logger.info("Knowledge graph plugin registered: %s", kg.stats())