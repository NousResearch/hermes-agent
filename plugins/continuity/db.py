"""
Continuity Plugin — SQLite DAG metadata store.

Structures the DAG of session_shard → daily_journal → weekly_summary → monthly_summary
nodes, edges, and provenance. Migration-aware (schema versioning).

Path: ~/.hermes/continuum/continuity.db
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SQL_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL CHECK(node_type IN ('session_shard','daily','weekly','monthly')),
    date_key TEXT NOT NULL,            -- e.g. '2026-05-31', '2026-W22', '2026-05'
    title TEXT DEFAULT '',
    markdown_path TEXT DEFAULT '',
    token_count INTEGER DEFAULT 0,
    compression_depth INTEGER DEFAULT 0,
    provider TEXT DEFAULT '',
    model TEXT DEFAULT '',
    source_session_id TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    author_mode TEXT DEFAULT 'cron' CHECK(author_mode IN ('primary','cron','system')),
    operational_tokens INTEGER DEFAULT 0,
    relational_tokens INTEGER DEFAULT 0,
    provenance_hash TEXT DEFAULT '',
    content_checksum TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
    parent_node_id TEXT NOT NULL,
    child_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'synthesized_from'
        CHECK(edge_type IN ('synthesized_from','expanded_to','supersedes')),
    PRIMARY KEY (parent_node_id, child_node_id),
    FOREIGN KEY (parent_node_id) REFERENCES nodes(node_id),
    FOREIGN KEY (child_node_id) REFERENCES nodes(node_id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_date ON nodes(date_key);
CREATE INDEX IF NOT EXISTS idx_nodes_type_date ON nodes(node_type, date_key);
CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(source_session_id);
CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_node_id);
"""

SQL_MIGRATIONS: Dict[int, str] = {
    # Future migrations go here as dict entries
}


def get_continuity_home() -> Path:
    """Return ~/.hermes/continuum/ directory, creating if needed."""
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    continuity_home = hermes_home / "continuum"
    continuity_home.mkdir(parents=True, exist_ok=True)
    return continuity_home


def get_db_path() -> Path:
    """Return path to continuity.db."""
    return get_continuity_home() / "continuity.db"


# ---------------------------------------------------------------------------
# Thread-safe connection management
# ---------------------------------------------------------------------------

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = get_db_path()
        _local.conn = sqlite3.connect(str(db_path))
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def close_connection() -> None:
    """Close the thread-local connection."""
    if hasattr(_local, "conn") and _local.conn is not None:
        _local.conn.close()
        _local.conn = None


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def migrate() -> bool:
    """Create/upgrade schema. Returns True if migration was applied."""
    conn = _get_conn()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
    if cursor.fetchone() is None:
        # Fresh install: create tables
        conn.executescript(SQL_CREATE_TABLES)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
        logger.info("Continuity DB schema v%d created at %s", SCHEMA_VERSION, get_db_path())
        return True

    # Check current version
    row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
    current_version = row["v"] if row and row["v"] else 0

    if current_version < SCHEMA_VERSION:
        for v in range(current_version + 1, SCHEMA_VERSION + 1):
            if v in SQL_MIGRATIONS:
                conn.executescript(SQL_MIGRATIONS[v])
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (v,))
                logger.info("Applied migration v%d", v)
        conn.commit()
        return True

    return False


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_key_for(node_type: str, dt: date = None) -> str:
    """Generate date_key based on node_type."""
    if dt is None:
        dt = date.today()
    if node_type in ("session_shard", "daily"):
        return dt.isoformat()
    elif node_type == "weekly":
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    elif node_type == "monthly":
        return f"{dt.year}-{dt.month:02d}"
    return dt.isoformat()


def _generate_node_id(node_type: str, date_key: str, session_id: str = "") -> str:
    """Generate a deterministic node ID (same inputs → same ID)."""
    from hashlib import md5
    raw = f"{node_type}:{date_key}:{session_id}"
    return f"{node_type}_{md5(raw.encode()).hexdigest()[:12]}"


def upsert_node(
    node_type: str,
    date_key: str,
    title: str = "",
    markdown_path: str = "",
    token_count: int = 0,
    compression_depth: int = 0,
    provider: str = "",
    model: str = "",
    source_session_id: str = "",
    author_mode: str = "cron",
    operational_tokens: int = 0,
    relational_tokens: int = 0,
    provenance_hash: str = "",
    content_checksum: str = "",
    node_id: str = "",              # optional explicit ID
) -> str:
    """Insert or update a node. Returns node_id.

    If node_id is provided, uses it directly. Otherwise derives a
    deterministic node_id from (node_type, date_key, source_session_id).
    Repeated calls with the same logical inputs UPDATE the existing row.
    """
    if not node_id:
        node_id = _generate_node_id(node_type, date_key, source_session_id)
    now = _now()
    conn = _get_conn()
    conn.execute(
        """INSERT INTO nodes (
            node_id, node_type, date_key, title, markdown_path,
            token_count, compression_depth, provider, model,
            source_session_id, created_at, author_mode,
            operational_tokens, relational_tokens,
            provenance_hash, content_checksum
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            title           = excluded.title,
            markdown_path   = excluded.markdown_path,
            token_count     = excluded.token_count,
            compression_depth = excluded.compression_depth,
            provider        = excluded.provider,
            model           = excluded.model,
            author_mode     = excluded.author_mode,
            operational_tokens = excluded.operational_tokens,
            relational_tokens  = excluded.relational_tokens,
            provenance_hash = excluded.provenance_hash,
            content_checksum = excluded.content_checksum""",
        (
            node_id, node_type, date_key, title, markdown_path,
            token_count, compression_depth, provider, model,
            source_session_id, now, author_mode,
            operational_tokens, relational_tokens,
            provenance_hash, content_checksum,
        ),
    )
    conn.commit()
    return node_id


def add_edge(parent_id: str, child_id: str, edge_type: str = "synthesized_from") -> bool:
    """Add a DAG edge. Returns True on success, False if duplicate."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO edges (parent_node_id, child_node_id, edge_type) VALUES (?, ?, ?)",
            (parent_id, child_id, edge_type),
        )
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("add_edge failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------

def get_node(node_id: str) -> Optional[Dict[str, Any]]:
    """Get a single node by ID."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def get_nodes_by_type(node_type: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Get nodes by type, most recent first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM nodes WHERE node_type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (node_type, limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def get_nodes_by_date_range(node_type: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Get nodes of a type within a date range."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM nodes WHERE node_type = ? AND date_key >= ? AND date_key <= ? ORDER BY date_key ASC",
        (node_type, start_date, end_date),
    ).fetchall()
    return [dict(r) for r in rows]


def get_latest_node(node_type: str) -> Optional[Dict[str, Any]]:
    """Get the most recent node of a given type."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM nodes WHERE node_type = ? ORDER BY created_at DESC LIMIT 1",
        (node_type,),
    ).fetchone()
    return dict(row) if row else None


def get_children(parent_id: str) -> List[Dict[str, Any]]:
    """Get all child nodes of a parent (DAG walk down)."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT n.* FROM nodes n
           JOIN edges e ON n.node_id = e.child_node_id
           WHERE e.parent_node_id = ?
           ORDER BY n.created_at ASC""",
        (parent_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_parents(child_id: str) -> List[Dict[str, Any]]:
    """Get all parent nodes of a child (DAG walk up)."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT n.* FROM nodes n
           JOIN edges e ON n.node_id = e.parent_node_id
           WHERE e.child_node_id = ?
           ORDER BY n.created_at DESC""",
        (child_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def trace_provenance(node_id: str) -> List[Dict[str, Any]]:
    """Trace a node down to its leaf-level sources (session_shards).

    Walks the DAG recursively to find all session_shard ancestors.
    Uses get_parents() to walk UP the DAG (monthly → weekly → daily → shards).
    """
    visited = set()
    results = []

    def _walk(nid: str):
        if nid in visited:
            return
        visited.add(nid)
        node = get_node(nid)
        if node and node["node_type"] == "session_shard":
            results.append(node)
            return
        parents = get_parents(nid)
        if not parents:
            # Leaf of the DAG that isn't a session_shard — still a source
            if node and node["node_type"] == "session_shard":
                results.append(node)
            return
        for parent in parents:
            _walk(parent["node_id"])

    _walk(node_id)
    return results


def get_recent_daily_journals(days: int = 7) -> List[Dict[str, Any]]:
    """Get recent daily journal nodes within N days (inclusive)."""
    today = date.today()
    start = (today - timedelta(days=days - 1)).isoformat()
    return get_nodes_by_date_range("daily", start, today.isoformat())


def get_current_weekly() -> Optional[Dict[str, Any]]:
    """Get the current week's weekly summary."""
    today = date.today()
    iso_year, iso_week, _ = today.isocalendar()
    wk = f"{iso_year}-W{iso_week:02d}"
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM nodes WHERE node_type = 'weekly' AND date_key = ? ORDER BY created_at DESC LIMIT 1",
        (wk,),
    ).fetchone()
    return dict(row) if row else None


def get_current_monthly() -> Optional[Dict[str, Any]]:
    """Get the current month's monthly summary."""
    today = date.today()
    mo = f"{today.year}-{today.month:02d}"
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM nodes WHERE node_type = 'monthly' AND date_key = ? ORDER BY created_at DESC LIMIT 1",
        (mo,),
    ).fetchone()
    return dict(row) if row else None


def get_open_threads() -> List[Dict[str, Any]]:
    """Get nodes with open threads (from markdown_path parsing)."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM nodes WHERE node_type = 'daily'
           ORDER BY created_at DESC LIMIT 14"""
    ).fetchall()
    return [dict(r) for r in rows]


def count_session_shards_for_date(target_date: str) -> int:
    """Count session_shards for a specific date."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM nodes WHERE node_type = 'session_shard' AND date_key = ?",
        (target_date,),
    ).fetchone()
    return row["cnt"] if row else 0


def get_session_shards_for_date(target_date: str) -> List[Dict[str, Any]]:
    """Get all session_shards for a specific date."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM nodes WHERE node_type = 'session_shard' AND date_key = ? ORDER BY created_at ASC",
        (target_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_total_token_count() -> int:
    """Total tokens stored across all nodes."""
    conn = _get_conn()
    row = conn.execute("SELECT COALESCE(SUM(token_count), 0) as total FROM nodes").fetchone()
    return row["total"] if row else 0


# ---------------------------------------------------------------------------
# Integrity / maintenance
# ---------------------------------------------------------------------------

def integrity_sweep() -> Dict[str, Any]:
    """Check for orphaned nodes, broken edges, and token drift.

    Returns a dict with findings.
    """
    conn = _get_conn()
    findings = {}

    # Orphaned nodes (no edges at all)
    orphaned = conn.execute(
        """SELECT n.node_id, n.node_type, n.date_key FROM nodes n
           WHERE n.node_type != 'session_shard'
           AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.parent_node_id = n.node_id OR e.child_node_id = n.node_id)"""
    ).fetchall()
    findings["orphaned_nodes"] = [dict(r) for r in orphaned]

    # Broken provenance: monthly/weekly with no parent edges (can't trace upstream)
    broken = conn.execute(
        """SELECT n.node_id, n.node_type, n.date_key FROM nodes n
           WHERE n.node_type IN ('weekly', 'monthly')
           AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.child_node_id = n.node_id)"""
    ).fetchall()
    findings["broken_provenance"] = [dict(r) for r in broken]

    # Token drift: daily nodes where operational+relational far exceed total
    # (indicating stale sub-counts after manual edits)
    drifted = conn.execute(
        """SELECT node_id, node_type, token_count, operational_tokens, relational_tokens,
                  (operational_tokens + relational_tokens) as computed
           FROM nodes
           WHERE node_type = 'daily'
           AND operational_tokens + relational_tokens > token_count + 200"""
    ).fetchall()
    findings["token_drift"] = [dict(r) for r in drifted]

    findings["total_nodes"] = conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
    findings["total_edges"] = conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]

    return findings


def delete_node(node_id: str) -> bool:
    """Delete a node and its edges. Returns True if deleted."""
    conn = _get_conn()
    conn.execute("DELETE FROM edges WHERE parent_node_id = ? OR child_node_id = ?", (node_id, node_id))
    conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
    conn.commit()
    return True


def close():
    """Close the DB connection."""
    close_connection()
