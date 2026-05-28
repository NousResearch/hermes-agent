#!/usr/bin/env python3
"""
HERMES MEMORY PALACE — Persistent Memory Engine
Replaces fragile in-memory 2200 char limit with structured SQLite persistence.

Architecture:
- episodic_memory: timestamped events, conversations, decisions
- semantic_memory: facts, relationships, knowledge extracted from episodes
- working_memory: active session state, current task context
- palace_index: metadata for fast retrieval (tags, timestamps, importance)
"""

import sqlite3
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("memory_palace")

DB_PATH = os.path.expanduser("~/.hermes/memory-palace/palace.db")
MAX_DB_SIZE_BYTES = 500 * 1024  # 500 KB hard ceiling
SNAPSHOT_MAX_CHARS = 400        # cap context_snapshot length

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    session_id TEXT,
    category TEXT, -- 'decision', 'action', 'observation', 'feedback', 'error', 'insight', 'state_change'
    content TEXT NOT NULL,
    context_snapshot TEXT, -- JSON blob of relevant context at time of event
    importance INTEGER DEFAULT 0, -- 0-10, higher = more important for retention
    tags TEXT, -- JSON array of tags for retrieval
    expires_at REAL, -- NULL = permanent, timestamp = auto-expire
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS semantic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    relationships TEXT, -- JSON: {related_concept: strength, ...}
    source_episodes TEXT, -- JSON: [episode_ids]
    confidence REAL DEFAULT 0.5, -- 0.0-1.0
    last_updated REAL NOT NULL DEFAULT (julianday('now')),
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS working_memory (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL, -- JSON blob
    expires_at REAL, -- NULL = session-scoped, timestamp = absolute expiry
    updated_at REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_memory(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_category ON episodic_memory(category);
CREATE INDEX IF NOT EXISTS idx_episodic_importance ON episodic_memory(importance DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_concept ON semantic_memory(concept);
CREATE INDEX IF NOT EXISTS idx_working_memory ON working_memory(key);
"""


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# ─── EPISODIC MEMORY ────────────────────────────────────────────

def _check_capacity():
    """Pre-write capacity guard: prune if DB is near the hard limit.

    Called before any write operation to prevent write failures that
    cascade into error-logging failures (the crash loop we hit).
    """
    if not os.path.exists(DB_PATH):
        return
    if os.path.getsize(DB_PATH) >= MAX_DB_SIZE_BYTES * 0.9:
        # Aggressive prune when approaching 90% of cap
        try:
            auto_prune()
        except Exception:
            pass  # Never let capacity management crash the caller


def store_episode(session_id: str, category: str, content: str,
                  context: dict | None = None, importance: int = 0,
                  tags: list | None = None, expires_hours: float | None = None,
                  compact_context: bool = True):
    """Store an episodic memory entry. No-op if DB is critically full.

    If compact_context is True (default), context dicts are truncated
    via compact_snapshot() to prevent DB bloat from large JSON blobs.
    """
    try:
        _check_capacity()
        conn = get_db()
        expires = None
        if expires_hours:
            expires = time.time() + (expires_hours * 3600)
        snap = None
        if context:
            snap = compact_snapshot(context) if compact_context else json.dumps(context)
        conn.execute(
            """INSERT INTO episodic_memory (timestamp, session_id, category, content,
               context_snapshot, importance, tags, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), session_id, category, content[:1000],
             snap, importance, json.dumps(tags or []), expires)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # Silently drop — never let memory writes crash the pipeline
        logger.warning("store_episode failed (DB full?): %s", e)


def recall_episodes(hours: float = 24, category: str = None,
                    min_importance: int = 0, tags: list = None,
                    limit: int = 50) -> list:
    """Retrieve recent episodic memories with optional filters."""
    conn = get_db()
    cutoff = time.time() - (hours * 3600)
    query = "SELECT * FROM episodic_memory WHERE timestamp >= ? AND importance >= ?"
    params = [cutoff, min_importance]

    if category:
        query += " AND category = ?"
        params.append(category)
    if tags:
        for tag in tags:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")

    query += f" ORDER BY importance DESC, timestamp DESC LIMIT {limit}"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    columns = ["id", "timestamp", "session_id", "category", "content",
               "context_snapshot", "importance", "tags", "expires_at"]
    return [dict(zip(columns, row)) for row in rows]


# ─── SEMANTIC MEMORY ────────────────────────────────────────────

def store_fact(concept: str, description: str, relationships: dict = None,
               source_ids: list = None, confidence: float = 0.5):
    """Store or update a semantic fact."""
    try:
        conn = get_db()
        conn.execute(
            """INSERT INTO semantic_memory (concept, description, relationships,
               source_episodes, confidence)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(concept) DO UPDATE SET
               description = excluded.description,
               relationships = excluded.relationships,
               source_episodes = excluded.source_episodes,
               confidence = excluded.confidence,
               last_updated = julianday('now')""",
            (concept, description,
             json.dumps(relationships or {}),
             json.dumps(source_ids or []),
             confidence)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("store_fact failed: %s", e)


def recall_facts(query: str, limit: int = 20) -> list:
    """Search semantic memory by concept or description."""
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM semantic_memory
           WHERE concept LIKE ? OR description LIKE ?
           ORDER BY confidence DESC, last_updated DESC LIMIT ?""",
        (f"%{query}%", f"%{query}%", limit)
    ).fetchall()
    conn.close()
    columns = ["id", "concept", "description", "relationships",
               "source_episodes", "confidence", "last_updated", "created_at"]
    return [dict(zip(columns, row)) for row in rows]


# ─── WORKING MEMORY ─────────────────────────────────────────────

def set_working(key: str, value: dict, expires_hours: float = None):
    """Set working memory entry. No-op on failure."""
    try:
        conn = get_db()
        expires = None
        if expires_hours:
            expires = time.time() + (expires_hours * 3600)
        conn.execute(
            """INSERT OR REPLACE INTO working_memory (key, value, expires_at)
               VALUES (?, ?, ?)""",
            (key, json.dumps(value), expires)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("set_working failed: %s", e)


def get_working(key: str) -> Optional[dict]:
    """Get working memory entry."""
    conn = get_db()
    row = conn.execute(
        "SELECT value FROM working_memory WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
        (key, time.time())
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def clear_working():
    """Clear all working memory (session reset)."""
    conn = get_db()
    conn.execute("DELETE FROM working_memory")
    conn.commit()
    conn.close()


# ─── MAINTENANCE ────────────────────────────────────────────────

def prune_expired():
    """Remove expired episodes and working memory."""
    conn = get_db()
    now = time.time()
    conn.execute("DELETE FROM episodic_memory WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
    conn.execute("DELETE FROM working_memory WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
    deleted = conn.total_changes
    conn.commit()
    conn.close()
    return deleted


def _enforce_size_limit():
    """Hard-cap DB size by repeatedly pruning oldest low-importance episodes."""
    if not os.path.exists(DB_PATH):
        return
    if os.path.getsize(DB_PATH) <= MAX_DB_SIZE_BYTES:
        return
    conn = get_db()
    # Loop: delete batches until under cap or no more deletable rows
    for _ in range(20):  # safety cap on iterations
        if os.path.getsize(DB_PATH) <= MAX_DB_SIZE_BYTES:
            break
        conn.execute(
            "DELETE FROM episodic_memory WHERE importance < 5 "
            "AND rowid IN (SELECT rowid FROM episodic_memory "
            "WHERE importance < 5 ORDER BY timestamp ASC LIMIT 200)"
        )
        conn.commit()
        if conn.total_changes == 0:
            break  # nothing left to delete
    # If still over, clear working memory non-essentials
    if os.path.getsize(DB_PATH) > MAX_DB_SIZE_BYTES:
        conn.execute("DELETE FROM working_memory WHERE expires_at IS NOT NULL")
        conn.commit()
    # If STILL over, VACUUM to reclaim freed pages
    if os.path.getsize(DB_PATH) > MAX_DB_SIZE_BYTES:
        conn.close()
        vac = sqlite3.connect(DB_PATH)
        vac.isolation_level = None
        vac.execute("VACUUM")
        vac.close()
        return
    conn.close()


def auto_prune(max_db_kb: int = None) -> dict:
    """
    Prune expired entries + enforce size cap.
    Called by context_orchestrator at session start/end and by Night Council.
    WAL file is checked and VACUUMed if it exceeds the main DB size.
    """
    if max_db_kb:
        global MAX_DB_SIZE_BYTES
        MAX_DB_SIZE_BYTES = max_db_kb * 1024
    conn = get_db()
    now = time.time()
    # 1. Expire old entries
    expired_e = conn.execute(
        "DELETE FROM episodic_memory WHERE expires_at IS NOT NULL AND expires_at < ?",
        (now,)
    ).rowcount
    expired_w = conn.execute(
        "DELETE FROM working_memory WHERE expires_at IS NOT NULL AND expires_at < ?",
        (now,)
    ).rowcount
    conn.commit()
    # 2. Aggressive compaction: wipe low-importance episodic entries older than 48h
    #    regardless of size — keeps the DB lean for crash resilience
    cutoff_48h = time.time() - (48 * 3600)
    conn.execute(
        "DELETE FROM episodic_memory WHERE importance < 3 AND timestamp < ?",
        (cutoff_48h,)
    )
    conn.commit()
    # 3. VACUUM WAL if it's bloated (WAL > 2x main DB is a red flag)
    wal_path = DB_PATH + "-wal"
    if os.path.exists(wal_path) and os.path.exists(DB_PATH):
        wal_size = os.path.getsize(wal_path)
        db_size = os.path.getsize(DB_PATH)
        if wal_size > db_size * 2:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()
    # 4. Enforce hard size cap (includes VACUUM if needed)
    _enforce_size_limit()
    conn.close()
    return {"expired_episodes": expired_e, "expired_working": expired_w,
            "db_size_bytes": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0}


def get_stats() -> dict:
    """Return memory statistics."""
    conn = get_db()
    stats = {
        "episodic_count": conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0],
        "semantic_count": conn.execute("SELECT COUNT(*) FROM semantic_memory").fetchone()[0],
        "working_count": conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0],
        "db_size_bytes": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0,
        "wal_size_bytes": 0,
    }
    # Check WAL size for bloat detection
    wal_path = DB_PATH + "-wal"
    if os.path.exists(wal_path):
        stats["wal_size_bytes"] = os.path.getsize(wal_path)
    conn.close()
    return stats


def compact_snapshot(data: dict | str, max_chars: int = SNAPSHOT_MAX_CHARS) -> str:
    """Compress a context snapshot to a bounded string."""
    if isinstance(data, dict):
        # Keep only top-level keys, stringify with length limit
        compact = {k: str(v)[:100] for k, v in data.items()}
        text = json.dumps(compact)
    else:
        text = str(data)
    return text[:max_chars]


if __name__ == "__main__":
    # Self-test
    print("Initializing memory palace...")
    stats = get_stats()
    print(f"Episodic: {stats['episodic_count']}, Semantic: {stats['semantic_count']}, Working: {stats['working_count']}")
    print(f"DB size: {stats['db_size_bytes']} bytes")

    # Quick write/read test
    store_episode("test-001", "test", "Memory palace self-test",
                  context={"architect": True}, importance=5, tags=["test"])
    store_fact("Hermes Agent", "Self-improving agent stack on Mac M2 + Linux RTX3060",
               {"platform": "Mac M2 32GB", "backend": "Linux RTX3060"})
    set_working("active_task", {"task": "implementation", "phase": "infrastructure"})

    episodes = recall_episodes(hours=1)
    facts = recall_facts("Hermes")
    working = get_working("active_task")

    print(f"\nEpisodes found: {len(episodes)}")
    print(f"Facts found: {len(facts)}")
    print(f"Working memory: {working}")

    prune_expired()
    final_stats = get_stats()
    print(f"\nFinal stats: {final_stats}")
    print("Memory palace ready. ✅")