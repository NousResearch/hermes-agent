"""Episodic Memory Store — SQLite-backed persistent memory with FTS5 search.

Follows the same patterns as hermes_state.py (WAL mode, jitter retry, FTS5 triggers).
"""

import json
import logging
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from hermes_constants import get_hermes_home
from memory.config import DB_PATH

logger = logging.getLogger(__name__)
T = TypeVar("T")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    turn_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    profile_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_confirmed_at REAL
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    topic TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_decisions TEXT,
    unresolved TEXT,
    participants TEXT,
    created_at REAL NOT NULL,
    source_turns_json TEXT,
    episode_type TEXT NOT NULL DEFAULT 'raw'
);

CREATE TABLE IF NOT EXISTS dag_nodes (
    id TEXT PRIMARY KEY,
    parent_ids TEXT DEFAULT '[]',
    depth INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    source_range TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS health (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_updated ON entities(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_dag_depth ON dag_nodes(depth);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id TEXT NOT NULL REFERENCES entities(id),
    target_entity_id TEXT NOT NULL REFERENCES entities(id),
    relation_type TEXT NOT NULL,
    attributes TEXT DEFAULT '{}',
    observed_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    session_id TEXT REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);

CREATE TABLE IF NOT EXISTS fact_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(id),
    field_path TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT 'UPDATE',
    observed_at REAL NOT NULL,
    session_id TEXT REFERENCES sessions(id),
    confidence TEXT DEFAULT 'medium'
);

CREATE INDEX IF NOT EXISTS idx_fact_entity ON fact_history(entity_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_fact_field ON fact_history(entity_id, field_path);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    topic,
    summary,
    key_decisions,
    unresolved,
    participants,
    content=episodes,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS episodes_fts_insert AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, topic, summary, key_decisions, unresolved, participants)
    VALUES (new.id, new.topic, new.summary, new.key_decisions, new.unresolved, new.participants);
END;

CREATE TRIGGER IF NOT EXISTS episodes_fts_delete AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, topic, summary, key_decisions, unresolved, participants)
    VALUES('delete', old.id, old.topic, old.summary, old.key_decisions, old.unresolved, old.participants);
END;

CREATE TRIGGER IF NOT EXISTS episodes_fts_update AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, topic, summary, key_decisions, unresolved, participants)
    VALUES('delete', old.id, old.topic, old.summary, old.key_decisions, old.unresolved, old.participants);
    INSERT INTO episodes_fts(rowid, topic, summary, key_decisions, unresolved, participants)
    VALUES (new.id, new.topic, new.summary, new.key_decisions, new.unresolved, new.participants);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    name,
    type,
    profile_json,
    content=entities,
    content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS entities_fts_insert AFTER INSERT ON entities BEGIN
    INSERT INTO entities_fts(rowid, name, type, profile_json)
    VALUES (new.rowid, new.name, new.type, new.profile_json);
END;

CREATE TRIGGER IF NOT EXISTS entities_fts_delete AFTER DELETE ON entities BEGIN
    INSERT INTO entities_fts(entities_fts, rowid, name, type, profile_json)
    VALUES('delete', old.rowid, old.name, old.type, old.profile_json);
END;

CREATE TRIGGER IF NOT EXISTS entities_fts_update AFTER UPDATE ON entities BEGIN
    INSERT INTO entities_fts(entities_fts, rowid, name, type, profile_json)
    VALUES('delete', old.rowid, old.name, old.type, old.profile_json);
    INSERT INTO entities_fts(rowid, name, type, profile_json)
    VALUES (new.rowid, new.name, new.type, new.profile_json);
END;
"""


class EpisodicStore:
    """SQLite-backed episodic memory store with FTS5 search.

    Thread-safe via WAL mode and application-level jitter retry.
    Follows the same patterns as hermes_state.SessionDB.
    """

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150
    _CHECKPOINT_EVERY_N_WRITES = 50

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._write_count = 0
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()
        self._ensure_fts_schema()
        self._ensure_episode_type_column()

    # ── Schema ────────────────────────────────────────────────────────────

    def _init_schema(self):
        self._conn.executescript(SCHEMA_SQL)
        self._conn.executescript(FTS_SQL)

    def _ensure_fts_schema(self):
        """Rebuild FTS tables when the indexed column set changes."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'episodes_fts'"
        ).fetchone()
        sql = str(row[0] if row else "")
        if "participants" in sql and "unresolved" in sql:
            return

        self._conn.executescript(
            """
            DROP TRIGGER IF EXISTS episodes_fts_insert;
            DROP TRIGGER IF EXISTS episodes_fts_delete;
            DROP TRIGGER IF EXISTS episodes_fts_update;
            DROP TABLE IF EXISTS episodes_fts;
            """
        )
        self._conn.executescript(FTS_SQL)
        self._conn.execute(
            """
            INSERT INTO episodes_fts(rowid, topic, summary, key_decisions, unresolved, participants)
            SELECT id, topic, summary, key_decisions, unresolved, participants
            FROM episodes
            """
        )

    def _ensure_episode_type_column(self):
        """Add episode_type column to existing DBs that predate the feature."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM pragma_table_info('episodes') WHERE name = 'episode_type'"
        ).fetchone()
        if row and row[0] > 0:
            return
        self._conn.execute(
            "ALTER TABLE episodes ADD COLUMN episode_type TEXT NOT NULL DEFAULT 'raw'"
        )

    # ── Write helper (same pattern as hermes_state.py) ────────────────────

    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    try:
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except Exception:
                        pass
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(
                            self._WRITE_RETRY_MIN_S,
                            self._WRITE_RETRY_MAX_S,
                        )
                        time.sleep(jitter)
                        continue
                raise
            except Exception:
                raise
        raise last_err  # type: ignore[misc]

    # ── Read helper ───────────────────────────────────────────────────────

    def _execute_read(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _execute_read_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # ── Session operations ────────────────────────────────────────────────

    def ensure_session(
        self, session_id: str, source: str = "", started_at: Optional[float] = None
    ) -> None:
        """Create session record if it doesn't exist."""

        def _do(conn: sqlite3.Connection):
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                (session_id, source, started_at or time.time()),
            )

        self._execute_write(_do)

    # ── Turn operations ───────────────────────────────────────────────────

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[list] = None,
        tool_name: Optional[str] = None,
        token_count: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Insert a turn into the SQLite index. Returns the turn ID."""
        ts = timestamp or time.time()
        tc_json = json.dumps(tool_calls) if tool_calls else None

        def _do(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "INSERT INTO turns (session_id, role, content, tool_calls, tool_name, timestamp, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, role, content, tc_json, tool_name, ts, token_count),
            )
            conn.execute(
                "UPDATE sessions SET turn_count = turn_count + 1 WHERE id = ?",
                (session_id,),
            )
            return cur.lastrowid  # type: ignore[return-value]

        return self._execute_write(_do)

    def get_turns_for_session(self, session_id: str, limit: int = 500) -> List[dict]:
        """Get turns for a session, ordered by timestamp."""
        rows = self._execute_read(
            "SELECT id, role, content, tool_calls, tool_name, timestamp, token_count "
            "FROM turns WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
            (session_id, limit),
        )
        return [dict(r) for r in rows]

    # ── Episode operations ────────────────────────────────────────────────

    def create_episode(
        self,
        session_id: str,
        topic: str,
        summary: str,
        key_decisions: Optional[str] = None,
        unresolved: Optional[str] = None,
        participants: Optional[str] = None,
        source_turns: Optional[List[int]] = None,
        episode_type: str = "raw",
    ) -> int:
        """Create an episode. Returns the episode ID."""
        now = time.time()
        st_json = json.dumps(source_turns) if source_turns else None

        def _do(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "INSERT INTO episodes (session_id, topic, summary, key_decisions, "
                "unresolved, participants, created_at, source_turns_json, episode_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, topic, summary, key_decisions, unresolved,
                 participants, now, st_json, episode_type),
            )
            return cur.lastrowid  # type: ignore[return-value]

        return self._execute_write(_do)

    def update_episode_type(self, episode_id: int, episode_type: str) -> None:
        """Update episode_type for an existing episode."""

        def _do(conn: sqlite3.Connection):
            conn.execute(
                "UPDATE episodes SET episode_type = ? WHERE id = ?",
                (episode_type, episode_id),
            )

        self._execute_write(_do)

    def get_episode(self, episode_id: int) -> Optional[dict]:
        row = self._execute_read_one(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        )
        return dict(row) if row else None

    def get_episode_turns(self, episode_id: int) -> List[dict]:
        """Get the raw turns that an episode was distilled from."""
        ep = self.get_episode(episode_id)
        if not ep or not ep.get("source_turns_json"):
            return []
        turn_ids = json.loads(ep["source_turns_json"])
        if not turn_ids:
            return []
        placeholders = ",".join("?" * len(turn_ids))
        rows = self._execute_read(
            f"SELECT * FROM turns WHERE id IN ({placeholders}) ORDER BY timestamp ASC",
            tuple(turn_ids),
        )
        return [dict(r) for r in rows]

    def search_episodes(self, query: str, limit: int = 5) -> List[dict]:
        """FTS5 search over episode summaries and topics."""
        rows = self._execute_read(
            "SELECT e.id, e.session_id, e.topic, e.summary, e.key_decisions, "
            "e.created_at, rank "
            "FROM episodes_fts f "
            "JOIN episodes e ON e.id = f.rowid "
            "WHERE episodes_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query, limit),
        )
        return [dict(r) for r in rows]

    def get_recent_episodes(self, limit: int = 10) -> List[dict]:
        rows = self._execute_read(
            "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    # ── Entity operations ─────────────────────────────────────────────────

    def upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        profile_json: dict,
    ) -> None:
        """Insert or update an entity. Merges profile_json on update."""
        now = time.time()

        def _do(conn: sqlite3.Connection):
            existing = conn.execute(
                "SELECT profile_json FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            if existing:
                old_profile = json.loads(existing["profile_json"])
                # Deep merge: new keys override old, lists get extended
                merged = _merge_profiles(old_profile, profile_json)
                conn.execute(
                    "UPDATE entities SET profile_json = ?, updated_at = ?, name = ?, type = ? "
                    "WHERE id = ?",
                    (json.dumps(merged, ensure_ascii=False), now, name, entity_type, entity_id),
                )
            else:
                conn.execute(
                    "INSERT INTO entities (id, type, name, profile_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (entity_id, entity_type, name,
                     json.dumps(profile_json, ensure_ascii=False), now, now),
                )

        self._execute_write(_do)

    def get_entity(self, entity_id: str) -> Optional[dict]:
        row = self._execute_read_one(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        )
        if row:
            d = dict(row)
            d["profile_json"] = json.loads(d["profile_json"])
            return d
        return None

    def search_entities(self, query: str, limit: int = 5) -> List[dict]:
        """FTS5 search over entity names and profiles."""
        rows = self._execute_read(
            "SELECT e.id, e.type, e.name, e.profile_json, e.updated_at, rank "
            "FROM entities_fts f "
            "JOIN entities e ON e.rowid = f.rowid "
            "WHERE entities_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query, limit),
        )
        results = []
        for r in rows:
            d = dict(r)
            d["profile_json"] = json.loads(d["profile_json"])
            results.append(d)
        return results

    def get_recent_entities(self, limit: int = 10) -> List[dict]:
        rows = self._execute_read(
            "SELECT * FROM entities ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        results = []
        for r in rows:
            d = dict(r)
            d["profile_json"] = json.loads(d["profile_json"])
            results.append(d)
        return results

    def confirm_entity(self, entity_id: str) -> None:
        """Update last_confirmed_at to now."""
        now = time.time()

        def _do(conn: sqlite3.Connection):
            conn.execute(
                "UPDATE entities SET last_confirmed_at = ? WHERE id = ?",
                (now, entity_id),
            )

        self._execute_write(_do)

    # ── DAG node operations ───────────────────────────────────────────────

    def create_dag_node(
        self,
        node_id: str,
        parent_ids: List[str],
        depth: int,
        content: str,
        source_range: Optional[dict] = None,
    ) -> None:
        now = time.time()
        sr_json = json.dumps(source_range) if source_range else None

        def _do(conn: sqlite3.Connection):
            conn.execute(
                "INSERT INTO dag_nodes (id, parent_ids, depth, content, source_range, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, json.dumps(parent_ids), depth, content, sr_json, now),
            )

        self._execute_write(_do)

    def get_dag_node(self, node_id: str) -> Optional[dict]:
        row = self._execute_read_one(
            "SELECT * FROM dag_nodes WHERE id = ?", (node_id,)
        )
        if row:
            d = dict(row)
            d["parent_ids"] = json.loads(d["parent_ids"])
            if d.get("source_range"):
                d["source_range"] = json.loads(d["source_range"])
            return d
        return None

    def get_dag_nodes_at_depth(self, depth: int) -> List[dict]:
        rows = self._execute_read(
            "SELECT * FROM dag_nodes WHERE depth = ? ORDER BY created_at ASC",
            (depth,),
        )
        results = []
        for r in rows:
            d = dict(r)
            d["parent_ids"] = json.loads(d["parent_ids"])
            if d.get("source_range"):
                d["source_range"] = json.loads(d["source_range"])
            results.append(d)
        return results

    # ── Relationship operations ──────────────────────────────────────────

    def add_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        attributes: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Add a relationship between two entities. Returns the relationship ID."""
        now = time.time()
        attrs_json = json.dumps(attributes or {}, ensure_ascii=False)

        def _do(conn: sqlite3.Connection) -> int:
            # Check for existing relationship
            existing = conn.execute(
                "SELECT id, attributes FROM relationships "
                "WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?",
                (source_entity_id, target_entity_id, relation_type),
            ).fetchone()
            if existing:
                # Update existing
                old_attrs = json.loads(existing["attributes"]) if existing["attributes"] else {}
                merged = {**old_attrs, **(attributes or {})}
                conn.execute(
                    "UPDATE relationships SET attributes = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(merged, ensure_ascii=False), now, existing["id"]),
                )
                return existing["id"]
            cur = conn.execute(
                "INSERT INTO relationships "
                "(source_entity_id, target_entity_id, relation_type, attributes, "
                "observed_at, updated_at, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (source_entity_id, target_entity_id, relation_type, attrs_json,
                 now, now, session_id),
            )
            return cur.lastrowid  # type: ignore[return-value]

        return self._execute_write(_do)

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Get relationships for an entity.

        Args:
            entity_id: The entity to find relationships for.
            direction: 'outgoing', 'incoming', or 'both'.
            relation_type: Optional filter by relation type.
            limit: Max results.
        """
        clauses = []
        params: list = []

        if direction == "outgoing":
            clauses.append("source_entity_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            clauses.append("target_entity_id = ?")
            params.append(entity_id)
        else:
            clauses.append("(source_entity_id = ? OR target_entity_id = ?)")
            params.extend([entity_id, entity_id])

        if relation_type:
            clauses.append("relation_type = ?")
            params.append(relation_type)

        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)

        rows = self._execute_read(
            f"SELECT * FROM relationships WHERE {where} ORDER BY updated_at DESC LIMIT ?",
            tuple(params),
        )
        results = []
        for r in rows:
            d = dict(r)
            if d.get("attributes"):
                d["attributes"] = json.loads(d["attributes"])
            else:
                d["attributes"] = {}
            results.append(d)
        return results

    def get_all_relationship_types(self) -> List[str]:
        """Get all distinct relationship types in the store."""
        rows = self._execute_read(
            "SELECT DISTINCT relation_type FROM relationships ORDER BY relation_type"
        )
        return [r["relation_type"] for r in rows]

    # ── Fact history operations ──────────────────────────────────────────

    def record_fact_change(
        self,
        entity_id: str,
        field_path: str,
        old_value: Optional[str],
        new_value: str,
        operation: str = "UPDATE",
        session_id: Optional[str] = None,
        confidence: str = "medium",
    ) -> int:
        """Record a fact change for temporal tracking. Returns the history ID."""
        now = time.time()

        def _do(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "INSERT INTO fact_history "
                "(entity_id, field_path, old_value, new_value, operation, "
                "observed_at, session_id, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (entity_id, field_path, old_value, new_value, operation,
                 now, session_id, confidence),
            )
            return cur.lastrowid  # type: ignore[return-value]

        return self._execute_write(_do)

    def get_fact_history(
        self,
        entity_id: str,
        field_path: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Get fact change history for an entity."""
        if field_path:
            rows = self._execute_read(
                "SELECT * FROM fact_history WHERE entity_id = ? AND field_path = ? "
                "ORDER BY observed_at DESC LIMIT ?",
                (entity_id, field_path, limit),
            )
        else:
            rows = self._execute_read(
                "SELECT * FROM fact_history WHERE entity_id = ? "
                "ORDER BY observed_at DESC LIMIT ?",
                (entity_id, limit),
            )
        return [dict(r) for r in rows]

    # ── Temporal queries ──────────────────────────────────────────────────

    def get_stale_entities(
        self,
        threshold_seconds: float,
        limit: int = 20,
    ) -> List[dict]:
        """Get entities not confirmed in threshold_seconds.

        Returns entities where last_confirmed_at is NULL or older than threshold.
        """
        cutoff = time.time() - threshold_seconds
        rows = self._execute_read(
            "SELECT * FROM entities "
            "WHERE last_confirmed_at IS NULL OR last_confirmed_at < ? "
            "ORDER BY updated_at ASC LIMIT ?",
            (cutoff, limit),
        )
        results = []
        for r in rows:
            d = dict(r)
            d["profile_json"] = json.loads(d["profile_json"])
            results.append(d)
        return results

    def get_stale_facts(
        self,
        entity_id: str,
        threshold_seconds: float,
    ) -> List[dict]:
        """Get facts for an entity not observed in threshold_seconds.

        Returns fact_history entries that haven't been re-observed recently.
        """
        cutoff = time.time() - threshold_seconds
        # Get the latest observation of each field_path
        rows = self._execute_read(
            "SELECT field_path, MAX(observed_at) as last_observed, "
            "new_value, operation "
            "FROM fact_history "
            "WHERE entity_id = ? "
            "GROUP BY field_path "
            "HAVING last_observed < ? "
            "ORDER BY last_observed ASC",
            (entity_id, cutoff),
        )
        return [dict(r) for r in rows]

    def get_potential_contradictions(
        self,
        entity_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[dict]:
        """Find potential contradictions — same field_path changed multiple times.

        A contradiction is when the same entity+field has been UPDATEd
        with different values, suggesting conflicting information.
        """
        if entity_id:
            rows = self._execute_read(
                "SELECT entity_id, field_path, COUNT(*) as change_count, "
                "GROUP_CONCAT(new_value, ' | ') as values_seen, "
                "MAX(observed_at) as last_change "
                "FROM fact_history "
                "WHERE entity_id = ? AND operation = 'UPDATE' "
                "GROUP BY entity_id, field_path "
                "HAVING change_count >= 2 "
                "ORDER BY change_count DESC LIMIT ?",
                (entity_id, limit),
            )
        else:
            rows = self._execute_read(
                "SELECT entity_id, field_path, COUNT(*) as change_count, "
                "GROUP_CONCAT(new_value, ' | ') as values_seen, "
                "MAX(observed_at) as last_change "
                "FROM fact_history "
                "WHERE operation = 'UPDATE' "
                "GROUP BY entity_id, field_path "
                "HAVING change_count >= 2 "
                "ORDER BY change_count DESC LIMIT ?",
                (limit,),
            )
        results = []
        for r in rows:
            d = dict(r)
            d["values_seen"] = d["values_seen"].split(" | ") if d.get("values_seen") else []
            # Get the entity name
            entity = self._execute_read_one(
                "SELECT name FROM entities WHERE id = ?", (d["entity_id"],)
            )
            d["entity_name"] = entity["name"] if entity else d["entity_id"]
            results.append(d)
        return results

    def get_entity_relationships_graph(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> dict:
        """Get a relationship graph around an entity.

        Returns a dict with 'nodes' and 'edges' for graph traversal.
        Depth 1 = direct relationships only.
        """
        visited = {entity_id}
        frontier = [entity_id]
        all_edges = []
        seen_edges = set()  # For deduplication
        all_nodes = {}

        for _ in range(depth):
            next_frontier = []
            for eid in frontier:
                # Get entity info
                entity = self.get_entity(eid)
                if entity and eid not in all_nodes:
                    all_nodes[eid] = {
                        "id": eid,
                        "name": entity["name"],
                        "type": entity["type"],
                    }

                # Get outgoing relationships
                rels = self.get_relationships(eid, direction="outgoing")
                for rel in rels:
                    target = rel["target_entity_id"]
                    edge_key = (rel["source_entity_id"], target, rel["relation_type"])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        all_edges.append({
                            "source": rel["source_entity_id"],
                            "target": target,
                            "type": rel["relation_type"],
                            "attributes": rel.get("attributes", {}),
                        })
                    if target not in visited:
                        visited.add(target)
                        next_frontier.append(target)

                # Get incoming relationships
                rels = self.get_relationships(eid, direction="incoming")
                for rel in rels:
                    source = rel["source_entity_id"]
                    edge_key = (source, rel["target_entity_id"], rel["relation_type"])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        all_edges.append({
                            "source": source,
                            "target": rel["target_entity_id"],
                            "type": rel["relation_type"],
                            "attributes": rel.get("attributes", {}),
                        })
                    if source not in visited:
                        visited.add(source)
                        next_frontier.append(source)

            frontier = next_frontier

        # Fetch remaining node info
        for eid in frontier:
            if eid not in all_nodes:
                entity = self.get_entity(eid)
                if entity:
                    all_nodes[eid] = {
                        "id": eid,
                        "name": entity["name"],
                        "type": entity["type"],
                    }

        return {
            "nodes": list(all_nodes.values()),
            "edges": all_edges,
            "center": entity_id,
            "depth": depth,
        }

    # ── Health check ──────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Write test value, read it back, confirm match. Return status."""
        test_key = "_health_test"
        test_value = f"ok_{time.time()}"
        status = {
            "db_writable": False,
            "db_readable": False,
            "fts_episodes": False,
            "fts_entities": False,
            "round_trip": False,
            "error": None,
        }
        try:
            # Write
            def _do(conn: sqlite3.Connection):
                conn.execute(
                    "INSERT OR REPLACE INTO health (key, value, updated_at) VALUES (?, ?, ?)",
                    (test_key, test_value, time.time()),
                )
            self._execute_write(_do)
            status["db_writable"] = True

            # Read back
            row = self._execute_read_one(
                "SELECT value FROM health WHERE key = ?", (test_key,)
            )
            if row and row["value"] == test_value:
                status["db_readable"] = True
                status["round_trip"] = True

            # FTS smoke test
            self._execute_read("SELECT rowid FROM episodes_fts LIMIT 1")
            status["fts_episodes"] = True
            self._execute_read("SELECT rowid FROM entities_fts LIMIT 1")
            status["fts_entities"] = True

        except Exception as e:
            status["error"] = str(e)
            logger.error("Episodic memory health check failed: %s", e)

        # Update health record
        try:
            def _do2(conn: sqlite3.Connection):
                conn.execute(
                    "INSERT OR REPLACE INTO health (key, value, updated_at) VALUES (?, ?, ?)",
                    ("_last_health_check", json.dumps(status), time.time()),
                )
            self._execute_write(_do2)
        except Exception:
            pass

        return status

    def get_health(self) -> Optional[dict]:
        row = self._execute_read_one(
            "SELECT value, updated_at FROM health WHERE key = '_last_health_check'"
        )
        if row:
            return {
                "status": json.loads(row["value"]),
                "checked_at": row["updated_at"],
            }
        return None

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        stats = {}
        for table in ("turns", "episodes", "entities", "dag_nodes", "sessions"):
            row = self._execute_read_one(f"SELECT COUNT(*) as cnt FROM {table}")
            stats[table] = row["cnt"] if row else 0
        stats["db_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
        stats["db_size_mb"] = round(stats["db_size_bytes"] / (1024 * 1024), 2)
        return stats

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


# ── Utility ───────────────────────────────────────────────────────────────

def _merge_profiles(old: dict, new: dict) -> dict:
    """Deep merge two entity profiles. New values override old, lists extend."""
    merged = old.copy()
    for key, value in new.items():
        if key in merged and isinstance(merged[key], list) and isinstance(value, list):
            # Extend lists, deduplicate strings
            seen = set(str(v) for v in merged[key])
            for item in value:
                if str(item) not in seen:
                    merged[key].append(item)
                    seen.add(str(item))
        elif key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_profiles(merged[key], value)
        else:
            merged[key] = value
    return merged
