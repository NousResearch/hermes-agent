"""Local structured subconscious memory store for Hermes.

SQLite-only, profile-scoped, no external credentials.  The store keeps four
cognitive layers: working, episodic, semantic, and procedural.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

LAYERS = {"working", "episodic", "semantic", "procedural"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    layer TEXT NOT NULL CHECK(layer IN ('working','episodic','semantic','procedural')),
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    tags TEXT DEFAULT '',
    source TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    confidence REAL DEFAULT 0.6,
    ttl_days INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    normalized_hash TEXT DEFAULT '',
    UNIQUE(layer, content, source, session_id)
);

CREATE INDEX IF NOT EXISTS idx_subconscious_layer ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_subconscious_updated ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_subconscious_session ON memories(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, summary, tags, content='memories', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary, tags)
    VALUES (new.id, new.content, new.summary, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags)
    VALUES ('delete', old.id, old.content, old.summary, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags)
    VALUES ('delete', old.id, old.content, old.summary, old.tags);
    INSERT INTO memories_fts(rowid, content, summary, tags)
    VALUES (new.id, new.content, new.summary, new.tags);
END;

CREATE TABLE IF NOT EXISTS consolidation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    stats_json TEXT DEFAULT '{}',
    error TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS memory_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id INTEGER NOT NULL,
    target_memory_id INTEGER NOT NULL,
    relation TEXT NOT NULL DEFAULT 'related',
    weight REAL DEFAULT 0.5,
    source TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    UNIQUE(source_memory_id, target_memory_id, relation),
    FOREIGN KEY(source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(target_memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_subconscious_edges_source ON memory_edges(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_subconscious_edges_target ON memory_edges(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_subconscious_edges_relation ON memory_edges(relation);

CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL UNIQUE,
    total_memories INTEGER DEFAULT 0,
    working_count INTEGER DEFAULT 0,
    episodic_count INTEGER DEFAULT 0,
    semantic_count INTEGER DEFAULT 0,
    procedural_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    duplicate_groups INTEGER DEFAULT 0,
    avg_confidence REAL DEFAULT 0.0,
    recall_precision_at_5 REAL DEFAULT 0.0,
    stale_memory_count INTEGER DEFAULT 0,
    conflict_count INTEGER DEFAULT 0,
    autonomy_violations INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detected_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id_1 INTEGER NOT NULL,
    memory_id_2 INTEGER NOT NULL,
    conflict_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('low','medium','high')),
    detected_at TEXT NOT NULL,
    resolved BOOLEAN DEFAULT 0,
    UNIQUE(memory_id_1, memory_id_2, conflict_type),
    FOREIGN KEY(memory_id_1) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(memory_id_2) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON detected_conflicts(resolved);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _normalized_hash(layer: str, content: str, metadata: dict[str, Any] | None) -> str:
    scope = ""
    if metadata:
        scope = "|".join(str(metadata.get(key) or "") for key in ("platform", "chat_id", "topic_id"))
    value = f"{layer}|{scope}|{_normalize_text(content)}"
    return sha256(value.encode("utf-8")).hexdigest()


def _recency_score(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 86400.0)
    return 1.0 / (1.0 + age_days / 14.0)


def default_ttl_days(layer: str) -> int | None:
    if layer == "working":
        return 7
    if layer == "procedural":
        return 90
    return None


class SubconsciousStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="subconscious.db")
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._ensure_column("memories", "normalized_hash", "TEXT DEFAULT ''")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_subconscious_norm ON memories(layer, normalized_hash)")
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cols = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add_memory(
        self,
        layer: str,
        content: str,
        *,
        summary: str = "",
        tags: Iterable[str] | str = "",
        source: str = "",
        session_id: str = "",
        confidence: float = 0.6,
        ttl_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        layer = layer.strip().lower()
        if layer not in LAYERS:
            raise ValueError(f"unknown layer: {layer}")
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        if not isinstance(tags, str):
            tags = ",".join(t.strip() for t in tags if str(t).strip())
        if ttl_days is None:
            ttl_days = default_ttl_days(layer)
        ts = now_iso()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        norm_hash = _normalized_hash(layer, content, metadata or {})
        with self._lock:
            duplicate = self._conn.execute(
                "SELECT id, confidence FROM memories WHERE layer=? AND normalized_hash=? ORDER BY confidence DESC, updated_at DESC LIMIT 1",
                (layer, norm_hash),
            ).fetchone()
            if duplicate:
                new_conf = min(1.0, max(float(duplicate["confidence"]), confidence) + 0.02)
                self._conn.execute(
                    "UPDATE memories SET last_seen_at=?, updated_at=?, confidence=? WHERE id=?",
                    (ts, ts, new_conf, duplicate["id"]),
                )
                self._conn.commit()
                return int(duplicate["id"])
            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO memories
                    (layer, content, summary, tags, source, session_id, confidence, ttl_days,
                     created_at, updated_at, last_seen_at, metadata_json, normalized_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (layer, content, summary, tags, source, session_id, confidence, ttl_days,
                     ts, ts, ts, metadata_json, norm_hash),
                )
                self._conn.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                row = self._conn.execute(
                    "SELECT id, confidence FROM memories WHERE layer=? AND content=? AND source=? AND session_id=?",
                    (layer, content, source, session_id),
                ).fetchone()
                new_conf = min(1.0, max(float(row["confidence"]), confidence) + 0.02)
                self._conn.execute(
                    "UPDATE memories SET last_seen_at=?, updated_at=?, confidence=? WHERE id=?",
                    (ts, ts, new_conf, row["id"]),
                )
                self._conn.commit()
                return int(row["id"])

    def search(
        self,
        query: str,
        *,
        layer: str | None = None,
        limit: int = 8,
        topic_id: str | None = None,
        chat_id: str | None = None,
        platform: str | None = None,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        limit_value = 8 if limit is None else limit
        limit = max(1, min(int(limit_value), 25))
        fetch_limit = limit if not any((topic_id, chat_id, platform)) else min(limit * 5, 100)
        params: list[Any] = []
        where = ""
        if layer:
            layer = layer.strip().lower()
            if layer not in LAYERS:
                raise ValueError(f"unknown layer: {layer}")
            where = "WHERE m.layer = ?"
            params.append(layer)
        with self._lock:
            if query:
                # FTS MATCH is brittle for punctuation-heavy chat text; fall back to LIKE if needed.
                like = f"%{query}%"
                rows = self._conn.execute(
                    f"""
                    SELECT m.* FROM memories m
                    {where + (' AND' if where else 'WHERE')} (m.content LIKE ? OR m.summary LIKE ? OR m.tags LIKE ?)
                    ORDER BY m.confidence DESC, m.updated_at DESC LIMIT ?
                    """,
                    (*params, like, like, like, fetch_limit),
                ).fetchall()
                if not rows:
                    tokens = [token for token in query.replace('"', " ").split() if len(token) > 2][:6]
                    if tokens:
                        token_clause = " OR ".join(["m.content LIKE ? OR m.summary LIKE ? OR m.tags LIKE ?" for _ in tokens])
                        token_params: list[Any] = []
                        for token in tokens:
                            token_like = f"%{token}%"
                            token_params.extend([token_like, token_like, token_like])
                        rows = self._conn.execute(
                            f"""
                            SELECT m.* FROM memories m
                            {where + (' AND' if where else 'WHERE')} ({token_clause})
                            ORDER BY m.confidence DESC, m.updated_at DESC LIMIT ?
                            """,
                            (*params, *token_params, fetch_limit),
                        ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT m.* FROM memories m {where} ORDER BY m.updated_at DESC LIMIT ?",
                    (*params, fetch_limit),
                ).fetchall()
        filtered = [self._score_row(dict(r), topic_id=topic_id, graph_weight=0.0) for r in rows if self._matches_scope(dict(r), topic_id=topic_id, chat_id=chat_id, platform=platform)]
        filtered.sort(key=lambda row: row["score"], reverse=True)
        return filtered[:limit]

    def _score_row(self, row: dict[str, Any], *, topic_id: str | None = None, graph_weight: float = 0.0) -> dict[str, Any]:
        confidence = float(row.get("confidence") or 0.0)
        recency = _recency_score(str(row.get("last_seen_at") or row.get("updated_at") or ""))
        try:
            metadata = json.loads(row.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        topic_boost = 0.15 if topic_id is not None and str(metadata.get("topic_id") or "") == str(topic_id) else 0.0
        ttl_days = row.get("ttl_days")
        decay = 1.0
        if ttl_days:
            try:
                created = datetime.fromisoformat(str(row.get("created_at") or "").replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds() / 86400.0)
                decay = max(0.05, 1.0 - age_days / max(float(ttl_days), 1.0))
            except (TypeError, ValueError):
                decay = 1.0
        score = (confidence * 0.55 + recency * 0.20 + graph_weight * 0.20 + topic_boost) * decay
        row["score"] = round(score, 6)
        row["decay"] = round(decay, 6)
        return row

    def _matches_scope(
        self,
        row: dict[str, Any],
        *,
        topic_id: str | None = None,
        chat_id: str | None = None,
        platform: str | None = None,
    ) -> bool:
        if not any((topic_id, chat_id, platform)):
            return True
        try:
            metadata = json.loads(row.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        if topic_id is not None and str(metadata.get("topic_id") or row.get("topic_id") or "") != str(topic_id):
            return False
        if chat_id is not None and str(metadata.get("chat_id") or "") != str(chat_id):
            return False
        if platform is not None and str(metadata.get("platform") or "") != str(platform):
            return False
        return True

    def add_edge(
        self,
        source_memory_id: int,
        target_memory_id: int,
        *,
        relation: str = "related",
        weight: float = 0.5,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        relation = (relation or "related").strip().lower()
        if not relation:
            raise ValueError("relation must not be empty")
        ts = now_iso()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO memory_edges
                (source_memory_id, target_memory_id, relation, weight, source, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_memory_id, target_memory_id, relation)
                DO UPDATE SET weight=excluded.weight, source=excluded.source, metadata_json=excluded.metadata_json
                """,
                (int(source_memory_id), int(target_memory_id), relation, float(weight), source, ts, metadata_json),
            )
            self._conn.commit()
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = self._conn.execute(
                "SELECT id FROM memory_edges WHERE source_memory_id=? AND target_memory_id=? AND relation=?",
                (int(source_memory_id), int(target_memory_id), relation),
            ).fetchone()
            return int(row["id"])

    def related(self, memory_id: int, *, limit: int = 8) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 8), 25))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.relation, e.weight, e.source AS edge_source, e.metadata_json AS edge_metadata_json, m.*
                FROM memory_edges e
                JOIN memories m ON m.id = CASE
                    WHEN e.source_memory_id = ? THEN e.target_memory_id
                    ELSE e.source_memory_id
                END
                WHERE e.source_memory_id = ? OR e.target_memory_id = ?
                ORDER BY e.weight DESC, m.confidence DESC, m.updated_at DESC
                LIMIT ?
                """,
                (int(memory_id), int(memory_id), int(memory_id), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def hybrid_search(
        self,
        query: str,
        *,
        layer: str | None = None,
        limit: int = 8,
        topic_id: str | None = None,
        chat_id: str | None = None,
        platform: str | None = None,
    ) -> list[dict[str, Any]]:
        direct = self.search(query, layer=layer, limit=limit, topic_id=topic_id, chat_id=chat_id, platform=platform)
        seen = {int(row["id"]) for row in direct}
        results = [{**row, "retrieval": "direct"} for row in direct]
        for row in direct:
            for related in self.related(int(row["id"]), limit=limit):
                related_id = int(related["id"])
                if related_id in seen:
                    continue
                if not self._matches_scope(related, topic_id=topic_id, chat_id=chat_id, platform=platform):
                    continue
                seen.add(related_id)
                results.append({**self._score_row(related, topic_id=topic_id, graph_weight=float(related.get("weight") or 0.0)), "retrieval": "graph"})
                if len(results) >= limit:
                    results.sort(key=lambda item: item["score"], reverse=True)
                    return results[:limit]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def skill_candidates(self, *, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 20), 50))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.* FROM memories m
                WHERE m.layer IN ('procedural','semantic')
                  AND (m.tags LIKE '%workflow%' OR m.tags LIKE '%procedural%' OR m.tags LIKE '%skill%' OR m.content LIKE '%Run %' OR m.content LIKE '%command%' OR m.content LIKE '%implementation%')
                ORDER BY m.confidence DESC, m.last_seen_at DESC, m.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._score_row(dict(r)) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT layer, COUNT(*) AS n FROM memories GROUP BY layer").fetchall()
        counts = {layer: 0 for layer in sorted(LAYERS)}
        counts.update({r["layer"]: int(r["n"]) for r in rows})
        return counts

    def capture_metrics_snapshot(self) -> dict[str, Any]:
        """Capture a daily observability snapshot for the local memory store."""
        ts = now_iso()
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            counts = self.counts()
            total = sum(counts.values())
            edge_count = int(self._conn.execute("SELECT COUNT(*) AS n FROM memory_edges").fetchone()["n"])
            duplicate_groups = int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM (SELECT normalized_hash FROM memories WHERE normalized_hash != '' GROUP BY layer, normalized_hash HAVING COUNT(*) > 1)"
            ).fetchone()["n"])
            avg_row = self._conn.execute("SELECT AVG(confidence) AS avg_confidence FROM memories").fetchone()
            avg_confidence = float(avg_row["avg_confidence"] or 0.0)
            stale_memory_count = int(self._conn.execute(
                """
                SELECT COUNT(*) AS n FROM memories
                WHERE ttl_days IS NOT NULL
                  AND julianday('now') - julianday(created_at) > ttl_days
                """
            ).fetchone()["n"])
            conflict_count = int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM detected_conflicts WHERE resolved = 0"
            ).fetchone()["n"])
            self._conn.execute(
                """
                INSERT INTO metrics_snapshots
                (snapshot_date, total_memories, working_count, episodic_count, semantic_count, procedural_count,
                 edge_count, duplicate_groups, avg_confidence, stale_memory_count, conflict_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_date) DO UPDATE SET
                    total_memories=excluded.total_memories,
                    working_count=excluded.working_count,
                    episodic_count=excluded.episodic_count,
                    semantic_count=excluded.semantic_count,
                    procedural_count=excluded.procedural_count,
                    edge_count=excluded.edge_count,
                    duplicate_groups=excluded.duplicate_groups,
                    avg_confidence=excluded.avg_confidence,
                    stale_memory_count=excluded.stale_memory_count,
                    conflict_count=excluded.conflict_count,
                    created_at=excluded.created_at
                """,
                (
                    snapshot_date,
                    total,
                    counts["working"],
                    counts["episodic"],
                    counts["semantic"],
                    counts["procedural"],
                    edge_count,
                    duplicate_groups,
                    avg_confidence,
                    stale_memory_count,
                    conflict_count,
                    ts,
                ),
            )
            self._conn.commit()
        return {
            "date": snapshot_date,
            "total_memories": total,
            "counts": counts,
            "edge_count": edge_count,
            "duplicate_groups": duplicate_groups,
            "avg_confidence": avg_confidence,
            "stale_memory_count": stale_memory_count,
            "conflict_count": conflict_count,
        }

    def expire_stale_memories(self) -> dict[str, Any]:
        """Delete memories that have exceeded their TTL."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id FROM memories
                WHERE ttl_days IS NOT NULL
                  AND julianday('now') - julianday(created_at) > ttl_days
                """
            ).fetchall()
            expired_ids = [int(row["id"]) for row in rows]
            if not expired_ids:
                return {"success": True, "expired_count": 0, "expired_ids": []}
            placeholders = ",".join("?" for _ in expired_ids)
            self._conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", expired_ids)
            self._conn.commit()
        return {"success": True, "expired_count": len(expired_ids), "expired_ids": expired_ids}

    def detect_conflicts(self) -> dict[str, Any]:
        """Find simple contradictory semantic/procedural memories."""
        negation_pairs = {
            "must": "must not",
            "always": "never",
            "should": "should not",
            "requires": "prohibits",
            "allowed": "blocked",
            "executes": "coordinates only",
            "implements": "delegates",
        }
        with self._lock:
            rows = [
                dict(row)
                for row in self._conn.execute(
                    """
                    SELECT id, layer, content, confidence
                    FROM memories
                    WHERE layer IN ('semantic', 'procedural')
                    ORDER BY layer, confidence DESC, id ASC
                    """
                ).fetchall()
            ]

        indexed: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            content = str(row.get("content") or "").lower()
            for positive, negative in negation_pairs.items():
                if re.search(rf"\b{re.escape(negative)}\b", content):
                    indexed.setdefault((row["layer"], positive, "negative"), []).append(row)
                elif re.search(rf"\b{re.escape(positive)}\b", content):
                    indexed.setdefault((row["layer"], positive, "positive"), []).append(row)

        candidates: list[dict[str, Any]] = []
        seen_pairs: set[tuple[int, int, str]] = set()
        for layer in ("semantic", "procedural"):
            for positive, negative in negation_pairs.items():
                positives = indexed.get((layer, positive, "positive"), [])
                negatives = indexed.get((layer, positive, "negative"), [])
                for pos_row in positives:
                    for neg_row in negatives:
                        first_id, second_id = sorted((int(pos_row["id"]), int(neg_row["id"])))
                        conflict_type = f"{positive}_vs_{negative.replace(' ', '_')}"
                        key = (first_id, second_id, conflict_type)
                        if first_id == second_id or key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        confidence = max(float(pos_row.get("confidence") or 0.0), float(neg_row.get("confidence") or 0.0))
                        candidates.append({
                            "memory_id_1": first_id,
                            "memory_id_2": second_id,
                            "conflict_type": conflict_type,
                            "severity": "high" if confidence >= 0.7 else "medium",
                        })

        inserted = 0
        ts = now_iso()
        with self._lock:
            for conflict in candidates:
                cur = self._conn.execute(
                    """
                    INSERT INTO detected_conflicts
                    (memory_id_1, memory_id_2, conflict_type, severity, detected_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id_1, memory_id_2, conflict_type) DO NOTHING
                    """,
                    (
                        conflict["memory_id_1"],
                        conflict["memory_id_2"],
                        conflict["conflict_type"],
                        conflict["severity"],
                        ts,
                    ),
                )
                inserted += int(cur.rowcount > 0)
            unresolved_count = int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM detected_conflicts WHERE resolved = 0"
            ).fetchone()["n"])
            self._conn.commit()
        return {
            "success": True,
            "detected_count": len(candidates),
            "new_conflict_count": inserted,
            "conflict_count": unresolved_count,
            "conflicts": candidates,
        }

    def begin_run(self, run_key: str) -> bool:
        ts = now_iso()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO consolidation_runs(run_key, status, started_at) VALUES (?, 'running', ?)",
                    (run_key, ts),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def finish_run(self, run_key: str, *, status: str, stats: dict[str, Any] | None = None, error: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE consolidation_runs SET status=?, finished_at=?, stats_json=?, error=? WHERE run_key=?",
                (status, now_iso(), json.dumps(stats or {}, ensure_ascii=False, sort_keys=True), error, run_key),
            )
            self._conn.commit()

    def last_run(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM consolidation_runs ORDER BY started_at DESC, id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def status(self) -> dict[str, Any]:
        with self._lock:
            edge_count = int(self._conn.execute("SELECT COUNT(*) AS n FROM memory_edges").fetchone()["n"])
            duplicate_count = int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM (SELECT normalized_hash FROM memories WHERE normalized_hash != '' GROUP BY layer, normalized_hash HAVING COUNT(*) > 1)"
            ).fetchone()["n"])
            conflict_count = int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM detected_conflicts WHERE resolved = 0"
            ).fetchone()["n"])
        return {
            "db_path": str(self.db_path),
            "counts": self.counts(),
            "edge_count": edge_count,
            "duplicate_groups": duplicate_count,
            "conflict_count": conflict_count,
            "last_run": self.last_run(),
        }
