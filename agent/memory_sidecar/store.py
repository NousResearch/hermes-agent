from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Optional

from hermes_constants import get_hermes_home
from hermes_state import apply_wal_with_fallback

from .models import ContextQuery, ContextResult, IngestSource, Observation, ObservationFile, SessionFact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_sources (
    source_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    last_offset INTEGER NOT NULL DEFAULT 0,
    partial_line TEXT NOT NULL DEFAULT '',
    last_event_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_id TEXT,
    role TEXT,
    event_ts TEXT NOT NULL,
    observation_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    detail TEXT NOT NULL,
    privacy_status TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observation_concepts (
    observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    concept TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observation_files (
    observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    change_kind TEXT
);

CREATE TABLE IF NOT EXISTS session_facts (
    session_id TEXT PRIMARY KEY,
    user_goal TEXT,
    latest_summary TEXT,
    last_seen_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    title,
    summary,
    detail,
    concepts,
    file_paths,
    tokenize='unicode61'
);

CREATE INDEX IF NOT EXISTS idx_observations_session_id ON observations(session_id);
CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(observation_type);
CREATE INDEX IF NOT EXISTS idx_observations_event_ts ON observations(event_ts);
CREATE INDEX IF NOT EXISTS idx_observation_files_path ON observation_files(file_path);
CREATE INDEX IF NOT EXISTS idx_observation_concepts_concept ON observation_concepts(concept);
"""


class MemorySidecarStore:
    """SQLite-backed store for normalized transcript observations."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = get_hermes_home() / "memory_sidecar.db"
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("PRAGMA foreign_keys=ON")
        apply_wal_with_fallback(self._conn, db_label="memory_sidecar.db")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get_checkpoint(self, source_id: str) -> Optional[IngestSource]:
        with self._lock:
            row = self._conn.execute(
                "SELECT source_id, path, last_offset, partial_line, last_event_at, updated_at FROM ingest_sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return self._row_to_ingest_source(row) if row else None

    def upsert_checkpoint(self, checkpoint: IngestSource) -> IngestSource:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ingest_sources (source_id, path, last_offset, partial_line, last_event_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    path = excluded.path,
                    last_offset = excluded.last_offset,
                    partial_line = excluded.partial_line,
                    last_event_at = excluded.last_event_at,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.source_id,
                    checkpoint.path,
                    checkpoint.last_offset,
                    checkpoint.partial_line,
                    checkpoint.last_event_at,
                    checkpoint.updated_at or checkpoint.last_event_at or "",
                ),
            )
            self._conn.commit()
        return self.get_checkpoint(checkpoint.source_id) or checkpoint

    def insert_observation(self, observation: Observation) -> Observation:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO observations (
                    session_id, message_id, role, event_ts, observation_type,
                    title, summary, detail, privacy_status, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.session_id,
                    observation.message_id,
                    observation.role,
                    observation.event_ts,
                    observation.observation_type,
                    observation.title,
                    observation.summary,
                    observation.detail,
                    observation.privacy_status,
                    observation.confidence,
                    observation.created_at or observation.event_ts,
                ),
            )
            lastrowid = cur.lastrowid
            if lastrowid is None:
                raise RuntimeError("insert_observation() did not return a row id")
            observation_id = int(lastrowid)
            self._replace_concepts(observation_id, observation.concepts)
            self._replace_files(observation_id, observation.files)
            self._upsert_fts(observation_id, observation)
            self._conn.commit()
        return Observation(
            id=observation_id,
            session_id=observation.session_id,
            message_id=observation.message_id,
            role=observation.role,
            event_ts=observation.event_ts,
            observation_type=observation.observation_type,
            title=observation.title,
            summary=observation.summary,
            detail=observation.detail,
            concepts=observation.concepts,
            files=observation.files,
            privacy_status=observation.privacy_status,
            confidence=observation.confidence,
            created_at=observation.created_at or observation.event_ts,
        )

    def upsert_session_fact(self, fact: SessionFact) -> SessionFact:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO session_facts (session_id, user_goal, latest_summary, last_seen_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_goal = excluded.user_goal,
                    latest_summary = excluded.latest_summary,
                    last_seen_at = excluded.last_seen_at
                """,
                (fact.session_id, fact.user_goal, fact.latest_summary, fact.last_seen_at),
            )
            self._conn.commit()
        return self.get_session_fact(fact.session_id) or fact

    def get_session_fact(self, session_id: str) -> Optional[SessionFact]:
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, user_goal, latest_summary, last_seen_at FROM session_facts WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return SessionFact(
            session_id=row["session_id"],
            user_goal=row["user_goal"],
            latest_summary=row["latest_summary"],
            last_seen_at=row["last_seen_at"],
        )

    def query_context(self, query: ContextQuery) -> ContextResult:
        where: list[str] = []
        params: list[object] = []
        joins: list[str] = []
        order_clause = "o.event_ts DESC, o.id DESC"

        if query.query:
            joins.append("JOIN observations_fts fts ON fts.rowid = o.id")
            where.append("observations_fts MATCH ?")
            params.append(query.query)
            if query.time_bias == "relevant":
                order_clause = "bm25(observations_fts), o.event_ts DESC, o.id DESC"

        if query.session_id:
            where.append("o.session_id = ?")
            params.append(query.session_id)

        if query.types:
            placeholders = ",".join("?" * len(query.types))
            where.append(f"o.observation_type IN ({placeholders})")
            params.extend(query.types)

        if query.concepts:
            joins.append("JOIN observation_concepts oc ON oc.observation_id = o.id")
            placeholders = ",".join("?" * len(query.concepts))
            where.append(f"oc.concept IN ({placeholders})")
            params.extend(query.concepts)

        if query.file_path:
            joins.append("JOIN observation_files ofi ON ofi.observation_id = o.id")
            where.append("ofi.file_path = ?")
            params.append(query.file_path)

        sql = f"""
            SELECT DISTINCT o.id, o.session_id, o.message_id, o.role, o.event_ts,
                   o.observation_type, o.title, o.summary, o.detail,
                   o.privacy_status, o.confidence, o.created_at
            FROM observations o
            {' '.join(joins)}
            {'WHERE ' + ' AND '.join(where) if where else ''}
            ORDER BY {order_clause}
            LIMIT ?
        """
        params.append(query.limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            observations = tuple(self._hydrate_observation(row) for row in rows)
            decisions = tuple(obs for obs in observations if obs.observation_type == "decision")
            changed_files = tuple(self._collect_changed_files(query, observations))
            session_fact = self.get_session_fact(query.session_id) if query.session_id else None

        return ContextResult(
            observations=observations,
            decisions=decisions,
            changed_files=changed_files,
            session_fact=session_fact,
            suggested_follow_ups=self._suggest_follow_ups(observations),
        )

    def _collect_changed_files(self, query: ContextQuery, observations: tuple[Observation, ...]) -> list[str]:
        if not observations:
            return []
        ids = [obs.id for obs in observations if obs.id is not None]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        sql = f"SELECT DISTINCT file_path FROM observation_files WHERE observation_id IN ({placeholders}) ORDER BY file_path"
        rows = self._conn.execute(sql, ids).fetchall()
        files = [str(r["file_path"]) for r in rows]
        if query.file_path and query.file_path not in files:
            files.insert(0, query.file_path)
        return files

    def _replace_concepts(self, observation_id: int, concepts: Iterable[str]) -> None:
        self._conn.execute("DELETE FROM observation_concepts WHERE observation_id = ?", (observation_id,))
        values = [(observation_id, str(concept).strip()) for concept in concepts if str(concept).strip()]
        if values:
            self._conn.executemany(
                "INSERT INTO observation_concepts (observation_id, concept) VALUES (?, ?)",
                values,
            )

    def _replace_files(self, observation_id: int, files: Iterable[ObservationFile]) -> None:
        self._conn.execute("DELETE FROM observation_files WHERE observation_id = ?", (observation_id,))
        values = [
            (observation_id, entry.file_path, entry.change_kind)
            for entry in files
            if str(entry.file_path).strip()
        ]
        if values:
            self._conn.executemany(
                "INSERT INTO observation_files (observation_id, file_path, change_kind) VALUES (?, ?, ?)",
                values,
            )

    def _upsert_fts(self, observation_id: int, observation: Observation) -> None:
        self._conn.execute("DELETE FROM observations_fts WHERE rowid = ?", (observation_id,))
        self._conn.execute(
            "INSERT INTO observations_fts(rowid, title, summary, detail, concepts, file_paths) VALUES (?, ?, ?, ?, ?, ?)",
            (
                observation_id,
                observation.title,
                observation.summary,
                observation.detail,
                " ".join(observation.concepts),
                " ".join(file.file_path for file in observation.files),
            ),
        )

    def _hydrate_observation(self, row: sqlite3.Row) -> Observation:
        observation_id = int(row["id"])
        concepts = tuple(
            entry["concept"]
            for entry in self._conn.execute(
                "SELECT concept FROM observation_concepts WHERE observation_id = ? ORDER BY concept",
                (observation_id,),
            ).fetchall()
        )
        files = tuple(
            ObservationFile(file_path=entry["file_path"], change_kind=entry["change_kind"])
            for entry in self._conn.execute(
                "SELECT file_path, change_kind FROM observation_files WHERE observation_id = ? ORDER BY file_path",
                (observation_id,),
            ).fetchall()
        )
        return Observation(
            id=observation_id,
            session_id=row["session_id"],
            message_id=row["message_id"],
            role=row["role"],
            event_ts=row["event_ts"],
            observation_type=row["observation_type"],
            title=row["title"],
            summary=row["summary"],
            detail=row["detail"],
            concepts=concepts,
            files=files,
            privacy_status=row["privacy_status"],
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_ingest_source(row: sqlite3.Row) -> IngestSource:
        return IngestSource(
            source_id=row["source_id"],
            path=row["path"],
            last_offset=int(row["last_offset"]),
            partial_line=row["partial_line"] or "",
            last_event_at=row["last_event_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _suggest_follow_ups(observations: tuple[Observation, ...]) -> tuple[str, ...]:
        suggestions: list[str] = []
        if any(obs.observation_type == "next_step" for obs in observations):
            suggestions.append("What is the next concrete step from the latest session?")
        if any(obs.observation_type == "decision" for obs in observations):
            suggestions.append("Which prior decisions still constrain the current work?")
        if any(obs.files for obs in observations):
            suggestions.append("Which changed files should be opened first?")
        return tuple(suggestions)
