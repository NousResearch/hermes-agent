"""SQLite FTS index/search layer for Memory v2."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from .schemas import (
    CandidateMemory,
    GateDecision,
    MemoryItem,
    MemoryType,
    ProjectCard,
    ProjectStatus,
    SourceRef,
    utc_now_iso,
)
from .store import MemoryV2Store


class MemoryV2Index:
    """SQLite-backed keyword index for Memory v2 canonical records."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id TEXT PRIMARY KEY,
                  type TEXT NOT NULL,
                  title TEXT,
                  subject TEXT,
                  predicate TEXT,
                  value TEXT,
                  body TEXT,
                  summary TEXT,
                  status TEXT,
                  confidence REAL,
                  importance REAL,
                  created_at TEXT,
                  updated_at TEXT NOT NULL,
                  valid_from TEXT,
                  valid_until TEXT,
                  expires_at TEXT,
                  source_refs TEXT,
                  supersedes TEXT,
                  superseded_by TEXT,
                  tags TEXT,
                  file_path TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                  id UNINDEXED,
                  title,
                  subject,
                  predicate,
                  value,
                  body,
                  summary,
                  tags
                );

                CREATE TABLE IF NOT EXISTS source_refs (
                  id TEXT PRIMARY KEY,
                  type TEXT NOT NULL,
                  uri TEXT NOT NULL,
                  title TEXT,
                  observed_at TEXT,
                  quote TEXT
                );

                CREATE TABLE IF NOT EXISTS retrieval_log (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query TEXT NOT NULL,
                  query_hash TEXT,
                  route TEXT,
                  retrieved_ids TEXT,
                  created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "retrieval_log", "query_hash", "TEXT")
            for column, column_type in {
                "subject": "TEXT",
                "predicate": "TEXT",
                "value": "TEXT",
                "confidence": "REAL",
                "importance": "REAL",
                "created_at": "TEXT",
                "valid_from": "TEXT",
                "valid_until": "TEXT",
                "expires_at": "TEXT",
                "supersedes": "TEXT",
                "superseded_by": "TEXT",
            }.items():
                self._ensure_column(conn, "memories", column, column_type)
            self._ensure_fts_schema(conn)

    @staticmethod
    def _ensure_fts_schema(conn: sqlite3.Connection) -> None:
        expected_columns = ["id", "title", "subject", "predicate", "value", "body", "summary", "tags"]
        existing_columns = [row[1] for row in conn.execute("PRAGMA table_info(memories_fts)").fetchall()]
        if existing_columns == expected_columns:
            return
        conn.execute("DROP TABLE IF EXISTS memories_fts")
        conn.execute(
            """
            CREATE VIRTUAL TABLE memories_fts USING fts5(
              id UNINDEXED,
              title,
              subject,
              predicate,
              value,
              body,
              summary,
              tags
            )
            """
        )
        conn.execute(
            """
            INSERT INTO memories_fts (id, title, subject, predicate, value, body, summary, tags)
            SELECT
              id,
              COALESCE(title, ''),
              COALESCE(subject, ''),
              COALESCE(predicate, ''),
              COALESCE(value, ''),
              COALESCE(body, ''),
              COALESCE(summary, ''),
              COALESCE(tags, '')
            FROM memories
            """
        )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        allowed = {
            "retrieval_log": {"query_hash": "TEXT"},
            "memories": {
                "subject": "TEXT",
                "predicate": "TEXT",
                "value": "TEXT",
                "confidence": "REAL",
                "importance": "REAL",
                "created_at": "TEXT",
                "valid_from": "TEXT",
                "valid_until": "TEXT",
                "expires_at": "TEXT",
                "supersedes": "TEXT",
                "superseded_by": "TEXT",
            },
        }
        if allowed.get(table, {}).get(column) != column_type:
            raise ValueError("unsupported schema migration")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(" + table + ")").fetchall()}
        if column not in columns:
            conn.execute("ALTER TABLE " + table + " ADD COLUMN " + column + " " + column_type)

    def table_names(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual') ORDER BY name").fetchall()
        return [row[0] for row in rows]

    def count_memories(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return int(row[0])

    def index_source_ref(self, source: SourceRef) -> None:
        """Index source metadata used to ground recalled memory items."""
        payload = source.to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO source_refs (id, type, uri, title, observed_at, quote)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  type=excluded.type,
                  uri=excluded.uri,
                  title=excluded.title,
                  observed_at=excluded.observed_at,
                  quote=excluded.quote
                """,
                (
                    payload["id"],
                    payload["type"],
                    payload["uri"],
                    payload.get("title", ""),
                    payload.get("observed_at", ""),
                    payload.get("quote", ""),
                ),
            )

    def source_ref(self, source_id: str) -> Dict[str, Any] | None:
        """Return indexed source metadata by id, if present."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, type, uri, title, observed_at, quote FROM source_refs WHERE id = ?",
                (str(source_id),),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "type": row[1],
            "uri": row[2],
            "title": row[3] or "",
            "observed_at": row[4] or "",
            "quote": row[5] or "",
        }

    def source_refs(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Return source metadata for the given ids in input order."""
        sources: List[Dict[str, Any]] = []
        for source_id in source_ids:
            if source := self.source_ref(source_id):
                sources.append(source)
        return sources

    def index_memory_item(self, item: MemoryItem, *, file_path: str | Path | None = None) -> None:
        """Index a canonical semantic/core memory item from fixture or store data."""
        payload = item.to_dict()
        body_parts = [
            payload.get("subject"),
            payload.get("predicate"),
            payload.get("value"),
            payload.get("body"),
            " ".join(payload.get("supersedes") or []),
            str(payload.get("superseded_by") or ""),
        ]
        title = " ".join(str(part) for part in (payload.get("subject"), payload.get("predicate")) if part)
        self.index_record(
            id=payload["id"],
            type=payload["type"],
            title=title,
            subject=str(payload.get("subject") or ""),
            predicate=str(payload.get("predicate") or ""),
            value=str(payload.get("value") or ""),
            body="\n".join(str(part) for part in body_parts if part),
            summary=str(payload.get("summary") or payload.get("value") or payload.get("body") or ""),
            status=payload["status"],
            confidence=payload.get("confidence"),
            importance=payload.get("importance"),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            valid_from=payload.get("valid_from"),
            valid_until=payload.get("valid_until"),
            expires_at=payload.get("expires_at"),
            source_refs=list(payload.get("source_refs") or []),
            supersedes=list(payload.get("supersedes") or []),
            superseded_by=payload.get("superseded_by"),
            tags=list(payload.get("tags") or []),
            file_path=str(file_path or ""),
        )

    def index_project_card(self, card: ProjectCard, *, file_path: str | Path | None = None) -> None:
        body_parts = [
            card.goal,
            card.why_it_matters,
            card.current_state,
            "\n".join(card.decisions),
            "\n".join(card.open_questions),
            "\n".join(card.next_actions),
            "\n".join(card.related_entities),
        ]
        structured_value = {
            "goal": card.goal,
            "why_it_matters": card.why_it_matters,
            "current_state": card.current_state,
            "decisions": list(card.decisions),
            "open_questions": list(card.open_questions),
            "next_actions": list(card.next_actions),
            "related_entities": list(card.related_entities),
        }
        self.index_record(
            id=card.id,
            type="project_state",
            title=card.name,
            body="\n".join(part for part in body_parts if part),
            summary=card.current_state or card.goal,
            status=cast(ProjectStatus, card.status).value,
            value=json.dumps(structured_value, ensure_ascii=False, sort_keys=True),
            source_refs=card.source_refs,
            tags=["project", card.id],
            file_path=str(file_path) if file_path else "",
        )

    def index_candidate(self, candidate: CandidateMemory) -> None:
        candidate_type = cast(MemoryType, candidate.type).value
        gate_decision = cast(GateDecision, candidate.gate_decision).value
        body_parts = [
            candidate.claim,
            candidate.promotion_reason,
            candidate.decision_reason,
        ]
        self.index_record(
            id=candidate.id,
            type="candidate",
            title=candidate.id,
            subject=candidate_type,
            predicate="gate_decision",
            value=candidate.decision_reason,
            body="\n".join(part for part in body_parts if part),
            summary=candidate.claim,
            status=gate_decision,
            confidence=candidate.confidence,
            importance=candidate.importance,
            created_at=candidate.created_at,
            source_refs=candidate.source_refs,
            tags=["candidate", candidate_type, gate_decision],
            file_path="inbox/candidates.jsonl",
        )

    def index_raw_event(self, event: Dict[str, Any]) -> bool:
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            return False
        title = f"Raw event {event_id}".strip()
        body = "\n".join(
            str(event.get(key) or "")
            for key in ("type", "session_id", "user_content", "assistant_content", "tool", "content")
            if event.get(key)
        )
        self.index_record(
            id=event_id,
            type="raw_event",
            title=title,
            body=body,
            summary=str(event.get("user_content") or event.get("content") or ""),
            status="archived",
            source_refs=[event_id] if event_id else [],
            tags=["raw_event", str(event.get("type") or "")],
            file_path="inbox/raw_events.jsonl",
        )
        return True

    def index_record(
        self,
        *,
        id: str,
        type: str,
        title: str = "",
        subject: str = "",
        predicate: str = "",
        value: str = "",
        body: str = "",
        summary: str = "",
        status: str = "active",
        confidence: float | None = None,
        importance: float | None = None,
        created_at: str = "",
        updated_at: str = "",
        valid_from: str | None = None,
        valid_until: str | None = None,
        expires_at: str | None = None,
        source_refs: Optional[List[str]] = None,
        supersedes: Optional[List[str]] = None,
        superseded_by: str | None = None,
        tags: Optional[List[str]] = None,
        file_path: str = "",
    ) -> None:
        record_id = str(id).strip()
        if not record_id:
            raise ValueError("indexed record id is required")
        source_refs = [str(ref) for ref in (source_refs or [])]
        supersedes = [str(ref) for ref in (supersedes or [])]
        tags = [str(tag) for tag in (tags or []) if str(tag)]
        now = utc_now_iso()
        stored_updated_at = str(updated_at or now)
        stored_created_at = str(created_at or stored_updated_at)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                  id, type, title, subject, predicate, value, body, summary, status,
                  confidence, importance, created_at, updated_at, valid_from, valid_until, expires_at,
                  source_refs, supersedes, superseded_by, tags, file_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  type=excluded.type,
                  title=excluded.title,
                  subject=excluded.subject,
                  predicate=excluded.predicate,
                  value=excluded.value,
                  body=excluded.body,
                  summary=excluded.summary,
                  status=excluded.status,
                  confidence=excluded.confidence,
                  importance=excluded.importance,
                  created_at=excluded.created_at,
                  updated_at=excluded.updated_at,
                  valid_from=excluded.valid_from,
                  valid_until=excluded.valid_until,
                  expires_at=excluded.expires_at,
                  source_refs=excluded.source_refs,
                  supersedes=excluded.supersedes,
                  superseded_by=excluded.superseded_by,
                  tags=excluded.tags,
                  file_path=excluded.file_path
                """,
                (
                    record_id,
                    str(type),
                    str(title or ""),
                    str(subject or ""),
                    str(predicate or ""),
                    str(value or ""),
                    str(body or ""),
                    str(summary or ""),
                    str(status or ""),
                    confidence,
                    importance,
                    stored_created_at,
                    stored_updated_at,
                    valid_from,
                    valid_until,
                    expires_at,
                    json.dumps(source_refs, ensure_ascii=False),
                    json.dumps(supersedes, ensure_ascii=False),
                    str(superseded_by or ""),
                    json.dumps(tags, ensure_ascii=False),
                    str(file_path or ""),
                ),
            )
            conn.execute("DELETE FROM memories_fts WHERE id = ?", (record_id,))
            conn.execute(
                "INSERT INTO memories_fts (id, title, subject, predicate, value, body, summary, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record_id,
                    str(title or ""),
                    str(subject or ""),
                    str(predicate or ""),
                    str(value or ""),
                    str(body or ""),
                    str(summary or ""),
                    " ".join(tags),
                ),
            )

    def search(self, query: str, *, route: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        safe_limit = self._coerce_limit(limit)
        fts_queries = self._fts_queries(query_text)
        rows = []
        sql_limit = min(50, max(safe_limit, safe_limit * 5))
        with self._connect() as conn:
            for fts_query in fts_queries:
                rows = conn.execute(
                    """
                    SELECT
                      m.id, m.type, m.title, m.subject, m.predicate, m.value, m.body, m.summary, m.status,
                      m.confidence, m.importance, m.created_at, m.updated_at, m.valid_from, m.valid_until, m.expires_at,
                      m.source_refs, m.supersedes, m.superseded_by, m.tags, m.file_path,
                      bm25(memories_fts) AS rank
                    FROM memories_fts
                    JOIN memories m ON m.id = memories_fts.id
                    WHERE memories_fts MATCH ?
                    ORDER BY
                      CASE m.status
                        WHEN 'active' THEN 0
                        WHEN 'uncertain' THEN 1
                        WHEN 'pending' THEN 2
                        WHEN 'promoted' THEN 3
                        WHEN 'archived_only' THEN 4
                        WHEN 'archived' THEN 5
                        WHEN 'superseded' THEN 6
                        WHEN 'rejected' THEN 7
                        ELSE 8
                      END,
                      rank
                    LIMIT ?
                    """,
                    (fts_query, sql_limit),
                ).fetchall()
                if rows:
                    break
        results = [result for result in (self._row_to_result(row) for row in rows) if self._is_temporally_visible(result)]
        results = results[:safe_limit]
        self.log_retrieval(query_text, route=route, retrieved_ids=[result["id"] for result in results])
        return results

    def rebuild_from_store(self, store: MemoryV2Store) -> Dict[str, int]:
        """Clear and rebuild the index from the current file-store contents."""
        self.initialize()
        with self._connect() as conn:
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM memories_fts")
            conn.execute("DELETE FROM source_refs")
        counts = {"project_cards": 0, "candidates": 0, "raw_events": 0, "source_refs": 0, "memory_items": 0}
        for source in store.list_source_refs():
            self.index_source_ref(source)
            counts["source_refs"] += 1
        for item in store.list_memory_items():
            self.index_memory_item(item, file_path=store._memory_item_path(item.id))
            counts["memory_items"] += 1
        for card in store.list_project_cards():
            self.index_project_card(card, file_path=store._project_card_path(card.id))
            counts["project_cards"] += 1
        for candidate in store.list_candidates():
            self.index_candidate(candidate)
            counts["candidates"] += 1
        for event in store.read_raw_events():
            if self.index_raw_event(event):
                counts["raw_events"] += 1
        return counts

    def log_retrieval(self, query: str, *, route: str = "", retrieved_ids: Optional[List[str]] = None) -> None:
        query_text = str(query or "")
        query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        query_for_log = self._redacted_query_for_log(query_text)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO retrieval_log (query, query_hash, route, retrieved_ids, created_at) VALUES (?, ?, ?, ?, ?)",
                (query_for_log, query_hash, route, json.dumps(retrieved_ids or [], ensure_ascii=False), utc_now_iso()),
            )

    def retrieval_logs(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, query, query_hash, route, retrieved_ids, created_at FROM retrieval_log ORDER BY id"
            ).fetchall()
        return [
            {
                "id": row[0],
                "query": row[1],
                "query_hash": row[2] or "",
                "route": row[3],
                "retrieved_ids": json.loads(row[4] or "[]"),
                "created_at": row[5],
            }
            for row in rows
        ]

    @staticmethod
    def _is_temporally_visible(result: Dict[str, Any]) -> bool:
        now = datetime.now(timezone.utc)
        valid_from = MemoryV2Index._parse_time(result.get("valid_from"))
        valid_until = MemoryV2Index._parse_time(result.get("valid_until"))
        expires_at = MemoryV2Index._parse_time(result.get("expires_at"))
        if valid_from is not None and valid_from > now:
            return False
        if valid_until is not None and valid_until < now:
            return False
        if expires_at is not None and expires_at < now:
            return False
        return True

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _redacted_query_for_log(query: str) -> str:
        sensitive_patterns = (
            r"(?i)password\s*(?:is|=|:)\s*\S+",
            r"(?i)password\s+\S+",
            r"(?i)passwd\s*(?:is|=|:)\s*\S+",
            r"(?i)passwd\s+\S+",
            r"(?i)token\s*(?:is|=|:)\s*\S+",
            r"(?i)token\s+\S+",
            r"(?i)secret\s*(?:is|=|:)\s*\S+",
            r"(?i)secret\s+\S+",
            r"(?i)api[_ -]?key\s*(?:is|=|:)\s*\S+",
            r"(?i)api[_ -]?key\s+\S+",
            r"(?i)authorization\s*:\s*bearer\s+\S+",
            r"(?i)bearer\s+\S+",
            r"(?i)[A-Z0-9_]*API[_-]?KEY\s*=\s*\S+",
            r"(?i)[A-Z0-9_]*API[_-]?KEY\s+\S+",
        )
        if any(re.search(pattern, query) for pattern in sensitive_patterns):
            return "[REDACTED sensitive query]"
        return query[:500]

    @staticmethod
    def _coerce_limit(limit: Any) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer") from exc
        return max(1, min(value, 50))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _fts_queries(query: str) -> List[str]:
        terms = MemoryV2Index._fts_terms(query)
        if not terms:
            return []
        strict = " AND ".join(f'"{term}"' for term in terms)
        relaxed_terms = [term for term in terms if term not in MemoryV2Index._STOPWORDS]
        if len(relaxed_terms) < 2:
            return [strict]
        relaxed_and = " AND ".join(f'"{term}"' for term in relaxed_terms)
        relaxed_or = " OR ".join(f'"{term}"' for term in relaxed_terms)
        queries = [strict]
        if relaxed_and != strict:
            queries.append(relaxed_and)
        queries.append(relaxed_or)
        return queries

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "is",
        "it",
        "me",
        "of",
        "or",
        "should",
        "that",
        "the",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
    }

    @staticmethod
    def _fts_terms(query: str) -> List[str]:
        return [term for term in "".join(ch if ch.isalnum() else " " for ch in query).split() if term]

    @staticmethod
    def _fts_query(query: str) -> str:
        # Backwards-compatible strict AND query used by older tests/callers.
        terms = MemoryV2Index._fts_terms(query)
        if not terms:
            return '""'
        return " AND ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _row_to_result(row: sqlite3.Row | tuple) -> Dict[str, Any]:
        return {
            "id": row[0],
            "type": row[1],
            "title": row[2] or "",
            "subject": row[3] or "",
            "predicate": row[4] or "",
            "value": row[5] or "",
            "body": row[6] or "",
            "summary": row[7] or "",
            "status": row[8] or "",
            "confidence": row[9],
            "importance": row[10],
            "created_at": row[11] or "",
            "updated_at": row[12] or "",
            "valid_from": row[13] or "",
            "valid_until": row[14] or "",
            "expires_at": row[15] or "",
            "source_refs": json.loads(row[16] or "[]"),
            "supersedes": json.loads(row[17] or "[]"),
            "superseded_by": row[18] or "",
            "tags": json.loads(row[19] or "[]"),
            "file_path": row[20] or "",
            "rank": row[21],
        }
