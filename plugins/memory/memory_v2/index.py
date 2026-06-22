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
            importance=card.importance,
            updated_at=card.updated_at,
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
            created_at=str(event.get("created_at") or ""),
            updated_at=str(event.get("updated_at") or ""),
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

    def active_project_cards(self, *, limit: int = 5) -> List[Dict[str, Any]]:
        """Return active project-card records for broad continuity recall."""
        safe_limit = self._coerce_limit(limit)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  id, type, title, subject, predicate, value, body, summary, status,
                  confidence, importance, created_at, updated_at, valid_from, valid_until, expires_at,
                  source_refs, supersedes, superseded_by, tags, file_path,
                  0.0 AS rank
                FROM memories
                WHERE type = 'project_state' AND status = 'active'
                ORDER BY COALESCE(importance, 0.0) DESC, updated_at DESC, title ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [result for result in (self._row_to_result(row) for row in rows) if self._is_temporally_visible(result)]

    def search(self, query: str, *, route: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        safe_limit = self._coerce_limit(limit)
        fts_queries = self._fts_queries(query_text)
        rows = []
        sql_limit = min(100, max(safe_limit * 4, safe_limit + 10))
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
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, sql_limit),
                ).fetchall()
                # Preserve the old high-precision strict -> relaxed fallback
                # behavior. Hybrid scoring reranks within the first plausible
                # candidate pool rather than flooding precise queries with
                # broad OR matches.
                if rows:
                    break
        results = [result for result in (self._row_to_result(row) for row in rows) if self._is_temporally_visible(result)]
        if str(route or "").strip().lower() in {"deep_recall", "past_conversation_exact"}:
            results = self._annotate_hybrid_scores(query_text, route=route, results=results)
        else:
            results = self._hybrid_rank_results(query_text, route=route, results=results)
        results = results[:safe_limit]
        self.log_retrieval(query_text, route=route, retrieved_ids=[result["id"] for result in results])
        return results

    @classmethod
    def _annotate_hybrid_scores(cls, query: str, *, route: str = "", results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_terms = cls._content_terms(query)
        query_phrase = " ".join(query_terms)
        annotated: List[Dict[str, Any]] = []
        for result in results:
            components = cls._score_components(result, query_terms=query_terms, query_phrase=query_phrase, route=route)
            enriched = dict(result)
            enriched["hybrid_score"] = sum(components.values())
            enriched["score_components"] = components
            annotated.append(enriched)
        return annotated

    @classmethod
    def _hybrid_rank_results(cls, query: str, *, route: str = "", results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_terms = cls._content_terms(query)
        query_phrase = " ".join(query_terms)
        ranked: List[Dict[str, Any]] = []
        for result in results:
            components = cls._score_components(result, query_terms=query_terms, query_phrase=query_phrase, route=route)
            score = sum(components.values())
            enriched = dict(result)
            enriched["hybrid_score"] = score
            enriched["score_components"] = components
            ranked.append(enriched)
        ranked.sort(
            key=lambda item: (
                float(item.get("hybrid_score") or 0.0),
                -cls._status_order(str(item.get("status") or "")),
                str(item.get("updated_at") or ""),
                str(item.get("id") or ""),
            ),
            reverse=True,
        )
        return ranked

    @classmethod
    def _score_components(
        cls,
        result: Dict[str, Any],
        *,
        query_terms: List[str],
        query_phrase: str,
        route: str,
    ) -> Dict[str, float]:
        rank = result.get("rank")
        try:
            # SQLite FTS5 bm25() returns smaller/more-negative scores for
            # stronger lexical matches. Keep that signal dominant for raw
            # evidence retrieval (for example LoCoMo), while still allowing
            # route/type and structured-field boosts to break close ties.
            fts_score = min(20.0, max(0.0, -float(rank or 0.0) * 5_000_000.0))
        except (TypeError, ValueError):
            fts_score = 0.0
        token_overlap = cls._token_overlap_score(result, query_terms)
        phrase_boost = cls._phrase_boost(result, query_phrase)
        route_type_boost = cls._route_type_boost(str(route or ""), str(result.get("type") or ""))
        status_boost = cls._status_boost(str(result.get("status") or ""))
        confidence_boost = cls._bounded_float(result.get("confidence"), default=0.5) * 0.15
        importance_boost = cls._bounded_float(result.get("importance"), default=0.5) * 0.25
        return {
            "fts": round(fts_score, 6),
            "token_overlap": round(token_overlap, 6),
            "phrase": round(phrase_boost, 6),
            "route_type_boost": round(route_type_boost, 6),
            "status": round(status_boost, 6),
            "confidence": round(confidence_boost, 6),
            "importance": round(importance_boost, 6),
        }

    @classmethod
    def _token_overlap_score(cls, result: Dict[str, Any], query_terms: List[str]) -> float:
        if not query_terms:
            return 0.0
        query_set = set(query_terms)
        field_weights = {
            "title": 1.25,
            "subject": 1.1,
            "predicate": 1.0,
            "value": 1.2,
            "summary": 1.0,
            "body": 0.75,
            "tags": 0.8,
        }
        matched_weight = 0.0
        max_weight = sum(field_weights.values())
        for field, weight in field_weights.items():
            value = result.get(field)
            if isinstance(value, list):
                text = " ".join(str(part) for part in value)
            else:
                text = str(value or "")
            field_terms = set(cls._content_terms(text))
            if field_terms:
                matched_weight += weight * (len(query_set & field_terms) / len(query_set))
        return 4.0 * (matched_weight / max_weight)

    @classmethod
    def _phrase_boost(cls, result: Dict[str, Any], query_phrase: str) -> float:
        if not query_phrase or len(query_phrase) < 6:
            return 0.0
        haystack = " ".join(
            str(result.get(field) or "")
            for field in ("title", "subject", "predicate", "value", "summary", "body")
        ).lower()
        if query_phrase in " ".join(cls._content_terms(haystack)):
            return 1.0
        return 0.0

    @staticmethod
    def _route_type_boost(route: str, memory_type: str) -> float:
        route_key = route.strip().lower()
        type_key = memory_type.strip().lower()
        preferred: Dict[str, Dict[str, float]] = {
            "project_continuity": {"project_state": 2.0, "preference": 0.25, "fact": 0.2, "raw_event": 0.1},
            "preference_recall": {"preference": 2.0, "fact": 0.35, "project_state": 0.15, "raw_event": 0.1},
            "environment_fact": {"environment": 2.0, "fact": 0.8, "raw_event": 0.1},
            "procedure_lookup": {"procedure_ref": 2.0, "fact": 0.3, "raw_event": 0.1},
            "deep_recall": {"raw_event": 0.25, "project_state": 0.15, "fact": 0.1, "preference": 0.1},
            "past_conversation_exact": {"raw_event": 0.5},
        }
        return preferred.get(route_key, {}).get(type_key, 0.0)

    @staticmethod
    def _status_boost(status: str) -> float:
        return {
            "active": 0.8,
            "uncertain": 0.45,
            "pending": 0.35,
            "promoted": 0.3,
            "archived_only": 0.1,
            "archived": 0.0,
            "superseded": -1.0,
            "rejected": -1.5,
        }.get(status, 0.0)

    @staticmethod
    def _status_order(status: str) -> int:
        return {
            "active": 0,
            "uncertain": 1,
            "pending": 2,
            "promoted": 3,
            "archived_only": 4,
            "archived": 5,
            "superseded": 6,
            "rejected": 7,
        }.get(status, 8)

    @staticmethod
    def _bounded_float(value: Any, *, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(0.0, min(1.0, numeric))

    @classmethod
    def _content_terms(cls, text: str) -> List[str]:
        return [term.lower() for term in cls._fts_terms(text) if term.lower() not in cls._STOPWORDS]

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
            r"(?i)client\s+secret\s*(?:is|=|:)\s*\S+",
            r"(?i)client\s+secret\s+\S+",
            r"(?i)credential\s*(?:is|=|:)\s*\S+",
            r"(?i)credential\s+\S+",
            r"(?i)private\s+key\s*(?:is|=|:)\s*\S+",
            r"(?i)private\s+key\s+\S+",
            r"(?i)api[_ -]?key\s*(?:is|=|:)\s*\S+",
            r"(?i)api[_ -]?key\s+\S+",
            r"(?i)authorization\s*:\s*bearer\s+\S+",
            r"(?i)bearer\s+\S+",
            r"(?i)[A-Z0-9_]*API[_-]?KEY\s*=\s*\S+",
            r"(?i)[A-Z0-9_]*API[_-]?KEY\s+\S+",
            r"(?i)\bsk-[A-Za-z0-9][A-Za-z0-9_-]{8,}\b",
            r"(?i)\bgh[pousr]_[A-Za-z0-9_]{8,}\b",
            r"(?i)\bxox[baprs]-[A-Za-z0-9-]{8,}\b",
            r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
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
