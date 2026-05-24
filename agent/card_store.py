"""Card Store — SQLite-backed storage for memory cards and knowledge cards.

Profile-aware: uses ``get_hermes_home()`` for path resolution so each
profile gets its own ``card_store.db``.

Schema
------
memory_cards
  id, type, title, body, tags[], created_at, updated_at, source, project,
  context, confidence, agent_created, pinned, body_hash
  types: decision, rule, incident, preference, active_context

knowledge_cards
  id, title, body, source, evidence[], truth_level, project_fit, status,
  domains[], created_at, updated_at, review_status, duplicate_of,
  origin_project, promoted, body_hash
  truth_levels: verified, probable, speculative, disproven
  review_status: pending_review, approved, rejected, deferred, duplicate

FTS5 virtual tables provide full-text search on both tables.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_NAME = "card_store.db"

MEMORY_CARD_TYPES = {"decision", "rule", "incident", "preference", "active_context"}
KNOWLEDGE_TRUTH_LEVELS = {"verified", "probable", "speculative", "disproven"}
KNOWLEDGE_REVIEW_STATUSES = {"pending_review", "approved", "rejected", "deferred", "duplicate", "revision_requested"}


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class CardRef:
    """Lightweight reference to a card (for context packs)."""
    card_id: str
    card_type: str  # "memory" | "knowledge"
    title: str
    tier: str = ""
    relevance_score: float = 0.0
    token_cost: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionReason:
    """Why a card/file was selected for a context pack."""
    session_id: str
    card_id: str
    card_type: str
    tier: str
    reason: str
    relevance_score: float
    token_cost: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── CardStore ────────────────────────────────────────────────────────

class CardStore:
    """SQLite-backed card store with FTS5 search.

    Profile-aware — each profile (HERMES_HOME) gets its own database.
    """

    def __init__(self, hermes_home: Optional[Path] = None) -> None:
        if hermes_home is None:
            hermes_home = get_hermes_home()
        self._db_path = hermes_home / _DB_NAME
        self._init_db()

    # ── Connection management ────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """Yield a connection with row_factory set."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Initialization ───────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create tables and FTS5 virtual tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            # Memory cards table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_cards (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL CHECK(type IN ('decision','rule','incident','preference','active_context')),
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                    updated_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                    source TEXT DEFAULT '',
                    project TEXT DEFAULT '',
                    context TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.5,
                    agent_created INTEGER DEFAULT 1,
                    pinned INTEGER DEFAULT 0,
                    body_hash TEXT DEFAULT ''
                )
            """)

            # Knowledge cards table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_cards (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    source TEXT DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '[]',
                    truth_level TEXT NOT NULL DEFAULT 'probable'
                        CHECK(truth_level IN ('verified','probable','speculative','disproven')),
                    project_fit REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'draft',
                    domains TEXT NOT NULL DEFAULT '[]',
                    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                    updated_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                    review_status TEXT DEFAULT 'pending_review'
                        CHECK(review_status IN ('pending_review','approved','rejected','deferred','duplicate','revision_requested')),
                    duplicate_of TEXT DEFAULT '',
                    origin_project TEXT DEFAULT '',
                    promoted INTEGER DEFAULT 0,
                    body_hash TEXT DEFAULT ''
                )
            """)

            # Context pack log table (for "why selected" trace)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_pack_log (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    card_type TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    reason TEXT DEFAULT '',
                    relevance_score REAL DEFAULT 0.0,
                    token_cost INTEGER DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Metrics snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics_snapshots (
                    id TEXT PRIMARY KEY,
                    snapshot_type TEXT NOT NULL,
                    period TEXT NOT NULL,
                    data TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # FTS5 virtual table for memory cards
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_cards_fts
                USING fts5(body, title, tags, context, project, source,
                           tokenize='unicode61')
            """)

            # FTS5 virtual table for knowledge cards
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_cards_fts
                USING fts5(body, title, source, domains, origin_project,
                           tokenize='unicode61')
            """)

            # Triggers for FTS5 auto-sync on INSERT
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memory_cards_ai AFTER INSERT ON memory_cards
                BEGIN
                    INSERT INTO memory_cards_fts(rowid, body, title, tags, context, project, source)
                    VALUES (new.rowid, new.body, new.title, new.tags, new.context, new.project, new.source);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS knowledge_cards_ai AFTER INSERT ON knowledge_cards
                BEGIN
                    INSERT INTO knowledge_cards_fts(rowid, body, title, source, domains, origin_project)
                    VALUES (new.rowid, new.body, new.title, new.source, new.domains, new.origin_project);
                END
            """)

            # Triggers for FTS5 auto-sync on UPDATE
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memory_cards_au AFTER UPDATE ON memory_cards
                BEGIN
                    DELETE FROM memory_cards_fts WHERE rowid = old.rowid;
                    INSERT INTO memory_cards_fts(rowid, body, title, tags, context, project, source)
                    VALUES (new.rowid, new.body, new.title, new.tags, new.context, new.project, new.source);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS knowledge_cards_au AFTER UPDATE ON knowledge_cards
                BEGIN
                    DELETE FROM knowledge_cards_fts WHERE rowid = old.rowid;
                    INSERT INTO knowledge_cards_fts(rowid, body, title, source, domains, origin_project)
                    VALUES (new.rowid, new.body, new.title, new.source, new.domains, new.origin_project);
                END
            """)

            # Triggers for FTS5 auto-sync on DELETE
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memory_cards_ad AFTER DELETE ON memory_cards
                BEGIN
                    DELETE FROM memory_cards_fts WHERE rowid = old.rowid;
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS knowledge_cards_ad AFTER DELETE ON knowledge_cards
                BEGIN
                    DELETE FROM knowledge_cards_fts WHERE rowid = old.rowid;
                END
            """)

            # Indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_cards_type ON memory_cards(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_cards_project ON memory_cards(project)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_cards_created ON memory_cards(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_cards_pinned ON memory_cards(pinned)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_cards_review ON knowledge_cards(review_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_cards_truth ON knowledge_cards(truth_level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_cards_domains ON knowledge_cards(domains)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_cards_created ON knowledge_cards(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_context_pack_log_session ON context_pack_log(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_type ON metrics_snapshots(snapshot_type)")
            self._repair_fts_objects(conn)

    def _repair_fts_objects(self, conn: sqlite3.Connection) -> None:
        """Repair legacy FTS triggers/indexes without touching source rows.

        Older local card stores used FTS5 special ``delete`` inserts in update
        and delete triggers. If the FTS table drifted from the base table, those
        triggers raised ``sqlite3.OperationalError: SQL logic error`` on normal
        review actions. Recreate the FTS tables from source tables once when a
        legacy trigger is detected.
        """
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name IN ('memory_cards_au', 'memory_cards_ad', "
            "'knowledge_cards_au', 'knowledge_cards_ad')"
        ).fetchall()
        legacy = any("_fts, rowid" in (row["sql"] or "") for row in rows)
        if not legacy:
            return

        for name in (
            "memory_cards_ai", "memory_cards_au", "memory_cards_ad",
            "knowledge_cards_ai", "knowledge_cards_au", "knowledge_cards_ad",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")

        conn.execute("DROP TABLE IF EXISTS memory_cards_fts")
        conn.execute("DROP TABLE IF EXISTS knowledge_cards_fts")

        conn.execute("""
            CREATE VIRTUAL TABLE memory_cards_fts
            USING fts5(body, title, tags, context, project, source,
                       tokenize='unicode61')
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE knowledge_cards_fts
            USING fts5(body, title, source, domains, origin_project,
                       tokenize='unicode61')
        """)

        conn.execute("""
            INSERT INTO memory_cards_fts(rowid, body, title, tags, context, project, source)
            SELECT rowid, body, title, tags, context, project, source FROM memory_cards
        """)
        conn.execute("""
            INSERT INTO knowledge_cards_fts(rowid, body, title, source, domains, origin_project)
            SELECT rowid, body, title, source, domains, origin_project FROM knowledge_cards
        """)

        conn.execute("""
            CREATE TRIGGER memory_cards_ai AFTER INSERT ON memory_cards
            BEGIN
                INSERT INTO memory_cards_fts(rowid, body, title, tags, context, project, source)
                VALUES (new.rowid, new.body, new.title, new.tags, new.context, new.project, new.source);
            END
        """)
        conn.execute("""
            CREATE TRIGGER knowledge_cards_ai AFTER INSERT ON knowledge_cards
            BEGIN
                INSERT INTO knowledge_cards_fts(rowid, body, title, source, domains, origin_project)
                VALUES (new.rowid, new.body, new.title, new.source, new.domains, new.origin_project);
            END
        """)
        conn.execute("""
            CREATE TRIGGER memory_cards_au AFTER UPDATE ON memory_cards
            BEGIN
                DELETE FROM memory_cards_fts WHERE rowid = old.rowid;
                INSERT INTO memory_cards_fts(rowid, body, title, tags, context, project, source)
                VALUES (new.rowid, new.body, new.title, new.tags, new.context, new.project, new.source);
            END
        """)
        conn.execute("""
            CREATE TRIGGER knowledge_cards_au AFTER UPDATE ON knowledge_cards
            BEGIN
                DELETE FROM knowledge_cards_fts WHERE rowid = old.rowid;
                INSERT INTO knowledge_cards_fts(rowid, body, title, source, domains, origin_project)
                VALUES (new.rowid, new.body, new.title, new.source, new.domains, new.origin_project);
            END
        """)
        conn.execute("""
            CREATE TRIGGER memory_cards_ad AFTER DELETE ON memory_cards
            BEGIN
                DELETE FROM memory_cards_fts WHERE rowid = old.rowid;
            END
        """)
        conn.execute("""
            CREATE TRIGGER knowledge_cards_ad AFTER DELETE ON knowledge_cards
            BEGIN
                DELETE FROM knowledge_cards_fts WHERE rowid = old.rowid;
            END
        """)

    # ── CRUD: Memory Cards ───────────────────────────────────────────

    def create_memory_card(
        self,
        card_type: str,
        title: str,
        body: str,
        tags: Optional[List[str]] = None,
        source: str = "",
        project: str = "",
        context: str = "",
        confidence: float = 0.5,
        agent_created: bool = True,
        pinned: bool = False,
    ) -> str:
        """Create a memory card. Returns the card ID."""
        if card_type not in MEMORY_CARD_TYPES:
            raise ValueError(f"Invalid memory card type: {card_type}. Valid: {MEMORY_CARD_TYPES}")

        card_id = str(uuid.uuid4())[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body_hash = hashlib.sha256(body.encode()).hexdigest()[:16]

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO memory_cards
                (id, type, title, body, tags, created_at, updated_at,
                 source, project, context, confidence, agent_created, pinned, body_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card_id, card_type, title, body,
                json.dumps(tags or []), now, now,
                source, project, context, confidence,
                1 if agent_created else 0,
                1 if pinned else 0,
                body_hash,
            ))

        logger.info("Memory card created: %s (%s) — %s", card_id, card_type, title)
        return card_id

    def get_memory_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory card by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memory_cards WHERE id = ?", (card_id,)
            ).fetchone()
        if row:
            return dict(row)
        return None

    def update_memory_card(self, card_id: str, **kwargs) -> bool:
        """Update a memory card. Returns True if updated."""
        allowed = {"title", "body", "tags", "source", "project", "context",
                   "confidence", "pinned"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updates["updated_at"] = now

        if "body" in updates:
            updates["body_hash"] = hashlib.sha256(updates["body"].encode()).hexdigest()[:16]

        if "tags" in updates and isinstance(updates["tags"], list):
            updates["tags"] = json.dumps(updates["tags"])
        if "pinned" in updates:
            updates["pinned"] = 1 if updates["pinned"] else 0

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [card_id]

        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE memory_cards SET {set_clause} WHERE id = ?", values
            )
        return cur.rowcount > 0

    def delete_memory_card(self, card_id: str) -> bool:
        """Delete a memory card. Returns True if deleted."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memory_cards WHERE id = ?", (card_id,))
        return cur.rowcount > 0

    # ── CRUD: Knowledge Cards ────────────────────────────────────────

    def create_knowledge_card(
        self,
        title: str,
        body: str,
        source: str = "",
        evidence: Optional[List[str]] = None,
        truth_level: str = "probable",
        project_fit: float = 0.5,
        status: str = "draft",
        domains: Optional[List[str]] = None,
        review_status: str = "pending_review",
        duplicate_of: str = "",
        origin_project: str = "",
        promoted: bool = False,
    ) -> str:
        """Create a knowledge card. Returns the card ID."""
        if truth_level not in KNOWLEDGE_TRUTH_LEVELS:
            raise ValueError(f"Invalid truth level: {truth_level}. Valid: {KNOWLEDGE_TRUTH_LEVELS}")
        if review_status not in KNOWLEDGE_REVIEW_STATUSES:
            raise ValueError(f"Invalid review status: {review_status}. Valid: {KNOWLEDGE_REVIEW_STATUSES}")

        card_id = str(uuid.uuid4())[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body_hash = hashlib.sha256(body.encode()).hexdigest()[:16]

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO knowledge_cards
                (id, title, body, source, evidence, truth_level, project_fit, status,
                 domains, created_at, updated_at, review_status, duplicate_of,
                 origin_project, promoted, body_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card_id, title, body, source,
                json.dumps(evidence or []), truth_level, project_fit, status,
                json.dumps(domains or []), now, now,
                review_status, duplicate_of, origin_project,
                1 if promoted else 0,
                body_hash,
            ))

        logger.info("Knowledge card created: %s — %s", card_id, title)
        return card_id

    def get_knowledge_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Get a knowledge card by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_cards WHERE id = ?", (card_id,)
            ).fetchone()
        if row:
            return dict(row)
        return None

    def update_knowledge_card(self, card_id: str, **kwargs) -> bool:
        """Update a knowledge card. Returns True if updated."""
        allowed = {"title", "body", "source", "evidence", "truth_level",
                   "project_fit", "status", "domains", "review_status",
                   "duplicate_of", "origin_project", "promoted"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updates["updated_at"] = now

        if "body" in updates:
            updates["body_hash"] = hashlib.sha256(updates["body"].encode()).hexdigest()[:16]

        for json_field in ("evidence", "domains"):
            if json_field in updates and isinstance(updates[json_field], list):
                updates[json_field] = json.dumps(updates[json_field])
        if "promoted" in updates:
            updates["promoted"] = 1 if updates["promoted"] else 0

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [card_id]

        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE knowledge_cards SET {set_clause} WHERE id = ?", values
            )
        return cur.rowcount > 0

    def delete_knowledge_card(self, card_id: str) -> bool:
        """Delete a knowledge card. Returns True if deleted."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM knowledge_cards WHERE id = ?", (card_id,))
        return cur.rowcount > 0

    # ── Search ──────────────────────────────────────────────────────

    def search_cards(
        self,
        query: str,
        card_type: Optional[str] = None,
        domains: Optional[List[str]] = None,
        status: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search memory + knowledge cards using FTS5.

        Returns ranked results with relevance score.
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        results: List[Dict[str, Any]] = []

        with self._conn() as conn:
            # Search memory cards
            mem_where = []
            mem_params: list = []
            if card_type and card_type != "knowledge":
                mem_where.append("m.type = ?")
                mem_params.append(card_type)

            mem_query = f"""
                SELECT m.*, memory_cards_fts.rank as relevance_score
                FROM memory_cards m
                JOIN memory_cards_fts ON m.rowid = memory_cards_fts.rowid
                WHERE memory_cards_fts MATCH ?
                {'AND ' + ' AND '.join(mem_where) if mem_where else ''}
                ORDER BY memory_cards_fts.rank
                LIMIT ?
            """
            mem_params = [query] + mem_params + [limit]
            for row in conn.execute(mem_query, mem_params).fetchall():
                d = dict(row)
                d["card_type"] = "memory"
                results.append(d)

            # Search knowledge cards
            know_where = []
            know_params: list = []
            if domains:
                # Check if any domain matches
                domain_conditions = []
                for d in domains:
                    domain_conditions.append("k.domains LIKE ?")
                    know_params.append(f'%"{d}"%')
                know_where.append(f"({' OR '.join(domain_conditions)})")
            if review_status:
                know_where.append("k.review_status = ?")
                know_params.append(review_status)

            know_query = f"""
                SELECT k.*, knowledge_cards_fts.rank as relevance_score
                FROM knowledge_cards k
                JOIN knowledge_cards_fts ON k.rowid = knowledge_cards_fts.rowid
                WHERE knowledge_cards_fts MATCH ?
                {'AND ' + ' AND '.join(know_where) if know_where else ''}
                ORDER BY knowledge_cards_fts.rank
                LIMIT ?
            """
            know_params = [query] + know_params + [limit]
            for row in conn.execute(know_query, know_params).fetchall():
                d = dict(row)
                d["card_type"] = "knowledge"
                results.append(d)

        # Sort by relevance score (lower = better in FTS5)
        results.sort(key=lambda x: x.get("relevance_score", 999))
        return results[:limit]

    # ── List ─────────────────────────────────────────────────────────

    def list_memory_cards(
        self,
        card_type: Optional[str] = None,
        status_filter: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List memory cards with pagination. Returns (cards, total_count)."""
        where_clauses = []
        params: list = []

        if card_type and card_type in MEMORY_CARD_TYPES:
            where_clauses.append("type = ?")
            params.append(card_type)
        if project:
            where_clauses.append("project = ?")
            params.append(project)

        where = " AND ".join(where_clauses) if where_clauses else "1=1"

        with self._conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM memory_cards WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM memory_cards WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

        cards = [dict(r) for r in rows]
        return cards, total

    def list_knowledge_cards(
        self,
        review_status: Optional[str] = None,
        truth_level: Optional[str] = None,
        domains: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List knowledge cards with pagination. Returns (cards, total_count)."""
        where_clauses = []
        params: list = []

        if review_status and review_status in KNOWLEDGE_REVIEW_STATUSES:
            where_clauses.append("review_status = ?")
            params.append(review_status)
        if truth_level and truth_level in KNOWLEDGE_TRUTH_LEVELS:
            where_clauses.append("truth_level = ?")
            params.append(truth_level)
        if domains:
            for d in domains:
                where_clauses.append("domains LIKE ?")
                params.append(f'%"{d}"%')

        where = " AND ".join(where_clauses) if where_clauses else "1=1"

        with self._conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM knowledge_cards WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM knowledge_cards WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

        cards = [dict(r) for r in rows]
        return cards, total

    # ── Duplicate detection ──────────────────────────────────────────

    def find_duplicate_memory_cards(
        self,
        card_type: str,
        title: str,
        body: str,
        similarity_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Find existing memory cards that are duplicates of the proposed card.

        Uses hash-based exact match + keyword overlap for semantic similarity.
        """
        candidates = []
        body_hash = hashlib.sha256(body.encode()).hexdigest()[:16]

        with self._conn() as conn:
            # Exact hash match
            rows = conn.execute("""
                SELECT * FROM memory_cards
                WHERE type = ? AND body_hash = ?
            """, (card_type, body_hash)).fetchall()
            for row in rows:
                d = dict(row)
                d["similarity"] = 1.0
                candidates.append(d)

        # Keyword overlap for semantic similarity
        if not candidates:
            existing = self.list_memory_cards(card_type=card_type, limit=200)[0]
            body_words = set(_tokenize(body))
            title_words = set(_tokenize(title))
            for card in existing:
                existing_words = set(_tokenize(card.get("body", "")))
                existing_title = set(_tokenize(card.get("title", "")))
                if not existing_words and not existing_title:
                    continue
                all_words = body_words | title_words | existing_words | existing_title
                if not all_words:
                    continue
                overlap = (body_words & existing_words) | (title_words & existing_title)
                similarity = len(overlap) / len(all_words) if all_words else 0
                if similarity >= similarity_threshold:
                    card["similarity"] = round(similarity, 3)
                    candidates.append(card)

        candidates.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return candidates

    def find_duplicate_knowledge_cards(
        self,
        title: str,
        body: str,
        domains: Optional[List[str]] = None,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find existing knowledge cards that are duplicates of the proposed card."""
        candidates = []
        body_hash = hashlib.sha256(body.encode()).hexdigest()[:16]

        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM knowledge_cards
                WHERE body_hash = ?
            """, (body_hash,)).fetchall()
            for row in rows:
                d = dict(row)
                d["similarity"] = 1.0
                candidates.append(d)

        if not candidates:
            existing = self.list_knowledge_cards(limit=200)[0]
            body_words = set(_tokenize(body))
            title_words = set(_tokenize(title))
            for card in existing:
                existing_words = set(_tokenize(card.get("body", "")))
                existing_title = set(_tokenize(card.get("title", "")))
                all_words = body_words | title_words | existing_words | existing_title
                if not all_words:
                    continue
                overlap = (body_words & existing_words) | (title_words & existing_title)
                similarity = len(overlap) / len(all_words) if all_words else 0
                if similarity >= similarity_threshold:
                    card["similarity"] = round(similarity, 3)
                    candidates.append(card)

        candidates.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return candidates

    # ── Context pack log ─────────────────────────────────────────────

    def log_context_pack_selection(
        self,
        session_id: str,
        card_id: str,
        card_type: str,
        tier: str,
        reason: str,
        relevance_score: float,
        token_cost: int,
    ) -> str:
        """Log why a card was selected for a context pack."""
        entry_id = str(uuid.uuid4())[:12]
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO context_pack_log
                (id, session_id, card_id, card_type, tier, reason, relevance_score, token_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry_id, session_id, card_id, card_type, tier, reason, relevance_score, token_cost))
        return entry_id

    def get_context_pack_trace(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the context pack selection trace for a session."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM context_pack_log
                WHERE session_id = ?
                ORDER BY created_at
            """, (session_id,)).fetchall()
        return [dict(r) for r in rows]

    # ── Metrics snapshots ────────────────────────────────────────────

    def save_metrics_snapshot(
        self,
        snapshot_type: str,
        period: str,
        data: Dict[str, Any],
    ) -> str:
        """Save a metrics snapshot."""
        entry_id = str(uuid.uuid4())[:12]
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO metrics_snapshots (id, snapshot_type, period, data)
                VALUES (?, ?, ?, ?)
            """, (entry_id, snapshot_type, period, json.dumps(data)))
        return entry_id

    def get_metrics_snapshots(
        self,
        snapshot_type: str,
        period: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics snapshots."""
        params = [snapshot_type]
        where = "snapshot_type = ?"
        if period:
            where += " AND period = ?"
            params.append(period)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM metrics_snapshots WHERE {where} ORDER BY created_at",
                params,
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"])
            results.append(d)
        return results


# ── Helpers ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for keyword overlap."""
    import re
    return re.findall(r'\b\w+\b', text.lower())
