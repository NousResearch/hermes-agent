"""Episodic Timeline — what-where-when binding for memory.

Binds facts into ordered episodes with temporal context, enabling
queries like "what happened before/after X?" and "what was I working
on when this happened?"

Scientific basis:
- Tulving (1972): episodic = what-where-when binding
- Park et al. (2023): memory stream with recency × importance × relevance
- Clayton & Dickinson (1998): first non-human evidence for what-where-when

All methods are best-effort: exceptions caught and logged, never blocking.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Additional schema for episodic tables (added to store.py _SCHEMA)
EPISODE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS episodes (
    episode_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    title         TEXT DEFAULT '',
    started_at    TIMESTAMP NOT NULL,
    ended_at      TIMESTAMP,
    context_hash  TEXT DEFAULT '',
    summary       TEXT DEFAULT '',
    salience_peak REAL DEFAULT 0.0,
    fact_count    INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episode_facts (
    episode_id    INTEGER REFERENCES episodes(episode_id),
    fact_id       INTEGER REFERENCES facts(fact_id),
    ordinal       INTEGER NOT NULL,
    role          TEXT DEFAULT 'event',
    linked_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (episode_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at);
CREATE INDEX IF NOT EXISTS idx_episode_facts_ord ON episode_facts(episode_id, ordinal);
"""


class EpisodicTimeline:
    """Binds facts into ordered episodes with temporal context.

    Episodes are session-scoped sequences of facts. Each fact gets an
    ordinal position within its episode. Retrieval can follow temporal
    links: "what happened around X?" or "what happened before X?"

    Usage::

        timeline = EpisodicTimeline(conn, lock)
        ep_id = timeline.start_episode("session-1")
        timeline.append_fact(ep_id, fact_id=42)
        timeline.append_fact(ep_id, fact_id=43)
        timeline.close_episode(ep_id, summary="Discussed auth refactor")

        # Later: retrieve context around a fact
        window = timeline.get_context_window(fact_id=42, window=3)
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = conn
        self._lock = lock
        self._active_episode: int | None = None
        self._current_ordinal: int = 0

    def init_tables(self) -> None:
        """Create episode tables if they don't exist."""
        try:
            with self._lock:
                self._conn.executescript(EPISODE_SCHEMA)
                self._conn.commit()
        except Exception as e:
            logger.debug("EpisodicTimeline init_tables failed: %s", e)

    def start_episode(self, session_id: str,
                      context_hash: str = "") -> int | None:
        """Create a new episode. Returns episode_id."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with self._lock:
                cursor = self._conn.execute(
                    "INSERT INTO episodes (session_id, started_at, context_hash) "
                    "VALUES (?, ?, ?)",
                    (session_id, now, context_hash),
                )
                self._conn.commit()
                self._active_episode = cursor.lastrowid
                self._current_ordinal = 0
                return self._active_episode
        except Exception as e:
            logger.debug("start_episode failed: %s", e)
            return None

    def append_fact(self, fact_id: int,
                    episode_id: int | None = None,
                    role: str = "event") -> int | None:
        """Add a fact to an episode in sequence. Returns ordinal position."""
        ep_id = episode_id or self._active_episode
        if ep_id is None:
            return None
        try:
            with self._lock:
                self._current_ordinal += 1
                self._conn.execute(
                    "INSERT OR IGNORE INTO episode_facts "
                    "(episode_id, fact_id, ordinal, role) VALUES (?, ?, ?, ?)",
                    (ep_id, fact_id, self._current_ordinal, role),
                )
                self._conn.execute(
                    "UPDATE episodes SET fact_count = "
                    "(SELECT COUNT(*) FROM episode_facts WHERE episode_id = ?) "
                    "WHERE episode_id = ?",
                    (ep_id, ep_id),
                )
                self._conn.commit()
                return self._current_ordinal
        except Exception as e:
            logger.debug("append_fact failed: %s", e)
            return None

    def close_episode(self, episode_id: int | None = None,
                      summary: str = "") -> None:
        """Close an episode, setting ended_at and summary."""
        ep_id = episode_id or self._active_episode
        if ep_id is None:
            return
        try:
            now = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._conn.execute(
                    "UPDATE episodes SET ended_at = ?, summary = ? "
                    "WHERE episode_id = ?",
                    (now, summary, ep_id),
                )
                self._conn.commit()
            if ep_id == self._active_episode:
                self._active_episode = None
        except Exception as e:
            logger.debug("close_episode failed: %s", e)

    def get_timeline(self, session_id: str | None = None,
                     limit: int = 20) -> list[dict]:
        """Return episodes in chronological order."""
        try:
            with self._lock:
                if session_id:
                    rows = self._conn.execute(
                        "SELECT * FROM episodes WHERE session_id = ? "
                        "ORDER BY started_at DESC LIMIT ?",
                        (session_id, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("get_timeline failed: %s", e)
            return []

    def get_context_window(self, fact_id: int,
                           window: int = 3) -> list[dict]:
        """Return N facts before and after a given fact within its episode."""
        try:
            with self._lock:
                link = self._conn.execute(
                    "SELECT episode_id, ordinal FROM episode_facts "
                    "WHERE fact_id = ?",
                    (fact_id,),
                ).fetchone()
                if not link:
                    return []
                ep_id = link["episode_id"]
                ordinal = link["ordinal"]
                rows = self._conn.execute(
                    "SELECT ef.fact_id, ef.ordinal, ef.role, "
                    "f.content, f.category, f.trust_score, f.strength "
                    "FROM episode_facts ef "
                    "JOIN facts f ON ef.fact_id = f.fact_id "
                    "WHERE ef.episode_id = ? "
                    "AND ef.ordinal BETWEEN ? AND ? "
                    "ORDER BY ef.ordinal",
                    (ep_id, max(1, ordinal - window), ordinal + window),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("get_context_window failed: %s", e)
            return []

    def get_predecessors(self, episode_id: int,
                         limit: int = 5) -> list[dict]:
        """Return episodes that ended before this one started."""
        try:
            with self._lock:
                ep = self._conn.execute(
                    "SELECT started_at FROM episodes WHERE episode_id = ?",
                    (episode_id,),
                ).fetchone()
                if not ep:
                    return []
                rows = self._conn.execute(
                    "SELECT * FROM episodes "
                    "WHERE started_at < ? AND episode_id != ? "
                    "ORDER BY started_at DESC LIMIT ?",
                    (ep["started_at"], episode_id, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("get_predecessors failed: %s", e)
            return []

    def detect_topic_shift(self, new_entities: set[str],
                           recent_entities: set[str],
                           threshold: float = 0.2) -> bool:
        """Detect if a topic shift has occurred.

        Returns True if entity overlap between new and recent is below threshold.
        """
        if not recent_entities:
            return False
        overlap = len(new_entities & recent_entities) / max(
            1, len(new_entities | recent_entities))
        return overlap < threshold

    def get_recent_entities(self, episode_id: int | None = None,
                           limit: int = 20) -> set[str]:
        """Get entities from recent facts in the episode."""
        ep_id = episode_id or self._active_episode
        if ep_id is None:
            return set()
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT e.name FROM entities e "
                    "JOIN fact_entities fe ON e.entity_id = fe.entity_id "
                    "JOIN episode_facts ef ON fe.fact_id = ef.fact_id "
                    "WHERE ef.episode_id = ? "
                    "ORDER BY ef.ordinal DESC LIMIT ?",
                    (ep_id, limit),
                ).fetchall()
                return {r["name"] for r in rows}
        except Exception as e:
            logger.debug("get_recent_entities failed: %s", e)
            return set()

    def link_episodes(self, source_id: int, target_id: int,
                      relation: str = "caused_by") -> None:
        """Store a causal/temporal link between episodes."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO cross_domain_links "
                    "(entity, domain_a, domain_b, fact_refs_a, fact_refs_b) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (f"episode_link:{relation}",
                     f"episode:{source_id}", f"episode:{target_id}",
                     str(source_id), str(target_id)),
                )
                self._conn.commit()
        except Exception as e:
            logger.debug("link_episodes failed: %s", e)
