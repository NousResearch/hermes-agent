"""Dream Engine — structured selective replay for memory consolidation.

During idle periods, replays episodes in structured patterns to strengthen
schemas, discover cross-episode bridges, and surface novel hypotheses.

This is NOT random recombination. Hippocampal replay is structured:
temporally ordered, salience-selected, entity-linked.
(Wilson & McNaughton 1994 Science; Foster & Wilson 2006 Nature)

Three modes, ordered by safety:
  Mode 1: Sequential Replay — strengthen existing schemas from top episodes
  Mode 2: Cross-Episode Pattern Discovery — find shared entities across episodes
  Mode 3: Schema-Driven Hypothesis — generate testable predictions from schemas

Scientific basis:
- Wagner et al. (2004) Nature: sleep consolidates gist (59.5% vs 22.7%)
- McClelland et al. (1995): hippocampal replay trains neocortical schemas
- Buzsaki (2015): replay is selective (novelty, reward, emotional valence)
- Goldstein et al. (2023) Nature Comms: sleep-like replay improves AI consolidation
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DreamResult:
    """Result of a dream cycle."""
    mode: str = "auto"
    episodes_used: int = 0
    facts_replayed: int = 0
    schemas_boosted: int = 0
    schemas_created: int = 0
    hypotheses: int = 0
    duration_ms: int = 0


# Dream-specific schema
_DREAM_SCHEMA = """\
CREATE TABLE IF NOT EXISTS dream_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mode            TEXT NOT NULL,
    episodes_used   INTEGER DEFAULT 0,
    facts_replayed  INTEGER DEFAULT 0,
    schemas_boosted INTEGER DEFAULT 0,
    schemas_created INTEGER DEFAULT 0,
    hypotheses      INTEGER DEFAULT 0,
    duration_ms     INTEGER DEFAULT 0,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dream_hypotheses (
    hypothesis_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    source_schema_id INTEGER,
    source_fact_ids  TEXT DEFAULT '',
    content          TEXT NOT NULL,
    confidence       REAL DEFAULT 0.2,
    verified         INTEGER DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dream_hyp_verified ON dream_hypotheses(verified);
"""


class DreamEngine:
    """Structured selective replay for memory consolidation.

    Three modes, ordered by safety:

    Mode 1 (Sequential Replay): Select top-K salient episodes, replay
    facts in chronological order, boost matching schemas.

    Mode 2 (Cross-Episode Patterns): Find facts from different episodes
    that share entities, create new schemas from their combination.

    Mode 3 (Hypothesis Generation): Use high-confidence schemas to
    generate predictions about unconsolidated facts.

    All modes are rule-based (no LLM in Mode 1-2). Mode 3 has optional
    LLM fallback but defaults to rule-based.
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock,
                 cooldown_hours: float = 1.0,
                 mode1_top_k: int = 10,
                 mode2_top_k: int = 5,
                 mode3_idle_hours: float = 24.0,
                 mode3_min_schema_conf: float = 0.7) -> None:
        self._conn = conn
        self._lock = lock
        self._cooldown_hours = cooldown_hours
        self._mode1_top_k = mode1_top_k
        self._mode2_top_k = mode2_top_k
        self._mode3_idle_hours = mode3_idle_hours
        self._mode3_min_conf = mode3_min_schema_conf
        self._last_dream_time: float = 0.0

    def init_tables(self) -> None:
        """Create dream tables if they don't exist."""
        try:
            with self._lock:
                self._conn.executescript(_DREAM_SCHEMA)
                self._conn.commit()
        except Exception as e:
            logger.debug("DreamEngine init_tables failed: %s", e)

    def should_dream(self) -> bool:
        """Check if a dream cycle should run.

        Conditions: cooldown elapsed AND there are episodes + schemas.
        """
        now = time.time()
        hours_since_last = (now - self._last_dream_time) / 3600.0
        if hours_since_last < self._cooldown_hours:
            return False

        try:
            with self._lock:
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM episodes"
                ).fetchone()[0]
                if count == 0:
                    return False
                schema_count = self._conn.execute(
                    "SELECT COUNT(*) FROM schemas"
                ).fetchone()[0]
                return schema_count > 0
        except Exception as e:
            logger.debug("should_dream check failed: %s", e)
            return False

    def dream_cycle(self, session_id: str = "",
                    mode: str = "auto") -> DreamResult:
        """Run one dream cycle. Returns summary.

        In 'auto' mode: runs Mode 1, then Mode 2 if low yield,
        then Mode 3 if gated conditions met.
        """
        start = time.time()
        result = DreamResult(mode=mode)

        try:
            if mode == "auto":
                r1 = self._mode1_sequential_replay()
                result.episodes_used += r1.episodes_used
                result.facts_replayed += r1.facts_replayed
                result.schemas_boosted += r1.schemas_boosted

                if r1.schemas_boosted < 2:
                    r2 = self._mode2_cross_episode_patterns()
                    result.episodes_used += r2.episodes_used
                    result.schemas_created += r2.schemas_created

                if self._should_run_mode3():
                    r3 = self._mode3_hypothesis_generation()
                    result.hypotheses += r3.hypotheses
            elif mode == "replay":
                r1 = self._mode1_sequential_replay()
                result.episodes_used = r1.episodes_used
                result.facts_replayed = r1.facts_replayed
                result.schemas_boosted = r1.schemas_boosted
            elif mode == "patterns":
                r2 = self._mode2_cross_episode_patterns()
                result.episodes_used = r2.episodes_used
                result.schemas_created = r2.schemas_created
            elif mode == "hypotheses":
                r3 = self._mode3_hypothesis_generation()
                result.hypotheses = r3.hypotheses

            result.duration_ms = int((time.time() - start) * 1000)
            self._last_dream_time = time.time()
            self._log_run(result, session_id)

        except Exception as e:
            logger.debug("dream_cycle failed: %s", e)

        return result

    def _mode1_sequential_replay(self) -> DreamResult:
        """Mode 1: Select top-K salient episodes, replay facts,
        boost matching schemas."""
        result = DreamResult(mode="replay")
        try:
            with self._lock:
                episodes = self._conn.execute(
                    "SELECT episode_id, salience_peak, fact_count "
                    "FROM episodes WHERE fact_count > 0 "
                    "ORDER BY salience_peak DESC LIMIT ?",
                    (self._mode1_top_k,),
                ).fetchall()

            for ep in episodes:
                ep_id = ep["episode_id"]
                result.episodes_used += 1

                with self._lock:
                    facts = self._conn.execute(
                        "SELECT ef.fact_id, f.content, f.trust_score, f.strength "
                        "FROM episode_facts ef "
                        "JOIN facts f ON ef.fact_id = f.fact_id "
                        "WHERE ef.episode_id = ? "
                        "ORDER BY ef.ordinal",
                        (ep_id,),
                    ).fetchall()

                    for fact in facts:
                        result.facts_replayed += 1
                        content = fact["content"]

                        schemas = self._conn.execute(
                            "SELECT schema_id, content, confidence "
                            "FROM schemas WHERE confidence > 0.3"
                        ).fetchall()

                        for schema in schemas:
                            overlap = self._entity_overlap(
                                content, schema["content"])
                            if overlap > 0.3:
                                boost = 0.02 * fact["trust_score"]
                                self._conn.execute(
                                    "UPDATE schemas SET "
                                    "confidence = MIN(1.0, confidence + ?), "
                                    "source_count = source_count + 1, "
                                    "updated_at = CURRENT_TIMESTAMP "
                                    "WHERE schema_id = ?",
                                    (boost, schema["schema_id"]),
                                )
                                result.schemas_boosted += 1

                    self._conn.commit()

        except Exception as e:
            logger.debug("Mode 1 replay failed: %s", e)
        return result

    def _mode2_cross_episode_patterns(self) -> DreamResult:
        """Mode 2: Find facts from different episodes sharing entities,
        create new schemas from their combination."""
        result = DreamResult(mode="patterns")
        try:
            with self._lock:
                episodes = self._conn.execute(
                    "SELECT episode_id FROM episodes WHERE fact_count > 1 "
                    "ORDER BY salience_peak DESC LIMIT ?",
                    (self._mode2_top_k,),
                ).fetchall()

                if len(episodes) < 2:
                    return result

                ep_ids = [e["episode_id"] for e in episodes]
                placeholders = ",".join("?" * len(ep_ids))

                shared_entities = self._conn.execute(
                    f"SELECT e.name, COUNT(DISTINCT ef.episode_id) as ep_count "
                    f"FROM entities e "
                    f"JOIN fact_entities fe ON e.entity_id = fe.entity_id "
                    f"JOIN episode_facts ef ON fe.fact_id = ef.fact_id "
                    f"WHERE ef.episode_id IN ({placeholders}) "
                    f"GROUP BY e.name HAVING ep_count >= 2",
                    ep_ids,
                ).fetchall()

                for entity_row in shared_entities:
                    entity = entity_row["name"]

                    facts = self._conn.execute(
                        "SELECT DISTINCT f.fact_id, f.content, f.trust_score, "
                        "ef.episode_id "
                        "FROM facts f "
                        "JOIN fact_entities fe ON f.fact_id = fe.fact_id "
                        "JOIN entities e ON fe.entity_id = e.entity_id "
                        "JOIN episode_facts ef ON f.fact_id = ef.fact_id "
                        "WHERE e.name = ? "
                        "AND ef.episode_id IN ({}) "
                        "AND f.strength > 0.5 "
                        "ORDER BY f.trust_score DESC LIMIT 5".format(placeholders),
                        [entity] + ep_ids,
                    ).fetchall()

                    if len(facts) >= 2:
                        contents = [f["content"][:100] for f in facts[:3]]
                        combined = f"[Cross-episode: {entity}] " + " | ".join(contents)

                        existing = self._conn.execute(
                            "SELECT schema_id FROM schemas "
                            "WHERE content LIKE ? LIMIT 1",
                            (f"%{entity}%",),
                        ).fetchone()

                        if not existing:
                            self._conn.execute(
                                "INSERT INTO schemas "
                                "(content, domain, confidence, source_count) "
                                "VALUES (?, ?, ?, ?)",
                                (combined, "cross_episode", 0.3, len(facts)),
                            )
                            result.schemas_created += 1

                result.episodes_used = len(ep_ids)
                self._conn.commit()

        except Exception as e:
            logger.debug("Mode 2 pattern discovery failed: %s", e)
        return result

    def _mode3_hypothesis_generation(self) -> DreamResult:
        """Mode 3: Use high-confidence schemas to generate predictions
        about unconsolidated facts."""
        result = DreamResult(mode="hypotheses")
        try:
            with self._lock:
                schemas = self._conn.execute(
                    "SELECT schema_id, content, confidence, domain "
                    "FROM schemas WHERE confidence >= ? "
                    "ORDER BY confidence DESC LIMIT 10",
                    (self._mode3_min_conf,),
                ).fetchall()

                for schema in schemas:
                    entities = set(re.findall(
                        r'\b[A-Z][a-z]{2,}\b', schema["content"]))

                    if not entities:
                        continue

                    for entity in list(entities)[:3]:
                        facts = self._conn.execute(
                            "SELECT f.fact_id, f.content "
                            "FROM facts f "
                            "JOIN fact_entities fe ON f.fact_id = fe.fact_id "
                            "JOIN entities e ON fe.entity_id = e.entity_id "
                            "WHERE e.name = ? "
                            "AND f.strength > 0.3 "
                            "AND f.fact_id NOT IN "
                            "  (SELECT fact_id FROM episode_facts) "
                            "LIMIT 3",
                            (entity,),
                        ).fetchall()

                        for fact in facts:
                            existing = self._conn.execute(
                                "SELECT hypothesis_id FROM dream_hypotheses "
                                "WHERE content LIKE ? LIMIT 1",
                                (f"%{entity}%",),
                            ).fetchone()

                            if not existing:
                                hypothesis_content = (
                                    f"[Hypothesis from schema '{schema['content'][:50]}'] "
                                    f"Entity '{entity}' may relate to: "
                                    f"{fact['content'][:80]}"
                                )
                                self._conn.execute(
                                    "INSERT INTO dream_hypotheses "
                                    "(source_schema_id, source_fact_ids, content, confidence) "
                                    "VALUES (?, ?, ?, ?)",
                                    (schema["schema_id"],
                                     str(fact["fact_id"]),
                                     hypothesis_content,
                                     0.2),
                                )
                                result.hypotheses += 1

                self._conn.commit()

        except Exception as e:
            logger.debug("Mode 3 hypothesis generation failed: %s", e)
        return result

    def _should_run_mode3(self) -> bool:
        """Check if Mode 3 should run (idle for 24+ hours)."""
        now = time.time()
        hours_since_last = (now - self._last_dream_time) / 3600.0
        return hours_since_last >= self._mode3_idle_hours

    def _entity_overlap(self, text_a: str, text_b: str) -> float:
        """Compute entity overlap between two texts (Jaccard on capitalized words)."""
        entities_a = set(re.findall(r'\b[A-Z][a-z]{2,}\b', text_a))
        entities_b = set(re.findall(r'\b[A-Z][a-z]{2,}\b', text_b))
        if not entities_a or not entities_b:
            return 0.0
        return len(entities_a & entities_b) / len(entities_a | entities_b)

    def _log_run(self, result: DreamResult, session_id: str) -> None:
        """Log the dream run to the database."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO dream_runs "
                    "(mode, episodes_used, facts_replayed, schemas_boosted, "
                    "schemas_created, hypotheses, duration_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (result.mode, result.episodes_used, result.facts_replayed,
                     result.schemas_boosted, result.schemas_created,
                     result.hypotheses, result.duration_ms),
                )
                self._conn.commit()
        except Exception as e:
            logger.debug("Dream run logging failed: %s", e)

    def get_hypotheses(self, verified: bool = False,
                       limit: int = 10) -> list[dict]:
        """Return hypotheses for review."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM dream_hypotheses "
                    "WHERE verified = ? ORDER BY created_at DESC LIMIT ?",
                    (1 if verified else 0, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("get_hypotheses failed: %s", e)
            return []

    def verify_hypothesis(self, hypothesis_id: int) -> None:
        """Mark a hypothesis as verified (human confirmed)."""
        try:
            with self._lock:
                hyp = self._conn.execute(
                    "SELECT content, confidence FROM dream_hypotheses "
                    "WHERE hypothesis_id = ?",
                    (hypothesis_id,),
                ).fetchone()
                if hyp:
                    self._conn.execute(
                        "INSERT INTO schemas (content, domain, confidence) "
                        "VALUES (?, ?, ?)",
                        (hyp["content"], "verified_hypothesis",
                         min(0.7, hyp["confidence"] + 0.3)),
                    )
                    self._conn.execute(
                        "UPDATE dream_hypotheses SET verified = 1 "
                        "WHERE hypothesis_id = ?",
                        (hypothesis_id,),
                    )
                    self._conn.commit()
        except Exception as e:
            logger.debug("verify_hypothesis failed: %s", e)
