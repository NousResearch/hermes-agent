"""MemoryPipeline -- organic memory infrastructure inside MemoryManager.

NOT a MemoryProvider.  No name, no tools, no system_prompt_block.
Pure interceptor: executes organic logic before/after MemoryManager lifecycle methods.

All methods are best-effort: exceptions are caught and logged at debug level,
never blocking upstream providers.

Design philosophy (浑然一体):
    Memory's organic properties (salience gating, silent engrams, consolidation,
    reconsolidation, predictive feedback, spreading activation) are infrastructure
    of the entire cognitive system -- not features of a specific storage backend.
    Just as synaptic plasticity is a universal property of neural circuits, not
    a "plugin" for the hippocampus, the MemoryPipeline lives inside MemoryManager
    and operates on ALL memory pathways regardless of which provider is active.

Architecture:
    MemoryManager
        └── MemoryPipeline (interceptor layer, NOT a provider)
            ├── SalienceScorer      (Layer 1: sensory gate)
            ├── SilentEngramEngine  (Layer 2: availability continuum)
            ├── ConsolidationEngine (Layer 3: sleep-like consolidation)
            ├── ReconsolidationEngine (Layer 4: prediction-error updates)
            ├── FeedbackCoordinator (Layer 5: predictive processing + learning)
            └── ActivationGraph     (Layer 6: spreading activation)
        └── providers[] (builtin + one external)

5 Architectural Invariants Preserved:
    1. MemoryProvider ABC contract unchanged
    2. Single external provider limit unchanged
    3. Tool registry unchanged (pipeline exposes no tools)
    4. ContextEngine orthogonality preserved
    5. run_agent.py integration points unchanged
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# Layer 1: SalienceScorer (sensory gate)
# ===========================================================================

_EMOTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"[!！]{2,}"), 0.6),
    (re.compile(r"\b(urgent|critical|emergency|broken|crash|bug|fail)\b", re.I), 0.5),
    (re.compile(r"\b(down|outage|corrupt|overload|timeout|deadlock)\b", re.I), 0.45),
    (re.compile(r"\b(love|hate|amazing|terrible|awesome|awful)\b", re.I), 0.3),
    (re.compile(r"\b(worried|excited|frustrated|angry|happy|sad)\b", re.I), 0.35),
    (re.compile(r"\b(important|crucial|vital|essential|key)\b", re.I), 0.4),
]

_IMPORTANCE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(decided|decision|agreed|confirmed|final)\b", re.I), 0.7),
    (re.compile(r"\b(requirement|spec|specification|constraint)\b", re.I), 0.6),
    (re.compile(r"\b(deploy|release|production|launch)\b", re.I), 0.6),
    (re.compile(r"\b(architecture|design|refactor|migrat)\b", re.I), 0.5),
    (re.compile(r"\b(remember|note|important|don't forget)\b", re.I), 0.8),
    (re.compile(r"\b(prefer|always|never|usually)\b", re.I), 0.5),
    (re.compile(r"\b(bug|issue|error|problem)\b", re.I), 0.4),
]

_TRIVIAL_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"^(hi|hello|hey|thanks|ok|yes|no|sure)\s*[.!?]?\s*$", re.I), 0.9),
    (re.compile(r"^(good morning|good night|bye|see you)", re.I), 0.8),
    (re.compile(r"^(what time|what date|weather)", re.I), 0.5),
]


@dataclass
class SalienceResult:
    """Multi-dimensional salience score for a message."""
    overall: float = 0.0
    emotion: float = 0.0
    novelty: float = 0.5
    importance: float = 0.0
    repetition_penalty: float = 1.0
    is_trivial: bool = False


@dataclass
class _RepetitionDetector:
    """Detects topic repetition using content hashing (F3 power-law penalty)."""
    window_size: int = 50
    _recent: deque = field(default_factory=lambda: deque(maxlen=50))
    _topic_counts: dict[str, int] = field(default_factory=dict)

    def _fuzzy_bucket(self, text: str) -> str:
        words = [w for w in re.sub(r"[^\w\s]", "", text.lower()).split() if len(w) > 2]
        return " ".join(words[:5])

    def observe(self, text: str) -> float:
        bucket = self._fuzzy_bucket(text)
        if not bucket:
            return 1.0
        self._topic_counts[bucket] = self._topic_counts.get(bucket, 0) + 1
        self._recent.append(bucket)
        if len(self._recent) == self._recent.maxlen or len(self._topic_counts) > self._recent.maxlen * 2:
            window_counts: dict[str, int] = {}
            for b in self._recent:
                window_counts[b] = window_counts.get(b, 0) + 1
            for topic in list(self._topic_counts):
                if topic not in window_counts:
                    del self._topic_counts[topic]
                else:
                    self._topic_counts[topic] = window_counts[topic]
        n = self._topic_counts.get(bucket, 1)
        return max(0.1, 1.0 / math.sqrt(n))

    def reset(self) -> None:
        self._recent.clear()
        self._topic_counts.clear()


class SalienceScorer:
    """Multi-dimensional salience scorer — the sensory gate.

    Pure rule-based — no LLM calls, O(message_length) time.
    Scientific basis: F4 (CREB/excitability allocation).
    Thread-safe: all mutable state protected by _lock.
    """

    def __init__(self, novelty_window: int = 50) -> None:
        self._rep = _RepetitionDetector(window_size=novelty_window)
        self._lock = threading.Lock()

    def score(self, message: str) -> SalienceResult:
        if not message or not message.strip():
            return SalienceResult(overall=0.0, is_trivial=True)
        text = message.strip()
        with self._lock:
            trivial_penalty = 1.0
            for pattern, weight in _TRIVIAL_PATTERNS:
                if pattern.search(text):
                    trivial_penalty = min(trivial_penalty, 1.0 - weight)
            is_trivial = trivial_penalty < 0.3
            emotion = 0.0
            for pattern, weight in _EMOTION_PATTERNS:
                if pattern.search(text):
                    emotion = max(emotion, weight)
            if len(text) < 20:
                emotion *= 0.5
            importance = 0.0
            for pattern, weight in _IMPORTANCE_PATTERNS:
                if pattern.search(text):
                    importance = max(importance, weight)
            if len(text) > 200:
                importance = min(1.0, importance + 0.1)
            freshness = self._rep.observe(text)
            novelty = freshness
            rep_factor = freshness
            raw = (0.25 * emotion + 0.30 * novelty + 0.30 * importance
                   + 0.15 * min(1.0, len(text) / 200))
            adjusted = raw * rep_factor * (1.0 - (1.0 - trivial_penalty) * 0.8)
            overall = max(0.0, min(1.0, adjusted))
            return SalienceResult(
                overall=overall, emotion=emotion, novelty=novelty,
                importance=importance, repetition_penalty=rep_factor,
                is_trivial=is_trivial,
            )

    def reset(self) -> None:
        self._rep = _RepetitionDetector(window_size=self._rep.window_size)


# ===========================================================================
# Layer 2: SilentEngramEngine (availability continuum)
# ===========================================================================

class SilentEngramEngine:
    """Manages memory strength decay and recovery.

    Memories decay via power-law but NEVER reach zero.  Forgotten facts
    become "silent engrams" that can be recovered via context similarity.
    Scientific basis: F5 (Ryan et al. 2015 Science — forgetting ≠ erasure).

    Thresholds:
        active:      strength > 0.5
        semi_active:  0.2 < strength <= 0.5
        silent:       0.05 < strength <= 0.2
        buried:       strength <= 0.05
    """

    ACTIVE = 0.5
    SEMI_ACTIVE = 0.2
    SILENT = 0.05

    def __init__(self, half_life_hours: float = 720.0) -> None:
        self._half_life = half_life_hours

    def apply_decay(self, state: 'PipelineState', hours_elapsed: float = 1.0) -> int:
        """Apply power-law decay to all engram strengths. Returns affected rows."""
        if not state:
            return 0
        try:
            decay_factor = 0.5 ** (hours_elapsed / self._half_life)
            with state._lock:
                cursor = state._conn.execute(
                    "UPDATE engram_strengths SET "
                    "strength = MAX(0.001, strength * ?), "
                    "last_accessed = CURRENT_TIMESTAMP "
                    "WHERE strength > 0.001",
                    (decay_factor,),
                )
                state._conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.debug("Engram decay failed: %s", e)
            return 0

    def strengthen(self, state: 'PipelineState', memory_ref: str,
                   delta: float = 0.03) -> float:
        """Strengthen an engram on retrieval (spacing effect). Returns new strength."""
        if not state:
            return 0.0
        try:
            with state._lock:
                row = state._conn.execute(
                    "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
                    (memory_ref,),
                ).fetchone()
                if row:
                    new_str = min(1.0, row["strength"] + delta)
                    state._conn.execute(
                        "UPDATE engram_strengths SET strength = ?, "
                        "last_accessed = CURRENT_TIMESTAMP, "
                        "access_count = access_count + 1 "
                        "WHERE memory_ref = ?",
                        (new_str, memory_ref),
                    )
                else:
                    new_str = min(1.0, 1.0 + delta)
                    state._conn.execute(
                        "INSERT INTO engram_strengths "
                        "(memory_ref, provider, strength) VALUES (?, 'unknown', ?)",
                        (memory_ref, new_str),
                    )
                state._conn.commit()
                return new_str
        except Exception as e:
            logger.debug("Engram strengthen failed: %s", e)
            return 0.0

    def classify(self, strength: float) -> str:
        """Classify strength into accessibility level."""
        if strength > self.ACTIVE:
            return "active"
        elif strength > self.SEMI_ACTIVE:
            return "semi_active"
        elif strength > self.SILENT:
            return "silent"
        return "buried"


# ===========================================================================
# Layer 3: ConsolidationEngine (sleep-like consolidation)
# ===========================================================================

class ConsolidationEngine:
    """Consolidates episodic memories into semantic schemas.

    Three-phase process mimicking sleep consolidation:
    1. Select: pick salient unconsolidated facts
    2. Transfer: group by entity/category, create schema candidates
    3. Integrate: merge with existing schemas or create new ones
    Scientific basis: F6 (Diekelmann & Born 2019 Nature Reviews Neuroscience).
    """

    def __init__(self, min_facts: int = 5) -> None:
        self._min_facts = min_facts

    def consolidate(self, state: 'PipelineState',
                    facts: list[dict] | None = None) -> dict:
        """Run consolidation. Returns summary dict.

        In Phase 1-2, this operates on pipeline_state.db schemas.
        In Phase 3+, it will pull facts from providers.
        """
        if not state:
            return {"schemas_created": 0, "schemas_updated": 0}
        created, updated = 0, 0
        try:
            with state._lock:
                # Consolidation runs whenever we have enough new facts
                if facts and len(facts) >= self._min_facts:
                    # Get existing schema contents for dedup
                    existing = state._conn.execute(
                        "SELECT content FROM schemas ORDER BY updated_at DESC LIMIT 20"
                    ).fetchall()
                    existing_contents = {r["content"][:50] for r in existing}

                    for fact in facts[:10]:
                        content = fact.get("content", "")
                        domain = fact.get("domain", "general")
                        if not content or len(content) < 10:
                            continue
                        # Simple dedup: skip if similar content already exists
                        if content[:50] in existing_contents:
                            updated += 1
                            continue
                        state._conn.execute(
                            "INSERT INTO schemas (content, domain, confidence) "
                            "VALUES (?, ?, ?)",
                            (content, domain, 0.5),
                        )
                        created += 1

                # Log the run (single commit for atomicity)
                state._conn.execute(
                    "INSERT INTO consolidation_runs "
                    "(session_id, memories_processed, schemas_created, schemas_updated) "
                    "VALUES (?, ?, ?, ?)",
                    (self._session_id or "", len(facts or []), created, updated),
                )
                state._conn.commit()
        except Exception as e:
            logger.debug("Consolidation failed: %s", e)
        return {"schemas_created": created, "schemas_updated": updated}

    def extract_insights(self, messages: list) -> str:
        """Extract key facts from messages about to be discarded by compression."""
        insights = []
        for msg in messages[-5:]:  # last 5 messages
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if len(content) > 50:
                # Extract first sentence as insight
                first_sentence = content.split(".")[0][:200]
                if first_sentence.strip():
                    insights.append(f"- {first_sentence.strip()}")
        return "\n".join(insights) if insights else ""


# ===========================================================================
# Layer 4: ReconsolidationEngine (prediction-error updates)
# ===========================================================================

class ReconsolidationEngine:
    """Prediction-error driven memory updates.

    When new information contradicts existing memories, the system enters
    a "reconsolidation" mode: evaluating the conflict and updating.
    Scientific basis: F8 (Sinclair & Barense 2019 Trends in Neurosciences).
    """

    def __init__(self, error_threshold: float = 0.3) -> None:
        self._threshold = error_threshold

    def check_retrieval(self, state: 'PipelineState',
                        query: str, result: str,
                        engrams: 'SilentEngramEngine | None' = None) -> None:
        """Record retrieval event for potential reconsolidation."""
        if not state:
            return
        try:
            ref = sha256(query.encode()).hexdigest()[:16]
            if engrams:
                engrams.strengthen(state, ref)
            else:
                SilentEngramEngine().strengthen(state, ref)
        except Exception as e:
            logger.debug("Reconsolidation check failed: %s", e)

    def detect_conflict(self, new_content: str,
                        existing_contents: list[str]) -> float:
        """Detect prediction error between new and existing content.

        Returns error score [0, 1]. High error = high conflict.
        Simple token-overlap heuristic.
        """
        if not existing_contents:
            return 0.0
        new_tokens = set(new_content.lower().split())
        max_overlap = 0.0
        for existing in existing_contents:
            existing_tokens = set(existing.lower().split())
            if not new_tokens or not existing_tokens:
                continue
            overlap = len(new_tokens & existing_tokens) / max(
                1, len(new_tokens | existing_tokens))
            max_overlap = max(max_overlap, overlap)
        # High overlap = low conflict, low overlap = high conflict
        return 1.0 - max_overlap


# ===========================================================================
# Layer 5: FeedbackCoordinator (predictive processing + learning)
# ===========================================================================

class FeedbackCoordinator:
    """Three interconnected feedback loops.

    1. SalienceLearner: learns which signals predict useful memories
    2. PredictiveModel: generates expectations from schemas
    3. CrossDomainBridge: discovers unexpected connections
    Scientific basis: Predictive coding (Friston 2010).
    Thread-safe: _pending_predictions protected by _lock.
    """

    def __init__(self) -> None:
        self._pending_predictions: list[str] = []
        self._lock = threading.Lock()

    def predict(self, state: 'PipelineState', context: str) -> list[str]:
        """Generate predictions from existing schemas."""
        if not state:
            return []
        try:
            with state._lock:
                rows = state._conn.execute(
                    "SELECT content, confidence FROM schemas "
                    "WHERE confidence > 0.3 ORDER BY confidence DESC LIMIT 3"
                ).fetchall()
            predictions = []
            for row in rows:
                predictions.append(
                    f"Expected pattern (conf={row['confidence']:.2f}): "
                    f"{row['content'][:100]}"
                )
            with self._lock:
                self._pending_predictions = predictions
            return predictions
        except Exception as e:
            logger.debug("Prediction failed: %s", e)
            return []

    def observe_outcome(self, state: 'PipelineState',
                        actual: str) -> float:
        """Compare predictions against actual outcome. Returns error score."""
        with self._lock:
            pending = list(self._pending_predictions)
        if not pending or not state:
            return 0.0
        try:
            actual_tokens = set(actual.lower().split())
            max_error = 0.0
            for pred in pending:
                pred_tokens = set(pred.lower().split())
                if not pred_tokens or not actual_tokens:
                    continue
                overlap = len(pred_tokens & actual_tokens) / max(
                    1, len(pred_tokens | actual_tokens))
                error = 1.0 - overlap
                max_error = max(max_error, error)

            # Update schema confidence based on prediction error
            if max_error > 0.5:
                # High error: schema was wrong, decrease confidence
                with state._lock:
                    state._conn.execute(
                        "UPDATE schemas SET confidence = MAX(0.1, confidence - 0.05) "
                        "WHERE confidence > 0.3"
                    )
                    state._conn.commit()
            elif max_error < 0.2:
                # Low error: schema was right, increase confidence
                with state._lock:
                    state._conn.execute(
                        "UPDATE schemas SET confidence = MIN(1.0, confidence + 0.03) "
                        "WHERE confidence > 0.3"
                    )
                    state._conn.commit()

            with self._lock:
                self._pending_predictions = []
            return max_error
        except Exception as e:
            logger.debug("Observe outcome failed: %s", e)
            return 0.0

    def discover_bridges(self, state: 'PipelineState') -> int:
        """Discover cross-domain connections between entities."""
        if not state:
            return 0
        try:
            with state._lock:
                # Find entities that appear in multiple domains
                rows = state._conn.execute(
                    "SELECT entity, COUNT(DISTINCT domain_a) as domain_count "
                    "FROM cross_domain_links GROUP BY entity "
                    "HAVING domain_count >= 2"
                ).fetchall()
                return len(rows)
        except Exception as e:
            logger.debug("Bridge discovery failed: %s", e)
            return 0


# ===========================================================================
# Layer 6: ActivationGraph (spreading activation)
# ===========================================================================

class ActivationGraph:
    """Hebbian co-activation graph for spreading activation.

    When entities are co-retrieved, their connection strengthens.
    Activation spreads through the graph to pre-activate related memories.
    Scientific basis: Collins & Loftus (1975) spreading activation.
    """

    def __init__(self, edge_decay_hours: float = 168.0) -> None:
        self._decay_hours = edge_decay_hours

    def record_co_activation(self, state: 'PipelineState',
                             entities: list[str], delta: float = 0.1) -> None:
        """Strengthen edges between co-activated entities (Hebbian learning)."""
        if not state or len(entities) < 2:
            return
        try:
            with state._lock:
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        a, b = sorted([entities[i], entities[j]])
                        state._conn.execute(
                            "INSERT INTO activation_edges "
                            "(source_entity, target_entity, strength, co_activation_count) "
                            "VALUES (?, ?, ?, 1) "
                            "ON CONFLICT(source_entity, target_entity) DO UPDATE SET "
                            "strength = MIN(1.0, strength + ?), "
                            "co_activation_count = co_activation_count + 1, "
                            "last_activated = CURRENT_TIMESTAMP",
                            (a, b, delta, delta),
                        )
                state._conn.commit()
        except Exception as e:
            logger.debug("Co-activation recording failed: %s", e)

    def get_neighbors(self, state: 'PipelineState',
                      entity: str, min_strength: float = 0.3,
                      limit: int = 5) -> list[dict]:
        """Get strongly connected neighbors of an entity."""
        if not state:
            return []
        try:
            with state._lock:
                rows = state._conn.execute(
                    "SELECT target_entity AS neighbor, strength FROM activation_edges "
                    "WHERE source_entity = ? AND strength >= ? "
                    "UNION ALL "
                    "SELECT source_entity AS neighbor, strength FROM activation_edges "
                    "WHERE target_entity = ? AND strength >= ? "
                    "ORDER BY strength DESC LIMIT ?",
                    (entity, min_strength, entity, min_strength, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("Get neighbors failed: %s", e)
            return []

    def expand_query(self, state: 'PipelineState',
                     query: str, limit: int = 3) -> list[str]:
        """Expand a query using spreading activation.

        Extracts entities from query, finds their neighbors, returns
        additional context strings.
        """
        if not state:
            return []
        try:
            # Simple entity extraction: capitalized words
            entities = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
            expansions = []
            for entity in entities[:3]:
                neighbors = self.get_neighbors(state, entity, limit=limit)
                for n in neighbors:
                    expansions.append(
                        f"[co-activated: {entity} → {n['neighbor']} "
                        f"(strength={n['strength']:.2f})]"
                    )
            return expansions
        except Exception as e:
            logger.debug("Query expansion failed: %s", e)
            return []

    def decay_edges(self, state: 'PipelineState',
                    hours_elapsed: float = 1.0) -> int:
        """Decay all edge strengths. Returns affected rows."""
        if not state:
            return 0
        try:
            decay_factor = 0.5 ** (hours_elapsed / self._decay_hours)
            with state._lock:
                cursor = state._conn.execute(
                    "UPDATE activation_edges SET "
                    "strength = MAX(0.01, strength * ?) "
                    "WHERE strength > 0.01",
                    (decay_factor,),
                )
                state._conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.debug("Edge decay failed: %s", e)
            return 0


# ===========================================================================
# Pipeline Schema (database provisioning)
# ===========================================================================

_PIPELINE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS salience_weights (
    signal_type TEXT PRIMARY KEY, weight REAL NOT NULL,
    sample_count INTEGER DEFAULT 0, success_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS salience_encoding_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL,
    fact_ref TEXT, emotion_score REAL, novelty_score REAL,
    importance_score REAL, overall_score REAL, was_helpful INTEGER DEFAULT -1,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS engram_strengths (
    memory_ref TEXT PRIMARY KEY, provider TEXT NOT NULL,
    strength REAL DEFAULT 1.0, last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decay_half_life_hours REAL DEFAULT 720.0, access_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS schemas (
    schema_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL,
    domain TEXT DEFAULT 'general', confidence REAL DEFAULT 0.5,
    source_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, hrr_vector BLOB
);
CREATE TABLE IF NOT EXISTS schema_sources (
    schema_id INTEGER REFERENCES schemas(schema_id),
    memory_ref TEXT NOT NULL, provider TEXT NOT NULL,
    contribution REAL DEFAULT 1.0, PRIMARY KEY (schema_id, memory_ref)
);
CREATE TABLE IF NOT EXISTS reconsolidation_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_ref TEXT,
    old_content TEXT, new_content TEXT, prediction_error REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS consolidation_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
    memories_processed INTEGER DEFAULT 0, schemas_created INTEGER DEFAULT 0,
    schemas_updated INTEGER DEFAULT 0, conflicts_found INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, schema_id INTEGER,
    prediction TEXT NOT NULL, context TEXT DEFAULT '', outcome TEXT DEFAULT '',
    error_score REAL DEFAULT 0.0, resolved INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, resolved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS salience_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_ref TEXT,
    signal_type TEXT, signal_value REAL, was_helpful INTEGER DEFAULT 0,
    was_retrieved INTEGER DEFAULT 0, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS activation_edges (
    source_entity TEXT NOT NULL, target_entity TEXT NOT NULL,
    strength REAL DEFAULT 0.1, co_activation_count INTEGER DEFAULT 1,
    last_activated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_entity, target_entity)
);
CREATE TABLE IF NOT EXISTS cross_domain_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT NOT NULL,
    domain_a TEXT NOT NULL, domain_b TEXT NOT NULL,
    fact_refs_a TEXT DEFAULT '', fact_refs_b TEXT DEFAULT '',
    strength REAL DEFAULT 0.5, discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_engram_strength ON engram_strengths(strength DESC);
CREATE INDEX IF NOT EXISTS idx_engram_provider ON engram_strengths(provider);
CREATE INDEX IF NOT EXISTS idx_schemas_domain ON schemas(domain);
CREATE INDEX IF NOT EXISTS idx_schemas_confidence ON schemas(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_activation_source ON activation_edges(source_entity);
CREATE INDEX IF NOT EXISTS idx_activation_target ON activation_edges(target_entity);
CREATE INDEX IF NOT EXISTS idx_cross_links_entity ON cross_domain_links(entity);
CREATE INDEX IF NOT EXISTS idx_salience_feedback_ref ON salience_feedback(memory_ref, was_retrieved);
"""


# ===========================================================================
# PipelineState -- persistent storage
# ===========================================================================

class PipelineState:
    """Persistent state for the memory pipeline (pipeline_state.db).

    Design: single connection + threading.RLock, WAL mode, independent
    from any provider's database.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "pipeline_state.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=10.0,
        )
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="pipeline_state.db")
        with self._lock:
            self._conn.executescript(_PIPELINE_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ===========================================================================
# MemoryPipeline -- the interceptor layer
# ===========================================================================

class MemoryPipeline:
    """Organic memory pipeline -- internal infrastructure of MemoryManager.

    NOT a MemoryProvider.  Pure interceptor wrapping MemoryManager lifecycle.
    All methods best-effort: exceptions caught at debug level, never blocking.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config or {}
        self._state: PipelineState | None = None
        self._enabled: bool = self._config.get("enabled", True)
        self._session_id: str = ""
        # All 6 layers + episodic + dreaming
        self._salience: SalienceScorer | None = None
        self._engrams: SilentEngramEngine | None = None
        self._consolidation: ConsolidationEngine | None = None
        self._reconsolidation: ReconsolidationEngine | None = None
        self._feedback: FeedbackCoordinator | None = None
        self._activation: ActivationGraph | None = None
        self._episodic = None   # EpisodicTimeline (from holographic plugin)
        self._dreaming = None   # DreamEngine (from holographic plugin)

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize pipeline state and all organic modules."""
        if not self._enabled:
            return
        self._session_id = session_id
        db_path = self._config.get("db_path") or None
        self._state = PipelineState(db_path=db_path)

        # Layer 1: SalienceScorer
        sal_cfg = self._config.get("salience", {})
        if sal_cfg.get("enabled", True):
            self._salience = SalienceScorer(
                novelty_window=sal_cfg.get("novelty_window", 50))

        # Layer 2: SilentEngramEngine
        eng_cfg = self._config.get("silent_engram", {})
        if eng_cfg.get("enabled", True):
            self._engrams = SilentEngramEngine(
                half_life_hours=eng_cfg.get("half_life_hours", 720.0))

        # Layer 3: ConsolidationEngine
        con_cfg = self._config.get("consolidation", {})
        if con_cfg.get("enabled", True):
            self._consolidation = ConsolidationEngine(
                min_facts=con_cfg.get("min_facts_for_consolidation", 5))

        # Layer 4: ReconsolidationEngine
        rec_cfg = self._config.get("reconsolidation", {})
        if rec_cfg.get("enabled", True):
            self._reconsolidation = ReconsolidationEngine(
                error_threshold=rec_cfg.get("prediction_error_threshold", 0.3))

        # Layer 5: FeedbackCoordinator
        if self._config.get("feedback", {}).get("enabled", True):
            self._feedback = FeedbackCoordinator()

        # Layer 6: ActivationGraph
        act_cfg = self._config.get("activation", {})
        if act_cfg.get("enabled", True):
            self._activation = ActivationGraph(
                edge_decay_hours=act_cfg.get("edge_decay_hours", 168.0))

        # Layer 7: EpisodicTimeline (what-where-when binding)
        epi_cfg = self._config.get("episodic", {})
        if epi_cfg.get("enabled", False):
            try:
                import importlib.util
                _spec = importlib.util.spec_from_file_location(
                    "holographic_episodic",
                    str(Path(__file__).resolve().parent.parent
                        / "plugins" / "memory" / "holographic" / "episodic.py"))
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                self._episodic = _mod.EpisodicTimeline(self._state._conn, self._state._lock)
                self._episodic.init_tables()
            except Exception as e:
                logger.debug("EpisodicTimeline init failed: %s", e)

        # Layer 8: DreamEngine (structured selective replay)
        dream_cfg = self._config.get("dreaming", {})
        if dream_cfg.get("enabled", False):
            try:
                import importlib.util, sys as _sys
                _spec = importlib.util.spec_from_file_location(
                    "holographic_dreaming",
                    str(Path(__file__).resolve().parent.parent
                        / "plugins" / "memory" / "holographic" / "dreaming.py"))
                _mod = importlib.util.module_from_spec(_spec)
                _sys.modules[_spec.name] = _mod
                _spec.loader.exec_module(_mod)
                self._dreaming = _mod.DreamEngine(
                    self._state._conn, self._state._lock,
                    cooldown_hours=dream_cfg.get("cooldown_hours", 1.0),
                    mode1_top_k=dream_cfg.get("mode1_top_k", 10),
                    mode2_top_k=dream_cfg.get("mode2_top_k", 5),
                    mode3_idle_hours=dream_cfg.get("mode3_idle_hours", 24.0),
                    mode3_min_schema_conf=dream_cfg.get("mode3_min_schema_conf", 0.7),
                )
                self._dreaming.init_tables()
            except Exception as e:
                logger.debug("DreamEngine init failed: %s", e)

        logger.debug("MemoryPipeline initialized (session=%s, layers=%d)",
                      session_id, sum(1 for x in [self._salience, self._engrams,
                      self._consolidation, self._reconsolidation,
                      self._feedback, self._activation,
                      self._episodic, self._dreaming] if x))

    def shutdown(self) -> None:
        """Flush and close pipeline state."""
        if self._state is not None:
            self._state.close()
            self._state = None
        logger.debug("MemoryPipeline shut down")

    # -- Pre-interceptors --

    def pre_turn_start(self, turn: int, message: str) -> None:
        """Reset salience novelty window periodically, decay activation edges."""
        if self._salience and turn > 0 and turn % 100 == 0:
            try:
                self._salience.reset()
            except Exception as e:
                logger.debug("SalienceScorer reset failed: %s", e)
        if self._activation and self._state:
            try:
                self._activation.decay_edges(self._state, hours_elapsed=0.1)
            except Exception as e:
                logger.debug("Activation decay failed: %s", e)

    def pre_sync(self, user: str, asst: str) -> dict | None:
        """Score user content for salience, persist signals."""
        if not self._salience:
            return None
        try:
            result = self._salience.score(user)
            if self._state:
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO salience_encoding_log "
                        "(source, emotion_score, novelty_score, importance_score, overall_score) "
                        "VALUES (?, ?, ?, ?, ?)",
                        ("builtin", result.emotion, result.novelty,
                         result.importance, result.overall),
                    )
                    self._state._conn.commit()
            return {
                "salience_overall": result.overall,
                "salience_emotion": result.emotion,
                "salience_novelty": result.novelty,
                "salience_importance": result.importance,
                "salience_is_trivial": result.is_trivial,
            }
        except Exception as e:
            logger.debug("SalienceScorer.score failed: %s", e)
            return None

    def pre_memory_write(
        self, action: str, target: str, content: str, metadata: dict
    ) -> dict | None:
        """Salience gate — score content, attach metadata."""
        if not self._salience or action not in ("add", "replace"):
            return None
        try:
            result = self._salience.score(content)
            if self._state:
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO salience_encoding_log "
                        "(source, fact_ref, emotion_score, novelty_score, "
                        "importance_score, overall_score) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (target, content[:100], result.emotion, result.novelty,
                         result.importance, result.overall),
                    )
                    self._state._conn.commit()
            return {
                **metadata,
                "pipeline_salience": result.overall,
                "pipeline_emotion": result.emotion,
                "pipeline_novelty": result.novelty,
                "pipeline_importance": result.importance,
            }
        except Exception as e:
            logger.debug("SalienceScorer pre_memory_write failed: %s", e)
            return None

    def pre_compress(self, messages: list) -> str:
        """Extract key insights before context compression."""
        if not self._consolidation:
            return ""
        try:
            return self._consolidation.extract_insights(messages)
        except Exception as e:
            logger.debug("Consolidation extract_insights failed: %s", e)
            return ""

    # -- Post-interceptors --

    def post_prefetch(self, query: str, provider_results: list[str]) -> str:
        """Augment prefetch with predictions and spreading activation."""
        parts = []
        try:
            # Layer 5: predictions from schemas
            if self._feedback and self._state:
                predictions = self._feedback.predict(self._state, query)
                for pred in predictions:
                    parts.append(pred)

            # Layer 6: spreading activation
            if self._activation and self._state:
                expansions = self._activation.expand_query(self._state, query)
                parts.extend(expansions)
        except Exception as e:
            logger.debug("Pipeline post_prefetch failed: %s", e)
        return "\n".join(parts)

    def post_tool_call(self, name: str, args: dict, result: str) -> None:
        """Record retrieval for reconsolidation, co-activation."""
        if not self._state:
            return
        try:
            # Layer 4: reconsolidation check
            if self._reconsolidation and name == "fact_store":
                action = args.get("action", "")
                if action in ("search", "probe"):
                    self._reconsolidation.check_retrieval(
                        self._state, args.get("query", ""), result,
                        engrams=self._engrams)

            # Layer 6: record co-activation from search results
            if self._activation and name == "fact_store":
                query = args.get("query", "")
                entities = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
                if len(entities) >= 2:
                    self._activation.record_co_activation(self._state, entities)
        except Exception as e:
            logger.debug("Pipeline post_tool_call failed: %s", e)

    def post_session_end(self, messages: list) -> None:
        """Consolidation, engram decay, bridge discovery, dreaming."""
        if not self._state:
            return
        try:
            # Layer 2: apply engram decay (1 hour worth)
            if self._engrams:
                self._engrams.apply_decay(self._state, hours_elapsed=1.0)

            # Layer 3: run consolidation
            if self._consolidation:
                facts = []
                for msg in messages[-10:]:
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    if content and len(content) > 20:
                        facts.append({"content": content, "domain": "general"})
                self._consolidation.consolidate(self._state, facts)

            # Layer 5: discover cross-domain bridges
            if self._feedback:
                self._feedback.discover_bridges(self._state)

            # Layer 6: decay activation edges
            if self._activation:
                self._activation.decay_edges(self._state, hours_elapsed=1.0)

            # Layer 7: close episodic episode
            if self._episodic:
                try:
                    summary = f"Session {self._session_id}: {len(messages)} messages"
                    self._episodic.close_episode(summary=summary)
                except Exception as e:
                    logger.debug("Episodic close_episode failed: %s", e)

            # Layer 8: run dream cycle if conditions met
            if self._dreaming:
                try:
                    if self._dreaming.should_dream():
                        import threading as _t
                        _t.Thread(
                            target=self._dreaming.dream_cycle,
                            args=(self._session_id,),
                            daemon=True,
                        ).start()
                except Exception as e:
                    logger.debug("Dream cycle failed: %s", e)
        except Exception as e:
            logger.debug("Pipeline post_session_end failed: %s", e)

    def post_session_switch(self, new_id: str, **kwargs) -> None:
        """No-op for now. Phase 2+: update per-session consolidation state."""
        pass

    def post_delegation(self, task: str, result: str, **kwargs) -> None:
        """No-op for now. Phase 2+: score subagent result."""
        pass

    def augment_system_prompt(self) -> str:
        """Inject organic memory status into system prompt."""
        if not self._state:
            return ""
        try:
            with self._state._lock:
                engram_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM engram_strengths"
                ).fetchone()[0]
                schema_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM schemas"
                ).fetchone()[0]
                edge_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM activation_edges"
                ).fetchone()[0]
            if engram_count == 0 and schema_count == 0:
                return ""
            return (
                f"[Organic Memory: {engram_count} engrams, "
                f"{schema_count} schemas, {edge_count} activation edges]"
            )
        except Exception as e:
            logger.debug("augment_system_prompt failed: %s", e)
            return ""


# ===========================================================================
# Config loader
# ===========================================================================

def _load_pipeline_config() -> dict:
    """Load memory.pipeline config from $HERMES_HOME/config.yaml."""
    try:
        from hermes_cli.config import cfg_get, load_config
        config = load_config()
        return cfg_get(config, "memory", "pipeline", default={}) or {}
    except Exception as e:
        logger.debug("Failed to load pipeline config: %s", e)
        return {}
