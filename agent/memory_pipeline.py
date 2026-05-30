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
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Schema (full database provisioning)
# ---------------------------------------------------------------------------

_PIPELINE_SCHEMA = """\
-- Layer 1: Salience learning state
CREATE TABLE IF NOT EXISTS salience_weights (
    signal_type     TEXT PRIMARY KEY,
    weight          REAL NOT NULL,
    sample_count    INTEGER DEFAULT 0,
    success_count   INTEGER DEFAULT 0,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS salience_encoding_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    fact_ref        TEXT,
    emotion_score   REAL,
    novelty_score   REAL,
    importance_score REAL,
    overall_score   REAL,
    was_helpful     INTEGER DEFAULT -1,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Layer 2: Silent Engram state (cross-provider strength tracking)
CREATE TABLE IF NOT EXISTS engram_strengths (
    memory_ref      TEXT PRIMARY KEY,
    provider        TEXT NOT NULL,
    strength        REAL DEFAULT 1.0,
    last_accessed   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decay_half_life_hours REAL DEFAULT 720.0,
    access_count    INTEGER DEFAULT 0
);

-- Layer 3: Schema Store (neocortical semantic knowledge)
CREATE TABLE IF NOT EXISTS schemas (
    schema_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL,
    domain          TEXT DEFAULT 'general',
    confidence      REAL DEFAULT 0.5,
    source_count    INTEGER DEFAULT 1,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS schema_sources (
    schema_id       INTEGER REFERENCES schemas(schema_id),
    memory_ref      TEXT NOT NULL,
    provider        TEXT NOT NULL,
    contribution    REAL DEFAULT 1.0,
    PRIMARY KEY (schema_id, memory_ref)
);

CREATE TABLE IF NOT EXISTS reconsolidation_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_ref      TEXT,
    old_content     TEXT,
    new_content     TEXT,
    prediction_error REAL,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS consolidation_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    memories_processed INTEGER DEFAULT 0,
    schemas_created INTEGER DEFAULT 0,
    schemas_updated INTEGER DEFAULT 0,
    conflicts_found INTEGER DEFAULT 0,
    duration_ms     INTEGER DEFAULT 0,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Layer 4: Prediction state
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_id       INTEGER,
    prediction      TEXT NOT NULL,
    context         TEXT DEFAULT '',
    outcome         TEXT DEFAULT '',
    error_score     REAL DEFAULT 0.0,
    resolved        INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at     TIMESTAMP
);

-- Layer 5: Salience Feedback
CREATE TABLE IF NOT EXISTS salience_feedback (
    feedback_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_ref      TEXT,
    signal_type     TEXT,
    signal_value    REAL,
    was_helpful     INTEGER DEFAULT 0,
    was_retrieved   INTEGER DEFAULT 0,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Layer 6: Co-Activation Graph (spreading activation)
CREATE TABLE IF NOT EXISTS activation_edges (
    source_entity   TEXT NOT NULL,
    target_entity   TEXT NOT NULL,
    strength        REAL DEFAULT 0.1,
    co_activation_count INTEGER DEFAULT 1,
    last_activated  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_entity, target_entity)
);

CREATE TABLE IF NOT EXISTS cross_domain_links (
    link_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity          TEXT NOT NULL,
    domain_a        TEXT NOT NULL,
    domain_b        TEXT NOT NULL,
    fact_refs_a     TEXT DEFAULT '',
    fact_refs_b     TEXT DEFAULT '',
    strength        REAL DEFAULT 0.5,
    discovered_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_engram_strength ON engram_strengths(strength DESC);
CREATE INDEX IF NOT EXISTS idx_engram_provider ON engram_strengths(provider);
CREATE INDEX IF NOT EXISTS idx_schemas_domain ON schemas(domain);
CREATE INDEX IF NOT EXISTS idx_schemas_confidence ON schemas(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_activation_source ON activation_edges(source_entity);
CREATE INDEX IF NOT EXISTS idx_activation_target ON activation_edges(target_entity);
CREATE INDEX IF NOT EXISTS idx_cross_links_entity ON cross_domain_links(entity);
CREATE INDEX IF NOT EXISTS idx_salience_feedback_ref ON salience_feedback(memory_ref, was_retrieved);
"""


# ---------------------------------------------------------------------------
# PipelineState -- persistent storage for organic memory modules
# ---------------------------------------------------------------------------

class PipelineState:
    """Persistent state for the memory pipeline (pipeline_state.db).

    Design constraints:
    - Single connection + threading.RLock (same pattern as MemoryStore in store.py)
    - WAL mode (same as store.py via apply_wal_with_fallback)
    - Independent from any provider's database connection

    References to provider memories use ``memory_ref`` (format:
    ``{provider_name}:{native_id}`` or content hash), NOT foreign keys.
    This ensures cross-provider decoupling.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "pipeline_state.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        """Create all pipeline tables if they do not exist. Enable WAL mode."""
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="pipeline_state.db")
        with self._lock:
            self._conn.executescript(_PIPELINE_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection. Idempotent."""
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# MemoryPipeline -- the interceptor layer
# ---------------------------------------------------------------------------

class MemoryPipeline:
    """Organic memory pipeline -- internal infrastructure of MemoryManager.

    NOT a MemoryProvider.  Has no name, does not expose tools, does not
    produce system_prompt_block.  Pure interceptor: wraps MemoryManager
    lifecycle methods to execute organic logic before/after providers.

    All methods are best-effort: exceptions are caught and logged at
    debug level, never blocking upstream providers.

    Phase 1: All methods are no-op stubs.  The pipeline skeleton establishes
    the plumbing so that Phase 2 (extracting modules from Holographic) can
    proceed without ever touching MemoryManager again.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config or {}
        self._state: PipelineState | None = None
        self._enabled: bool = self._config.get("enabled", True)
        # Layer references -- populated in Phase 2+
        self._salience = None
        self._silent_engram = None
        self._consolidation = None
        self._reconsolidation = None
        self._feedback = None
        self._activation = None

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize pipeline state database.

        Called from MemoryManager.initialize_all() BEFORE providers init.
        """
        if not self._enabled:
            return
        db_path = self._config.get("db_path") or None
        self._state = PipelineState(db_path=db_path)
        logger.debug("MemoryPipeline initialized (session=%s)", session_id)

    def shutdown(self) -> None:
        """Flush and close pipeline state.

        Called from MemoryManager.shutdown_all() BEFORE providers shutdown.
        """
        if self._state is not None:
            self._state.close()
            self._state = None
        logger.debug("MemoryPipeline shut down")

    # -- Pre-interceptors (called BEFORE provider operations) --

    def pre_turn_start(self, turn: int, message: str) -> None:
        """Called before providers' on_turn_start.

        Phase 1: no-op.
        Phase 2: reset novelty window, decay activation edges.
        """
        pass

    def pre_sync(self, user: str, asst: str) -> dict | None:
        """Called before providers' sync_turn.

        Returns salience metadata for providers that support it.
        Phase 1: returns None (no metadata).
        Phase 2: score user content, return salience signals.
        """
        return None

    def pre_memory_write(
        self, action: str, target: str, content: str, metadata: dict
    ) -> dict | None:
        """Called before providers' on_memory_write.

        Returns modified metadata or None to pass-through.
        Phase 1: returns None.
        Phase 2: salience gate -- score content, potentially block/score writes.
        """
        return None

    def pre_compress(self, messages: list) -> str:
        """Called before providers' on_pre_compress.

        Returns insights text to include in compression summary.
        Phase 1: returns empty string.
        Phase 2: extract key facts from messages about to be discarded.
        """
        return ""

    # -- Post-interceptors (called AFTER provider operations) --

    def post_prefetch(self, query: str, provider_results: list[str]) -> str:
        """Called after providers' prefetch.

        Returns augmented context to append to prefetch results.
        Phase 1: returns empty string (no augmentation).
        Phase 2: spontaneous recovery, predictions, spreading activation.
        """
        return ""

    def post_tool_call(self, name: str, args: dict, result: str) -> None:
        """Called after provider's handle_tool_call.

        Phase 1: no-op.
        Phase 2: record retrieval for reconsolidation, feedback learning,
        co-activation recording.
        """
        pass

    def post_session_end(self, messages: list) -> None:
        """Called BEFORE providers' on_session_end.

        Phase 1: no-op.
        Phase 2: apply engram decay, run consolidation, discover bridges.
        """
        pass

    def post_session_switch(self, new_id: str, **kwargs) -> None:
        """Called after providers' on_session_switch.

        Phase 1: no-op.
        Phase 2: update per-session consolidation state.
        """
        pass

    def post_delegation(self, task: str, result: str, **kwargs) -> None:
        """Called after providers' on_delegation.

        Phase 1: no-op.
        Phase 2: score subagent result, extract high-salience facts.
        """
        pass

    def augment_system_prompt(self) -> str:
        """Called after providers' system_prompt_block.

        Returns text to append to the system prompt.
        Phase 1: returns empty string.
        Phase 2: inject organic memory status (silent count, schema count).
        """
        return ""


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_pipeline_config() -> dict:
    """Load memory.pipeline config from $HERMES_HOME/config.yaml.

    Returns an empty dict if the section is missing or config is unreadable.
    Uses lazy imports to avoid circular dependency with hermes_cli.config.
    """
    try:
        from hermes_cli.config import cfg_get, load_config
        config = load_config()
        return cfg_get(config, "memory", "pipeline", default={}) or {}
    except Exception as e:
        logger.debug("Failed to load pipeline config: %s", e)
        return {}
