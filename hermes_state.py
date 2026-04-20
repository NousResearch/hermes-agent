#!/usr/bin/env python3
"""
SQLite State Store for Hermes Agent.

Provides persistent session storage with FTS5 full-text search, replacing
the per-session JSONL file approach. Stores session metadata, full message
history, and model configuration for CLI and gateway sessions.

Key design decisions:
- WAL mode for concurrent readers + one writer (gateway multi-platform)
- FTS5 virtual table for fast text search across all session messages
- Compression-triggered session splitting via parent_session_id chains
- Batch runner and RL trajectories are NOT stored here (separate systems)
- Session source tagging ('cli', 'telegram', 'discord', etc.) for filtering
"""

import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"

SCHEMA_VERSION = 18

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    session_state TEXT,
    owner_fingerprint TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    message_text TEXT NOT NULL,
    message_char_count INTEGER,
    message_word_count INTEGER,
    has_code_block INTEGER,
    has_url INTEGER,
    complex_keyword_hit TEXT,
    conversation_depth INTEGER,
    routed_model TEXT NOT NULL,
    routing_reason TEXT,
    quality_score REAL,
    response_latency_ms INTEGER,
    response_cost_usd REAL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_session ON routing_decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_created ON routing_decisions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_model ON routing_decisions(routed_model);

CREATE TABLE IF NOT EXISTS cron_output_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    output_path TEXT NOT NULL,
    char_count INTEGER,
    line_count INTEGER,
    word_count INTEGER,
    entity_count INTEGER,
    numeric_values TEXT,
    sentiment_score REAL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cron_output_stats_job ON cron_output_stats(job_id);
CREATE INDEX IF NOT EXISTS idx_cron_output_stats_created ON cron_output_stats(created_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    prediction_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicted_value TEXT NOT NULL,
    predicted_at REAL NOT NULL,
    resolution_target_at REAL,
    actual_value TEXT,
    resolved_at REAL,
    correct INTEGER,
    confidence REAL,
    source_output_path TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_job ON predictions(job_id);
CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(resolved_at) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type, subject);

CREATE TABLE IF NOT EXISTS knowledge_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT,
    content TEXT NOT NULL,
    source TEXT,
    session_id TEXT,
    file_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT,
    name TEXT NOT NULL,
    role TEXT,
    organization TEXT,
    details TEXT,
    file_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    description TEXT,
    file_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT,
    title TEXT NOT NULL,
    rationale TEXT,
    status TEXT DEFAULT 'active',
    file_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_tags_tag ON knowledge_tags(tag);
CREATE INDEX IF NOT EXISTS idx_knowledge_tags_entity ON knowledge_tags(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_people_name ON knowledge_people(name);
CREATE INDEX IF NOT EXISTS idx_knowledge_projects_status ON knowledge_projects(name, status);
CREATE INDEX IF NOT EXISTS idx_knowledge_notes_created ON knowledge_notes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_decisions_status ON knowledge_decisions(status);
-- NOTE: uuid indexes are NOT created here because executescript() runs BEFORE
-- the migration loop. On a pre-v14 database the uuid column doesn't exist yet,
-- so these indexes would fail. They are created in the v14 migration block
-- below, after the ALTER TABLE statements add the column.

CREATE TABLE IF NOT EXISTS implicit_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type TEXT NOT NULL,
    context_id TEXT,
    session_id TEXT,
    signal_value REAL NOT NULL,
    signal_source TEXT NOT NULL,
    message_text TEXT,
    metadata TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_implicit_signals_type ON implicit_signals(signal_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_implicit_signals_session ON implicit_signals(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS provider_health_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    event_type TEXT NOT NULL,
    latency_ms REAL,
    error TEXT,
    health_score REAL,
    consecutive_failures INTEGER,
    recovery_time_seconds REAL,
    session_id TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_provider_health_provider ON provider_health_log(provider, created_at DESC);

CREATE TABLE IF NOT EXISTS obsidian_managed_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT,
    vault_relative_path TEXT NOT NULL UNIQUE,
    managed_relative_path TEXT NOT NULL,
    entity_type TEXT,
    wiki_page_type TEXT,
    file_ext TEXT,
    content_hash TEXT,
    last_vault_mtime REAL,
    last_vault_size INTEGER,
    last_db_revision_id INTEGER,
    last_sync_direction TEXT NOT NULL DEFAULT 'vault_scan',
    sync_status TEXT NOT NULL DEFAULT 'synced',
    conflict_state TEXT NOT NULL DEFAULT 'none',
    source_origin TEXT NOT NULL DEFAULT 'managed',
    tombstoned INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obsidian_managed_uuid ON obsidian_managed_files(uuid);
CREATE INDEX IF NOT EXISTS idx_obsidian_managed_status ON obsidian_managed_files(sync_status, tombstoned);

CREATE TABLE IF NOT EXISTS obsidian_file_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_relative_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_text TEXT,
    source TEXT NOT NULL,
    actor TEXT,
    metadata_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obsidian_file_revisions_path ON obsidian_file_revisions(vault_relative_path, created_at DESC);

CREATE TABLE IF NOT EXISTS obsidian_sync_checkpoints (
    scope TEXT PRIMARY KEY,
    last_started_at REAL,
    last_completed_at REAL,
    last_status TEXT,
    last_error TEXT,
    last_scan_count INTEGER NOT NULL DEFAULT 0,
    last_change_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS obsidian_sync_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    path TEXT,
    direction TEXT,
    status TEXT NOT NULL,
    detail TEXT,
    metadata_json TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obsidian_sync_events_created ON obsidian_sync_events(created_at DESC);

CREATE TABLE IF NOT EXISTS obsidian_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_relative_path TEXT NOT NULL,
    uuid TEXT,
    entity_type TEXT,
    conflict_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    summary TEXT NOT NULL,
    db_snapshot TEXT,
    vault_snapshot TEXT,
    resolution TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    resolved_at REAL
);

CREATE INDEX IF NOT EXISTS idx_obsidian_conflicts_status ON obsidian_conflicts(status, created_at DESC);

CREATE TABLE IF NOT EXISTS obsidian_attachment_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_path TEXT NOT NULL,
    target_path TEXT NOT NULL,
    target_type TEXT NOT NULL,
    exists_flag INTEGER NOT NULL DEFAULT 0,
    mime_type TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(owner_path, target_path, target_type)
);

CREATE INDEX IF NOT EXISTS idx_obsidian_attachment_owner ON obsidian_attachment_index(owner_path);

CREATE TABLE IF NOT EXISTS obsidian_canvas_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canvas_path TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    target_path TEXT,
    broken INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT,
    updated_at REAL NOT NULL,
    UNIQUE(canvas_path, node_id)
);

CREATE INDEX IF NOT EXISTS idx_obsidian_canvas_path ON obsidian_canvas_index(canvas_path);

CREATE TABLE IF NOT EXISTS obsidian_import_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_relative_path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    file_uuid TEXT,
    content_hash TEXT,
    last_vault_mtime REAL,
    imported INTEGER NOT NULL DEFAULT 0,
    imported_managed_path TEXT,
    metadata_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obsidian_import_imported ON obsidian_import_candidates(imported, updated_at DESC);

CREATE TABLE IF NOT EXISTS obsidian_plugin_connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL UNIQUE,
    client_name TEXT NOT NULL,
    client_version TEXT,
    vault_name TEXT,
    status TEXT NOT NULL DEFAULT 'connected',
    metadata_json TEXT,
    connected_at REAL,
    disconnected_at REAL,
    last_seen_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obsidian_plugin_status ON obsidian_plugin_connections(status, last_seen_at DESC);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_notes_fts USING fts5(
    content,
    content=knowledge_notes,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS knowledge_notes_fts_insert AFTER INSERT ON knowledge_notes BEGIN
    INSERT INTO knowledge_notes_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_notes_fts_delete AFTER DELETE ON knowledge_notes BEGIN
    INSERT INTO knowledge_notes_fts(knowledge_notes_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_notes_fts_update AFTER UPDATE ON knowledge_notes BEGIN
    INSERT INTO knowledge_notes_fts(knowledge_notes_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO knowledge_notes_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


from agent.state.obsidian import _ObsidianMixin  # W5.A scoped split  # noqa: E402


class SessionDB(_ObsidianMixin):
    """
    SQLite-backed session storage with FTS5 search.

    Thread-safe for the common gateway pattern (multiple reader threads,
    single writer via WAL mode). Each method opens its own cursor.
    """

    # ── Write-contention tuning ──
    # With multiple hermes processes (gateway + CLI sessions + worktree agents)
    # all sharing one state.db, WAL write-lock contention causes visible TUI
    # freezes.  SQLite's built-in busy handler uses a deterministic sleep
    # schedule that causes convoy effects under high concurrency.
    #
    # Instead, we keep the SQLite timeout short (1s) and handle retries at the
    # application level with random jitter, which naturally staggers competing
    # writers and avoids the convoy.
    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020   # 20ms
    _WRITE_RETRY_MAX_S = 0.150   # 150ms
    # Attempt a PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 50

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._write_count = 0
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            # Short timeout — application-level retry with random jitter
            # handles contention instead of sitting in SQLite's internal
            # busy handler for up to 30s.
            timeout=1.0,
            # Autocommit mode: Python's default isolation_level="" auto-starts
            # transactions on DML, which conflicts with our explicit
            # BEGIN IMMEDIATE.  None = we manage transactions ourselves.
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()

    # ── Core write helper ──

    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        *fn* receives the connection and should perform INSERT/UPDATE/DELETE
        statements.  The caller must NOT call ``commit()`` — that's handled
        here after *fn* returns.

        BEGIN IMMEDIATE acquires the WAL write lock at transaction start
        (not at commit time), so lock contention surfaces immediately.
        On ``database is locked``, we release the Python lock, sleep a
        random 20-150ms, and retry — breaking the convoy pattern that
        SQLite's built-in deterministic backoff creates.

        Returns whatever *fn* returns.
        """
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
                # Success — periodic best-effort checkpoint.
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
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
                # Non-lock error or retries exhausted — propagate.
                raise
        # Retries exhausted (shouldn't normally reach here).
        raise last_err or sqlite3.OperationalError(
            "database is locked after max retries"
        )

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint.  Never blocks, never raises.

        Flushes committed WAL frames back into the main DB file for any
        frames that no other connection currently needs.  Keeps the WAL
        from growing unbounded when many processes hold persistent
        connections.
        """
        try:
            with self._lock:
                result = self._conn.execute(
                    "PRAGMA wal_checkpoint(PASSIVE)"
                ).fetchone()
                if result and result[1] > 0:
                    logger.debug(
                        "WAL checkpoint: %d/%d pages checkpointed",
                        result[2], result[1],
                    )
        except Exception:
            pass  # Best effort — never fatal.

    def close(self):
        """Close the database connection.

        Attempts a PASSIVE WAL checkpoint first so that exiting processes
        help keep the WAL file from growing unbounded.
        """
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    def _init_schema(self):
        """Create tables and FTS if they don't exist, run migrations."""
        cursor = self._conn.cursor()

        cursor.executescript(SCHEMA_SQL)

        # Check schema version and run migrations
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        else:
            current_version = row["version"] if isinstance(row, sqlite3.Row) else row[0]
            if current_version < 2:
                # v2: add finish_reason column to messages
                try:
                    cursor.execute("ALTER TABLE messages ADD COLUMN finish_reason TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 2")
            if current_version < 3:
                # v3: add title column to sessions
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 3")
            if current_version < 4:
                # v4: add unique index on title (NULLs allowed, only non-NULL must be unique)
                try:
                    cursor.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
                        "ON sessions(title) WHERE title IS NOT NULL"
                    )
                except sqlite3.OperationalError:
                    pass  # Index already exists
                cursor.execute("UPDATE schema_version SET version = 4")
            if current_version < 5:
                new_columns = [
                    ("cache_read_tokens", "INTEGER DEFAULT 0"),
                    ("cache_write_tokens", "INTEGER DEFAULT 0"),
                    ("reasoning_tokens", "INTEGER DEFAULT 0"),
                    ("billing_provider", "TEXT"),
                    ("billing_base_url", "TEXT"),
                    ("billing_mode", "TEXT"),
                    ("estimated_cost_usd", "REAL"),
                    ("actual_cost_usd", "REAL"),
                    ("cost_status", "TEXT"),
                    ("cost_source", "TEXT"),
                    ("pricing_version", "TEXT"),
                ]
                for name, column_type in new_columns:
                    try:
                        # name and column_type come from the hardcoded tuple above,
                        # not user input. Double-quote identifier escaping is applied
                        # as defense-in-depth; SQLite DDL cannot be parameterized.
                        safe_name = name.replace('"', '""')
                        cursor.execute(f'ALTER TABLE sessions ADD COLUMN "{safe_name}" {column_type}')
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 5")
            if current_version < 6:
                # v6: add reasoning columns to messages table — preserves assistant
                # reasoning text and structured reasoning_details across gateway
                # session turns.  Without these, reasoning chains are lost on
                # session reload, breaking multi-turn reasoning continuity for
                # providers that replay reasoning (OpenRouter, OpenAI, Nous).
                for col_name, col_type in [
                    ("reasoning", "TEXT"),
                    ("reasoning_details", "TEXT"),
                    ("codex_reasoning_items", "TEXT"),
                ]:
                    try:
                        safe = col_name.replace('"', '""')
                        cursor.execute(
                            f'ALTER TABLE messages ADD COLUMN "{safe}" {col_type}'
                        )
                    except sqlite3.OperationalError:
                        pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 6")
            if current_version < 7:
                # v7: add routing_decisions and cron_output_stats tables for ML pipeline
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS routing_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        message_text TEXT NOT NULL,
                        message_char_count INTEGER,
                        message_word_count INTEGER,
                        has_code_block INTEGER,
                        has_url INTEGER,
                        complex_keyword_hit TEXT,
                        conversation_depth INTEGER,
                        routed_model TEXT NOT NULL,
                        routing_reason TEXT,
                        quality_score REAL,
                        response_latency_ms INTEGER,
                        response_cost_usd REAL,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_routing_decisions_session ON routing_decisions(session_id);
                    CREATE INDEX IF NOT EXISTS idx_routing_decisions_created ON routing_decisions(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_routing_decisions_model ON routing_decisions(routed_model);

                    CREATE TABLE IF NOT EXISTS cron_output_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        output_path TEXT NOT NULL,
                        char_count INTEGER,
                        line_count INTEGER,
                        word_count INTEGER,
                        entity_count INTEGER,
                        numeric_values TEXT,
                        sentiment_score REAL,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_cron_output_stats_job ON cron_output_stats(job_id);
                    CREATE INDEX IF NOT EXISTS idx_cron_output_stats_created ON cron_output_stats(created_at DESC);
                """)
                cursor.execute("UPDATE schema_version SET version = 7")
            if current_version < 8:
                # v8: add predictions table for market forecast tracking
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        prediction_type TEXT NOT NULL,
                        subject TEXT NOT NULL,
                        predicted_value TEXT NOT NULL,
                        predicted_at REAL NOT NULL,
                        resolution_target_at REAL,
                        actual_value TEXT,
                        resolved_at REAL,
                        correct INTEGER,
                        confidence REAL,
                        source_output_path TEXT,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_predictions_job ON predictions(job_id);
                    CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(resolved_at) WHERE resolved_at IS NULL;
                    CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type, subject);
                """)
                cursor.execute("UPDATE schema_version SET version = 8")
            if current_version < 9:
                # v9: add knowledge tables for structured personal knowledge
                # (notes, people, projects, decisions) with tag-based cross-linking
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS knowledge_notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        source TEXT,
                        session_id TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS knowledge_people (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        role TEXT,
                        organization TEXT,
                        details TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS knowledge_projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        status TEXT DEFAULT 'active',
                        description TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS knowledge_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        rationale TEXT,
                        status TEXT DEFAULT 'active',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS knowledge_tags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_type TEXT NOT NULL,
                        entity_id INTEGER NOT NULL,
                        tag TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_knowledge_tags_tag ON knowledge_tags(tag);
                    CREATE INDEX IF NOT EXISTS idx_knowledge_tags_entity ON knowledge_tags(entity_type, entity_id);
                    CREATE INDEX IF NOT EXISTS idx_knowledge_people_name ON knowledge_people(name);
                    CREATE INDEX IF NOT EXISTS idx_knowledge_projects_status ON knowledge_projects(name, status);
                    CREATE INDEX IF NOT EXISTS idx_knowledge_notes_created ON knowledge_notes(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_knowledge_decisions_status ON knowledge_decisions(status);
                """)
                # FTS5 for knowledge notes full-text search
                try:
                    cursor.executescript("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_notes_fts USING fts5(
                            content,
                            content=knowledge_notes,
                            content_rowid=id
                        );

                        CREATE TRIGGER IF NOT EXISTS knowledge_notes_fts_insert AFTER INSERT ON knowledge_notes BEGIN
                            INSERT INTO knowledge_notes_fts(rowid, content) VALUES (new.id, new.content);
                        END;

                        CREATE TRIGGER IF NOT EXISTS knowledge_notes_fts_delete AFTER DELETE ON knowledge_notes BEGIN
                            INSERT INTO knowledge_notes_fts(knowledge_notes_fts, rowid, content) VALUES('delete', old.id, old.content);
                        END;

                        CREATE TRIGGER IF NOT EXISTS knowledge_notes_fts_update AFTER UPDATE ON knowledge_notes BEGIN
                            INSERT INTO knowledge_notes_fts(knowledge_notes_fts, rowid, content) VALUES('delete', old.id, old.content);
                            INSERT INTO knowledge_notes_fts(rowid, content) VALUES (new.id, new.content);
                        END;
                    """)
                except Exception:
                    pass  # FTS5 may not be available on all builds
                cursor.execute("UPDATE schema_version SET version = 9")
            if current_version < 10:
                # v10: add session_state column for persistent key-value state
                # that survives context compression (inspired by agno session_state)
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN session_state TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 10")

            if current_version < 11:
                # v11: action_log — append-only accountability log for
                # side-effecting tool invocations (separate from session traces)
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS action_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        tool_name TEXT NOT NULL,
                        tool_args TEXT,
                        result_summary TEXT,
                        mutates INTEGER NOT NULL DEFAULT 1,
                        platform TEXT,
                        approved_by TEXT,
                        timestamp REAL NOT NULL,
                        duration_ms INTEGER
                    );
                    CREATE INDEX IF NOT EXISTS idx_action_log_session
                        ON action_log(session_id, timestamp);
                    CREATE INDEX IF NOT EXISTS idx_action_log_tool
                        ON action_log(tool_name, timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_action_log_timestamp
                        ON action_log(timestamp DESC);
                """)
                cursor.execute("UPDATE schema_version SET version = 11")

            if current_version < 12:
                # v12: implicit_signals — automatic quality signals from conversation
                # patterns for continuous improvement of routing, reasoning, and skill
                # matching classifiers
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS implicit_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_type TEXT NOT NULL,
                        context_id TEXT,
                        session_id TEXT,
                        signal_value REAL NOT NULL,
                        signal_source TEXT NOT NULL,
                        message_text TEXT,
                        metadata TEXT,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_implicit_signals_type
                        ON implicit_signals(signal_type, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_implicit_signals_session
                        ON implicit_signals(session_id, created_at DESC);
                """)
                cursor.execute("UPDATE schema_version SET version = 12")

            if current_version < 13:
                # v13: provider_health_log — persists provider success/failure events
                # so learned recovery windows survive restarts
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS provider_health_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        provider TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        latency_ms REAL,
                        error TEXT,
                        health_score REAL,
                        consecutive_failures INTEGER,
                        recovery_time_seconds REAL,
                        session_id TEXT,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_provider_health_provider
                        ON provider_health_log(provider, created_at DESC);
                """)
                cursor.execute("UPDATE schema_version SET version = 13")

            if current_version < 14:
                # v14: add uuid and file_path to knowledge tables
                for table in ["knowledge_notes", "knowledge_people", "knowledge_projects", "knowledge_decisions"]:
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN uuid TEXT")
                    except sqlite3.OperationalError:
                        pass
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN file_path TEXT")
                    except sqlite3.OperationalError:
                        pass

                # Add indexes for uuid
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_notes_uuid ON knowledge_notes(uuid)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_people_uuid ON knowledge_people(uuid)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_projects_uuid ON knowledge_projects(uuid)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_decisions_uuid ON knowledge_decisions(uuid)")

                cursor.execute("UPDATE schema_version SET version = 14")
            if current_version < 15:
                # v15: add macro_metrics_history for time-series charting
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS macro_metrics_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_id TEXT NOT NULL,
                        category TEXT,
                        value REAL NOT NULL,
                        unit TEXT,
                        source TEXT,
                        timestamp REAL NOT NULL,
                        metadata TEXT,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_macro_metric_time ON macro_metrics_history(metric_id, timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_macro_metric_category ON macro_metrics_history(category);
                """)
                cursor.execute("UPDATE schema_version SET version = 15")
            if current_version < 16:
                # v16: Obsidian sync foundation — managed files, conflicts,
                # import queue, plugin heartbeats, and attachment/canvas indexes.
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS obsidian_managed_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        uuid TEXT,
                        vault_relative_path TEXT NOT NULL UNIQUE,
                        managed_relative_path TEXT NOT NULL,
                        entity_type TEXT,
                        wiki_page_type TEXT,
                        file_ext TEXT,
                        content_hash TEXT,
                        last_vault_mtime REAL,
                        last_vault_size INTEGER,
                        last_db_revision_id INTEGER,
                        last_sync_direction TEXT NOT NULL DEFAULT 'vault_scan',
                        sync_status TEXT NOT NULL DEFAULT 'synced',
                        conflict_state TEXT NOT NULL DEFAULT 'none',
                        source_origin TEXT NOT NULL DEFAULT 'managed',
                        tombstoned INTEGER NOT NULL DEFAULT 0,
                        metadata_json TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_obsidian_managed_uuid ON obsidian_managed_files(uuid);
                    CREATE INDEX IF NOT EXISTS idx_obsidian_managed_status ON obsidian_managed_files(sync_status, tombstoned);

                    CREATE TABLE IF NOT EXISTS obsidian_sync_checkpoints (
                        scope TEXT PRIMARY KEY,
                        last_started_at REAL,
                        last_completed_at REAL,
                        last_status TEXT,
                        last_error TEXT,
                        last_scan_count INTEGER NOT NULL DEFAULT 0,
                        last_change_count INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS obsidian_sync_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        path TEXT,
                        direction TEXT,
                        status TEXT NOT NULL,
                        detail TEXT,
                        metadata_json TEXT,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_obsidian_sync_events_created ON obsidian_sync_events(created_at DESC);

                    CREATE TABLE IF NOT EXISTS obsidian_conflicts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        vault_relative_path TEXT NOT NULL,
                        uuid TEXT,
                        entity_type TEXT,
                        conflict_type TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'open',
                        summary TEXT NOT NULL,
                        db_snapshot TEXT,
                        vault_snapshot TEXT,
                        resolution TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        resolved_at REAL
                    );
                    CREATE INDEX IF NOT EXISTS idx_obsidian_conflicts_status ON obsidian_conflicts(status, created_at DESC);

                    CREATE TABLE IF NOT EXISTS obsidian_attachment_index (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_path TEXT NOT NULL,
                        target_path TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        exists_flag INTEGER NOT NULL DEFAULT 0,
                        mime_type TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        UNIQUE(owner_path, target_path, target_type)
                    );
                    CREATE INDEX IF NOT EXISTS idx_obsidian_attachment_owner ON obsidian_attachment_index(owner_path);

                    CREATE TABLE IF NOT EXISTS obsidian_canvas_index (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        canvas_path TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        node_type TEXT NOT NULL,
                        target_path TEXT,
                        broken INTEGER NOT NULL DEFAULT 0,
                        metadata_json TEXT,
                        updated_at REAL NOT NULL,
                        UNIQUE(canvas_path, node_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_obsidian_canvas_path ON obsidian_canvas_index(canvas_path);

                    CREATE TABLE IF NOT EXISTS obsidian_import_candidates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        vault_relative_path TEXT NOT NULL UNIQUE,
                        title TEXT NOT NULL,
                        file_uuid TEXT,
                        content_hash TEXT,
                        last_vault_mtime REAL,
                        imported INTEGER NOT NULL DEFAULT 0,
                        imported_managed_path TEXT,
                        metadata_json TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_obsidian_import_imported ON obsidian_import_candidates(imported, updated_at DESC);

                    CREATE TABLE IF NOT EXISTS obsidian_plugin_connections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_id TEXT NOT NULL UNIQUE,
                        client_name TEXT NOT NULL,
                        client_version TEXT,
                        vault_name TEXT,
                        status TEXT NOT NULL DEFAULT 'connected',
                        metadata_json TEXT,
                        connected_at REAL,
                        disconnected_at REAL,
                        last_seen_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_obsidian_plugin_status ON obsidian_plugin_connections(status, last_seen_at DESC);
                """)
                cursor.execute("UPDATE schema_version SET version = 16")
            if current_version < 17:
                # v17: durable file revisions for Obsidian conflict snapshots
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS obsidian_file_revisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        vault_relative_path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        content_text TEXT,
                        source TEXT NOT NULL,
                        actor TEXT,
                        metadata_json TEXT,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_obsidian_file_revisions_path ON obsidian_file_revisions(vault_relative_path, created_at DESC);
                """)
                cursor.execute("UPDATE schema_version SET version = 17")

            if current_version < 18:
                # v18 (F-003): per-session owner fingerprint. X-Hermes-Session-Id
                # was previously a bearer token — any authenticated caller could
                # read any session they could enumerate. We now bind sessions to
                # a sha256-hash of the creating API key (first 16 hex chars) and
                # reject cross-owner reads at the gateway. Null owner_fingerprint
                # marks pre-v18 sessions as legacy ("anonymous") — they remain
                # readable only on loopback without an API key (backward-compat).
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN owner_fingerprint TEXT")
                except sqlite3.OperationalError:
                    pass
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_owner_fp ON sessions(owner_fingerprint)"
                )
                cursor.execute("UPDATE schema_version SET version = 18")

        # Unique title index — always ensure it exists (safe to run after migrations
        # since the title column is guaranteed to exist at this point)
        try:
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
                "ON sessions(title) WHERE title IS NOT NULL"
            )
        except sqlite3.OperationalError:
            pass  # Index already exists

        # FTS5 setup (separate because CREATE VIRTUAL TABLE can't be in executescript with IF NOT EXISTS reliably)
        try:
            cursor.execute("SELECT * FROM messages_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)

        self._conn.commit()

    def close(self):
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    def create_session(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        parent_session_id: str = None,
        owner_fingerprint: str = None,
    ) -> str:
        """Create a new session record. Returns the session_id.

        `owner_fingerprint` is used by the gateway (F-003) to bind sessions to
        the API key that created them; cross-owner X-Hermes-Session-Id reads
        are rejected. None means "legacy / anonymous" (pre-v18 sessions or
        un-keyed local callers).
        """
        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO sessions (id, source, user_id, model, model_config,
                   system_prompt, parent_session_id, started_at, owner_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source,
                    user_id,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt,
                    parent_session_id,
                    time.time(),
                    owner_fingerprint,
                ),
            )
        self._execute_write(_do)
        return session_id

    def get_session_owner(self, session_id: str) -> Optional[str]:
        """Return the owner_fingerprint for a session, or None if unset / missing.

        A caller that received X-Hermes-Session-Id must pass its own computed
        fingerprint and compare against this return. None here means "legacy
        anonymous session" — readable only when the caller is also anonymous
        (no API key configured).
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT owner_fingerprint FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        if not row:
            return None
        val = row["owner_fingerprint"] if isinstance(row, sqlite3.Row) else row[0]
        return val if val else None

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                (time.time(), end_reason, session_id),
            )
        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                (system_prompt, session_id),
            )
        self._execute_write(_do)

    # -----------------------------------------------------------------
    # Session state (persistent key-value dict surviving compression)
    # -----------------------------------------------------------------

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Load the session_state JSON dict for a session. Returns {} if unset."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT session_state FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        if not row:
            return {}
        raw = row["session_state"] if isinstance(row, sqlite3.Row) else row[0]
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    def set_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """Persist the full session_state dict (overwrites previous value)."""
        serialized = json.dumps(state, ensure_ascii=False)
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET session_state = ? WHERE id = ?",
                (serialized, session_id),
            )
        self._execute_write(_do)

    def update_session_state(self, session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge *updates* into the existing session_state and persist.

        Keys set to None are deleted. Returns the merged state.
        """
        current = self.get_session_state(session_id)
        for key, value in updates.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        self.set_session_state(session_id, current)
        return current

    def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        absolute: bool = False,
    ) -> None:
        """Update token counters and backfill model if not already set.

        When *absolute* is False (default), values are **incremented** — use
        this for per-API-call deltas (CLI path).

        When *absolute* is True, values are **set directly** — use this when
        the caller already holds cumulative totals (gateway path, where the
        cached agent accumulates across messages).
        """
        if absolute:
            sql = """UPDATE sessions SET
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   reasoning_tokens = ?,
                   estimated_cost_usd = COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?)
                   WHERE id = ?"""
        else:
            sql = """UPDATE sessions SET
                   input_tokens = input_tokens + ?,
                   output_tokens = output_tokens + ?,
                   cache_read_tokens = cache_read_tokens + ?,
                   cache_write_tokens = cache_write_tokens + ?,
                   reasoning_tokens = reasoning_tokens + ?,
                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE COALESCE(actual_cost_usd, 0) + ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?)
                   WHERE id = ?"""
        params = (
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            estimated_cost_usd,
            actual_cost_usd,
            actual_cost_usd,
            cost_status,
            cost_source,
            pricing_version,
            billing_provider,
            billing_base_url,
            billing_mode,
            model,
            session_id,
        )
        def _do(conn):
            conn.execute(sql, params)
        self._execute_write(_do)

    def ensure_session(
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
    ) -> None:
        """Ensure a session row exists, creating it with minimal metadata if absent.

        Used by _flush_messages_to_session_db to recover from a failed
        create_session() call (e.g. transient SQLite lock at agent startup).
        INSERT OR IGNORE is safe to call even when the row already exists.
        """
        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO sessions
                   (id, source, model, started_at)
                   VALUES (?, ?, ?, ?)""",
                (session_id, source, model, time.time()),
            )
        self._execute_write(_do)

    def set_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
    ) -> None:
        """Set token counters to absolute values (not increment).

        Use this when the caller provides cumulative totals from a completed
        conversation run (e.g. the gateway, where the cached agent's
        session_prompt_tokens already reflects the running total).
        """
        def _do(conn):
            conn.execute(
                """UPDATE sessions SET
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   reasoning_tokens = ?,
                   estimated_cost_usd = ?,
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?)
                   WHERE id = ?""",
                (
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    reasoning_tokens,
                    estimated_cost_usd,
                    actual_cost_usd,
                    actual_cost_usd,
                    cost_status,
                    cost_source,
                    pricing_version,
                    billing_provider,
                    billing_base_url,
                    billing_mode,
                    model,
                    session_id,
                ),
            )
        self._execute_write(_do)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Resolve an exact or uniquely prefixed session ID to the full ID.

        Returns the exact ID when it exists. Otherwise treats the input as a
        prefix and returns the single matching session ID if the prefix is
        unambiguous. Returns None for no matches or ambiguous prefixes.
        """
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]

        escaped = (
            session_id_or_prefix
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
                (f"{escaped}%",),
            )
            matches = [row["id"] for row in cursor.fetchall()]
        if len(matches) == 1:
            return matches[0]
        return None

    # Maximum length for session titles
    MAX_TITLE_LENGTH = 100

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Validate and sanitize a session title.

        - Strips leading/trailing whitespace
        - Removes ASCII control characters (0x00-0x1F, 0x7F) and problematic
          Unicode control chars (zero-width, RTL/LTR overrides, etc.)
        - Collapses internal whitespace runs to single spaces
        - Normalizes empty/whitespace-only strings to None
        - Enforces MAX_TITLE_LENGTH

        Returns the cleaned title string or None.
        Raises ValueError if the title exceeds MAX_TITLE_LENGTH after cleaning.
        """
        if not title:
            return None

        # Remove ASCII control characters (0x00-0x1F, 0x7F) but keep
        # whitespace chars (\t=0x09, \n=0x0A, \r=0x0D) so they can be
        # normalized to spaces by the whitespace collapsing step below
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', title)

        # Remove problematic Unicode control characters:
        # - Zero-width chars (U+200B-U+200F, U+FEFF)
        # - Directional overrides (U+202A-U+202E, U+2066-U+2069)
        # - Object replacement (U+FFFC), interlinear annotation (U+FFF9-U+FFFB)
        cleaned = re.sub(
            r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]',
            '', cleaned,
        )

        # Collapse internal whitespace runs and strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if not cleaned:
            return None

        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})"
            )

        return cleaned

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title.

        Returns True if session was found and title was set.
        Raises ValueError if title is already in use by another session,
        or if the title fails validation (too long, invalid characters).
        Empty/whitespace-only strings are normalized to None (clearing the title).
        """
        title = self.sanitize_title(title)
        def _do(conn):
            if title:
                # Check uniqueness (allow the same session to keep its own title)
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE title = ? AND id != ?",
                    (title, session_id),
                )
                conflict = cursor.fetchone()
                if conflict:
                    raise ValueError(
                        f"Title '{title}' is already in use by session {conflict['id']}"
                    )
            cursor = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
            return cursor.rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return row["title"] if row else None

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE title = ?", (title,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session ID, preferring the latest in a lineage.

        If the exact title exists, returns that session's ID.
        If not, searches for "title #N" variants and returns the latest one.
        If the exact title exists AND numbered variants exist, returns the
        latest numbered variant (the most recent continuation).
        """
        # First try exact match
        exact = self.get_session_by_title(title)

        # Also search for numbered variants: "title #2", "title #3", etc.
        # Escape SQL LIKE wildcards (%, _) in the title to prevent false matches
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, title, started_at FROM sessions "
                "WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
                (f"{escaped} #%",),
            )
            numbered = cursor.fetchall()

        if numbered:
            # Return the most recent numbered variant
            return numbered[0]["id"]
        elif exact:
            return exact["id"]
        return None

    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Generate the next title in a lineage (e.g., "my session" → "my session #2").

        Strips any existing " #N" suffix to find the base name, then finds
        the highest existing number and increments.
        """
        # Strip existing #N suffix to find the true base
        match = re.match(r'^(.*?) #(\d+)$', base_title)
        if match:
            base = match.group(1)
        else:
            base = base_title

        # Find all existing numbered variants
        # Escape SQL LIKE wildcards (%, _) in the base to prevent false matches
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
                (base, f"{escaped} #%"),
            )
            existing = [row["title"] for row in cursor.fetchall()]

        if not existing:
            return base  # No conflict, use the base name as-is

        # Find the highest number
        max_num = 1  # The unnumbered original counts as #1
        for t in existing:
            m = re.match(r'^.* #(\d+)$', t)
            if m:
                max_num = max(max_num, int(m.group(1)))

        return f"{base} #{max_num + 1}"

    def list_sessions_rich(
        self,
        source: str = None,
        exclude_sources: List[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (timestamp of last message).

        Uses a single query with correlated subqueries instead of N+2 queries.
        """
        where_clauses = []
        params = []

        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            {where_sql}
            ORDER BY s.started_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
        sessions = []
        for row in rows:
            s = dict(row)
            # Build the preview from the raw substring
            raw = s.pop("_preview_raw", "").strip()
            if raw:
                text = raw[:60]
                s["preview"] = text + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            sessions.append(s)

        return sessions

    # =========================================================================
    # Message storage
    # =========================================================================

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
    ) -> int:
        """
        Append a message to a session. Returns the message row ID.

        Also increments the session's message_count (and tool_call_count
        if role is 'tool' or tool_calls is present).
        """
        # Serialize structured fields to JSON before entering the write txn
        reasoning_details_json = (
            json.dumps(reasoning_details)
            if reasoning_details else None
        )
        codex_items_json = (
            json.dumps(codex_reasoning_items)
            if codex_reasoning_items else None
        )
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        # Pre-compute tool call count
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, timestamp, token_count, finish_reason,
                   reasoning, reasoning_details, codex_reasoning_items)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    content,
                    tool_call_id,
                    tool_calls_json,
                    tool_name,
                    time.time(),
                    token_count,
                    finish_reason,
                    reasoning,
                    reasoning_details_json,
                    codex_items_json,
                ),
            )
            msg_id = cursor.lastrowid

            # Update counters
            if num_tool_calls > 0:
                conn.execute(
                    """UPDATE sessions SET message_count = message_count + 1,
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                    (num_tool_calls, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (session_id,),
                )
            return msg_id

        return self._execute_write(_do)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all messages for a session, ordered by timestamp."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (session_id,),
            )
            rows = cursor.fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(msg)
        return result

    def get_messages_as_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT role, content, tool_call_id, tool_calls, tool_name, "
                "reasoning, reasoning_details, codex_reasoning_items "
                "FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (session_id,),
            )
            rows = cursor.fetchall()
        messages = []
        for row in rows:
            msg = {"role": row["role"], "content": row["content"]}
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            # Restore reasoning fields on assistant messages so providers
            # that replay reasoning (OpenRouter, OpenAI, Nous) receive
            # coherent multi-turn reasoning context.
            if row["role"] == "assistant":
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        pass
            messages.append(msg)
        return messages

    # =========================================================================
    # Search
    # =========================================================================

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}`` and bare boolean operators (``AND``, ``OR``,
        ``NOT``) have special meaning.  Passing raw user input directly to
        MATCH can cause ``sqlite3.OperationalError``.

        Strategy:
        - Preserve properly paired quoted phrases (``"exact phrase"``)
        - Strip unmatched FTS5-special characters that would cause errors
        - Wrap unquoted hyphenated terms in quotes so FTS5 matches them
          as exact phrases instead of splitting on the hyphen
        """
        # Step 1: Extract balanced double-quoted phrases and protect them
        # from further processing via numbered placeholders.
        _quoted_parts: list = []

        def _preserve_quoted(m: re.Match) -> str:
            _quoted_parts.append(m.group(0))
            return f"\x00Q{len(_quoted_parts) - 1}\x00"

        sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)

        # Step 2: Strip remaining (unmatched) FTS5-special characters
        sanitized = re.sub(r'[+{}()\"^]', " ", sanitized)

        # Step 3: Collapse repeated * (e.g. "***") into a single one,
        # and remove leading * (prefix-only needs at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

        # Step 4: Remove dangling boolean operators at start/end that would
        # cause syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

        # Step 5: Wrap unquoted hyphenated terms (e.g. ``chat-send``) in
        # double quotes.  FTS5's tokenizer splits on hyphens, turning
        # ``chat-send`` into ``chat AND send``.  Quoting preserves the
        # intended phrase match.
        sanitized = re.sub(r"\b(\w+(?:-\w+)+)\b", r'"\1"', sanitized)

        # Step 6: Restore preserved quoted phrases
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

        return sanitized.strip()

    # =========================================================================
    # ML Pipeline Data Collection
    # =========================================================================

    def log_routing_decision(
        self,
        routed_model: str,
        message_text: str,
        routing_reason: str = None,
        session_id: str = None,
        message_char_count: int = None,
        message_word_count: int = None,
        has_code_block: bool = False,
        has_url: bool = False,
        complex_keyword_hit: str = None,
        conversation_depth: int = None,
        response_latency_ms: int = None,
        response_cost_usd: float = None,
    ) -> int:
        """Log a model routing decision for ML training data collection.

        Returns the row ID of the inserted record.
        """
        result_holder = []

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO routing_decisions
                   (session_id, message_text, message_char_count, message_word_count,
                    has_code_block, has_url, complex_keyword_hit, conversation_depth,
                    routed_model, routing_reason, response_latency_ms,
                    response_cost_usd, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    message_text,
                    message_char_count,
                    message_word_count,
                    1 if has_code_block else 0,
                    1 if has_url else 0,
                    complex_keyword_hit,
                    conversation_depth,
                    routed_model,
                    routing_reason,
                    response_latency_ms,
                    response_cost_usd,
                    time.time(),
                ),
            )
            result_holder.append(cursor.lastrowid)

        self._execute_write(_do)
        return result_holder[0] if result_holder else 0

    def update_routing_quality_score(
        self,
        routing_id: int,
        quality_score: float,
    ) -> None:
        """Backfill the quality score for a routing decision."""
        def _do(conn):
            conn.execute(
                "UPDATE routing_decisions SET quality_score = ? WHERE id = ?",
                (quality_score, routing_id),
            )
        self._execute_write(_do)

    def get_routing_decisions(
        self,
        limit: int = 100,
        offset: int = 0,
        model_filter: str = None,
        scored_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve routing decisions for ML training.

        Args:
            limit: Max rows to return.
            offset: Skip this many rows.
            model_filter: Filter by routed_model.
            scored_only: Only return rows with a quality_score.
        """
        with self._lock:
            clauses = []
            params = []
            if model_filter:
                clauses.append("routed_model = ?")
                params.append(model_filter)
            if scored_only:
                clauses.append("quality_score IS NOT NULL")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.extend([limit, offset])
            rows = self._conn.execute(
                f"SELECT * FROM routing_decisions {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    # =========================================================================
    # Implicit Signals (automatic quality signals for ML classifiers)
    # =========================================================================

    def log_implicit_signal(
        self,
        signal_type: str,
        signal_value: float,
        signal_source: str,
        context_id: Optional[str] = None,
        session_id: Optional[str] = None,
        message_text: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> int:
        """Record an implicit quality signal from conversation patterns.

        Signal types:
        - routing_quality: Was the cheap/primary routing decision correct?
        - reasoning_effort: Was the reasoning effort level appropriate?
        - skill_match: Was the matched skill actually used?

        Signal sources:
        - implicit_continuation: Conversation continued normally (positive signal)
        - implicit_correction: User corrected or re-asked (negative signal)
        - implicit_skip: Suggested resource was ignored (mild negative)

        Returns the row ID.
        """
        result_holder = []

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO implicit_signals
                   (signal_type, context_id, session_id, signal_value,
                    signal_source, message_text, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal_type,
                    context_id,
                    session_id,
                    signal_value,
                    signal_source,
                    (message_text or "")[:2000] or None,
                    metadata,
                    time.time(),
                ),
            )
            result_holder.append(cursor.lastrowid)

        self._execute_write(_do)
        return result_holder[0] if result_holder else 0

    def get_implicit_signals(
        self,
        signal_type: str = None,
        session_id: str = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Retrieve implicit signals, optionally filtered by type or session."""
        with self._lock:
            clauses = []
            params = []
            if signal_type:
                clauses.append("signal_type = ?")
                params.append(signal_type)
            if session_id:
                clauses.append("session_id = ?")
                params.append(session_id)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.append(limit)
            rows = self._conn.execute(
                f"SELECT * FROM implicit_signals {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def count_implicit_signals(self, signal_type: str = None) -> int:
        """Count accumulated signals, optionally by type."""
        with self._lock:
            if signal_type:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM implicit_signals WHERE signal_type = ?",
                    (signal_type,),
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM implicit_signals").fetchone()
            return row[0] if row else 0

    # =========================================================================
    # Provider Health Log (persists recovery data across restarts)
    # =========================================================================

    def log_provider_health_event(
        self,
        provider: str,
        event_type: str,
        latency_ms: float = None,
        error: str = None,
        health_score: float = None,
        consecutive_failures: int = None,
        recovery_time_seconds: float = None,
        session_id: str = None,
    ) -> None:
        """Log a provider health event (success or failure)."""
        def _do(conn):
            conn.execute(
                """INSERT INTO provider_health_log
                   (provider, event_type, latency_ms, error, health_score,
                    consecutive_failures, recovery_time_seconds, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    provider,
                    event_type,
                    latency_ms,
                    (error or "")[:500] or None,
                    health_score,
                    consecutive_failures,
                    recovery_time_seconds,
                    session_id,
                    time.time(),
                ),
            )
        try:
            self._execute_write(_do)
        except Exception:
            pass  # Best-effort — never disrupt the agent

    def get_provider_recovery_history(
        self,
        provider: str,
        limit: int = 20,
    ) -> List[float]:
        """Get historical recovery times for a provider (seconds).

        Returns a list of recovery_time_seconds values, most recent first.
        Used to pre-populate ProviderHealthTracker._recovery_times on startup.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT recovery_time_seconds FROM provider_health_log
                   WHERE provider = ? AND event_type = 'recovery'
                   AND recovery_time_seconds IS NOT NULL
                   ORDER BY created_at DESC LIMIT ?""",
                (provider, limit),
            ).fetchall()
            return [row[0] for row in rows]

    # =========================================================================
    # Action log (accountability log for side-effecting tool invocations)
    # =========================================================================

    def log_action(
        self,
        session_id: str = None,
        tool_name: str = "",
        tool_args: str = None,
        result_summary: str = None,
        mutates: bool = True,
        platform: str = None,
        approved_by: str = None,
        timestamp: float = None,
        duration_ms: int = None,
    ) -> None:
        """Append to the action_log table.  Fire-and-forget: errors are logged, not raised."""
        ts = timestamp or time.time()
        truncated_args = (tool_args or "")[:4096] or None
        truncated_result = (result_summary or "")[:512] or None
        try:
            def _do(conn):
                conn.execute(
                    """INSERT INTO action_log
                       (session_id, tool_name, tool_args, result_summary,
                        mutates, platform, approved_by, timestamp, duration_ms)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        tool_name,
                        truncated_args,
                        truncated_result,
                        1 if mutates else 0,
                        platform,
                        approved_by,
                        ts,
                        duration_ms,
                    ),
                )
            self._execute_write(_do)
        except Exception as e:
            logger.debug("action_log insert failed: %s", e)

    def get_action_log(
        self,
        session_id: str = None,
        tool_name: str = None,
        mutates_only: bool = True,
        limit: int = 100,
        since: float = None,
    ) -> List[Dict[str, Any]]:
        """Query action_log for self-evolution dataset building.

        Args:
            session_id: Filter by session.
            tool_name: Filter by tool.
            mutates_only: If True, only return side-effecting actions.
            limit: Max rows.
            since: Only actions after this unix timestamp.
        """
        with self._lock:
            clauses = []
            params = []
            if session_id:
                clauses.append("session_id = ?")
                params.append(session_id)
            if tool_name:
                clauses.append("tool_name = ?")
                params.append(tool_name)
            if mutates_only:
                clauses.append("mutates = 1")
            if since:
                clauses.append("timestamp > ?")
                params.append(since)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.append(limit)
            rows = self._conn.execute(
                f"SELECT * FROM action_log {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def log_cron_output_stats(
        self,
        job_id: str,
        output_path: str,
        char_count: int = None,
        line_count: int = None,
        word_count: int = None,
        entity_count: int = None,
        numeric_values: Dict[str, Any] = None,
        sentiment_score: float = None,
    ) -> int:
        """Log statistical summary of a cron job output for anomaly detection.

        Returns the row ID of the inserted record.
        """
        result_holder = []
        numeric_json = json.dumps(numeric_values) if numeric_values else None

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO cron_output_stats
                   (job_id, output_path, char_count, line_count, word_count,
                    entity_count, numeric_values, sentiment_score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    output_path,
                    char_count,
                    line_count,
                    word_count,
                    entity_count,
                    numeric_json,
                    sentiment_score,
                    time.time(),
                ),
            )
            result_holder.append(cursor.lastrowid)

        self._execute_write(_do)
        return result_holder[0] if result_holder else 0

    def get_cron_output_stats(
        self,
        job_id: str,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent output stats for a cron job (for anomaly detection baseline)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM cron_output_stats WHERE job_id = ? ORDER BY created_at DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # =========================================================================
    # Prediction Tracking
    # =========================================================================

    def log_prediction(
        self,
        job_id: str,
        prediction_type: str,
        subject: str,
        predicted_value: str,
        predicted_at: float = None,
        resolution_target_at: float = None,
        confidence: float = None,
        source_output_path: str = None,
    ) -> int:
        """Log a prediction extracted from a cron job output.

        Returns the row ID of the inserted record.
        """
        result_holder = []
        now = time.time()

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO predictions
                   (job_id, prediction_type, subject, predicted_value,
                    predicted_at, resolution_target_at, confidence,
                    source_output_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    prediction_type,
                    subject,
                    predicted_value,
                    predicted_at or now,
                    resolution_target_at,
                    confidence,
                    source_output_path,
                    now,
                ),
            )
            result_holder.append(cursor.lastrowid)

        self._execute_write(_do)
        return result_holder[0] if result_holder else 0

    def get_unresolved_predictions(
        self,
        before_timestamp: float = None,
    ) -> List[Dict[str, Any]]:
        """Get predictions that haven't been resolved yet."""
        with self._lock:
            if before_timestamp:
                rows = self._conn.execute(
                    "SELECT * FROM predictions WHERE resolved_at IS NULL "
                    "AND (resolution_target_at IS NULL OR resolution_target_at <= ?) "
                    "ORDER BY predicted_at",
                    (before_timestamp,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM predictions WHERE resolved_at IS NULL ORDER BY predicted_at",
                ).fetchall()
            return [dict(r) for r in rows]

    def resolve_prediction(
        self,
        prediction_id: int,
        actual_value: str,
        correct: bool,
    ) -> None:
        """Mark a prediction as resolved with its actual outcome."""
        def _do(conn):
            conn.execute(
                "UPDATE predictions SET actual_value = ?, resolved_at = ?, correct = ? WHERE id = ?",
                (actual_value, time.time(), 1 if correct else 0, prediction_id),
            )
        self._execute_write(_do)

    def get_prediction_accuracy(
        self,
        subject: str = None,
        prediction_type: str = None,
        window_days: int = 30,
    ) -> Dict[str, Any]:
        """Get prediction accuracy stats over a time window.

        Returns dict with total, correct, accuracy, by_type breakdown.
        """
        with self._lock:
            cutoff = time.time() - (window_days * 86400)
            clauses = ["resolved_at IS NOT NULL", "created_at >= ?"]
            params: list = [cutoff]

            if subject:
                clauses.append("subject = ?")
                params.append(subject)
            if prediction_type:
                clauses.append("prediction_type = ?")
                params.append(prediction_type)

            where = " AND ".join(clauses)

            row = self._conn.execute(
                f"SELECT COUNT(*) as total, SUM(correct) as correct_count "
                f"FROM predictions WHERE {where}",
                params,
            ).fetchone()

            total = row[0] if row else 0
            correct_count = row[1] if row and row[1] else 0
            accuracy = correct_count / total if total > 0 else None

            return {
                "total": total,
                "correct": correct_count,
                "accuracy": accuracy,
                "window_days": window_days,
            }

    # =========================================================================
    # Macro Metrics History (Time-Series)
    # =========================================================================

    def log_macro_metric(
        self,
        metric_id: str,
        value: float,
        category: str = None,
        unit: str = None,
        source: str = None,
        timestamp: float = None,
        metadata: Dict[str, Any] = None,
    ) -> int:
        """Log a numeric macro metric for historical charting."""
        result_holder = []
        now = time.time()
        meta_json = json.dumps(metadata) if metadata else None

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO macro_metrics_history
                   (metric_id, category, value, unit, source, timestamp, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    metric_id,
                    category,
                    value,
                    unit,
                    source,
                    timestamp or now,
                    meta_json,
                    now,
                ),
            )
            result_holder.append(cursor.lastrowid)

        self._execute_write(_do)
        return result_holder[0] if result_holder else 0

    def get_macro_history(
        self,
        metric_id: str,
        limit: int = 100,
        days: int = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical time-series for a specific macro metric."""
        with self._lock:
            query = "SELECT * FROM macro_metrics_history WHERE metric_id = ?"
            params: list = [metric_id]

            if days:
                cutoff = time.time() - (days * 86400)
                query += " AND timestamp >= ?"
                params.append(cutoff)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = self._conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_macro_latest(
        self,
        category: str = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Get the latest value for each metric, optionally filtered by category."""
        with self._lock:
            query = """
                SELECT m1.* FROM macro_metrics_history m1
                JOIN (
                    SELECT metric_id, MAX(timestamp) as max_ts
                    FROM macro_metrics_history
                    GROUP BY metric_id
                ) m2 ON m1.metric_id = m2.metric_id AND m1.timestamp = m2.max_ts
            """
            params = []
            if category:
                query += " WHERE m1.category = ?"
                params.append(category)

            rows = self._conn.execute(query, params).fetchall()
            return {r["metric_id"]: dict(r) for r in rows}

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        owner_fingerprint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across session messages using FTS5.

        Supports FTS5 query syntax:
          - Simple keywords: "docker deployment"
          - Phrases: '"exact phrase"'
          - Boolean: "docker OR kubernetes", "python NOT java"
          - Prefix: "deploy*"

        Returns matching messages with session metadata, content snippet,
        and surrounding context (1 message before and after the match).

        ``owner_fingerprint``: when set, results are restricted to sessions
        owned by that fingerprint (matches F-003 gateway isolation). Pass
        ``None`` (default) for admin/CLI scope — sees all sessions.
        """
        if not query or not query.strip():
            return []

        query = self._sanitize_fts5_query(query)
        if not query:
            return []

        # Build WHERE clauses dynamically
        where_clauses = ["messages_fts MATCH ?"]
        params: list = [query]

        if source_filter is not None:
            source_placeholders = ",".join("?" for _ in source_filter)
            where_clauses.append(f"s.source IN ({source_placeholders})")
            params.extend(source_filter)

        if exclude_sources is not None:
            exclude_placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({exclude_placeholders})")
            params.extend(exclude_sources)

        if role_filter:
            role_placeholders = ",".join("?" for _ in role_filter)
            where_clauses.append(f"m.role IN ({role_placeholders})")
            params.extend(role_filter)

        if owner_fingerprint is not None:
            # Match sessions explicitly owned by this fingerprint. Legacy rows
            # with NULL owner_fingerprint are treated as "unbound" and EXCLUDED
            # here: F-003 binds them on first authenticated use, not on read.
            where_clauses.append("s.owner_fingerprint = ?")
            params.append(owner_fingerprint)

        where_sql = " AND ".join(where_clauses)
        params.extend([limit, offset])

        sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                m.content,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {where_sql}
            ORDER BY rank
            LIMIT ? OFFSET ?
        """

        with self._lock:
            try:
                cursor = self._conn.execute(sql, params)
            except sqlite3.OperationalError:
                # FTS5 query syntax error despite sanitization — return empty
                return []
            matches = [dict(row) for row in cursor.fetchall()]

        # Add surrounding context (1 message before + after each match).
        # Done outside the lock so we don't hold it across N sequential queries.
        for match in matches:
            try:
                with self._lock:
                    ctx_cursor = self._conn.execute(
                        """SELECT role, content FROM messages
                           WHERE session_id = ? AND id >= ? - 1 AND id <= ? + 1
                           ORDER BY id""",
                        (match["session_id"], match["id"], match["id"]),
                    )
                    context_msgs = [
                        {"role": r["role"], "content": (r["content"] or "")[:200]}
                        for r in ctx_cursor.fetchall()
                    ]
                match["context"] = context_msgs
            except Exception:
                match["context"] = []

        # Remove full content from result (snippet is enough, saves tokens)
        for match in matches:
            match.pop("content", None)

        return matches

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
        owner_fingerprint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source and/or owner fingerprint.

        ``owner_fingerprint``: when set, only sessions owned by that fingerprint
        are returned (F-003 isolation). ``None`` (default) = admin scope.
        """
        with self._lock:
            where: list[str] = []
            params: list = []
            if source:
                where.append("source = ?")
                params.append(source)
            if owner_fingerprint is not None:
                where.append("owner_fingerprint = ?")
                params.append(owner_fingerprint)
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""
            params.extend([limit, offset])
            cursor = self._conn.execute(
                f"SELECT * FROM sessions{where_sql} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Utility
    # =========================================================================

    def session_count(self, source: str = None) -> int:
        """Count sessions, optionally filtered by source."""
        with self._lock:
            if source:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE source = ?", (source,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM sessions")
            return cursor.fetchone()[0]

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]

    # =========================================================================
    # Export and cleanup
    # =========================================================================

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, "messages": messages}

    def export_all(
        self,
        source: str = None,
        owner_fingerprint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.

        ``owner_fingerprint``: when set, only sessions owned by that fingerprint
        are exported (F-003 isolation). ``None`` (default) = admin scope.
        """
        sessions = self.search_sessions(
            source=source, limit=100000, owner_fingerprint=owner_fingerprint,
        )
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        def _do(conn):
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages. Returns True if found."""
        def _do(conn):
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
            )
            if cursor.fetchone()[0] == 0:
                return False
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True
        return self._execute_write(_do)

    def prune_sessions(self, older_than_days: int = 90, source: str = None) -> int:
        """
        Delete sessions older than N days. Returns count of deleted sessions.
        Only prunes ended sessions (not active ones).
        """
        cutoff = time.time() - (older_than_days * 86400)

        def _do(conn):
            if source:
                cursor = conn.execute(
                    """SELECT id FROM sessions
                       WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?""",
                    (cutoff, source),
                )
            else:
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE started_at < ? AND ended_at IS NOT NULL",
                    (cutoff,),
                )
            session_ids = [row["id"] for row in cursor.fetchall()]

            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            return len(session_ids)

        return self._execute_write(_do)

    # =========================================================================
    # Knowledge Store — structured notes, people, projects, decisions with tags
    # =========================================================================

    def save_knowledge_note(
        self, content: str, tags: List[str] = None,
        source: str = "conversation", session_id: str = None,
        uuid: str = None, file_path: str = None,
    ) -> int:
        """Save or update a knowledge note and attach tags. Returns the note ID."""
        now = time.time()

        def _do(conn):
            # Try to match by UUID first
            row = None
            if uuid:
                row = conn.execute(
                    "SELECT id FROM knowledge_notes WHERE uuid = ?", (uuid,)
                ).fetchone()

            if row:
                note_id = row[0]
                updates = ["content = ?", "updated_at = ?"]
                params = [content, now]
                if source:
                    updates.append("source = ?")
                    params.append(source)
                if session_id:
                    updates.append("session_id = ?")
                    params.append(session_id)
                if file_path:
                    updates.append("file_path = ?")
                    params.append(file_path)
                
                params.append(note_id)
                conn.execute(
                    f"UPDATE knowledge_notes SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
            else:
                cursor = conn.execute(
                    "INSERT INTO knowledge_notes (uuid, content, source, session_id, file_path, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (uuid, content, source, session_id, file_path, now, now),
                )
                note_id = cursor.lastrowid
            
            if tags:
                for tag in tags:
                    normalized = tag.strip().lower()
                    if normalized:
                        # Skip if tag already exists for this note
                        existing = conn.execute(
                            "SELECT 1 FROM knowledge_tags WHERE entity_type = 'note' "
                            "AND entity_id = ? AND tag = ?",
                            (note_id, normalized),
                        ).fetchone()
                        if not existing:
                            conn.execute(
                                "INSERT INTO knowledge_tags (entity_type, entity_id, tag, created_at) "
                                "VALUES ('note', ?, ?, ?)",
                                (note_id, normalized, now),
                            )
            return note_id

        return self._execute_write(_do)

    def save_knowledge_person(
        self, name: str, role: str = None, organization: str = None,
        details: str = None, tags: List[str] = None,
        uuid: str = None, file_path: str = None,
    ) -> int:
        """Save or update a person entry. Returns the person ID."""
        now = time.time()

        def _do(conn):
            # Try to match by UUID first, then name
            row = None
            if uuid:
                row = conn.execute(
                    "SELECT id FROM knowledge_people WHERE uuid = ?", (uuid,)
                ).fetchone()
            
            if not row:
                row = conn.execute(
                    "SELECT id FROM knowledge_people WHERE LOWER(name) = LOWER(?)", (name,)
                ).fetchone()

            if row:
                person_id = row[0]
                # Update existing — only overwrite non-None fields
                updates = []
                params = []
                if uuid is not None:
                    updates.append("uuid = ?")
                    params.append(uuid)
                if role is not None:
                    updates.append("role = ?")
                    params.append(role)
                if organization is not None:
                    updates.append("organization = ?")
                    params.append(organization)
                if details is not None:
                    updates.append("details = ?")
                    params.append(details)
                if file_path is not None:
                    updates.append("file_path = ?")
                    params.append(file_path)
                updates.append("updated_at = ?")
                params.append(now)
                params.append(person_id)
                conn.execute(
                    f"UPDATE knowledge_people SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
            else:
                cursor = conn.execute(
                    "INSERT INTO knowledge_people (uuid, name, role, organization, details, file_path, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (uuid, name, role, organization, details, file_path, now, now),
                )
                person_id = cursor.lastrowid
            # Add any new tags (skip duplicates)
            if tags:
                for tag in tags:
                    normalized = tag.strip().lower()
                    if not normalized:
                        continue
                    existing = conn.execute(
                        "SELECT 1 FROM knowledge_tags WHERE entity_type = 'person' "
                        "AND entity_id = ? AND tag = ?",
                        (person_id, normalized),
                    ).fetchone()
                    if not existing:
                        conn.execute(
                            "INSERT INTO knowledge_tags (entity_type, entity_id, tag, created_at) "
                            "VALUES ('person', ?, ?, ?)",
                            (person_id, normalized, now),
                        )
            return person_id

        return self._execute_write(_do)

    def save_knowledge_project(
        self, name: str, description: str = None,
        status: str = "active", tags: List[str] = None,
        uuid: str = None, file_path: str = None,
    ) -> int:
        """Save or update a project entry. Returns the project ID."""
        now = time.time()

        def _do(conn):
            # Try to match by UUID first, then name
            row = None
            if uuid:
                row = conn.execute(
                    "SELECT id FROM knowledge_projects WHERE uuid = ?", (uuid,)
                ).fetchone()
                
            if not row:
                row = conn.execute(
                    "SELECT id FROM knowledge_projects WHERE LOWER(name) = LOWER(?)", (name,)
                ).fetchone()

            if row:
                project_id = row[0]
                updates = []
                params = []
                if uuid is not None:
                    updates.append("uuid = ?")
                    params.append(uuid)
                if description is not None:
                    updates.append("description = ?")
                    params.append(description)
                if status is not None:
                    updates.append("status = ?")
                    params.append(status)
                if file_path is not None:
                    updates.append("file_path = ?")
                    params.append(file_path)
                updates.append("updated_at = ?")
                params.append(now)
                params.append(project_id)
                conn.execute(
                    f"UPDATE knowledge_projects SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
            else:
                cursor = conn.execute(
                    "INSERT INTO knowledge_projects (uuid, name, status, description, file_path, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (uuid, name, status, description, file_path, now, now),
                )
                project_id = cursor.lastrowid
            if tags:
                for tag in tags:
                    normalized = tag.strip().lower()
                    if not normalized:
                        continue
                    existing = conn.execute(
                        "SELECT 1 FROM knowledge_tags WHERE entity_type = 'project' "
                        "AND entity_id = ? AND tag = ?",
                        (project_id, normalized),
                    ).fetchone()
                    if not existing:
                        conn.execute(
                            "INSERT INTO knowledge_tags (entity_type, entity_id, tag, created_at) "
                            "VALUES ('project', ?, ?, ?)",
                            (project_id, normalized, now),
                        )
            return project_id

        return self._execute_write(_do)

    def save_knowledge_decision(
        self, title: str, rationale: str = None,
        status: str = "active", tags: List[str] = None,
        uuid: str = None, file_path: str = None,
    ) -> int:
        """Save or update a decision. Returns the decision ID."""
        now = time.time()

        def _do(conn):
            # Try to match by UUID first, then title
            row = None
            if uuid:
                row = conn.execute(
                    "SELECT id FROM knowledge_decisions WHERE uuid = ?", (uuid,)
                ).fetchone()
            
            if not row:
                row = conn.execute(
                    "SELECT id FROM knowledge_decisions WHERE LOWER(title) = LOWER(?)", (title,)
                ).fetchone()

            if row:
                decision_id = row[0]
                updates = ["updated_at = ?"]
                params = [now]
                if uuid:
                    updates.append("uuid = ?")
                    params.append(uuid)
                if title:
                    updates.append("title = ?")
                    params.append(title)
                if rationale is not None:
                    updates.append("rationale = ?")
                    params.append(rationale)
                if status:
                    updates.append("status = ?")
                    params.append(status)
                if file_path:
                    updates.append("file_path = ?")
                    params.append(file_path)
                
                params.append(decision_id)
                conn.execute(
                    f"UPDATE knowledge_decisions SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
            else:
                cursor = conn.execute(
                    "INSERT INTO knowledge_decisions (uuid, title, rationale, status, file_path, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (uuid, title, rationale, status, file_path, now, now),
                )
                decision_id = cursor.lastrowid
            
            if tags:
                for tag in tags:
                    normalized = tag.strip().lower()
                    if not normalized:
                        continue
                    existing = conn.execute(
                        "SELECT 1 FROM knowledge_tags WHERE entity_type = 'decision' "
                        "AND entity_id = ? AND tag = ?",
                        (decision_id, normalized),
                    ).fetchone()
                    if not existing:
                        conn.execute(
                            "INSERT INTO knowledge_tags (entity_type, entity_id, tag, created_at) "
                            "VALUES ('decision', ?, ?, ?)",
                            (decision_id, normalized, now),
                        )
            return decision_id

        return self._execute_write(_do)


    _KNOWLEDGE_ENTITY_TYPES = {"note", "person", "project", "decision"}

    def search_knowledge(
        self, query: str = None, entity_type: str = None,
        tag: str = None, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search across all knowledge tables by tag and/or text query.

        Both query and tag can be combined — results must match both filters.
        Returns a list of dicts with 'type', 'id', 'summary', 'tags', 'created_at'.
        """
        limit = min(int(limit), 100)  # Cap to prevent unbounded result sets
        results = []
        with self._lock:
            conn = self._conn
            tag_lower = tag.strip().lower() if tag else None

            # Helper: get tags for an entity
            def _get_tags(etype, eid):
                rows = conn.execute(
                    "SELECT tag FROM knowledge_tags WHERE entity_type = ? AND entity_id = ?",
                    (etype, eid),
                ).fetchall()
                return [r[0] for r in rows]

            # If tag is given, find matching entity IDs first
            tag_entity_ids = {}
            if tag_lower:
                tag_rows = conn.execute(
                    "SELECT entity_type, entity_id FROM knowledge_tags WHERE tag = ?",
                    (tag_lower,),
                ).fetchall()
                for r in tag_rows:
                    tag_entity_ids.setdefault(r[0], set()).add(r[1])

            types_to_search = [entity_type] if entity_type else ["note", "person", "project", "decision"]
            like_query = f"%{query}%" if query else None

            for etype in types_to_search:
                if etype == "note":
                    if tag_lower and query:
                        # Combined: filter by tag IDs AND text match
                        ids = tag_entity_ids.get("note", set())
                        if not ids:
                            continue
                        placeholders = ",".join("?" for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, content, source, created_at FROM knowledge_notes "
                            f"WHERE id IN ({placeholders}) AND content LIKE ? "
                            f"ORDER BY created_at DESC LIMIT ?",
                            (*ids, like_query, limit),
                        ).fetchall()
                    elif query:
                        # Text search only — try FTS5, fall back to LIKE
                        try:
                            rows = conn.execute(
                                "SELECT n.id, n.content, n.source, n.created_at "
                                "FROM knowledge_notes n "
                                "JOIN knowledge_notes_fts f ON n.id = f.rowid "
                                "WHERE knowledge_notes_fts MATCH ? "
                                "ORDER BY rank LIMIT ?",
                                (query, limit),
                            ).fetchall()
                        except sqlite3.OperationalError as e:
                            logger.debug("FTS5 query failed, falling back to LIKE: %s", e)
                            rows = conn.execute(
                                "SELECT id, content, source, created_at FROM knowledge_notes "
                                "WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                                (like_query, limit),
                            ).fetchall()
                    elif tag_lower:
                        ids = tag_entity_ids.get("note", set())
                        if not ids:
                            continue
                        placeholders = ",".join("?" for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, content, source, created_at FROM knowledge_notes "
                            f"WHERE id IN ({placeholders}) ORDER BY created_at DESC LIMIT ?",
                            (*ids, limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT id, content, source, created_at FROM knowledge_notes "
                            "ORDER BY created_at DESC LIMIT ?",
                            (limit,),
                        ).fetchall()
                    for r in rows:
                        content_preview = r[1][:200] if r[1] else ""
                        results.append({
                            "type": "note", "id": r[0],
                            "summary": content_preview,
                            "source": r[2],
                            "tags": _get_tags("note", r[0]),
                            "created_at": r[3],
                        })

                elif etype == "person":
                    if tag_lower and query:
                        ids = tag_entity_ids.get("person", set())
                        if not ids:
                            continue
                        placeholders = ",".join("?" for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, name, role, organization, details, created_at "
                            f"FROM knowledge_people WHERE id IN ({placeholders}) "
                            f"AND (name LIKE ? OR organization LIKE ?) LIMIT ?",
                            (*ids, like_query, like_query, limit),
                        ).fetchall()
                    elif tag_lower:
                        ids = tag_entity_ids.get("person", set())
                        if not ids:
                            continue
                        placeholders = ",".join("?" for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, name, role, organization, details, created_at "
                            f"FROM knowledge_people WHERE id IN ({placeholders}) LIMIT ?",
                            (*ids, limit),
                        ).fetchall()
                    elif query:
                        rows = conn.execute(
                            "SELECT id, name, role, organization, details, created_at "
                            "FROM knowledge_people WHERE name LIKE ? OR organization LIKE ? "
                            "ORDER BY updated_at DESC LIMIT ?",
                            (like_query, like_query, limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT id, name, role, organization, details, created_at "
                            "FROM knowledge_people ORDER BY updated_at DESC LIMIT ?",
                            (limit,),
                        ).fetchall()
                    for r in rows:
                        summary_parts = [r[1]]
                        if r[2]:
                            summary_parts.append(r[2])
                        if r[3]:
                            summary_parts.append(f"@ {r[3]}")
                        results.append({
                            "type": "person", "id": r[0],
                            "summary": " — ".join(summary_parts),
                            "name": r[1], "role": r[2],
                            "organization": r[3], "details": r[4],
                            "tags": _get_tags("person", r[0]),
                            "created_at": r[5],
                        })

                elif etype == "project":
                    if tag_lower and query:
                        ids = tag_entity_ids.get("project", set())
                        if not ids:
                            continue
                        placeholders = ",".join("?" for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, name, status, description, created_at "
                            f"FROM knowledge_projects WHERE id IN ({placeholders}) "
                            f"AND (name LIKE ? OR description LIKE ?) LIMIT ?",
                            (*ids, like_query, like_query, limit),
                        ).fetchall()
                    elif tag_lower:
                        ids = tag_entity_ids.get("project", set())
                        if not ids:
                            continue
                        placeholders = ",".join("?" for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, name, status, description, created_at "
                            f"FROM knowledge_projects WHERE id IN ({placeholders}) LIMIT ?",
                            (*ids, limit),
                        ).fetchall()
                    elif query:
                        rows = conn.execute(
                            "SELECT id, name, status, description, created_at "
                            "FROM knowledge_projects WHERE name LIKE ? OR description LIKE ? "
                            "ORDER BY updated_at DESC LIMIT ?",
                            (like_query, like_query, limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT id, name, status, description, created_at "
                            "FROM knowledge_projects ORDER BY updated_at DESC LIMIT ?",
                            (limit,),
                        ).fetchall()
                    for r in rows:
                        results.append({
                            "type": "project", "id": r[0],
                            "summary": f"{r[1]} [{r[2]}]",
                            "name": r[1], "status": r[2],
                            "description": r[3],
                            "tags": _get_tags("project", r[0]),
                            "created_at": r[4],
                        })

                elif etype == "decision":
                    if tag_lower and query:
                        ids = tag_entity_ids.get("decision", set())
                        if not ids:
                            continue
                        placeholders = ",".join("?" for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, title, rationale, status, created_at "
                            f"FROM knowledge_decisions WHERE id IN ({placeholders}) "
                            f"AND (title LIKE ? OR rationale LIKE ?) LIMIT ?",
                            (*ids, like_query, like_query, limit),
                        ).fetchall()
                    elif tag_lower:
                        ids = tag_entity_ids.get("decision", set())
                        if not ids:
                            continue
                        placeholders = ",".join("?" for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, title, rationale, status, created_at "
                            f"FROM knowledge_decisions WHERE id IN ({placeholders}) LIMIT ?",
                            (*ids, limit),
                        ).fetchall()
                    elif query:
                        rows = conn.execute(
                            "SELECT id, title, rationale, status, created_at "
                            "FROM knowledge_decisions WHERE title LIKE ? OR rationale LIKE ? "
                            "ORDER BY created_at DESC LIMIT ?",
                            (like_query, like_query, limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT id, title, rationale, status, created_at "
                            "FROM knowledge_decisions ORDER BY created_at DESC LIMIT ?",
                            (limit,),
                        ).fetchall()
                    for r in rows:
                        results.append({
                            "type": "decision", "id": r[0],
                            "summary": f"{r[1]} [{r[3]}]",
                            "title": r[1], "rationale": r[2],
                            "status": r[3],
                            "tags": _get_tags("decision", r[0]),
                            "created_at": r[4],
                        })

        return results[:limit]

    def list_knowledge(
        self, entity_type: str, tag: str = None,
        status: str = None, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List knowledge entries of a specific type."""
        return self.search_knowledge(
            entity_type=entity_type, tag=tag, limit=limit,
        )

    def tag_entity(self, entity_type: str, entity_id: int, tags: List[str]) -> None:
        """Add tags to an existing entity (skips duplicates)."""
        if entity_type not in self._KNOWLEDGE_ENTITY_TYPES:
            raise ValueError(f"Invalid entity_type: {entity_type}")
        now = time.time()

        def _do(conn):
            for tag in tags:
                normalized = tag.strip().lower()
                if not normalized:
                    continue
                existing = conn.execute(
                    "SELECT 1 FROM knowledge_tags WHERE entity_type = ? "
                    "AND entity_id = ? AND tag = ?",
                    (entity_type, entity_id, normalized),
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO knowledge_tags (entity_type, entity_id, tag, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (entity_type, entity_id, normalized, now),
                    )

        self._execute_write(_do)
