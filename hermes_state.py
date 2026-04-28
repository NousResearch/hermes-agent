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
import random
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"


def _date_to_timestamp_range(date: str):
    """Return (start_ts, end_ts) Unix timestamps for a UTC calendar day.

    ``date`` must be a string in YYYY-MM-DD format.  Returns a tuple of
    floats representing the start (00:00:00 UTC) and end (00:00:00 UTC next
    day) of the requested day.
    """
    dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ts = dt.timestamp()
    end_ts = start_ts + 86400.0
    return start_ts, end_ts

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

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    agent_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    title TEXT,
    kind TEXT DEFAULT 'command',
    details TEXT,
    command TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    resolved_by TEXT,
    choice TEXT
);

CREATE INDEX IF NOT EXISTS idx_approvals_session ON approvals(session_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
CREATE INDEX IF NOT EXISTS idx_approvals_created ON approvals(created_at DESC);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    tool_call_id TEXT,
    tool_name TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL,
    diff TEXT DEFAULT '',
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    timestamp REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    code_session_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_artifacts_session_id ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_timestamp ON artifacts(timestamp);
CREATE INDEX IF NOT EXISTS idx_artifacts_code_session_id ON artifacts(code_session_id);

CREATE TABLE IF NOT EXISTS code_workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    repo_url TEXT,
    is_git_repo INTEGER DEFAULT 0,
    branch TEXT,
    detected_stack_json TEXT DEFAULT '[]',
    package_manager TEXT,
    commands_json TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_workspaces_path ON code_workspaces(path);
CREATE INDEX IF NOT EXISTS idx_code_workspaces_updated_at ON code_workspaces(updated_at DESC);

CREATE TABLE IF NOT EXISTS code_sessions (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    hermes_session_id TEXT,
    task_id TEXT,
    title TEXT,
    provider TEXT,
    model TEXT,
    branch TEXT,
    status TEXT NOT NULL DEFAULT 'planning',
    summary TEXT,
    metadata_json TEXT DEFAULT '{}',
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_sessions_workspace_id ON code_sessions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_code_sessions_hermes_session_id ON code_sessions(hermes_session_id);
CREATE INDEX IF NOT EXISTS idx_code_sessions_task_id ON code_sessions(task_id);
CREATE INDEX IF NOT EXISTS idx_code_sessions_status ON code_sessions(status);
CREATE INDEX IF NOT EXISTS idx_code_sessions_updated_at ON code_sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS code_session_events (
    id TEXT PRIMARY KEY,
    code_session_id TEXT NOT NULL,
    type TEXT NOT NULL,
    message TEXT,
    payload_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_session_events_session_id ON code_session_events(code_session_id);
CREATE INDEX IF NOT EXISTS idx_code_session_events_created_at ON code_session_events(created_at DESC);

CREATE TABLE IF NOT EXISTS code_commands (
    id TEXT PRIMARY KEY,
    code_session_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    command TEXT NOT NULL,
    argv_json TEXT DEFAULT '[]',
    cwd TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    safety TEXT NOT NULL DEFAULT 'safe',
    stdout TEXT DEFAULT '',
    stderr TEXT DEFAULT '',
    exit_code INTEGER,
    pid INTEGER,
    timeout_seconds INTEGER DEFAULT 120,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_commands_code_session_id ON code_commands(code_session_id);
CREATE INDEX IF NOT EXISTS idx_code_commands_workspace_id ON code_commands(workspace_id);
CREATE INDEX IF NOT EXISTS idx_code_commands_status ON code_commands(status);
CREATE INDEX IF NOT EXISTS idx_code_commands_created_at ON code_commands(created_at DESC);

CREATE TABLE IF NOT EXISTS code_git_snapshots (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    code_session_id TEXT,
    branch TEXT,
    remote_url TEXT,
    dirty INTEGER DEFAULT 0,
    summary_json TEXT DEFAULT '{}',
    files_json TEXT DEFAULT '[]',
    diff_stat TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_workspace_id ON code_git_snapshots(workspace_id);
CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_code_session_id ON code_git_snapshots(code_session_id);
CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_created_at ON code_git_snapshots(created_at DESC);

CREATE TABLE IF NOT EXISTS code_session_model_presets (
    id TEXT PRIMARY KEY,
    code_session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_session_model_presets_session_id ON code_session_model_presets(code_session_id);
CREATE INDEX IF NOT EXISTS idx_code_session_model_presets_name ON code_session_model_presets(code_session_id, name);

CREATE TABLE IF NOT EXISTS code_session_cost_entries (
    id TEXT PRIMARY KEY,
    code_session_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    task_type TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_session_id ON code_session_cost_entries(code_session_id);
CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_provider ON code_session_cost_entries(provider);
CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_model ON code_session_cost_entries(model);
CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_created_at ON code_session_cost_entries(created_at DESC);

CREATE TABLE IF NOT EXISTS code_diagnostics (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    code_session_id TEXT,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    diagnostics_json TEXT DEFAULT '[]',
    summary_json TEXT DEFAULT '{}',
    commands_json TEXT DEFAULT '[]',
    duration_ms INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_code_diagnostics_workspace_id ON code_diagnostics(workspace_id);
CREATE INDEX IF NOT EXISTS idx_code_diagnostics_code_session_id ON code_diagnostics(code_session_id);
CREATE INDEX IF NOT EXISTS idx_code_diagnostics_created_at ON code_diagnostics(created_at DESC);

CREATE TABLE IF NOT EXISTS code_agent_flows (
    id TEXT PRIMARY KEY,
    code_session_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    task_id TEXT,
    title TEXT,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'created',
    current_role TEXT,
    provider TEXT,
    model TEXT,
    preset TEXT,
    plan_json TEXT DEFAULT '{}',
    review_json TEXT,
    approval_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_code_agent_flows_code_session_id ON code_agent_flows(code_session_id);
CREATE INDEX IF NOT EXISTS idx_code_agent_flows_workspace_id ON code_agent_flows(workspace_id);
CREATE INDEX IF NOT EXISTS idx_code_agent_flows_status ON code_agent_flows(status);
CREATE INDEX IF NOT EXISTS idx_code_agent_flows_created_at ON code_agent_flows(created_at DESC);

CREATE TABLE IF NOT EXISTS code_agent_flow_steps (
    id TEXT PRIMARY KEY,
    flow_id TEXT NOT NULL,
    role TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    input_json TEXT DEFAULT '{}',
    output_json TEXT DEFAULT '{}',
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_code_agent_flow_steps_flow_id ON code_agent_flow_steps(flow_id);
CREATE INDEX IF NOT EXISTS idx_code_agent_flow_steps_status ON code_agent_flow_steps(status);

CREATE TABLE IF NOT EXISTS code_skill_runs (
    id TEXT PRIMARY KEY,
    skill_name TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    code_session_id TEXT,
    task_id TEXT,
    agent_flow_id TEXT,
    status TEXT NOT NULL DEFAULT 'created',
    input_json TEXT DEFAULT '{}',
    output_json TEXT DEFAULT '{}',
    summary TEXT,
    diagnostics_before_json TEXT,
    diagnostics_after_json TEXT,
    commands_json TEXT DEFAULT '[]',
    artifacts_json TEXT DEFAULT '[]',
    approval_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_code_skill_runs_workspace_id ON code_skill_runs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_code_skill_runs_code_session_id ON code_skill_runs(code_session_id);
CREATE INDEX IF NOT EXISTS idx_code_skill_runs_skill_name ON code_skill_runs(skill_name);
CREATE INDEX IF NOT EXISTS idx_code_skill_runs_status ON code_skill_runs(status);
CREATE INDEX IF NOT EXISTS idx_code_skill_runs_created_at ON code_skill_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'todo',
    priority TEXT NOT NULL DEFAULT 'medium',
    agent_id TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    run_id TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);

CREATE TABLE IF NOT EXISTS ledger_artifacts (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL DEFAULT '',
    format TEXT NOT NULL DEFAULT 'markdown',
    workspace_id TEXT,
    code_session_id TEXT,
    flow_id TEXT,
    command_id TEXT,
    orchestrated_run_id TEXT,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_code_session_id ON ledger_artifacts(code_session_id);
CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_workspace_id ON ledger_artifacts(workspace_id);
CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_category ON ledger_artifacts(category);
CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_created_at ON ledger_artifacts(created_at DESC);

CREATE TABLE IF NOT EXISTS orchestrated_runs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT,
    code_session_id TEXT,
    title TEXT,
    task_description TEXT,
    state TEXT NOT NULL DEFAULT 'intake',
    branch TEXT,
    worktree_path TEXT,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_orch_runs_workspace ON orchestrated_runs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_orch_runs_session ON orchestrated_runs(code_session_id);
CREATE INDEX IF NOT EXISTS idx_orch_runs_state ON orchestrated_runs(state);

CREATE TABLE IF NOT EXISTS orchestrated_run_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    type TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    message TEXT,
    payload_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orch_run_events_run_id ON orchestrated_run_events(run_id);
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
"""


class SessionDB:
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
    _WRITE_RETRY_MIN_S = 0.020  # 20ms
    _WRITE_RETRY_MAX_S = 0.150  # 150ms
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
                result = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if result and result[1] > 0:
                    logger.debug(
                        "WAL checkpoint: %d/%d pages checkpointed",
                        result[2],
                        result[1],
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

        # Create schema_version first so we can detect fresh vs existing DB
        # before running executescript (which fails on existing DBs that have
        # older table layouts missing columns referenced in SCHEMA_SQL indexes).
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        self._conn.commit()

        # Check schema version and run migrations
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            # Fresh database — safe to run full SCHEMA_SQL (all tables are new).
            cursor.executescript(SCHEMA_SQL)
            cursor.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
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
                        cursor.execute(
                            f'ALTER TABLE sessions ADD COLUMN "{safe_name}" {column_type}'
                        )
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
                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS approvals ("
                    "id TEXT PRIMARY KEY,"
                    "session_id TEXT,"
                    "agent_id TEXT,"
                    "status TEXT NOT NULL DEFAULT 'pending',"
                    "title TEXT,"
                    "kind TEXT DEFAULT 'command',"
                    "details TEXT,"
                    "command TEXT,"
                    "created_at TEXT NOT NULL,"
                    "updated_at TEXT NOT NULL,"
                    "resolved_at TEXT,"
                    "resolved_by TEXT,"
                    "choice TEXT"
                    ")"
                )
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_approvals_session ON approvals(session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status)",
                    "CREATE INDEX IF NOT EXISTS idx_approvals_created ON approvals(created_at DESC)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 7")
            if current_version < 8:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS artifacts (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        tool_call_id TEXT,
                        tool_name TEXT NOT NULL,
                        path TEXT NOT NULL,
                        status TEXT NOT NULL,
                        diff TEXT DEFAULT '',
                        additions INTEGER DEFAULT 0,
                        deletions INTEGER DEFAULT 0,
                        timestamp REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_artifacts_session_id ON artifacts(session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_artifacts_timestamp ON artifacts(timestamp)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 8")
            if current_version < 9:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS code_workspaces (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        path TEXT NOT NULL UNIQUE,
                        repo_url TEXT,
                        is_git_repo INTEGER DEFAULT 0,
                        branch TEXT,
                        detected_stack_json TEXT DEFAULT '[]',
                        package_manager TEXT,
                        commands_json TEXT DEFAULT '[]',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_code_workspaces_path ON code_workspaces(path)",
                    "CREATE INDEX IF NOT EXISTS idx_code_workspaces_updated_at ON code_workspaces(updated_at DESC)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 9")
            if current_version < 10:
                for ddl in [
                    """CREATE TABLE IF NOT EXISTS code_sessions (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL,
                        hermes_session_id TEXT,
                        task_id TEXT,
                        title TEXT,
                        provider TEXT,
                        model TEXT,
                        branch TEXT,
                        status TEXT NOT NULL DEFAULT 'planning',
                        summary TEXT,
                        metadata_json TEXT DEFAULT '{}',
                        started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        completed_at TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )""",
                    """CREATE TABLE IF NOT EXISTS code_session_events (
                        id TEXT PRIMARY KEY,
                        code_session_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        message TEXT,
                        payload_json TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )""",
                ]:
                    try:
                        cursor.execute(ddl)
                    except sqlite3.OperationalError:
                        pass
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_code_sessions_workspace_id ON code_sessions(workspace_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_sessions_hermes_session_id ON code_sessions(hermes_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_sessions_task_id ON code_sessions(task_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_sessions_status ON code_sessions(status)",
                    "CREATE INDEX IF NOT EXISTS idx_code_sessions_updated_at ON code_sessions(updated_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_code_session_events_session_id ON code_session_events(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_session_events_created_at ON code_session_events(created_at DESC)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                try:
                    cursor.execute(
                        "ALTER TABLE artifacts ADD COLUMN code_session_id TEXT"
                    )
                except sqlite3.OperationalError:
                    pass
                try:
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS idx_artifacts_code_session_id ON artifacts(code_session_id)"
                    )
                except sqlite3.OperationalError:
                    pass
                cursor.execute("UPDATE schema_version SET version = 10")
            if current_version < 11:
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS code_commands (
                        id TEXT PRIMARY KEY,
                        code_session_id TEXT NOT NULL,
                        workspace_id TEXT NOT NULL,
                        command TEXT NOT NULL,
                        argv_json TEXT DEFAULT '[]',
                        cwd TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        safety TEXT NOT NULL DEFAULT 'safe',
                        stdout TEXT DEFAULT '',
                        stderr TEXT DEFAULT '',
                        exit_code INTEGER,
                        pid INTEGER,
                        timeout_seconds INTEGER DEFAULT 120,
                        started_at TEXT,
                        completed_at TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_code_commands_code_session_id ON code_commands(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_commands_workspace_id ON code_commands(workspace_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_commands_status ON code_commands(status)",
                    "CREATE INDEX IF NOT EXISTS idx_code_commands_created_at ON code_commands(created_at DESC)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 11")
            if current_version < 12:
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS code_git_snapshots (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL,
                        code_session_id TEXT,
                        branch TEXT,
                        remote_url TEXT,
                        dirty INTEGER DEFAULT 0,
                        summary_json TEXT DEFAULT '{}',
                        files_json TEXT DEFAULT '[]',
                        diff_stat TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_workspace_id ON code_git_snapshots(workspace_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_code_session_id ON code_git_snapshots(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_created_at ON code_git_snapshots(created_at DESC)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 12")
            if current_version < 13:
                for ddl in [
                    """CREATE TABLE IF NOT EXISTS code_session_model_presets (
                        id TEXT PRIMARY KEY,
                        code_session_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        metadata_json TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )""",
                    """CREATE TABLE IF NOT EXISTS code_session_cost_entries (
                        id TEXT PRIMARY KEY,
                        code_session_id TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        task_type TEXT,
                        input_tokens INTEGER DEFAULT 0,
                        output_tokens INTEGER DEFAULT 0,
                        cache_read_tokens INTEGER DEFAULT 0,
                        cache_write_tokens INTEGER DEFAULT 0,
                        cost_usd REAL DEFAULT 0.0,
                        metadata_json TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )""",
                ]:
                    try:
                        cursor.execute(ddl)
                    except sqlite3.OperationalError:
                        pass
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_code_session_model_presets_session_id ON code_session_model_presets(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_session_model_presets_name ON code_session_model_presets(code_session_id, name)",
                    "CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_session_id ON code_session_cost_entries(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_provider ON code_session_cost_entries(provider)",
                    "CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_model ON code_session_cost_entries(model)",
                    "CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_created_at ON code_session_cost_entries(created_at DESC)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 13")
            if current_version < 14:
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS code_diagnostics (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL,
                        code_session_id TEXT,
                        source TEXT NOT NULL,
                        status TEXT NOT NULL,
                        diagnostics_json TEXT DEFAULT '[]',
                        summary_json TEXT DEFAULT '{}',
                        commands_json TEXT DEFAULT '[]',
                        duration_ms INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_code_diagnostics_workspace_id ON code_diagnostics(workspace_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_diagnostics_code_session_id ON code_diagnostics(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_diagnostics_created_at ON code_diagnostics(created_at DESC)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 14")
            if current_version < 15:
                for ddl in [
                    """CREATE TABLE IF NOT EXISTS code_agent_flows (
                        id TEXT PRIMARY KEY,
                        code_session_id TEXT NOT NULL,
                        workspace_id TEXT NOT NULL,
                        task_id TEXT,
                        title TEXT,
                        description TEXT,
                        status TEXT NOT NULL DEFAULT 'created',
                        current_role TEXT,
                        provider TEXT,
                        model TEXT,
                        preset TEXT,
                        plan_json TEXT DEFAULT '{}',
                        review_json TEXT,
                        approval_id TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT
                    )""",
                    """CREATE TABLE IF NOT EXISTS code_agent_flow_steps (
                        id TEXT PRIMARY KEY,
                        flow_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        input_json TEXT DEFAULT '{}',
                        output_json TEXT DEFAULT '{}',
                        error TEXT,
                        started_at TEXT,
                        completed_at TEXT,
                        created_at TEXT NOT NULL
                    )""",
                ]:
                    try:
                        cursor.execute(ddl)
                    except sqlite3.OperationalError:
                        pass
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_code_agent_flows_code_session_id ON code_agent_flows(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_agent_flows_workspace_id ON code_agent_flows(workspace_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_agent_flows_status ON code_agent_flows(status)",
                    "CREATE INDEX IF NOT EXISTS idx_code_agent_flows_created_at ON code_agent_flows(created_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_code_agent_flow_steps_flow_id ON code_agent_flow_steps(flow_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_agent_flow_steps_status ON code_agent_flow_steps(status)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 15")
            if current_version < 16:
                try:
                    cursor.execute(
                        """CREATE TABLE IF NOT EXISTS code_skill_runs (
                            id TEXT PRIMARY KEY,
                            skill_name TEXT NOT NULL,
                            workspace_id TEXT NOT NULL,
                            code_session_id TEXT,
                            task_id TEXT,
                            agent_flow_id TEXT,
                            status TEXT NOT NULL DEFAULT 'created',
                            input_json TEXT DEFAULT '{}',
                            output_json TEXT DEFAULT '{}',
                            summary TEXT,
                            diagnostics_before_json TEXT,
                            diagnostics_after_json TEXT,
                            commands_json TEXT DEFAULT '[]',
                            artifacts_json TEXT DEFAULT '[]',
                            approval_id TEXT,
                            error TEXT,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            completed_at TEXT
                        )"""
                    )
                except sqlite3.OperationalError:
                    pass
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_workspace_id ON code_skill_runs(workspace_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_code_session_id ON code_skill_runs(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_skill_name ON code_skill_runs(skill_name)",
                    "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_status ON code_skill_runs(status)",
                    "CREATE INDEX IF NOT EXISTS idx_code_skill_runs_created_at ON code_skill_runs(created_at DESC)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 16")
            if current_version < 17:
                # v17: add tasks table (previously absent from migration chain)
                try:
                    cursor.execute(
                        """CREATE TABLE IF NOT EXISTS tasks (
                            id TEXT PRIMARY KEY,
                            title TEXT NOT NULL,
                            description TEXT,
                            status TEXT NOT NULL DEFAULT 'todo',
                            priority TEXT NOT NULL DEFAULT 'medium',
                            agent_id TEXT,
                            session_id TEXT,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            completed_at TEXT,
                            run_id TEXT,
                            error_message TEXT
                        )"""
                    )
                except sqlite3.OperationalError:
                    pass  # Already exists
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
                cursor.execute("UPDATE schema_version SET version = 17")
            if current_version < 18:
                # v18: P0 Engineering Control Plane — ledger_artifacts, orchestrated_runs/events
                for ddl in [
                    """CREATE TABLE IF NOT EXISTS ledger_artifacts (
                        id TEXT PRIMARY KEY,
                        category TEXT NOT NULL,
                        title TEXT,
                        content TEXT NOT NULL DEFAULT '',
                        format TEXT NOT NULL DEFAULT 'markdown',
                        workspace_id TEXT,
                        code_session_id TEXT,
                        flow_id TEXT,
                        command_id TEXT,
                        orchestrated_run_id TEXT,
                        metadata_json TEXT DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )""",
                    """CREATE TABLE IF NOT EXISTS orchestrated_runs (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT,
                        code_session_id TEXT,
                        title TEXT,
                        task_description TEXT,
                        state TEXT NOT NULL DEFAULT 'intake',
                        branch TEXT,
                        worktree_path TEXT,
                        metadata_json TEXT DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT
                    )""",
                    """CREATE TABLE IF NOT EXISTS orchestrated_run_events (
                        id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        from_state TEXT,
                        to_state TEXT,
                        message TEXT,
                        payload_json TEXT DEFAULT '{}',
                        created_at TEXT NOT NULL
                    )""",
                ]:
                    try:
                        cursor.execute(ddl)
                    except sqlite3.OperationalError:
                        pass
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_code_session_id ON ledger_artifacts(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_workspace_id ON ledger_artifacts(workspace_id)",
                    "CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_category ON ledger_artifacts(category)",
                    "CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_created_at ON ledger_artifacts(created_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_orch_runs_workspace ON orchestrated_runs(workspace_id)",
                    "CREATE INDEX IF NOT EXISTS idx_orch_runs_session ON orchestrated_runs(code_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_orch_runs_state ON orchestrated_runs(state)",
                    "CREATE INDEX IF NOT EXISTS idx_orch_run_events_run_id ON orchestrated_run_events(run_id)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass
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
    ) -> str:
        """Create a new session record. Returns the session_id."""

        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO sessions (id, source, user_id, model, model_config,
                   system_prompt, parent_session_id, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source,
                    user_id,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt,
                    parent_session_id,
                    time.time(),
                ),
            )

        self._execute_write(_do)
        return session_id

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
            session_id_or_prefix.replace("\\", "\\\\")
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
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", title)

        # Remove problematic Unicode control characters:
        # - Zero-width chars (U+200B-U+200F, U+FEFF)
        # - Directional overrides (U+202A-U+202E, U+2066-U+2069)
        # - Object replacement (U+FFFC), interlinear annotation (U+FFF9-U+FFFB)
        cleaned = re.sub(
            r"[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]",
            "",
            cleaned,
        )

        # Collapse internal whitespace runs and strip
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

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
        match = re.match(r"^(.*?) #(\d+)$", base_title)
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
            m = re.match(r"^.* #(\d+)$", t)
            if m:
                max_num = max(max_num, int(m.group(1)))

        return f"{base} #{max_num + 1}"

    def list_sessions_rich(
        self,
        source: str = None,
        exclude_sources: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (timestamp of last message).

        Uses a single query with correlated subqueries instead of N+2 queries.

        By default, child sessions (subagent runs, compression continuations)
        are excluded.  Pass ``include_children=True`` to include them.
        """
        where_clauses = []
        params = []

        if not include_children:
            where_clauses.append("s.parent_session_id IS NULL")

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
            json.dumps(reasoning_details) if reasoning_details else None
        )
        codex_items_json = (
            json.dumps(codex_reasoning_items) if codex_reasoning_items else None
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
                    logger.warning(
                        "Failed to deserialize tool_calls in get_messages, falling back to []"
                    )
                    msg["tool_calls"] = []
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
                    logger.warning(
                        "Failed to deserialize tool_calls in conversation replay, falling back to []"
                    )
                    msg["tool_calls"] = []
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
                        logger.warning(
                            "Failed to deserialize reasoning_details, falling back to None"
                        )
                        msg["reasoning_details"] = None
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(
                            row["codex_reasoning_items"]
                        )
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "Failed to deserialize codex_reasoning_items, falling back to None"
                        )
                        msg["codex_reasoning_items"] = None
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
        - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
          matches them as exact phrases instead of splitting on the
          hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
        """
        # Step 1: Extract balanced double-quoted phrases and protect them
        # from further processing via numbered placeholders.
        _quoted_parts: list = []

        def _preserve_quoted(m: re.Match) -> str:
            _quoted_parts.append(m.group(0))
            return f"\x00Q{len(_quoted_parts) - 1}\x00"

        sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)

        # Step 2: Strip remaining (unmatched) FTS5-special characters
        sanitized = re.sub(r"[+{}()\"^]", " ", sanitized)

        # Step 3: Collapse repeated * (e.g. "***") into a single one,
        # and remove leading * (prefix-only needs at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

        # Step 4: Remove dangling boolean operators at start/end that would
        # cause syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

        # Step 5: Wrap unquoted dotted and/or hyphenated terms in double
        # quotes.  FTS5's tokenizer splits on dots and hyphens, turning
        # ``chat-send`` into ``chat AND send`` and ``P2.2`` into ``p2 AND 2``.
        # Quoting preserves phrase semantics.  A single pass avoids the
        # double-quoting bug that would occur if dotted and hyphenated
        # patterns were applied sequentially (e.g. ``my-app.config``).
        sanitized = re.sub(r"\b(\w+(?:[.-]\w+)+)\b", r'"\1"', sanitized)

        # Step 6: Restore preserved quoted phrases
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

        return sanitized.strip()

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
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
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source."""
        with self._lock:
            if source:
                cursor = self._conn.execute(
                    "SELECT * FROM sessions WHERE source = ? ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (source, limit, offset),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
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

    def sessions_today_count(self, date: str) -> int:
        """Count sessions started on the given date (YYYY-MM-DD, UTC)."""
        start_ts, end_ts = _date_to_timestamp_range(date)
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE started_at >= ? AND started_at < ?",
                (start_ts, end_ts),
            )
            return cursor.fetchone()[0]

    def active_sessions_count(self) -> int:
        """Count sessions that have no ended_at (still open)."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL"
            )
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

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.
        """
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""

        def _do(conn):
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )

        self._execute_write(_do)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Child sessions are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted, so they remain accessible independently.
        Returns True if the session was found and deleted.
        """

        def _do(conn):
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
            )
            if cursor.fetchone()[0] == 0:
                return False
            # Orphan child sessions so FK constraint is satisfied
            conn.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                "WHERE parent_session_id = ?",
                (session_id,),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True

        return self._execute_write(_do)

    def prune_sessions(self, older_than_days: int = 90, source: str = None) -> int:
        """Delete sessions older than N days. Returns count of deleted sessions.

        Only prunes ended sessions (not active ones).  Child sessions outside
        the prune window are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted.
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
            session_ids = set(row["id"] for row in cursor.fetchall())

            if not session_ids:
                return 0

            # Orphan any sessions whose parent is about to be deleted
            placeholders = ",".join("?" * len(session_ids))
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                list(session_ids),
            )

            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            return len(session_ids)

        return self._execute_write(_do)

    def create_artifact(
        self,
        session_id: str,
        tool_name: str,
        path: str,
        status: str,
        tool_call_id: Optional[str] = None,
        diff: str = "",
        additions: int = 0,
        deletions: int = 0,
        code_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        artifact_id = str(uuid.uuid4())
        now_ts = __import__("time").time()
        now_iso = datetime.now(timezone.utc).isoformat()

        if diff and additions == 0 and deletions == 0:
            additions = len([line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")])
            deletions = len([line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")])

        def _do(conn):
            conn.execute(
                """INSERT INTO artifacts
                   (id, session_id, tool_call_id, tool_name, path, status,
                    diff, additions, deletions, timestamp, created_at, code_session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id,
                    session_id,
                    tool_call_id,
                    tool_name,
                    path,
                    status,
                    diff,
                    additions,
                    deletions,
                    now_ts,
                    now_iso,
                    code_session_id,
                ),
            )

        self._execute_write(_do)
        cursor = self._conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_artifacts_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM artifacts WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        artifacts = [dict(row) for row in cursor.fetchall()]
        if not artifacts:
            cursor = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND role = 'tool' ORDER BY timestamp ASC",
                (session_id,)
            )
            for row in cursor.fetchall():
                try:
                    import json
                    content = json.loads(row["content"])
                    if "files_modified" in content:
                        for f in content["files_modified"]:
                            artifacts.append({
                                "id": f"legacy_{row['id']}",
                                "session_id": session_id,
                                "tool_call_id": row["tool_call_id"],
                                "tool_name": row["tool_name"],
                                "path": f,
                                "status": "modified",
                                "diff": content.get("diff", ""),
                                "additions": len([line for line in content.get("diff", "").splitlines() if line.startswith("+") and not line.startswith("+++")]),
                                "deletions": len([line for line in content.get("diff", "").splitlines() if line.startswith("-") and not line.startswith("---")]),
                                "timestamp": row["timestamp"],
                                "created_at": datetime.fromtimestamp(row["timestamp"], tz=timezone.utc).isoformat() if "timestamp" in row else None,
                                "code_session_id": None
                            })
                except Exception:
                    pass
        return artifacts



class WorkspaceDB:
    """SQLite-backed storage for code workspaces.

    Uses the same DB as SessionDB to enable foreign-key relationships.
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cursor = self._conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS code_workspaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                repo_url TEXT,
                is_git_repo INTEGER DEFAULT 0,
                branch TEXT,
                detected_stack_json TEXT DEFAULT '[]',
                package_manager TEXT,
                commands_json TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_code_workspaces_path ON code_workspaces(path)",
            "CREATE INDEX IF NOT EXISTS idx_code_workspaces_updated_at ON code_workspaces(updated_at DESC)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def upsert_workspace(
        self,
        path: str,
        name: str,
        is_git_repo: bool = False,
        branch: str = None,
        repo_url: str = None,
        detected_stack: List[str] = None,
        package_manager: str = None,
        commands: List[str] = None,
    ) -> dict:
        """Insert or update a workspace. Returns the workspace record."""
        now = datetime.now(timezone.utc).isoformat()
        workspace_id = str(uuid.uuid4())

        def _do(conn):
            existing = conn.execute(
                "SELECT id FROM code_workspaces WHERE path = ?", (path,)
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE code_workspaces SET
                        name = ?, is_git_repo = ?, branch = ?, repo_url = ?,
                        detected_stack_json = ?, package_manager = ?,
                        commands_json = ?, updated_at = ?
                       WHERE path = ?""",
                    (
                        name,
                        int(is_git_repo),
                        branch,
                        repo_url,
                        json.dumps(detected_stack or []),
                        package_manager,
                        json.dumps(commands or []),
                        now,
                        path,
                    ),
                )
                return conn.execute(
                    "SELECT * FROM code_workspaces WHERE path = ?", (path,)
                ).fetchone()
            else:
                conn.execute(
                    """INSERT INTO code_workspaces
                       (id, name, path, is_git_repo, branch, repo_url,
                        detected_stack_json, package_manager, commands_json,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        workspace_id,
                        name,
                        path,
                        int(is_git_repo),
                        branch,
                        repo_url,
                        json.dumps(detected_stack or []),
                        package_manager,
                        json.dumps(commands or []),
                        now,
                        now,
                    ),
                )
                return conn.execute(
                    "SELECT * FROM code_workspaces WHERE id = ?", (workspace_id,)
                ).fetchone()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = _do(self._conn)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return self._row_to_dict(row)

    def get_workspace(self, workspace_id: str) -> Optional[dict]:
        """Get a workspace by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM code_workspaces WHERE id = ?", (workspace_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_workspaces(self, limit: int = 500, offset: int = 0) -> list:
        cursor = self._conn.execute(
            "SELECT * FROM code_workspaces ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        result = dict(row)
        if result.get("detected_stack_json"):
            try:
                result["detected_stack"] = json.loads(result["detected_stack_json"])
            except (json.JSONDecodeError, TypeError):
                result["detected_stack"] = []
        else:
            result["detected_stack"] = []
        if result.get("commands_json"):
            try:
                result["commands"] = json.loads(result["commands_json"])
            except (json.JSONDecodeError, TypeError):
                result["commands"] = []
        else:
            result["commands"] = []
        if "is_git_repo" in result:
            result["is_git_repo"] = bool(result["is_git_repo"])
        return result


class CodeDiagnosticsDB:
    """SQLite-backed storage for code diagnostics results."""

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cursor = self._conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS code_diagnostics (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                code_session_id TEXT,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                diagnostics_json TEXT DEFAULT '[]',
                summary_json TEXT DEFAULT '{}',
                commands_json TEXT DEFAULT '[]',
                duration_ms INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_code_diagnostics_workspace_id ON code_diagnostics(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_diagnostics_code_session_id ON code_diagnostics(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_diagnostics_created_at ON code_diagnostics(created_at DESC)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _execute_write(self, fn: Callable) -> Any:
        last_err = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        self._conn.rollback()
                        raise
                    return result
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" not in str(e).lower():
                    raise
                time.sleep(
                    random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                )
        raise last_err

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def save_diagnostics(
        self,
        workspace_id: str,
        code_session_id: Optional[str],
        source: str,
        status: str,
        diagnostics: List[Dict[str, Any]],
        summary: Dict[str, int],
        commands: List[str],
        duration_ms: int,
    ) -> Dict[str, Any]:
        diag_id = str(uuid.uuid4())

        def _do(conn):
            conn.execute(
                """INSERT INTO code_diagnostics
                   (id, workspace_id, code_session_id, source, status,
                    diagnostics_json, summary_json, commands_json, duration_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    diag_id,
                    workspace_id,
                    code_session_id,
                    source,
                    status,
                    json.dumps(diagnostics),
                    json.dumps(summary),
                    json.dumps(commands),
                    duration_ms,
                ),
            )

        self._execute_write(_do)
        return self.get_diagnostics(diag_id)

    def get_diagnostics(self, diag_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_diagnostics WHERE id = ?", (diag_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_diagnostics(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            """SELECT * FROM code_diagnostics
               WHERE workspace_id = ?
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (workspace_id, limit, offset),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_latest_diagnostics(
        self, workspace_id: str, source: str = None
    ) -> Optional[Dict[str, Any]]:
        if source:
            cursor = self._conn.execute(
                """SELECT * FROM code_diagnostics
                   WHERE workspace_id = ? AND source = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (workspace_id, source),
            )
        else:
            cursor = self._conn.execute(
                """SELECT * FROM code_diagnostics
                   WHERE workspace_id = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (workspace_id,),
            )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        for old_field, new_field in (
            ("diagnostics_json", "diagnostics"),
            ("summary_json", "summary"),
            ("commands_json", "commands"),
        ):
            if result.get(old_field):
                try:
                    result[new_field] = json.loads(result[old_field])
                except (json.JSONDecodeError, TypeError):
                    result[new_field] = []
            else:
                result[new_field] = []
        return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def count_diff_changes(diff_text: str) -> tuple:
    """Count additions and deletions in a unified diff. Returns (additions, deletions)."""
    additions = 0
    deletions = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return additions, deletions


# ---------------------------------------------------------------------------
# CodeSessionDB
# ---------------------------------------------------------------------------

_CODE_SESSION_VALID_STATUSES = frozenset(
    {
        "created",
        "planning",
        "running",
        "coding",
        "reviewing",
        "waiting_approval",
        "completed",
        "done",
        "cancelled",
        "error",
        "failed",
    }
)


class CodeSessionDB:
    """SQLite-backed storage for code sessions and their timeline events."""

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cursor = self._conn.cursor()
        for ddl in [
            """CREATE TABLE IF NOT EXISTS code_sessions (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                hermes_session_id TEXT,
                task_id TEXT,
                title TEXT,
                provider TEXT,
                model TEXT,
                branch TEXT,
                status TEXT NOT NULL DEFAULT 'planning',
                summary TEXT,
                metadata_json TEXT DEFAULT '{}',
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS code_session_events (
                id TEXT PRIMARY KEY,
                code_session_id TEXT NOT NULL,
                type TEXT NOT NULL,
                message TEXT,
                payload_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
        ]:
            try:
                cursor.execute(ddl)
            except sqlite3.OperationalError:
                pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_code_sessions_workspace_id ON code_sessions(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_sessions_status ON code_sessions(status)",
            "CREATE INDEX IF NOT EXISTS idx_code_sessions_updated_at ON code_sessions(updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_code_session_events_session_id ON code_session_events(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_session_events_created_at ON code_session_events(created_at DESC)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _execute_write(self, fn: Callable) -> Any:
        last_err = None
        for _ in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        self._conn.rollback()
                        raise
                    return result
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" not in str(e).lower():
                    raise
                time.sleep(
                    random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                )
        raise last_err

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _session_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        for field, default in (("metadata_json", {}),):
            raw = result.pop(field, None)
            key = field[: -len("_json")]
            try:
                result[key] = json.loads(raw) if raw else default
            except (json.JSONDecodeError, TypeError):
                result[key] = default
        return result

    def create_session(
        self,
        workspace_id: str,
        hermes_session_id: Optional[str] = None,
        branch: Optional[str] = None,
        title: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata or {})

        def _do(conn):
            conn.execute(
                """INSERT INTO code_sessions
                   (id, workspace_id, hermes_session_id, task_id, title,
                    provider, model, branch, status, metadata_json,
                    started_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'planning', ?, ?, ?, ?)""",
                (
                    session_id,
                    workspace_id,
                    hermes_session_id,
                    task_id,
                    title,
                    provider,
                    model,
                    branch,
                    metadata_json,
                    now,
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._session_row_to_dict(row)

    def list_sessions(
        self,
        workspace_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM code_sessions WHERE 1=1"
        params: list = []
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = self._conn.execute(query, params)
        return [self._session_row_to_dict(row) for row in cursor.fetchall()]

    def update_session(
        self, session_id: str, fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "status" in fields and fields["status"] not in _CODE_SESSION_VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {fields['status']}. "
                f"Must be one of: {', '.join(sorted(_CODE_SESSION_VALID_STATUSES))}"
            )
        now = datetime.now(timezone.utc).isoformat()

        allowed = {
            "status", "summary", "provider", "model", "branch",
            "title", "hermes_session_id", "task_id",
            "completed_at", "metadata",
        }
        set_clauses = ["updated_at = ?"]
        values: list = [now]
        for key, val in fields.items():
            if key not in allowed:
                continue
            if key == "metadata":
                set_clauses.append("metadata_json = ?")
                values.append(json.dumps(val or {}))
            else:
                set_clauses.append(f"{key} = ?")
                values.append(val)
        values.append(session_id)

        def _do(conn):
            conn.execute(
                f"UPDATE code_sessions SET {', '.join(set_clauses)} WHERE id = ?",
                values,
            )

        self._execute_write(_do)
        return self.get_session(session_id)

    def add_event(
        self,
        code_session_id: str,
        event_type: str,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        def _do(conn):
            conn.execute(
                """INSERT INTO code_session_events
                   (id, code_session_id, type, message, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    code_session_id,
                    event_type,
                    message,
                    json.dumps(payload or {}),
                    now,
                ),
            )

        self._execute_write(_do)
        cursor = self._conn.execute(
            "SELECT * FROM code_session_events WHERE id = ?", (event_id,)
        )
        row = cursor.fetchone()
        result = dict(row)
        try:
            result["payload"] = json.loads(result.pop("payload_json", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            result["payload"] = {}
        return result

    def list_events(
        self, code_session_id: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            """SELECT * FROM code_session_events
               WHERE code_session_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (code_session_id, limit),
        )
        events = []
        for row in cursor.fetchall():
            result = dict(row)
            try:
                result["payload"] = json.loads(result.pop("payload_json", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                result["payload"] = {}
            events.append(result)
        return events

    def list_artifacts_for_code_session(
        self,
        code_session_id: str,
        hermes_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Direct link via code_session_id column on artifacts table
        try:
            cursor = self._conn.execute(
                "SELECT * FROM artifacts WHERE code_session_id = ? ORDER BY created_at ASC",
                (code_session_id,),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []

        if rows:
            return [dict(r) for r in rows]

        # Fallback: look up by hermes_session_id
        if hermes_session_id:
            try:
                cursor = self._conn.execute(
                    "SELECT * FROM artifacts WHERE session_id = ? ORDER BY created_at ASC",
                    (hermes_session_id,),
                )
                return [dict(r) for r in cursor.fetchall()]
            except sqlite3.OperationalError:
                pass

        return []

    def link_artifact_to_session(
        self, artifact_id: str, code_session_id: str
    ) -> Optional[Dict[str, Any]]:
        def _do(conn):
            cursor = conn.execute(
                "SELECT id FROM artifacts WHERE id = ?", (artifact_id,)
            )
            if not cursor.fetchone():
                return None
            conn.execute(
                "UPDATE artifacts SET code_session_id = ? WHERE id = ?",
                (code_session_id, artifact_id),
            )

        try:
            self._execute_write(_do)
        except sqlite3.OperationalError:
            return None

        try:
            cursor = self._conn.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.OperationalError:
            return None


# ---------------------------------------------------------------------------
# CodeCommandDB
# ---------------------------------------------------------------------------


class CodeCommandDB:
    """SQLite-backed storage for code commands."""

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS code_commands (
                    id TEXT PRIMARY KEY,
                    code_session_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    command TEXT NOT NULL,
                    argv_json TEXT DEFAULT '[]',
                    cwd TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    safety TEXT NOT NULL DEFAULT 'safe',
                    stdout TEXT DEFAULT '',
                    stderr TEXT DEFAULT '',
                    exit_code INTEGER,
                    pid INTEGER,
                    timeout_seconds INTEGER DEFAULT 120,
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
        except sqlite3.OperationalError:
            pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_code_commands_code_session_id ON code_commands(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_commands_workspace_id ON code_commands(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_commands_status ON code_commands(status)",
            "CREATE INDEX IF NOT EXISTS idx_code_commands_created_at ON code_commands(created_at DESC)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _execute_write(self, fn: Callable) -> Any:
        last_err = None
        for _ in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        self._conn.rollback()
                        raise
                    return result
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" not in str(e).lower():
                    raise
                time.sleep(
                    random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                )
        raise last_err

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        raw_argv = result.pop("argv_json", "[]")
        try:
            result["argv"] = json.loads(raw_argv) if raw_argv else []
        except (json.JSONDecodeError, TypeError):
            result["argv"] = []
        return result

    def create_command(
        self,
        code_session_id: str,
        workspace_id: str,
        command: str,
        argv: List[str],
        cwd: str,
        timeout_seconds: int = 120,
        status: str = "pending",
        safety: str = "safe",
    ) -> Dict[str, Any]:
        cmd_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        def _do(conn):
            conn.execute(
                """INSERT INTO code_commands
                   (id, code_session_id, workspace_id, command, argv_json,
                    cwd, status, safety, timeout_seconds, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cmd_id,
                    code_session_id,
                    workspace_id,
                    command,
                    json.dumps(argv),
                    cwd,
                    status,
                    safety,
                    timeout_seconds,
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        return self.get_command(cmd_id)

    def get_command(self, command_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_commands WHERE id = ?", (command_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_commands(
        self, code_session_id: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            """SELECT * FROM code_commands
               WHERE code_session_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (code_session_id, limit),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def update_command(self, command_id: str, **kwargs) -> Dict[str, Any]:
        allowed = {
            "status", "stdout", "stderr", "exit_code", "pid",
            "started_at", "completed_at",
        }
        now = datetime.now(timezone.utc).isoformat()
        set_clauses = ["updated_at = ?"]
        values: list = [now]
        for key, val in kwargs.items():
            if key in allowed:
                set_clauses.append(f"{key} = ?")
                values.append(val)
        values.append(command_id)

        def _do(conn):
            conn.execute(
                f"UPDATE code_commands SET {', '.join(set_clauses)} WHERE id = ?",
                values,
            )

        self._execute_write(_do)
        return self.get_command(command_id)


# ---------------------------------------------------------------------------
# GitSnapshotDB
# ---------------------------------------------------------------------------


class GitSnapshotDB:
    """SQLite-backed storage for git snapshots."""

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS code_git_snapshots (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    code_session_id TEXT,
                    branch TEXT,
                    remote_url TEXT,
                    dirty INTEGER DEFAULT 0,
                    summary_json TEXT DEFAULT '{}',
                    files_json TEXT DEFAULT '[]',
                    diff_stat TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
        except sqlite3.OperationalError:
            pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_workspace_id ON code_git_snapshots(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_code_session_id ON code_git_snapshots(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_git_snapshots_created_at ON code_git_snapshots(created_at DESC)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _execute_write(self, fn: Callable) -> Any:
        last_err = None
        for _ in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        self._conn.rollback()
                        raise
                    return result
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" not in str(e).lower():
                    raise
                time.sleep(
                    random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                )
        raise last_err

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        for field, default in (("summary_json", {}), ("files_json", [])):
            raw = result.pop(field, None)
            key = field[: -len("_json")]
            try:
                result[key] = json.loads(raw) if raw else default
            except (json.JSONDecodeError, TypeError):
                result[key] = default
        if "dirty" in result:
            result["dirty"] = bool(result["dirty"])
        return result

    def create_snapshot(
        self,
        workspace_id: str,
        code_session_id: Optional[str] = None,
        branch: Optional[str] = None,
        remote_url: Optional[str] = None,
        dirty: bool = False,
        summary: Optional[Dict[str, Any]] = None,
        files: Optional[List[Any]] = None,
        diff_stat: Optional[str] = None,
    ) -> Dict[str, Any]:
        snap_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        def _do(conn):
            conn.execute(
                """INSERT INTO code_git_snapshots
                   (id, workspace_id, code_session_id, branch, remote_url,
                    dirty, summary_json, files_json, diff_stat, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snap_id,
                    workspace_id,
                    code_session_id,
                    branch,
                    remote_url,
                    int(dirty),
                    json.dumps(summary or {}),
                    json.dumps(files or []),
                    diff_stat,
                    now,
                ),
            )

        self._execute_write(_do)
        cursor = self._conn.execute(
            "SELECT * FROM code_git_snapshots WHERE id = ?", (snap_id,)
        )
        return self._row_to_dict(cursor.fetchone())

    def list_snapshots(
        self,
        workspace_id: str,
        code_session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if code_session_id:
            cursor = self._conn.execute(
                """SELECT * FROM code_git_snapshots
                   WHERE workspace_id = ? AND code_session_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (workspace_id, code_session_id, limit),
            )
        else:
            cursor = self._conn.execute(
                """SELECT * FROM code_git_snapshots
                   WHERE workspace_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (workspace_id, limit),
            )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_git_snapshots WHERE id = ?", (snapshot_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None


# ---------------------------------------------------------------------------
# ProviderRouterDB
# ---------------------------------------------------------------------------


class ProviderRouterDB:
    """SQLite-backed storage for model presets and cost tracking."""

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cursor = self._conn.cursor()
        for ddl in [
            """CREATE TABLE IF NOT EXISTS code_session_model_presets (
                id TEXT PRIMARY KEY,
                code_session_id TEXT NOT NULL,
                name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS code_session_cost_entries (
                id TEXT PRIMARY KEY,
                code_session_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                task_type TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
        ]:
            try:
                cursor.execute(ddl)
            except sqlite3.OperationalError:
                pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_code_session_model_presets_session_id ON code_session_model_presets(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_session_model_presets_name ON code_session_model_presets(code_session_id, name)",
            "CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_session_id ON code_session_cost_entries(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_provider ON code_session_cost_entries(provider)",
            "CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_model ON code_session_cost_entries(model)",
            "CREATE INDEX IF NOT EXISTS idx_code_session_cost_entries_created_at ON code_session_cost_entries(created_at DESC)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _execute_write(self, fn: Callable) -> Any:
        last_err = None
        for _ in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        self._conn.rollback()
                        raise
                    return result
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" not in str(e).lower():
                    raise
                time.sleep(
                    random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                )
        raise last_err

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _preset_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        raw = result.pop("metadata_json", "{}")
        try:
            result["metadata"] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}
        return result

    def _cost_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        raw = result.pop("metadata_json", "{}")
        try:
            result["metadata"] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}
        return result

    # ── Presets ──

    def create_preset(
        self,
        code_session_id: str,
        name: str,
        provider: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        preset_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        def _do(conn):
            conn.execute(
                """INSERT INTO code_session_model_presets
                   (id, code_session_id, name, provider, model, metadata_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    preset_id,
                    code_session_id,
                    name,
                    provider,
                    model,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        cursor = self._conn.execute(
            "SELECT * FROM code_session_model_presets WHERE id = ?", (preset_id,)
        )
        return self._preset_row_to_dict(cursor.fetchone())

    def get_preset_by_name(
        self, code_session_id: str, name: str
    ) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_session_model_presets WHERE code_session_id = ? AND name = ?",
            (code_session_id, name),
        )
        row = cursor.fetchone()
        return self._preset_row_to_dict(row) if row else None

    def get_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_session_model_presets WHERE id = ?", (preset_id,)
        )
        row = cursor.fetchone()
        return self._preset_row_to_dict(row) if row else None

    def list_presets(self, code_session_id: str) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM code_session_model_presets WHERE code_session_id = ? ORDER BY name ASC",
            (code_session_id,),
        )
        return [self._preset_row_to_dict(row) for row in cursor.fetchall()]

    def update_preset(
        self,
        preset_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        set_clauses = ["updated_at = ?"]
        values: list = [now]
        if provider is not None:
            set_clauses.append("provider = ?")
            values.append(provider)
        if model is not None:
            set_clauses.append("model = ?")
            values.append(model)
        if metadata is not None:
            set_clauses.append("metadata_json = ?")
            values.append(json.dumps(metadata))
        values.append(preset_id)

        def _do(conn):
            conn.execute(
                f"UPDATE code_session_model_presets SET {', '.join(set_clauses)} WHERE id = ?",
                values,
            )

        self._execute_write(_do)
        return self.get_preset(preset_id)

    def delete_preset(self, preset_id: str) -> bool:
        def _do(conn):
            cursor = conn.execute(
                "DELETE FROM code_session_model_presets WHERE id = ?", (preset_id,)
            )
            return cursor.rowcount > 0

        return bool(self._execute_write(_do))

    # ── Cost tracking ──

    def add_cost_entry(
        self,
        code_session_id: str,
        provider: str,
        model: str,
        task_type: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        def _do(conn):
            conn.execute(
                """INSERT INTO code_session_cost_entries
                   (id, code_session_id, provider, model, task_type,
                    input_tokens, output_tokens, cache_read_tokens,
                    cache_write_tokens, cost_usd, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry_id,
                    code_session_id,
                    provider,
                    model,
                    task_type,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    cost_usd,
                    json.dumps(metadata or {}),
                    now,
                ),
            )

        self._execute_write(_do)
        cursor = self._conn.execute(
            "SELECT * FROM code_session_cost_entries WHERE id = ?", (entry_id,)
        )
        return self._cost_row_to_dict(cursor.fetchone())

    def get_cost_summary(self, code_session_id: str) -> Dict[str, Any]:
        cursor = self._conn.execute(
            """SELECT provider, model,
                      SUM(input_tokens) as input_tokens,
                      SUM(output_tokens) as output_tokens,
                      SUM(cache_read_tokens) as cache_read_tokens,
                      SUM(cache_write_tokens) as cache_write_tokens,
                      SUM(cost_usd) as cost_usd,
                      COUNT(*) as entry_count
               FROM code_session_cost_entries
               WHERE code_session_id = ?
               GROUP BY provider, model""",
            (code_session_id,),
        )
        rows = cursor.fetchall()

        by_provider: Dict[str, Any] = {}
        total_cost = 0.0
        total_entries = 0
        total_input = 0
        total_output = 0
        for row in rows:
            r = dict(row)
            prov = r["provider"]
            cost = r["cost_usd"] or 0.0
            inp = r["input_tokens"] or 0
            out = r["output_tokens"] or 0
            total_cost += cost
            total_entries += r["entry_count"]
            total_input += inp
            total_output += out
            if prov not in by_provider:
                by_provider[prov] = {
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "entry_count": 0,
                }
            by_provider[prov]["cost_usd"] += cost
            by_provider[prov]["input_tokens"] += inp
            by_provider[prov]["output_tokens"] += out
            by_provider[prov]["entry_count"] += r["entry_count"]

        return {
            "code_session_id": code_session_id,
            "total_cost_usd": round(total_cost, 8),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "entry_count": total_entries,
            "by_provider": by_provider,
        }

    def list_cost_entries(
        self,
        code_session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            """SELECT * FROM code_session_cost_entries
               WHERE code_session_id = ?
               ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (code_session_id, limit, offset),
        )
        return [self._cost_row_to_dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# TaskDB
# ---------------------------------------------------------------------------


class TaskDB:
    """SQLite-backed storage for user-facing tasks."""

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'todo',
                    priority TEXT NOT NULL DEFAULT 'medium',
                    agent_id TEXT,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    run_id TEXT,
                    error_message TEXT
                )"""
            )
        except sqlite3.OperationalError:
            pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
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
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                        time.sleep(jitter)
                        continue
                raise
        raise last_err or sqlite3.OperationalError("database is locked after max retries")

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row) if row else {}

    def close(self):
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    def list_tasks(
        self,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM tasks WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            cursor = self._conn.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def create_task(
        self,
        task_id: str,
        title: str,
        description: Optional[str] = None,
        priority: str = "medium",
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()

        def _do(conn):
            conn.execute(
                """INSERT INTO tasks
                   (id, title, description, status, priority, agent_id,
                    session_id, created_at, updated_at, run_id)
                   VALUES (?, ?, ?, 'todo', ?, ?, ?, ?, ?, ?)""",
                (task_id, title, description, priority, agent_id, session_id, now, now, run_id),
            )

        self._execute_write(_do)
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            return self._row_to_dict(cursor.fetchone())

    def update_task(
        self,
        task_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not updates:
            return None
        now = datetime.now(timezone.utc).isoformat()
        allowed = {"title", "description", "status", "priority", "agent_id",
                   "session_id", "run_id", "error_message", "completed_at"}
        safe = {k: v for k, v in updates.items() if k in allowed}
        if not safe:
            return None

        # Auto-set completed_at when status → done
        if safe.get("status") == "done" and "completed_at" not in safe:
            safe["completed_at"] = now

        safe["updated_at"] = now
        set_clause = ", ".join(f"{k} = ?" for k in safe)
        values = list(safe.values()) + [task_id]

        updated = [False]

        def _do(conn):
            cursor = conn.execute(
                f"UPDATE tasks SET {set_clause} WHERE id = ?", values  # noqa: S608
            )
            updated[0] = cursor.rowcount > 0

        self._execute_write(_do)
        if not updated[0]:
            return None
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            return self._row_to_dict(cursor.fetchone())

    def delete_task(self, task_id: str) -> bool:
        deleted = [False]

        def _do(conn):
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            deleted[0] = cursor.rowcount > 0

        self._execute_write(_do)
        return deleted[0]

    def count_by_status(self) -> Dict[str, int]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status"
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

    def completed_today_count(self, date: str) -> int:
        """Count tasks completed on the given date (YYYY-MM-DD prefix match)."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'done' AND completed_at LIKE ?",
                (f"{date}%",),
            )
            return cursor.fetchone()[0]


# ---------------------------------------------------------------------------
# ApprovalDB
# ---------------------------------------------------------------------------


class ApprovalDB:
    """SQLite-backed storage for approval requests."""

    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS approvals ("
                "id TEXT PRIMARY KEY,"
                "session_id TEXT,"
                "agent_id TEXT,"
                "status TEXT NOT NULL DEFAULT 'pending',"
                "title TEXT,"
                "kind TEXT DEFAULT 'command',"
                "details TEXT,"
                "command TEXT,"
                "created_at TEXT NOT NULL,"
                "updated_at TEXT NOT NULL,"
                "resolved_at TEXT,"
                "resolved_by TEXT,"
                "choice TEXT"
                ")"
            )
        except sqlite3.OperationalError:
            pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_approvals_session ON approvals(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status)",
            "CREATE INDEX IF NOT EXISTS idx_approvals_created ON approvals(created_at DESC)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _execute_write(self, fn: Callable) -> Any:
        last_err = None
        for _ in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        self._conn.rollback()
                        raise
                    return result
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" not in str(e).lower():
                    raise
                time.sleep(
                    random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
                )
        raise last_err

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    def create_approval(
        self,
        approval_id: str,
        session_id: Optional[str],
        agent_id: Optional[str],
        title: Optional[str],
        command: Optional[str],
        created_at: str,
        kind: str = "command",
        details: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at

        def _do(conn):
            conn.execute(
                """INSERT INTO approvals
                   (id, session_id, agent_id, status, title, kind, details,
                    command, created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)""",
                (
                    approval_id,
                    session_id,
                    agent_id,
                    title,
                    kind,
                    details,
                    command,
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def list_approvals(
        self,
        status: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM approvals WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = self._conn.execute(query, params)
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def resolve_approval(
        self,
        approval_id: str,
        status: str,
        resolved_by: Optional[str] = None,
        choice: Optional[str] = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        updated = [False]

        def _do(conn):
            cursor = conn.execute(
                """UPDATE approvals SET
                   status = ?, resolved_at = ?, resolved_by = ?,
                   choice = ?, updated_at = ?
                   WHERE id = ? AND status = 'pending'""",
                (status, now, resolved_by, choice, now, approval_id),
            )
            if cursor.rowcount > 0:
                updated[0] = True

        self._execute_write(_do)
        return updated[0]

    def get_pending_count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM approvals WHERE status = 'pending'")
        row = cursor.fetchone()
        return row[0] if row else 0

    def resolved_today_count(self, status: str, date: str) -> int:
        """Count approvals resolved with *status* on the given day (YYYY-MM-DD prefix)."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM approvals WHERE status = ? AND resolved_at LIKE ?",
            (status, f"{date}%"),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def upsert_from_queue(
        self,
        approval_id: str,
        session_id: str,
        agent_id: str,
        title: Optional[str],
        command: str,
        created_at: str,
        kind: str = "command",
        details: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def _do(conn):
            cursor = conn.execute("SELECT status FROM approvals WHERE id = ?", (approval_id,))
            row = cursor.fetchone()
            if row:
                if row[0] == 'pending':
                    return
                conn.execute(
                    """UPDATE approvals SET
                       session_id = ?, agent_id = ?, status = 'pending', title = ?,
                       kind = ?, details = ?, command = ?, created_at = ?,
                       updated_at = ?, resolved_at = NULL, resolved_by = NULL, choice = NULL
                       WHERE id = ?""",
                    (session_id, agent_id, title, kind, details, command, created_at, now, approval_id)
                )
            else:
                conn.execute(
                    """INSERT INTO approvals
                       (id, session_id, agent_id, status, title, kind, details, command, created_at, updated_at)
                       VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)""",
                    (approval_id, session_id, agent_id, title, kind, details, command, created_at, now)
                )

        self._execute_write(_do)
