"""The state.db DDL text, the SQL fragments it is built from, and the FTS layout checks that compare a database's stored tables with it."""

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Sequence

from hermes_state_common import routed_sessions_setting
from hermes_state_keys import CRON_PRINCIPAL_ID

# Pre-split logger identity so log filtering/capture is unchanged.
logger = logging.getLogger("hermes_state")


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _sql_json_extract(expression: str, path: str) -> str:
    """Build a non-throwing JSON marker lookup for a JSON TEXT column."""

    safe_json = (
        f"(CASE WHEN json_valid({expression}) "
        f"THEN {expression} ELSE json_object() END)"
    )
    return f"json_extract({safe_json}, {_sql_literal(path)})"

# ── Session lineage predicates ({a} = sessions alias) ───────────────────────

# /branch child (kept visible, never cascade-deleted): stable marker OR legacy end_reason heuristic.
_BRANCH_CHILD_SQL = (f"{_sql_json_extract('{a}.model_config', '$._branched_from')} IS NOT NULL"
    " OR EXISTS (SELECT 1 FROM sessions p            WHERE p.id = {a}.parent_session_id"
    "            AND p.end_reason = 'branched'            AND {a}.started_at >= p.ended_at)")
_COMPRESSION_CHILD_SQL = ("EXISTS (SELECT 1 FROM sessions p        WHERE p.id = {a}.parent_session_id"
    "        AND p.end_reason = 'compression')")

# 'session_switch' creates no child row today, but pre-marker DBs hold legacy reset children whose parent
# ended that way.  Must stay identical to the recovery fence in find_latest_gateway_session_for_peer.
_RESET_END_REASONS = ("session_reset", "session_switch", "idle", "daily", "suspended", "resume_pending_expired")
_RESET_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RESET_END_REASONS)


def _legacy_reset_child_sql(alias: str, reasons_sql: str) -> str:
    """Pre-marker reset-continuation heuristic: child rides its parent's exact non-empty routing key and the
    parent ended at a reset boundary.  Shared by ``_RESET_CHILD_SQL`` and ``reopen_session()`` so the two
    cannot drift; ``reasons_sql`` is a literal or placeholder list."""
    return (f"EXISTS (SELECT 1 FROM sessions p            WHERE p.id = {alias}.parent_session_id"
        f"            AND p.end_reason IN ({reasons_sql})            AND {alias}.session_key IS NOT NULL"
        f"            AND {alias}.session_key != ''            AND {alias}.session_key = p.session_key)")


# A reset starts a separate user-visible conversation though rows keep parent_session_id for lineage.
# Stable marker, or the same-key fallback for pre-marker rows (exact key keeps subagent children out).
_RESET_CHILD_SQL = (f"{_sql_json_extract('{a}.model_config', '$._reset_from')} IS NOT NULL"
    " OR " + _legacy_reset_child_sql("{a}", _RESET_END_REASONS_SQL))


def _ephemeral_child_sql(alias: str = "s") -> str:
    """Subagent runs, not branch, reset, or compression children."""
    return (f"({alias}.parent_session_id IS NOT NULL AND NOT ({_BRANCH_CHILD_SQL.format(a=alias)})"
        f" AND NOT ({_COMPRESSION_CHILD_SQL.format(a=alias)}) AND NOT ({_RESET_CHILD_SQL.format(a=alias)}))")


# v32 (#140315): no data migration. #106742 added runtime_epoch, session_admissions, worker_executions,
# worker_receipts and three columns to SCHEMA_SQL under v31. Drift is detected by the schema read probe
# (schema_read_probe_statements), never by this number, and its catch-up is the declarative
# executescript(SCHEMA_SQL) + _reconcile_columns() every writable open already runs (additive,
# idempotent). The bump only records that the declared shape changed; a new table or column needs no bump.
SCHEMA_VERSION = 32

# FTS storage-layout version, tracked INDEPENDENTLY of SCHEMA_VERSION in the
# state_meta key ``fts_storage_version``. The main schema version advances
# freely on open (so future migrations always land); the FTS *layout* only
# reaches the current version when a DB is either born fresh or explicitly
# optimized via ``hermes sessions optimize-storage``. A legacy DB sits at
# layout 0 (marker absent) with a working inline index until the user opts in.
#   1 = v23 external-content layout with a tool-row-excluded trigram
#   2 = trigram also excludes structured tool_calls JSON
#   3 = messages_fts source aligned to a stable projection view
#       (``messages_fts_src``): always-truncate tool rows to the prefix, no
#       moving high-water boundary. The external-content source now reads
#       back EXACTLY what the triggers indexed, so the rank=1
#       'integrity-check' probe cannot drift from the stored index (the
#       recurring fts5 "checksum mismatch" / leaked-token failures).
FTS_STORAGE_VERSION = 3

# Tool results are often multi-megabyte machine payloads. The base FTS index
# stores only a bounded prefix of every tool row; tool rows are skipped by
# default in search, and explicit tool-only search uses a LIKE fallback over
# the full stored content, so no search capability is lost. The projection
# below is STABLE — it depends only on the row being written, never on
# mutable ``state_meta`` markers — which is what keeps the external-content
# integrity checker and the trigger 'delete'/'update' commands in agreement
# with the stored index forever.
FTS_TOOL_CONTENT_PREFIX_CHARS = 8_192


def _fts_indexed_content_sql(alias: str) -> str:
    return f"""CASE WHEN {alias}.role = 'tool'
         THEN substr(COALESCE({alias}.content, ''), 1, {FTS_TOOL_CONTENT_PREFIX_CHARS})
         ELSE {alias}.content END"""


_FTS_NEW_INDEXED_CONTENT_SQL = _fts_indexed_content_sql("new")
_FTS_OLD_INDEXED_CONTENT_SQL = _fts_indexed_content_sql("old")


_FTS_TRIGGERS = ("messages_fts_insert", "messages_fts_delete", "messages_fts_update",
                 "messages_fts_trigram_insert", "messages_fts_trigram_delete", "messages_fts_trigram_update")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    runtime_revision INTEGER NOT NULL DEFAULT 0,
    runtime_generation INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    created_source TEXT,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    system_prompt_hash TEXT,
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
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    git_metadata_generation INTEGER NOT NULL DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at REAL,
    last_activity_description TEXT,
    last_activity_provenance TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    compression_recovery_deadline REAL,
    compression_overload_streak INTEGER NOT NULL DEFAULT 0,
    profile_name TEXT,
    transport_profile TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    auto_archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
    tool_names TEXT,
    transcript_revision INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    _compressed_summary INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT,
    display_identity BLOB,
    display_order INTEGER,
    message_uid TEXT,
    absorbed_message_uids TEXT,
    tool_call_uids TEXT,
    tool_call_uid TEXT
);

CREATE TABLE IF NOT EXISTS session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    billing_provider TEXT NOT NULL DEFAULT '',
    billing_base_url TEXT NOT NULL DEFAULT '',
    billing_mode TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    api_call_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
);

CREATE TABLE IF NOT EXISTS runtime_epoch (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    epoch INTEGER NOT NULL CHECK (epoch > 0),
    instance_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_admissions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    admission_id TEXT NOT NULL UNIQUE,
    request_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    lineage_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    intent TEXT NOT NULL CHECK (intent IN ('queue','steer','redirect')),
    status TEXT NOT NULL CHECK (status IN ('queued','started','unknown','terminal')),
    outcome TEXT,
    owner_epoch INTEGER,
    generation INTEGER,
    UNIQUE (principal_id, target_session_id, request_id),
    CHECK (status != 'started' OR (owner_epoch IS NOT NULL AND generation IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS session_admissions_pending
    ON session_admissions(target_session_id, status, seq);
-- Cron run history finds a job's unified-runtime runs by the scheduler principal's request ids
-- (``cron:<job>:<execution>``); without this a lookup walked
-- every cron admission of every job. Partial: interactive turns, the bulk of admissions, pay
-- nothing for it. The constant leading column gives the planner equality + range over a covering
-- index, so it beats the UNIQUE autoindex's bare principal_id prefix even without ANALYZE stats.
-- The WHERE is built from CRON_PRINCIPAL_ID, the literal the run queries inline, so they cannot drift.
CREATE INDEX IF NOT EXISTS idx_session_admissions_cron_request
    ON session_admissions(principal_id, request_id, target_session_id) WHERE principal_id = '""" + CRON_PRINCIPAL_ID + """';
CREATE TABLE IF NOT EXISTS worker_executions (
    execution_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('cron','child','compute','kanban')),
    owner_epoch INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('registered','running','unknown','terminal')),
    adoption_digest TEXT NOT NULL,
    last_sequence INTEGER NOT NULL DEFAULT 0 CHECK (last_sequence >= 0)
);
-- Lineage-wide worker exclusion (claim/register/reset) probes every row of a conversation's lineage
-- inside the write transaction; without this each probe scanned the whole (only-growing) table.
CREATE INDEX IF NOT EXISTS idx_worker_executions_session_status
    ON worker_executions(session_id, status);
CREATE TABLE IF NOT EXISTS worker_receipts (
    execution_id TEXT NOT NULL REFERENCES worker_executions(execution_id) ON DELETE RESTRICT,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    payload_digest TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (execution_id, sequence)
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
);

CREATE TABLE IF NOT EXISTS gateway_hygiene_state (
    session_key TEXT PRIMARY KEY,
    failure_streak INTEGER NOT NULL DEFAULT 0
);

-- Monotonic conversation generation per routing peer (#96811).
--
-- A host-declared conversation key (X-Hermes-Session-Key / build_session_key)
-- is per-CHAT and outlives any single conversation on it, so the prompt-cache
-- affinity scope derived from it must be qualified by which conversation is
-- currently live. Deriving that from the session rows themselves
-- (COUNT/MAX over _RESET_END_REASONS boundaries) cannot prove non-reuse:
-- delete_session() and bulk pruning remove ended rows, so an aggregate can
-- return a pair it already emitted and hand a new conversation a retired
-- affinity identity.
--
-- This counter lives outside prunable session history and only ever
-- increments, once per boundary actually written, so a generation can never
-- be reused for a peer even if every session row behind it is deleted.
--
-- These rows are deliberately NEVER garbage-collected, including when every
-- session row for the peer is gone. Collecting one resets that peer to "no
-- generation", so its next boundary writes generation = 1 again and re-issues
-- a gwk_ scope a retired conversation already used — exactly the ABA this
-- table exists to close. Do not add it to delete_session()'s cascade or to any
-- prune sweep. One (TEXT, TEXT, INTEGER) row per routing peer is the intended,
-- bounded cost.
CREATE TABLE IF NOT EXISTS conversation_generations (
    source TEXT NOT NULL,
    session_key TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, session_key)
);

-- Per-backend liveness heartbeat (#94895). Each serve / tui_gateway process
-- registers a row at startup and refreshes ``last_heartbeat`` periodically.
-- The startup orphan sweep (sessions.startup_orphan_reap) consults this
-- table to avoid reaping rows whose owning backend is still alive but
-- just idle (multi-backend state.db shared by isolated serve processes).
-- A backend whose ``last_heartbeat`` is older than the heartbeat staleness
-- window is treated as dead; rows without ANY matching heartbeat fall back
-- to the original staleness predicate so legacy deployments keep working.
CREATE TABLE IF NOT EXISTS gateway_heartbeats (
    backend_id TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    started_at REAL NOT NULL,
    last_heartbeat REAL NOT NULL,
    profile TEXT NOT NULL DEFAULT '',
    host TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_turn_leases (
    conversation_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS async_delegations (
    delegation_id TEXT PRIMARY KEY,
    origin_session TEXT NOT NULL,
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT,
    state TEXT NOT NULL,
    dispatched_at REAL NOT NULL,
    completed_at REAL,
    updated_at REAL NOT NULL,
    event_json TEXT,
    result_json TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    delivered_at REAL,
    owner_pid INTEGER,
    owner_started_at INTEGER,
    task_json TEXT,
    -- Owner-side execution fence (unified runtime).
    owner_execution_id TEXT,
    delivery_claim TEXT,
    delivery_claimed_at REAL,
    -- Mirrors the delegation tool's own CREATE TABLE (tools/async_delegation.py
    -- _initialize_schema). Keeping the canonical fresh-install shape identical
    -- to the tool's avoids a silent schema drift: the tool's lazy
    -- ALTER TABLE ADD COLUMN used to be the only source of this column, so two
    -- databases at the same schema_version had different
    -- async_delegations shapes depending on whether the delegation tool had
    -- ever run, breaking rebuild/replay pipelines that reconstruct state.db
    -- from the canonical schema (#94691).
    origin_session_id TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_source_id ON sessions(source, id);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
-- Partial index for the Insights assistant tool-call scan
-- (agent/insights.py _get_tool_usage / _get_skill_usage): those queries filter
-- messages by role='assistant' AND tool_calls IS NOT NULL, a small fraction of
-- rows on a large state.db. role and tool_calls are base columns, so this can
-- live in SCHEMA_SQL rather than DEFERRED_INDEX_SQL.
CREATE INDEX IF NOT EXISTS idx_messages_assistant_calls_by_session
    ON messages(session_id)
    WHERE role = 'assistant' AND tool_calls IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON compression_locks(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_turn_leases_expires ON session_turn_leases(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_model_usage_session ON session_model_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_session_model_usage_model ON session_model_usage(model);
CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery
    ON async_delegations(delivery_state, completed_at);
"""

# Indexes on later-added columns must run AFTER _reconcile_columns(), or executescript fails on legacy DBs.
# sessions_runtime_ledger_delete_guard: every delete path in this release retires a session's ledger rows
# (session_admissions / worker_executions, ON DELETE RESTRICT) first, so only a release that predates the
# gateway runtime reaches it. Such a release (schema_version 31 too, and it never refuses a newer schema)
# otherwise dies on the bare FK with no hint. It prints the RAISE text verbatim and buckets errors by phrase,
# so the text must stay free of the words it reads as storage damage, lock contention or disk trouble.
DEFERRED_INDEX_SQL = """
DROP TRIGGER IF EXISTS sessions_runtime_ledger_delete_guard;
CREATE TRIGGER IF NOT EXISTS sessions_runtime_ledger_delete_guard
BEFORE DELETE ON sessions
WHEN EXISTS (SELECT 1 FROM session_admissions WHERE target_session_id = old.id)
  OR EXISTS (SELECT 1 FROM worker_executions WHERE session_id = old.id)
BEGIN
    SELECT RAISE(ABORT, 'Refused to delete a session that has gateway runtime history: state.db was upgraded by a newer Hermes, and this older Hermes cannot delete such sessions. Downgrading is unsupported. Run hermes update to go back to the newer release, or restore the pre-update snapshot: https://hermes-agent.nousresearch.com/docs/getting-started/updating#rolling-back-to-an-older-release');
END;
DROP TRIGGER IF EXISTS messages_transcript_revision_update;
CREATE TRIGGER IF NOT EXISTS messages_transcript_revision_update
AFTER UPDATE OF role, content, api_content, tool_calls, tool_call_id, tool_name, reasoning, reasoning_content,
    reasoning_details ON messages
BEGIN
    -- An in-place rewrite (prompt expansion, multimodal context, a redaction) moves neither the
    -- active row count nor the highest id; this write-side revision is what a transcript-derived
    -- cache keys on (conversation_revision). Whoever wrote the row, old builds included.
    UPDATE sessions SET transcript_revision = transcript_revision + 1 WHERE id = new.session_id;
END;
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_display_page
    ON messages(session_id, display_order, active DESC, id DESC)
    WHERE active = 1 OR compacted = 1;
CREATE INDEX IF NOT EXISTS idx_messages_display_backfill
    ON messages(session_id) WHERE (display_order IS NULL OR display_identity IS NULL)
    AND (active = 1 OR compacted = 1);
CREATE INDEX IF NOT EXISTS idx_messages_display_identity
    ON messages(session_id, display_identity, display_order)
    WHERE display_identity IS NOT NULL AND (active = 1 OR compacted = 1);
DROP TRIGGER IF EXISTS messages_message_uid_insert;
CREATE TRIGGER IF NOT EXISTS messages_message_uid_insert
AFTER INSERT ON messages WHEN new.message_uid IS NULL
BEGIN
    -- Every row carries a message_uid, whoever wrote it: a build that predates the column binds NULL, so
    -- the store mints one on its behalf (the current build always binds a uid; this never fires for it).
    -- The note lives inside the body: a comment before DROP/CREATE would defeat the settled-trigger skip.
    UPDATE messages SET message_uid = lower(hex(randomblob(16))) WHERE id = new.id;
END;
DROP TRIGGER IF EXISTS messages_display_order_insert;
CREATE TRIGGER IF NOT EXISTS messages_display_order_insert
AFTER INSERT ON messages WHEN new.display_order IS NULL
BEGIN
    UPDATE messages SET display_order = COALESCE((
        SELECT display_order FROM messages
        WHERE session_id = new.session_id AND id <> new.id
          AND (active = 1 OR compacted = 1)
          AND display_identity = new.display_identity AND display_order IS NOT NULL
        ORDER BY display_order LIMIT 1
    ), new.id) WHERE id = new.id;
END;
DROP TRIGGER IF EXISTS messages_display_visibility_update;
CREATE TRIGGER IF NOT EXISTS messages_display_visibility_update
AFTER UPDATE OF active, compacted ON messages
WHEN (new.active = 1 OR new.compacted = 1) <> (old.active = 1 OR old.compacted = 1)
BEGIN
    UPDATE messages SET display_order = MIN(new.id, COALESCE((
        SELECT display_order FROM messages
        WHERE session_id = new.session_id AND id <> new.id
          AND (active = 1 OR compacted = 1)
          AND display_identity = new.display_identity AND display_order IS NOT NULL
        ORDER BY display_order LIMIT 1
    ), new.id)) WHERE id = new.id
      AND (new.active = 1 OR new.compacted = 1);
    UPDATE messages SET display_order = (SELECT display_order FROM messages WHERE id = new.id)
    WHERE session_id = new.session_id AND id <> new.id AND (active = 1 OR compacted = 1)
      AND display_identity = new.display_identity
      AND (new.active = 1 OR new.compacted = 1);
    UPDATE messages SET display_order = (
        SELECT MIN(peer.id) FROM messages AS peer
        WHERE peer.session_id = old.session_id AND (peer.active = 1 OR peer.compacted = 1)
          AND peer.display_identity = old.display_identity
    ) WHERE session_id = old.session_id AND (active = 1 OR compacted = 1)
      AND display_identity = old.display_identity
      AND NOT (new.active = 1 OR new.compacted = 1);
END;
DROP TRIGGER IF EXISTS messages_display_identity_update;
CREATE TRIGGER IF NOT EXISTS messages_display_identity_update
AFTER UPDATE OF role, content, timestamp, tool_call_id, tool_calls, tool_name,
                display_kind ON messages
WHEN new.role IS NOT old.role
  OR new.content IS NOT old.content
  OR new.timestamp IS NOT old.timestamp
  OR new.tool_call_id IS NOT old.tool_call_id
  OR new.tool_calls IS NOT old.tool_calls
  OR new.tool_name IS NOT old.tool_name
  OR new.display_kind IS NOT old.display_kind
BEGIN
    UPDATE messages SET display_identity = NULL, display_order = NULL
    WHERE id = new.id OR (
        session_id = old.session_id AND display_identity = old.display_identity
        AND (active = 1 OR compacted = 1)
    );
END;
DROP TRIGGER IF EXISTS messages_display_identity_delete;
CREATE TRIGGER IF NOT EXISTS messages_display_identity_delete
AFTER DELETE ON messages WHEN old.active = 1 OR old.compacted = 1
BEGIN
    UPDATE messages SET display_order = (
        SELECT MIN(peer.id) FROM messages AS peer
        WHERE peer.session_id = old.session_id AND (peer.active = 1 OR peer.compacted = 1)
          AND peer.display_identity = old.display_identity
    ) WHERE session_id = old.session_id AND (active = 1 OR compacted = 1)
      AND display_identity = old.display_identity;
END;
CREATE INDEX IF NOT EXISTS idx_messages_active_null
    ON messages(active) WHERE active IS NULL;
-- Logical-identity recovery (agent.transcript_repair._active_logical_message_row) runs inside the write lock.
CREATE INDEX IF NOT EXISTS idx_messages_session_uid
    ON messages(session_id, message_uid) WHERE active = 1;
CREATE INDEX IF NOT EXISTS idx_sessions_session_key
    ON sessions(session_key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_system_prompt_hash
    ON sessions(system_prompt_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_tool_names
    ON sessions(tool_names);
-- Recent-session browsing must never derive recency by scanning messages.
-- This expression is the durable, indexable approximation used to preselect
-- a small candidate set before compression-chain and preview hydration.
CREATE INDEX IF NOT EXISTS idx_sessions_effective_activity
    ON sessions(COALESCE(last_activity_at, started_at) DESC, started_at DESC);
-- CLI/TUI listings match gateway rows stored as source='local' by created_source,
-- before LIMIT. created_source is reconciled onto legacy DBs, so this index lives
-- here rather than in SCHEMA_SQL. A source-only index would walk every local row
-- (Desktop and TUI included) to find one surface.
CREATE INDEX IF NOT EXISTS idx_sessions_local_created_started
    ON sessions(created_source, started_at DESC)
    WHERE source = 'local';
"""


# ── Deferred FTS rebuild bookkeeping (schema v23) ──
# While a background index rebuild is pending, two state_meta keys define
# which message rows are currently IN the FTS indexes:
#
#   fts_rebuild_high_water  H — MAX(messages.id) at the moment the old
#                                indexes were dropped
#   fts_rebuild_progress    P — highest id the chunked backfill has indexed
#
# A row is indexed iff  id <= P  (backfilled)  OR  id > H  (inserted after
# the drop; ids are AUTOINCREMENT so new rows are always > H and the insert
# triggers index them live).  Rows in (P, H] are not yet indexed.
#
# Every trigger below gates on that same predicate: firing an FTS5
# external-content 'delete' for a row that is NOT in the index corrupts the
# index, and skipping it for a row that IS indexed leaves a stale entry.
# When no rebuild is pending both keys are absent and COALESCE turns the
# predicate into a tautology (id > -1 OR id <= -1), i.e. normal operation.
# The two state_meta PK probes per write are negligible next to the FTS
# insert itself.
#
# messages_fts_src: the base word index no longer reads raw `messages` as its
# external content. Tool rows are indexed as a bounded prefix, so the index
# must read that SAME projection back or FTS5's 'integrity-check' / 'delete'
# commands disagree with the stored tokens and corrupt the index (the
# recurring fts5 checksum-mismatch drift: the projection used to depend on a
# moving state_meta high-water key). The view/trigger/backfill all share the
# one expression in `_fts_indexed_content_sql` — a fixed per-row function
# with no marker lookups — so the boundary can never move again.
FTS_SQL = f"""
-- Stable projection the base word index reads and writes through: the view
-- computes EXACTLY what the triggers/backfill insert, so 'rebuild' and the
-- integrity checker always agree with the stored index.
CREATE VIEW IF NOT EXISTS messages_fts_src AS
    SELECT id,
           CASE WHEN role = 'tool'
                THEN substr(COALESCE(content, ''), 1, {FTS_TOOL_CONTENT_PREFIX_CHARS})
                ELSE content END AS content,
           tool_name, tool_calls
    FROM messages;

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_src',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages
WHEN (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (
        new.id,
        {_FTS_NEW_INDEXED_CONTENT_SQL},
        new.tool_name,
        new.tool_calls
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
WHEN (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES (
        'delete',
        old.id,
        {_FTS_OLD_INDEXED_CONTENT_SQL},
        old.tool_name,
        old.tool_calls
    );
END;

-- UPDATE OF skips the trigger entirely for non-content column writes
-- (status/compacted/observed/etc.), which is stronger than the WHEN gate
-- alone and avoids FTS I/O saturation on large state.db (#68858 / #73639).
CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES (
        'delete',
        old.id,
        {_FTS_OLD_INDEXED_CONTENT_SQL},
        old.tool_name,
        old.tool_calls
    );
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (
        new.id,
        {_FTS_NEW_INDEXED_CONTENT_SQL},
        new.tool_name,
        new.tool_calls
    );
END;
"""


# Trigram FTS5 table for CJK substring search.  The default unicode61
# tokenizer splits CJK characters into individual tokens, breaking phrase
# matching.  The trigram tokenizer creates overlapping 3-byte sequences so
# substring queries work natively for any script (CJK, Thai, etc.).
#
# The trigram index is the most expensive index in state.db (~2.6x the size
# of the text it covers). Tool output (~90% of message bytes, machine noise)
# and cron transcripts are excluded: the index reads through
# ``messages_fts_trigram_src``, a view that skips both classes. They stay
# fully stored in ``messages`` and searchable via the standard
# ``messages_fts`` index; they just don't get trigram (CJK substring)
# treatment. ``search_messages`` routes explicit tool/cron CJK searches to
# LIKE for the same reason. Structured ``tool_calls`` JSON likewise stays
# searchable through ``messages_fts``; excluding it here avoids indexing
# repetitive JSON syntax as trigrams (FTS_STORAGE_VERSION 2).
#
# Delegate-child (subagent) transcripts are excluded the same way (v30):
# on a fan-out-heavy install they were ~70% of all message bytes and
# ``session_search`` hides ``source='subagent'`` sessions anyway. A child
# is recognised by its source OR by the ``_delegate_from`` creation marker
# (children spawned under a gateway turn inherit the gateway's source).
# Compression/branch continuations of interactive sessions also carry
# ``parent_session_id`` but NOT the marker, so they stay trigram-indexed.
FTS_TRIGRAM_EXCLUDED_SOURCES = ("cron", "subagent")

def fts_trigram_session_sql(alias: str = "") -> str:
    """Predicate over a ``sessions`` row selecting sessions whose rows belong in
    the trigram index; ``alias`` qualifies every column for joins. Shared by the
    view, the sync triggers, and the deferred-backfill INSERT ... SELECTs so they
    can never disagree about the index boundary."""
    q = f"{alias}." if alias else ""
    return (
        f"{q}source NOT IN ("
        + ", ".join(f"'{src}'" for src in FTS_TRIGRAM_EXCLUDED_SOURCES)
        + f") AND {_sql_json_extract(q + 'model_config', '$._delegate_from')} IS NULL"
    )


FTS_TRIGRAM_SESSION_SQL = fts_trigram_session_sql()


FTS_TRIGRAM_SQL = f"""
CREATE VIEW IF NOT EXISTS messages_fts_trigram_src AS
    SELECT m.id, m.role, m.content, m.tool_name
    FROM messages AS m
    JOIN sessions AS s ON s.id = m.session_id
    WHERE m.role <> 'tool' AND {fts_trigram_session_sql('s')};

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tool_name,
    content='messages_fts_trigram_src',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND EXISTS (SELECT 1 FROM sessions
               WHERE id = new.session_id AND {FTS_TRIGRAM_SESSION_SQL})
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(rowid, content, tool_name)
    VALUES (new.id, new.content, new.tool_name);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND EXISTS (SELECT 1 FROM sessions
               WHERE id = old.session_id AND {FTS_TRIGRAM_SESSION_SQL})
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name)
    VALUES ('delete', old.id, old.content, old.tool_name);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update
AFTER UPDATE OF content, tool_name, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name)
    SELECT 'delete', old.id, old.content, old.tool_name
    WHERE old.role <> 'tool'
      AND EXISTS (SELECT 1 FROM sessions
                  WHERE id = old.session_id AND {FTS_TRIGRAM_SESSION_SQL});
    INSERT INTO messages_fts_trigram(rowid, content, tool_name)
    SELECT new.id, new.content, new.tool_name
    WHERE new.role <> 'tool'
      AND EXISTS (SELECT 1 FROM sessions
                  WHERE id = new.session_id AND {FTS_TRIGRAM_SESSION_SQL});
END;
"""

_FTS_CJK_TRIGGERS = ("messages_fts_cjk_insert", "messages_fts_cjk_delete", "messages_fts_cjk_update")

# Set when a tokenizer-less process dropped the cjk triggers to keep writes alive: the cjk index is missing rows
# and must not serve reads until `hermes sessions optimize-storage` rebuilds it on a capable host.
FTS_CJK_STALE_KEY = "fts_cjk_stale"

# Set when a base/trigram FTS index was detached after runtime corruption; startup must rebuild the complete
# index before reinstalling sync triggers (rows written while they were absent leave an unknown gap).
FTS_STALE_KEY = "fts_stale"

# Durable diagnostic for stale FTS recovery blocked across process restarts.
FTS_REBUILD_DEFERRAL_KEY = "fts_rebuild_deferral"


# ── Legacy (v22 / inline-content) FTS DDL ──────────────────────────────
# Used ONLY to keep an existing pre-v23 install's search working and its
# triggers repairable UNTIL the user opts into `hermes db optimize`. This is
# the exact inline shape v11..v22 shipped: each virtual table stores its own
# copy of ``content || tool_name || tool_calls`` and the trigram table indexes
# every row (including role='tool'). We never CREATE these on a fresh install —
# fresh installs are born on the v23 external-content schema above. These
# constants exist so a legacy DB is never accidentally handed the v23 DDL
# (which would create the external-content trigram source VIEW and leave the
# DB in a mixed, broken state). `optimize_fts_storage()` is what migrates a
# legacy DB to the v23 shape.
LEGACY_FTS_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE({_FTS_NEW_INDEXED_CONTENT_SQL}, '')
        || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE({_FTS_NEW_INDEXED_CONTENT_SQL}, '')
        || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""


LEGACY_FTS_TRIGRAM_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE({_FTS_NEW_INDEXED_CONTENT_SQL}, '')
        || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE({_FTS_NEW_INDEXED_CONTENT_SQL}, '')
        || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""


# ── CJK-bigram FTS index (replaces the trigram index when available) ────
# Trigram needs >=3 chars per term, so 1-2 char CJK terms fell through to a LIKE
# table scan; ``cjk_unicode61`` (native/fts5_cjk/, loadable) re-emits CJK runs as
# overlapping bigrams. Same v23 discipline as the trigram table: external-content
# over a tool-row-excluding view, triggers gated on a DEDICATED marker pair
# (fts_cjk_rebuild_high_water / _progress). The table exists ONLY when the
# tokenizer loads; a process that cannot load it drops the cjk triggers (writes
# keep working; the index goes stale until the next optimize-storage).
#
# Split DDL: the table/view is safe to ensure any time; triggers are created ONLY
# while the index is complete-or-marker-gated. A stale index must keep its
# triggers DROPPED — an external-content 'delete' for a rowid the index never
# held is the canonical FTS5 corruption hazard.
# The trigram tokenizer needs >=3 chars per query term, so 1-2 char CJK terms (ubiquitous in Korean/Chinese:
# 일본, 구글, 项目, ...) fall through to a LIKE full-table scan — measured 3-6s CPU per query on multi-GB installs
# and the dominant base cost of session_search on CJK workloads. ``cjk_unicode61`` (native/fts5_cjk/, a
# ~250-line loadable FTS5 tokenizer with no dependencies) wraps unicode61: maximal CJK runs are re-emitted
# as overlapping character bigrams (Lucene CJKAnalyzer semantics), everything else passes through unchanged.
# FTS5 phrase semantics turn a query term's consecutive bigrams into exact substring matching down to 2
# chars at index speed. Contributed by Soju06 (PR #65544).
FTS_CJK_TABLE_SQL = """
CREATE VIEW IF NOT EXISTS messages_fts_cjk_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_cjk USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_cjk_src',
    content_rowid='id',
    tokenize='cjk_unicode61'
);
"""

FTS_CJK_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;
"""


def fts5_cjk_so_path(home: Path) -> Path:
    """Location of the cjk_unicode61 loadable extension for the store in the folder ``home``."""
    env = os.getenv("HERMES_FTS5_CJK_SO")
    return Path(env).expanduser() if env else home / "lib" / "libfts5_cjk.so"


def _cjk_fts_config_enabled() -> bool:
    """config.yaml ``sessions.cjk_fts`` (default on) for the profile being served."""
    value = routed_sessions_setting("cjk_fts", "HERMES_CJK_FTS")
    return value is None or str(value).strip().lower() not in ("0", "false", "off", "no")


def load_fts5_cjk_extension(conn: sqlite3.Connection, home: Path) -> bool:
    """Best-effort load of the cjk_unicode61 tokenizer; False (never raises) when
    the .so is absent, ``sessions.cjk_fts`` is off, or loading is compiled out."""
    path = fts5_cjk_so_path(home)
    if not _cjk_fts_config_enabled() or not path.exists():
        return False
    try:
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(path))
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:
        logger.warning("fts5_cjk extension load failed (%s)", path, exc_info=True)
        return False


_FTS_DDL_TABLES = ("messages_fts", "messages_fts_trigram")


def _stored_form(sql: str) -> str:
    """``sqlite_master.sql`` as compared here: whitespace collapsed, none kept around parentheses and commas (the
    same definition written with different spacing is the same layout), and the object name unquoted (an ALTER
    TABLE RENAME round trip stores it as ``"name"`` without changing the layout)."""
    collapsed = re.sub(r"\s*([(),])\s*", r"\1", " ".join((sql or "").split()))
    return re.sub(r'^CREATE (VIRTUAL TABLE|VIEW) "([^"]+)"', r"CREATE \1 \2", collapsed)


def _expected_stored_sql(ddl: str, kind: str, name: str) -> str:
    """The stored form of the ``CREATE <kind> IF NOT EXISTS <name>`` statement in *ddl* (``IF NOT EXISTS``
    and ``;`` dropped), i.e. what ``sqlite_master.sql`` holds once the DDL has run."""
    start = ddl.index(f"CREATE {kind} IF NOT EXISTS {name} ")
    end = ddl.index(";", start)
    return _stored_form(ddl[start:end].replace(" IF NOT EXISTS", "", 1))


def _expected_vtable_sql(ddl: str, table: str) -> str:
    return _expected_stored_sql(ddl, "VIRTUAL TABLE", table)


# Every stored definition a released version has created, oldest first; the LAST entry is the one this code
# creates. optimize-storage rebuilds only the older entries: an unrecognised definition (a newer release's, after
# a downgrade) is left alone. Literal on purpose: a change to FTS_SQL / FTS_TRIGRAM_SQL fails
# test_the_layout_registry_ends_with_the_shipping_ddl until its new form is appended here, which keeps the
# previous form recognisable as old.
FTS_VTABLE_LAYOUTS: dict[str, tuple[str, ...]] = {
    "messages_fts": (
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content,content=messages,content_rowid=id)",
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content)",
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content,tool_name,tool_calls,content='messages',"
        "content_rowid='id')",
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content,tool_name,tool_calls,"
        "content='messages_fts_src',content_rowid='id')",
    ),
    "messages_fts_trigram": (
        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content,content=messages,content_rowid=id,"
        "tokenize='trigram')",
        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content,tokenize='trigram')",
        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content,tool_name,tool_calls,"
        "content='messages_fts_trigram_src',content_rowid='id',tokenize='trigram')",
        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content,tool_name,"
        "content='messages_fts_trigram_src',content_rowid='id',tokenize='trigram')",
    ),
}


def _stored_vtables(cursor: sqlite3.Cursor) -> dict[str, str]:
    rows = cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)", _FTS_DDL_TABLES).fetchall()
    return {name: _stored_form(sql) for name, sql in rows}


def _is_older_layout(table: str, stored: str) -> bool:
    """A definition an earlier release created. The two shape checks keep every store the pre-registry code
    treated as old (any base index without ``tool_name``, any trigram with ``tool_calls``) on the opt-in path."""
    if stored in FTS_VTABLE_LAYOUTS[table][:-1]:
        return True
    if table == "messages_fts":
        return "tool_name" not in stored
    return "tool_calls" in stored.lower()


def base_index_recognised(cursor: sqlite3.Cursor) -> bool:
    """``messages_fts`` holds a definition this code or an earlier release created; the open-time realign
    replaces nothing else."""
    return _stored_vtables(cursor).get("messages_fts") in FTS_VTABLE_LAYOUTS["messages_fts"]


def base_index_realigned_on_open(stored: str) -> bool:
    """External-content base index not reading ``messages_fts_src``: the next read-write open rebuilds it from
    the current DDL (``_migrate_misaligned_fts_source``), so it is never opt-in work."""
    return "tool_name" in stored and "messages_fts_src" not in stored


def fts_layout_drift(cursor: sqlite3.Cursor) -> list[str]:
    """FTS tables on an older release's definition that only ``optimize-storage`` replaces; empty otherwise.

    Shared by the update notice, doctor and optimize-storage's demote decision. An absent table is not drift
    (the trigram index is optional; the base index is created on open), nor is a base index the next open
    realigns, nor a definition this code does not recognise. The cjk index owns its own markers.
    """
    return [table for table, stored in _stored_vtables(cursor).items()
            if _is_older_layout(table, stored)
            and not (table == "messages_fts" and base_index_realigned_on_open(stored))]


def fts_layout_current(cursor: sqlite3.Cursor) -> bool:
    """True when every present FTS table, and the base index's source view, is exactly what this code creates:
    the only state the ``fts_storage_version`` stamp may be written over."""
    stored = _stored_vtables(cursor)
    if any(sql != FTS_VTABLE_LAYOUTS[table][-1] for table, sql in stored.items()):
        return False
    if "messages_fts" not in stored:
        return True
    row = cursor.execute("SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'messages_fts_src'").fetchone()
    return row is not None and _stored_form(row[0]) == _expected_stored_sql(FTS_SQL, "VIEW", "messages_fts_src")


def base_source_view_is_stale(cursor: sqlite3.Cursor) -> bool:
    """``messages_fts_src`` exists but differs from what this code creates (``CREATE VIEW IF NOT EXISTS`` never
    replaces it)."""
    row = cursor.execute("SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'messages_fts_src'").fetchone()
    return row is not None and _stored_form(row[0]) != _expected_stored_sql(FTS_SQL, "VIEW", "messages_fts_src")


# FTS5 shadow tables the virtual-table engine owns. `sqlite3 .recover` re-emits them
# as ordinary tables but cannot re-emit the CREATE VIRTUAL TABLE row, so every later
# CREATE VIRTUAL TABLE fails with "fts5: error creating shadow table <name>: table
# already exists" until the orphans are dropped (#103840).
_FTS5_SHADOW_SUFFIXES = ("content", "data", "docsize", "idx", "config")


def _drop_orphan_fts_shadow_tables(cursor: sqlite3.Cursor, families: Sequence[str]) -> list[str]:
    """Drop, per family, shadow tables whose virtual table row is absent from sqlite_master.

    Matches exact shadow names only (never a prefix LIKE, so the base family cannot reach
    ``messages_fts_trigram_*``) and leaves a family alone whenever its vtable is live. The
    shadows are derived index state; the caller recreates and rebuilds from ``messages``.
    Returns the families that were repaired.
    """
    repaired: list[str] = []
    for family in families:
        vtable_live = cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? "
            "AND sql LIKE 'CREATE VIRTUAL TABLE%'",
            (family,),
        ).fetchone()
        if vtable_live:
            continue
        shadows = [f"{family}_{suffix}" for suffix in _FTS5_SHADOW_SUFFIXES]
        orphans = [row[0] for row in cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({','.join('?' for _ in shadows)})",
            shadows,
        ).fetchall()]
        if not orphans:
            continue
        for name in orphans:
            cursor.execute(f'DROP TABLE "{name}"')
        logger.warning(
            "Dropped orphan FTS5 shadow tables of %s (%s); the index is recreated from messages",
            family, ", ".join(orphans),
        )
        repaired.append(family)
    return repaired
