"""Shared module-level constants for the SessionDB family of modules.

Extracted verbatim from hermes_state.py so the SessionDB mixin modules
(hermes_state_search / hermes_state_schema / hermes_state_portability) can
reference them without importing hermes_state (which would be a cycle).
hermes_state re-imports every name here for backward compatibility.
"""

import contextlib
import json
import logging
import os
import sys
import time
from typing import Any

from agent.skill_commands import (
    SKILL_EXCERPT_JOINT,
    SKILL_SCAFFOLD_SQL_LIKE,
    describe_skill_invocation,
)
from agent.context_compressor import (
    LEGACY_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)


# Session preview = the head of the first user message, shown wherever a
# session has no title (sidebar rows, pickers, exports, the desktop's
# `sessionTitle` fallback).
#
# A /skill invocation expands into a message that embeds the whole skill body,
# so the plain head of it previews the SKILL's opening prose as if the user had
# written it. Scaffolded rows therefore carry a wider excerpt so
# ``_shape_preview`` can hand it to ``describe_skill_invocation`` and recover
# ``/work — fix the title leak``: the whole message while it stays under the
# budget, and head + tail (where the typed instruction lands) once it doesn't.
_PREVIEW_HEAD_CHARS = 63


_PREVIEW_SCAFFOLD_WINDOW = 400


_PREVIEW_MAX_CHARS = 60


def escape_like(text: str) -> str:
    """Escape SQL LIKE wildcards so operator/session-derived text matches
    literally.  Pair with ``ESCAPE '\\'`` in the clause.

    ``%`` and ``_`` are wildcards to LIKE, and ``_`` in particular is common
    in the values these patterns run against (branch names, session titles,
    filesystem paths).  A match documented as substring/prefix must not
    silently widen.
    """
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_PREVIEW_CONTENT_SQL = "REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' ')"


_PREVIEW_SCAFFOLDED_SQL = f"m.content LIKE '{SKILL_SCAFFOLD_SQL_LIKE}'"


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


_SQL_WHITESPACE = "CHAR(9) || CHAR(10) || CHAR(13) || CHAR(32)"


def _sql_ltrim_whitespace(expression: str) -> str:
    return f"LTRIM({expression}, {_SQL_WHITESPACE})"


def _sql_trim_whitespace(expression: str) -> str:
    return f"TRIM({expression}, {_SQL_WHITESPACE})"


def _sql_starts_with(expression: str, prefixes: tuple[str, ...]) -> str:
    trimmed = _sql_ltrim_whitespace(expression)
    checks = [
        f"SUBSTR({trimmed}, 1, {len(prefix)}) = {_sql_literal(prefix)}"
        for prefix in prefixes
    ]
    return "(" + " OR ".join(checks) + ")"


# Current and historical long-form prefixes share this complete introduction;
# their stale-item guidance diverges only after it. Matching the whole intro
# avoids treating an ordinary user message that merely starts with the short
# bracketed label as a compaction carrier.
_PREVIEW_LONG_FORM_PREFIX = SUMMARY_PREFIX.split("Do NOT answer", 1)[0]
_PREVIEW_SUMMARY_PREFIXES = (
    _PREVIEW_LONG_FORM_PREFIX,
    LEGACY_SUMMARY_PREFIX,
)
_PREVIEW_STANDALONE_SUMMARY_SQL = _sql_starts_with(
    "m.content", _PREVIEW_SUMMARY_PREFIXES
)
_PREVIEW_MERGED_AFTER_SQL = (
    f"SUBSTR(m.content, INSTR(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)})"
    f" + {len(_MERGED_SUMMARY_DELIMITER)})"
)
_PREVIEW_MERGED_SUMMARY_SQL = (
    f"(INSTR(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) > 0"
    f" AND {_sql_starts_with(_PREVIEW_MERGED_AFTER_SQL, _PREVIEW_SUMMARY_PREFIXES)})"
)
_PREVIEW_MERGED_PRIOR_SQL = _sql_trim_whitespace(
    f"SUBSTR(m.content, 1, INSTR(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) - 1)"
)
_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL = _sql_ltrim_whitespace(
    _PREVIEW_MERGED_PRIOR_SQL
)
_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL = (
    f"CASE WHEN SUBSTR({_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL}, 1,"
    f" {len(_MERGED_PRIOR_CONTEXT_HEADER)}) = {_sql_literal(_MERGED_PRIOR_CONTEXT_HEADER)}"
    f" THEN {_sql_ltrim_whitespace(f'SUBSTR({_PREVIEW_MERGED_PRIOR_LTRIMMED_SQL}, {len(_MERGED_PRIOR_CONTEXT_HEADER) + 1})')}"
    f" ELSE {_PREVIEW_MERGED_PRIOR_SQL} END"
)
_PREVIEW_FORCE_USER_REMAINDER_SQL = (
    f"SUBSTR(m.content, INSTR(m.content, {_sql_literal(_SUMMARY_END_MARKER)})"
    f" + {len(_SUMMARY_END_MARKER)})"
)

# Session preview subqueries select their first eligible user-authored content.
# Pure compaction rows are ineligible; force-user-leading and merged carriers
# remain eligible only when authentic content survives the wire boundary.
_PREVIEW_ELIGIBLE_SQL = (
    f"((NOT {_PREVIEW_STANDALONE_SUMMARY_SQL} AND NOT {_PREVIEW_MERGED_SUMMARY_SQL})"
    f" OR ({_PREVIEW_STANDALONE_SUMMARY_SQL}"
    f" AND INSTR(m.content, {_sql_literal(_SUMMARY_END_MARKER)}) > 0"
    f" AND LENGTH({_sql_trim_whitespace(_PREVIEW_FORCE_USER_REMAINDER_SQL)}) > 0)"
    f" OR ({_PREVIEW_MERGED_SUMMARY_SQL}"
    f" AND LENGTH({_sql_trim_whitespace(_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL)}) > 0))"
)


# The shared ``_preview_raw`` SELECT expression, interpolated by every listing
# query. A scaffolded row gets a wider excerpt: the whole message while it fits
# the budget, else head + tail (where the typed instruction lands) spliced
# around SKILL_EXCERPT_JOINT.
_PREVIEW_RAW_SELECT = (
    f"CASE WHEN {_PREVIEW_STANDALONE_SUMMARY_SQL}"
    f" THEN {_PREVIEW_FORCE_USER_REMAINDER_SQL}"
    f" WHEN {_PREVIEW_MERGED_SUMMARY_SQL}"
    f" THEN {_PREVIEW_MERGED_PRIOR_UNWRAPPED_SQL}"
    f" WHEN {_PREVIEW_SCAFFOLDED_SQL}"
    f" AND LENGTH(m.content) > {_PREVIEW_SCAFFOLD_WINDOW * 2}"
    f" THEN SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_SCAFFOLD_WINDOW})"
    f" || '{SKILL_EXCERPT_JOINT}'"
    f" || SUBSTR({_PREVIEW_CONTENT_SQL}, -{_PREVIEW_SCAFFOLD_WINDOW})"
    f" WHEN {_PREVIEW_SCAFFOLDED_SQL}"
    f" THEN SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_SCAFFOLD_WINDOW * 2})"
    f" ELSE SUBSTR({_PREVIEW_CONTENT_SQL}, 1, {_PREVIEW_HEAD_CHARS}) END"
)


def _shape_preview(raw: Any) -> str:
    """Turn a ``_preview_raw`` column into the short preview callers show."""
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    described = describe_skill_invocation(text)
    text = described if described is not None else text.split(SKILL_EXCERPT_JOINT)[0]
    if len(text) > _PREVIEW_MAX_CHARS:
        return text[:_PREVIEW_MAX_CHARS] + "..."
    return text


# A child session counts as a /branch (kept visible, never cascade-deleted) if
# it carries the stable marker OR the legacy end_reason heuristic holds.
_BRANCH_CHILD_SQL = (
    "json_extract(COALESCE({a}.model_config, '{{}}'), '$._branched_from') IS NOT NULL"
    " OR EXISTS (SELECT 1 FROM sessions p"
    "            WHERE p.id = {a}.parent_session_id"
    "            AND p.end_reason = 'branched'"
    "            AND {a}.started_at >= p.ended_at)"
)


_COMPRESSION_CHILD_SQL = (
    "EXISTS (SELECT 1 FROM sessions p"
    "        WHERE p.id = {a}.parent_session_id"
    "        AND p.end_reason = 'compression')"
)


_RESET_END_REASONS = (
    "session_reset",
    # switch_session() never creates a child row, but pre-marker DBs can hold
    # legacy reset children whose parent later ended with 'session_switch'
    # (resumed then switched away before reopen-time stamping existed). Also
    # keeps this set identical to the recovery fence in
    # find_latest_gateway_session_for_peer, which interpolates
    # _RESET_END_REASONS_SQL so the two cannot drift.
    "session_switch",
    "idle",
    "daily",
    "suspended",
    "resume_pending_expired",
)
_RESET_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RESET_END_REASONS)

# Accidental end reasons that recovery treats as resumable (see
# docs/session-lifecycle.md "recoverable accidental reasons"). Interpolated
# into the recovery SQL below AND exposed as SessionDB.RECOVERABLE_END_REASONS
# so the tuple is the single source of truth — literals cannot drift.
_RECOVERABLE_END_REASONS = (
    "agent_close",
    "ws_orphan_reap",
    # A stale sentinel-parked runtime quietly superseded by a fresh
    # session.resume of the same stored session (no reclaimed broadcast);
    # the stored session stays resumable like any accidental end.
    "superseded_by_resume",
    # Startup sweep of rows orphaned by a dead gateway process (#65194):
    # the in-process ws-orphan grace timer died with the process, so the
    # row was closed at the next boot instead. Same accident class as
    # ws_orphan_reap — kept distinct for forensics — and equally resumable.
    "startup_orphan_reap",
)
_RECOVERABLE_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RECOVERABLE_END_REASONS)

# End reasons written by AUTOMATIC infrastructure cleanup (server shutdown,
# orphan reapers, idle/LRU eviction) rather than by a deliberate conversation
# boundary (compression, session_reset, session_switch, explicit user close).
# An automatic stamp records "some runtime went away", NOT "this conversation
# ended" — so a writer that can prove the conversation is still live (e.g. an
# active compression rotation holding the lease, #88197) may treat the stamp
# as stale and clear it. Superset of the recoverable set: those are already
# resumable accidents; the extra TUI reasons are the same accident class but
# were historically only known to tui_gateway's _AUTOMATIC_SESSION_END_REASONS.
_AUTOMATIC_END_REASONS = frozenset(_RECOVERABLE_END_REASONS) | {
    "tui_shutdown",
    "ws_disconnect",
    "idle_timeout",
    "lru_evict",
}


def is_automatic_end_reason(reason) -> bool:
    """True when *reason* is an automatic-cleanup end stamp (see above).

    Single owner of the "accidental vs deliberate end" predicate — every
    compression-liveness site must call this instead of re-implementing the
    reason taxonomy (#88197, never-patch-predicates).
    """
    return isinstance(reason, str) and reason in _AUTOMATIC_END_REASONS


def _legacy_reset_child_sql(alias: str, reasons_sql: str) -> str:
    """Pre-marker reset-continuation heuristic.

    A child is a legacy reset continuation when it rides its parent's exact
    non-empty routing key and the parent ended at a reset boundary. Shared by
    the listing predicate (``_RESET_CHILD_SQL``) and ``reopen_session()``'s
    marker-stamping UPDATE so the two sites cannot drift; ``reasons_sql`` is
    either the literal ``_RESET_END_REASONS_SQL`` or a bound-placeholder list.
    """
    return (
        f"EXISTS (SELECT 1 FROM sessions p"
        f"            WHERE p.id = {alias}.parent_session_id"
        f"            AND p.end_reason IN ({reasons_sql})"
        f"            AND {alias}.session_key IS NOT NULL"
        f"            AND {alias}.session_key != ''"
        f"            AND {alias}.session_key = p.session_key)"
    )


# A reset starts a separate user-visible conversation even though gateway rows
# retain parent_session_id for durable lineage. New rows carry the stable
# marker; the same-key fallback recovers rows written before the marker existed.
# Requiring the exact non-empty routing key keeps ordinary child/subagent rows
# out even when their parent is later reset.
_RESET_CHILD_SQL = (
    "json_extract(COALESCE({a}.model_config, '{{}}'), '$._reset_from') IS NOT NULL"
    " OR " + _legacy_reset_child_sql("{a}", _RESET_END_REASONS_SQL)
)


# Rows that surface in pickers: roots + branch/reset children. Subagent runs
# and compression continuations stay hidden.
_LISTABLE_CHILD_SQL = (
    f"(s.parent_session_id IS NULL OR {_BRANCH_CHILD_SQL.format(a='s')}"
    f" OR {_RESET_CHILD_SQL.format(a='s')})"
)


def _ephemeral_child_sql(alias: str = "s") -> str:
    """Subagent runs, not branch, reset, or compression children."""
    branch = _BRANCH_CHILD_SQL.format(a=alias)
    compression = _COMPRESSION_CHILD_SQL.format(a=alias)
    reset = _RESET_CHILD_SQL.format(a=alias)
    return (
        f"({alias}.parent_session_id IS NOT NULL"
        f" AND NOT ({branch})"
        f" AND NOT ({compression})"
        f" AND NOT ({reset}))"
    )


def _sql_session_last_active(alias: str = "s") -> str:
    """SQL expression for session recency used by list/status surfaces.

    Freshest of ``last_activity_at`` (mid-turn agent activity heartbeat) and
    the latest message timestamp, then fall back to ``started_at``.

    Must not prefer a stale heartbeat over a newer message: durable
    heartbeats are rate-limited (~60s), so after a turn writes messages
    ``last_activity_at`` can lag ``MAX(messages.timestamp)``.
    """
    msg_max = (
        f"(SELECT MAX(_act_m.timestamp) FROM messages _act_m "
        f"WHERE _act_m.session_id = {alias}.id)"
    )
    return (
        f"COALESCE("
        f"(SELECT MAX(_act_v.v) FROM ("
        f"SELECT {alias}.last_activity_at AS v "
        f"UNION ALL "
        f"SELECT {msg_max}"
        f") _act_v), "
        f"{alias}.started_at)"
    )


def _sql_session_last_active_by_id(session_id_expr: str) -> str:
    """Same freshest-of expression keyed by a session-id SQL expression."""
    msg_max = (
        f"(SELECT MAX(_act_m.timestamp) FROM messages _act_m "
        f"WHERE _act_m.session_id = {session_id_expr})"
    )
    activity = (
        f"(SELECT last_activity_at FROM sessions _act_s "
        f"WHERE _act_s.id = {session_id_expr})"
    )
    started = (
        f"(SELECT started_at FROM sessions _act_s "
        f"WHERE _act_s.id = {session_id_expr})"
    )
    return (
        f"COALESCE("
        f"(SELECT MAX(_act_v.v) FROM ("
        f"SELECT {activity} AS v "
        f"UNION ALL "
        f"SELECT {msg_max}"
        f") _act_v), "
        f"{started})"
    )


SCHEMA_VERSION = 27


# FTS storage-layout version, tracked INDEPENDENTLY of SCHEMA_VERSION in the
# state_meta key ``fts_storage_version``. The main schema version advances
# freely on open (so future migrations always land); the FTS *layout* only
# reaches the current version when a DB is either born fresh or explicitly
# optimized via ``hermes sessions optimize-storage``. A legacy DB sits at
# layout 0 (marker absent) with a working inline index until the user opts in.
#   1 = v23 external-content layout (content/tool_name/tool_calls,
#       tool-row-excluded trigram)
FTS_STORAGE_VERSION = 1


# Cap on user-controlled FTS5 query input before regex/sanitizer processing.
# Search queries do not need to be arbitrarily large, and bounding them keeps
# sanitizer/runtime behavior predictable under adversarial input.
MAX_FTS5_QUERY_CHARS = 2_048


_FTS_TRIGGERS = (
    "messages_fts_insert",
    "messages_fts_delete",
    "messages_fts_update",
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)


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
    source TEXT NOT NULL,
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
    profile_name TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
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
    display_metadata TEXT
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
    delivery_claim TEXT,
    delivery_claimed_at REAL
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


# Indexes that reference columns added in later schema versions must be
# created AFTER _reconcile_columns() has had a chance to ADD them on
# existing databases. SCHEMA_SQL above is run by sqlite executescript
# which would otherwise fail on legacy DBs ("no such column: active").
DEFERRED_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_active_null
    ON messages(active) WHERE active IS NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_session_key
    ON sessions(session_key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_system_prompt_hash
    ON sessions(system_prompt_hash);
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
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages
WHEN (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
WHEN (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

-- UPDATE OF skips the trigger entirely for non-content column writes
-- (status/compacted/observed/etc.), which is stronger than the WHEN gate
-- alone and avoids FTS I/O saturation on large state.db (#68858 / #73639).
CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;
"""


# Trigram FTS5 table for CJK substring search.  The default unicode61
# tokenizer splits CJK characters into individual tokens, breaking phrase
# matching.  The trigram tokenizer creates overlapping 3-byte sequences so
# substring queries work natively for any script (CJK, Thai, etc.).
#
# The trigram index is the most expensive index in state.db (~2.6x the size
# of the text it covers), and ``role='tool'`` rows are ~90% of message bytes
# while being almost entirely machine noise (base64 payloads, file dumps,
# delegation transcripts).  The index therefore reads through
# ``messages_fts_trigram_src``, a view that excludes tool rows — they stay
# fully stored in ``messages`` and fully searchable via the standard
# ``messages_fts`` index; they just don't get trigram (CJK substring)
# treatment.  ``search_messages`` routes CJK queries that filter on
# ``role='tool'`` to the LIKE fallback for the same reason.
FTS_TRIGRAM_SQL = """
CREATE VIEW IF NOT EXISTS messages_fts_trigram_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_trigram_src',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update
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
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;
"""


_FTS_CJK_TRIGGERS = (
    "messages_fts_cjk_insert",
    "messages_fts_cjk_delete",
    "messages_fts_cjk_update",
)


# state_meta breadcrumb set when a tokenizer-less process had to drop the
# cjk triggers to keep message writes alive: rows written from that moment
# on are missing from the cjk index, so it must not serve reads until
# `hermes sessions optimize-storage` rebuilds it on a capable host.
FTS_CJK_STALE_KEY = "fts_cjk_stale"


# Durable breadcrumb for a base/trigram FTS index that was detached from the
# canonical messages table after runtime corruption. While present, startup
# must rebuild the complete index before reinstalling sync triggers: rows may
# have been written while those triggers were absent, so merely recreating
# them would preserve an unknown index gap.
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
LEGACY_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""


LEGACY_FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update
AFTER UPDATE OF content, tool_name, tool_calls ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""


# ── Cross-process full-FTS-rebuild admission (single authority) ──────────────
#
# Several independent Hermes processes routinely share one state.db (gateway
# service, the Desktop app's `hermes serve` backend, interactive CLI sessions,
# the TUI slash worker). A full structural FTS rebuild — the FTS5 'rebuild'
# command or the drop/recreate script in `_recover_stale_fts` — must only ever
# run in ONE of them at a time: two concurrent rebuilds collide on write and
# have structurally corrupted state.db in production (PR #93200; the
# 2026-08-15 / 2026-08-23 incidents and issues #89293 / #90950).
#
# This is the single admission authority for every full structural rebuild
# entry point: `SessionSearchMixin.rebuild_fts()`,
# `SessionSchemaMixin._rebuild_fts_indexes()` (via `_init_schema`), and
# `SessionSchemaMixin._recover_stale_fts()`. The chunked deferred backfill
# (`fts_rebuild_step`) is deliberately NOT routed through it — it claims
# progress under `_execute_write`'s SQLite transaction authority and is
# intentionally multi-process.
#
# Semantics mirror `hermes_state._cross_process_repair_lock` (the schema-
# surgery authority): portable (msvcrt on Windows, flock elsewhere), bounded
# wait, and FAIL CLOSED — a caller that cannot acquire the lock must NOT
# rebuild. The kernel drops both lock types when the holder dies — UNLESS a
# forked child inherited the lock fd (flock rides the open file description,
# which fork() duplicates), in which case the orphaned descriptor holds the
# lock forever (issue #100108). `_acquire_db_flock` therefore records the
# holder's pid + start time under the lock and, when the recorded holder is
# provably dead, breaks the orphaned lock by unlinking and retaking it on a
# fresh inode; indeterminate liveness still defers. It lives here (not
# hermes_state) because the search/schema mixins cannot import hermes_state
# (cycle).
#
# The lock file is `<db>.fts_rebuild.lock`, distinct from `<db>.repair.lock`:
# schema surgery runs on an EXCLUSIVE offline connection and can legitimately
# take minutes in VACUUM, while runtime rebuilds run on live connections. The
# timeout is sized for a full 'rebuild' of both indexes on a large DB.

logger = logging.getLogger("hermes_state")

_FTS_REBUILD_LOCK_TIMEOUT_SECONDS = 120.0
_FTS_REBUILD_LOCK_POLL_SECONDS = 0.1
_IS_WINDOWS = sys.platform == "win32"

# Post-break re-acquire budget: once a provably-orphaned lock has been broken
# the fresh inode is uncontended (or contended only by live processes), so a
# short bounded wait suffices — never re-enter the full timeout.
_LOCK_BREAK_REACQUIRE_SECONDS = 5.0


def _proc_start_ticks(pid: int):
    """Kernel start time of *pid* in clock ticks, or None when unknowable.

    Field 22 of ``/proc/<pid>/stat`` (``starttime``) uniquely identifies a
    process together with its PID: a recycled PID gets a different start
    time. Returns None off Linux or on any read/parse failure — callers must
    treat None as "unknowable" and FAIL CLOSED.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            stat = fh.read()
        # comm (field 2) may contain spaces/parens; split after the LAST ')'.
        return int(stat.rsplit(b")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _read_lock_holder_record(handle):
    """Best-effort parse of the holder metadata JSON in a lock file."""
    try:
        handle.seek(0)
        raw = handle.read(4096)
    except (OSError, ValueError):
        return None
    if not raw:
        return None
    try:
        record = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _write_lock_holder_record(handle) -> None:
    """Record this process as the lock holder (advisory, best effort).

    Written under the flock so contenders that time out can tell an
    orphaned-fd holder (recorded process dead, flock inherited by a forked
    child — issue #100108) from a live wedged holder.
    """
    try:
        record = {
            "pid": os.getpid(),
            "start_ticks": _proc_start_ticks(os.getpid()),
            "acquired_at": time.time(),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(record, sort_keys=True).encode("utf-8"))
        handle.flush()
    except (OSError, ValueError):
        pass


def _clear_lock_holder_record(handle) -> None:
    """Erase holder metadata before a normal release.

    Guarantees that a surviving record always describes an ABNORMAL exit
    (holder died without releasing), which is the only condition under which
    a contender may break the lock.
    """
    try:
        handle.seek(0)
        handle.truncate()
        handle.flush()
    except (OSError, ValueError):
        pass


def _lock_holder_provably_dead(record) -> bool:
    """True ONLY when the recorded holder is provably dead or PID-recycled.

    Any indeterminate state (no record, malformed record, PID owned by
    another user, /proc unavailable, start-time unknowable) returns False —
    the caller must FAIL CLOSED and defer, never break a possibly-live
    holder's lock.
    """
    if not isinstance(record, dict):
        return False
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        # PermissionError et al.: the PID exists (or is unknowable) — closed.
        return False
    recorded_ticks = record.get("start_ticks")
    if recorded_ticks is None:
        return False
    current_ticks = _proc_start_ticks(pid)
    if current_ticks is None:
        return False
    # Same PID, different kernel start time: the recorded holder is dead and
    # its PID was recycled by an unrelated process.
    return current_ticks != recorded_ticks


def _acquire_db_flock(lock_path, handle, timeout_seconds, poll_seconds, description):
    """Bounded POSIX flock acquire with orphaned-holder staleness break.

    Returns ``(acquired, handle)``; *handle* may have been re-opened (the
    caller owns closing whichever handle comes back).

    Why breaking exists at all (issue #100108): ``flock`` belongs to the open
    file DESCRIPTION, which ``fork()`` duplicates into every child. A holder
    that forks (multiprocessing worker, daemonized helper) and then dies
    leaves the flock held by a child that will never release it — the
    kernel's holder-death release never triggers, and every contender defers
    forever. The recorded-holder liveness check distinguishes exactly that
    case: the process that ACQUIRED is provably dead (so its critical section
    died with it), yet the flock is still held. Only then is the lock file
    unlinked and retaken on a fresh inode; the orphan's flock stays on the
    old unlinked inode where it blocks nobody. Every successful acquire
    verifies its inode still names *lock_path*, so a racer that locked a dead
    inode retries instead of running concurrently with the breaker.
    Indeterminate liveness always defers (fail closed).
    """
    import fcntl

    deadline = time.monotonic() + timeout_seconds
    broke_lock = False
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            if time.monotonic() < deadline:
                time.sleep(poll_seconds)
                continue
            if broke_lock:
                return False, handle
            record = _read_lock_holder_record(handle)
            if not _lock_holder_provably_dead(record):
                return False, handle
            logger.warning(
                "%s %s is held by an orphaned file descriptor (recorded "
                "holder pid %s is dead — a forked child inherited the lock "
                "fd); breaking the stale lock and retaking it on a fresh "
                "file.",
                description,
                lock_path,
                (record or {}).get("pid"),
            )
            try:
                os.unlink(lock_path)
                handle.close()
                handle = open(lock_path, "a+b")
            except OSError as exc:
                logger.warning(
                    "Could not break stale %s %s (%s) — deferring.",
                    description,
                    lock_path,
                    exc,
                )
                return False, handle
            broke_lock = True
            deadline = time.monotonic() + _LOCK_BREAK_REACQUIRE_SECONDS
            continue
        # flock acquired — verify the path still names our inode: a breaker
        # may have unlinked/replaced the file while we waited, and a lock on
        # a dead inode excludes nobody.
        try:
            fd_stat = os.fstat(handle.fileno())
            path_stat = os.stat(lock_path)
            same_file = (
                fd_stat.st_dev == path_stat.st_dev
                and fd_stat.st_ino == path_stat.st_ino
            )
        except OSError:
            same_file = False
        if same_file:
            _write_lock_holder_record(handle)
            return True, handle
        try:
            handle.close()
            handle = open(lock_path, "a+b")
        except OSError:
            return False, handle
        if time.monotonic() >= deadline:
            return False, handle


def _describe_lock_holder(record) -> str:
    """Human-readable holder identity for deferral warnings."""
    if not isinstance(record, dict) or "pid" not in record:
        return "unknown (no holder record; pre-fix writer or non-Hermes)"
    pid = record.get("pid")
    acquired_at = record.get("acquired_at")
    age = ""
    try:
        if acquired_at is not None:
            age = f", acquired {time.time() - float(acquired_at):.0f}s ago"
    except (TypeError, ValueError):
        pass
    return f"pid {pid}{age}"


@contextlib.contextmanager
def fts_rebuild_admission(db_path):
    """Serialize full structural FTS rebuilds on *db_path* across processes.

    Yields True when this process holds the rebuild authority, False when the
    bounded acquire timed out or the lock file could not be opened at all. A
    caller that gets False must NOT perform a full rebuild — proceeding is
    exactly the concurrent-rebuild interleaving this lock exists to prevent
    (fail closed). The deferred/stale breadcrumb machinery already guarantees
    a skipped rebuild is retried later.

    ``db_path`` may be a str or Path; None (in-memory DB / tests without a
    file path) yields True — a private in-memory DB has no cross-process
    surface.
    """
    if db_path is None:
        yield True
        return
    lock_path = f"{db_path}.fts_rebuild.lock"
    try:
        handle = open(lock_path, "a+b")
    except OSError as exc:
        # Fail closed, exactly as a timed-out acquire does. A lock file we
        # cannot even open means the filesystem is out of space, inodes or
        # descriptors — and a sibling process that opened ITS handle before
        # the disk filled is still holding the authority and rebuilding.
        # Yielding True here handed every process on a full disk a concurrent
        # structural rebuild of the same live state.db with no cross-process
        # authority at all: the disk-full trigger and the re-corruption on
        # every multi-writer boot in #100368. Deferring costs nothing that
        # was reachable anyway — the breadcrumb retries, and on a read-only
        # directory the rebuild's own writes could not have committed either.
        logger.warning(
            "Could not open FTS rebuild lock %s (%s) — deferring this rebuild "
            "rather than running it without cross-process authority.",
            lock_path, exc,
        )
        yield False
        return

    acquired = False
    try:
        if _IS_WINDOWS:
            deadline = time.monotonic() + _FTS_REBUILD_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_FTS_REBUILD_LOCK_POLL_SECONDS)
        else:
            acquired, handle = _acquire_db_flock(
                lock_path,
                handle,
                _FTS_REBUILD_LOCK_TIMEOUT_SECONDS,
                _FTS_REBUILD_LOCK_POLL_SECONDS,
                "FTS rebuild lock",
            )
        if not acquired:
            record = None if _IS_WINDOWS else _read_lock_holder_record(handle)
            logger.warning(
                "FTS rebuild lock %s held by another process for more than "
                "%.0fs — deferring this rebuild to avoid racing the holder "
                "(the stale-FTS breadcrumb keeps it retryable). "
                "Recorded holder: %s.",
                lock_path, _FTS_REBUILD_LOCK_TIMEOUT_SECONDS,
                _describe_lock_holder(record),
            )
        yield acquired
    finally:
        try:
            if acquired:
                if _IS_WINDOWS:
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    _clear_lock_holder_record(handle)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - best effort release
            pass
        finally:
            handle.close()
