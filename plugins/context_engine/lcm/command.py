"""Slash-style /lcm command helpers for Hermes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import sqlite3
from typing import Any

from .db_bootstrap import (
    external_content_fts_needs_repair,
    inspect_lcm_schema_health,
    repair_external_content_fts,
)
from .diagnostics import _has_lifecycle_fragmentation, _state_db_path_for_engine
from .ingest_protection import (
    externalized_payload_stats,
    scan_externalized_payload_integrity,
    scan_sqlite_payload_risks,
    sensitive_pattern_status,
)
from .dag import build_nodes_fts_spec
from .presets import (
    explicit_operator_overrides,
    get_preset,
    invalid_operator_overrides,
    preset_env_diff,
    shipped_presets,
    suggest_preset_for_engine,
    unsupported_runtime_fields_text,
)
from .session_patterns import build_session_match_keys, matches_session_pattern
from .store import build_message_fts_spec

def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    precision = 0 if value >= 100 else 1 if value >= 10 else 2
    return f"{value:.{precision}f} {units[unit]}"


def _help_text(error: str | None = None) -> str:
    lines = []
    if error:
        lines.append(error)
        lines.append("")
    lines.extend([
        "LCM command help",
        "- /lcm or /lcm status: show current LCM runtime/session status",
        "- /lcm doctor: run read-only LCM health checks",
        "- /lcm doctor clean: best-effort scan of obvious junk/noise session candidates without deleting anything",
        "- /lcm doctor clean apply: backup-first cleanup for safe pattern-matched candidates only",
        "- /lcm doctor clean lifecycle: read-only scan for lifecycle rows with zero messages/nodes",
        "- /lcm doctor clean lifecycle apply: backup-first cleanup of empty lifecycle rows only",
        "- /lcm doctor repair: read-only scan for SQLite/FTS index repair needs",
        "- /lcm doctor repair apply: backup-first repair/rebuild of message and summary FTS indexes",
        "- /lcm doctor source: read-only scan for legacy blank-source rows",
        "- /lcm doctor source apply: backup-first normalization of legacy blank-source rows to unknown",
        "- /lcm doctor retention: read-only retention analysis for stored session footprint and age",
        "- /lcm backup: create a timestamped SQLite backup before any future cleanup workflow",
        "- /lcm rotate: preview a tail-preserving in-place compact of the active session (read-only)",
        "- /lcm rotate apply: backup-first rotate that advances the lifecycle frontier past pre-tail raw messages",
        "- /lcm preset show [name]: inspect shipped preset metadata and benchmark provenance",
        "- /lcm preset suggest: preview the best shipped preset for the current engine state",
        "- /lcm preset apply <name> --dry-run: preview env-var changes without mutating live config",
        "- /lcm help: show this help",
    ])
    return "\n".join(lines)


def _status_text(engine) -> str:
    status = engine.get_status()
    db_path = Path(engine._store.db_path)
    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    session_bound = bool(engine.current_session_id)
    source_stats = status.get("source_lineage") or {}
    runtime_identity = status.get("runtime_identity") or {}
    source_stats = {
        "messages_total": int(source_stats.get("messages_total", 0) or 0),
        "attributed_messages": int(source_stats.get("attributed_messages", 0) or 0),
        "normalized_unknown_messages": int(source_stats.get("normalized_unknown_messages", 0) or 0),
        "legacy_blank_source_messages": int(source_stats.get("legacy_blank_source_messages", 0) or 0),
        "effective_unknown_messages": int(source_stats.get("effective_unknown_messages", 0) or 0),
        **({"error": source_stats.get("error")} if source_stats.get("error") else {}),
    }
    protection = status.get("ingest_protection") or sensitive_pattern_status(engine._config)

    uninitialized = "(uninitialized)"
    unknown = "(unknown)"
    model = (engine.model or unknown) if session_bound else uninitialized
    provider = (engine.provider or unknown) if session_bound else uninitialized
    context_length_source = (
        (getattr(engine, "_context_length_source", "") or unknown)
        if session_bound
        else uninitialized
    )

    lines = [
        "LCM status",
        f"engine: {status.get('engine', engine.name)}",
        f"plugin_name: {runtime_identity.get('plugin_name', '(unknown)')}",
        f"plugin_version: {runtime_identity.get('plugin_version', '(unknown)')}",
        f"plugin_path: {runtime_identity.get('plugin_path', '(unknown)')}",
        f"module_path: {runtime_identity.get('module_path', '(unknown)')}",
        f"plugin_git_commit: {runtime_identity.get('plugin_git_commit') or '(unavailable)'}",
        f"plugin_git_branch: {runtime_identity.get('plugin_git_branch') or '(unavailable)'}",
        f"plugin_git_dirty: {runtime_identity.get('plugin_git_dirty') if runtime_identity.get('plugin_git_dirty') is not None else '(unavailable)'}",
        f"hermes_home: {runtime_identity.get('hermes_home', '') or '(unset)'}",
        f"session_id: {engine.current_session_id or '(unbound)'}",
        f"session_platform: {engine.current_session_platform or ('(unbound)' if not session_bound else '(unknown)')}",
        f"model: {model}",
        f"provider: {provider}",
        f"database_path: {db_path}",
        f"database_path_source: {runtime_identity.get('database_path_source', '(unknown)')}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"compression_count: {engine.compression_count}",
        f"last_compression_status: {status.get('last_compression_status', 'idle')}",
        f"last_compression_noop_reason: {status.get('last_compression_noop_reason', '') or '(none)'}",
        f"context_length: {engine.context_length if session_bound else '(uninitialized)'}",
        f"context_length_source: {context_length_source}",
        f"context_threshold: {engine._config.context_threshold}",
        f"threshold_tokens: {engine.threshold_tokens if session_bound else '(uninitialized)'}",
        f"cache_metrics_available: {_fmt_bool(status.get('cache_metrics_available'))}",
        f"last_input_tokens: {status.get('last_input_tokens', 0)}",
        f"last_output_tokens: {status.get('last_output_tokens', 0)}",
        f"last_cache_read_tokens: {status.get('last_cache_read_tokens', 0)}",
        f"last_cache_write_tokens: {status.get('last_cache_write_tokens', 0)}",
        f"last_reasoning_tokens: {status.get('last_reasoning_tokens', 0)}",
        f"cache_read_ratio: {float(status.get('cache_read_ratio', 0.0) or 0.0) * 100:.1f}%",
        f"sensitive_patterns_enabled: {_fmt_bool(protection.get('enabled'))}",
        f"sensitive_patterns: {', '.join(protection.get('patterns') or []) or '(none)'}",
        f"sensitive_patterns_source: {protection.get('source', 'default')}",
        # Filter classification for current_session_id (the foreground view).
        # When a side channel is in flight, get_status() reports the bound
        # session's flags; we read the engine properties instead so this row
        # stays consistent with the session_id row above.
        f"session_ignored: {_fmt_bool(engine.current_session_ignored)}",
        f"session_stateless: {_fmt_bool(engine.current_session_stateless)}",
        f"side_channel_active: {_fmt_bool(engine.side_channel_active)}",
        f"conversation_id: {runtime_identity.get('conversation_id', '') or '(unbound)'}",
        f"lifecycle_current_session_id: {runtime_identity.get('lifecycle_current_session_id', '') or '(none)'}",
        f"lifecycle_last_finalized_session_id: {runtime_identity.get('lifecycle_last_finalized_session_id', '') or '(none)'}",
        f"source_messages_total: {source_stats['messages_total']}",
        f"source_attributed_messages: {source_stats['attributed_messages']}",
        f"source_unknown_messages: {source_stats['normalized_unknown_messages']}",
        f"source_legacy_blank_messages: {source_stats['legacy_blank_source_messages']}",
        f"source_effective_unknown_messages: {source_stats['effective_unknown_messages']}",
    ]

    last_rotate_at = status.get("last_rotate_at")
    if last_rotate_at:
        lines.append(
            f"last_rotate_at: "
            f"{datetime.fromtimestamp(float(last_rotate_at), tz=timezone.utc).isoformat(timespec='seconds')}"
        )
        rotate_backup_size = int(status.get("rotate_backup_size", 0) or 0)
        if rotate_backup_size:
            lines.append(f"rotate_backup_size: {_fmt_size(rotate_backup_size)}")
    else:
        lines.append("last_rotate_at: (never)")
    if status.get("rotate_backup_path"):
        lines.append(f"rotate_backup_path: {status['rotate_backup_path']}")

    if session_bound:
        lines.extend([
            f"store_messages: {status.get('store_messages', 0)}",
            f"dag_nodes: {status.get('dag_nodes', 0)}",
        ])
    else:
        lines.append(
            "note: no active Hermes session has initialized LCM in this process yet — after a fresh restart, send one normal message first if you want live per-session runtime details"
        )

    if "ignore_session_patterns_source" in status:
        lines.append(
            f"ignore_session_patterns_source: {status.get('ignore_session_patterns_source')}"
        )
    if "stateless_session_patterns_source" in status:
        lines.append(
            f"stateless_session_patterns_source: {status.get('stateless_session_patterns_source')}"
        )
    if source_stats.get("error"):
        lines.append(f"source_lineage_error: {source_stats['error']}")
    return "\n".join(lines)


def _scan_clean_candidates(engine) -> dict[str, Any]:
    conn = engine._store._conn
    try:
        rows = conn.execute(
            """
            WITH session_ids AS (
                SELECT session_id FROM messages
                UNION
                SELECT session_id FROM summary_nodes
            ),
            message_stats AS (
                SELECT session_id,
                       COUNT(*) AS message_count,
                       COALESCE(SUM(token_estimate), 0) AS token_total
                FROM messages
                GROUP BY session_id
            ),
            node_stats AS (
                SELECT session_id, COUNT(*) AS node_count
                FROM summary_nodes
                GROUP BY session_id
            )
            SELECT s.session_id,
                   COALESCE(m.message_count, 0) AS message_count,
                   COALESCE(m.token_total, 0) AS token_total,
                   COALESCE(n.node_count, 0) AS node_count
            FROM session_ids s
            LEFT JOIN message_stats m ON m.session_id = s.session_id
            LEFT JOIN node_stats n ON n.session_id = s.session_id
            ORDER BY s.session_id
            """
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "error": str(exc),
            "candidates": [],
            "ignored_count": 0,
            "stateless_count": 0,
            "protected_count": 0,
        }

    candidates = []
    ignored_count = 0
    stateless_count = 0
    protected_count = 0

    for session_id, message_count, token_total, node_count in rows:
        keys = build_session_match_keys(session_id)
        matched_classes = []
        if matches_session_pattern(keys, engine._compiled_ignore_session_patterns):
            matched_classes.append("ignored-pattern")
            ignored_count += 1
        elif matches_session_pattern(keys, engine._compiled_stateless_session_patterns):
            matched_classes.append("stateless-pattern")
            stateless_count += 1
        if not matched_classes:
            continue
        # Protect the actively-bound session from cleanup, not the foreground
        # view. While a cron tick has rebound the engine, _session_id points
        # at the cron session and the engine is actively writing through it
        # via lifecycle hooks; deleting that data mid-flight would corrupt
        # the cleanup pass. current_session_id (foreground) is the wrong
        # field here.
        if session_id == getattr(engine, "_session_id", ""):
            protected_count += 1
            continue
        candidates.append(
            {
                "session_id": session_id,
                "classes": matched_classes,
                "message_count": int(message_count),
                "node_count": int(node_count),
                "token_total": int(token_total),
            }
        )

    return {
        "error": None,
        "candidates": candidates,
        "ignored_count": ignored_count,
        "stateless_count": stateless_count,
        "protected_count": protected_count,
    }


def _scan_retention_candidates(engine) -> dict[str, Any]:
    conn = engine._store._conn
    now = datetime.now().timestamp()
    # SQL is scoped to the foreground session so /lcm doctor retention
    # reports the operator's real conversation rather than whatever side
    # channel (cron tick, debug probe) currently owns engine._session_id.
    # The "protected" flag below still keys off engine._session_id (the
    # actively-bound row) because that is the row receiving live writes
    # from the concurrent run.
    session_id = engine.current_session_id
    if not session_id:
        return {
            "error": None,
            "sessions": [],
            "sessions_analyzed": 0,
            "stale_sessions_30d": 0,
            "stale_sessions_90d": 0,
            "retained_tokens_30d": 0,
            "retained_tokens_90d": 0,
            "protected_count": 0,
        }
    try:
        rows = conn.execute(
            """
            WITH session_ids AS (
                SELECT session_id FROM messages
                UNION
                SELECT session_id FROM summary_nodes
            ),
            message_stats AS (
                SELECT session_id,
                       COUNT(*) AS message_count,
                       COALESCE(SUM(token_estimate), 0) AS token_total,
                       MIN(timestamp) AS first_message_at,
                       MAX(timestamp) AS last_message_at
                FROM messages
                GROUP BY session_id
            ),
            node_stats AS (
                SELECT session_id,
                       COUNT(*) AS node_count,
                       COALESCE(SUM(token_count), 0) AS node_token_total,
                       MIN(COALESCE(earliest_at, created_at)) AS first_node_at,
                       MAX(COALESCE(latest_at, created_at)) AS last_node_at
                FROM summary_nodes
                GROUP BY session_id
            )
            SELECT s.session_id,
                   COALESCE(m.message_count, 0) AS message_count,
                   COALESCE(m.token_total, 0) AS token_total,
                   COALESCE(n.node_count, 0) AS node_count,
                   COALESCE(n.node_token_total, 0) AS node_token_total,
                   m.first_message_at,
                   m.last_message_at,
                   n.first_node_at,
                   n.last_node_at
            FROM session_ids s
            LEFT JOIN message_stats m ON m.session_id = s.session_id
            LEFT JOIN node_stats n ON n.session_id = s.session_id
            WHERE s.session_id = ?
            ORDER BY s.session_id
            """,
            (session_id,),
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "error": str(exc),
            "sessions": [],
            "sessions_analyzed": 0,
            "stale_sessions_30d": 0,
            "stale_sessions_90d": 0,
            "retained_tokens_30d": 0,
            "retained_tokens_90d": 0,
            "protected_count": 0,
        }

    sessions = []
    protected_count = 0
    stale_sessions_30d = 0
    stale_sessions_90d = 0
    retained_tokens_30d = 0
    retained_tokens_90d = 0

    for row in rows:
        (
            session_id,
            message_count,
            token_total,
            node_count,
            node_token_total,
            first_message_at,
            last_message_at,
            first_node_at,
            last_node_at,
        ) = row
        timestamps = [
            ts for ts in (first_message_at, last_message_at, first_node_at, last_node_at)
            if ts is not None
        ]
        if not timestamps:
            continue
        first_activity_at = min(float(ts) for ts in (first_message_at, first_node_at) if ts is not None)
        last_activity_at = max(float(ts) for ts in (last_message_at, last_node_at) if ts is not None)
        age_days = max(0.0, (now - last_activity_at) / 86400.0)
        # Bound (not foreground): protect the live session from retention
        # bookkeeping while the engine may still be writing to it.
        protected = session_id == getattr(engine, "_session_id", "")
        total_footprint_tokens = int(token_total) + int(node_token_total)
        if protected:
            protected_count += 1
        if age_days >= 30.0:
            stale_sessions_30d += 1
            retained_tokens_30d += total_footprint_tokens
        if age_days >= 90.0:
            stale_sessions_90d += 1
            retained_tokens_90d += total_footprint_tokens
        sessions.append(
            {
                "session_id": session_id,
                "protected": protected,
                "message_count": int(message_count),
                "node_count": int(node_count),
                "token_total": total_footprint_tokens,
                "raw_token_total": int(token_total),
                "summary_token_total": int(node_token_total),
                "first_activity_at": float(first_activity_at),
                "last_activity_at": float(last_activity_at),
                "age_days": age_days,
            }
        )

    sessions.sort(
        key=lambda item: (
            1 if item["protected"] else 0,
            0 if item["age_days"] >= 30.0 else 1,
            -item["token_total"],
            -item["node_count"],
            -item["message_count"],
            item["last_activity_at"],
            item["session_id"],
        )
    )

    return {
        "error": None,
        "sessions": sessions,
        "sessions_analyzed": len(sessions),
        "stale_sessions_30d": stale_sessions_30d,
        "stale_sessions_90d": stale_sessions_90d,
        "retained_tokens_30d": retained_tokens_30d,
        "retained_tokens_90d": retained_tokens_90d,
        "protected_count": protected_count,
    }


def _flush_engine_connections(engine) -> None:
    """Commit pending writes on every SQLite connection the engine owns.

    Shared by ``_backup_database`` (timestamped backup) and
    ``_rotate_backup_database`` (rolling backup) so the connection-flush
    contract stays in one place.
    """
    engine._store._conn.commit()
    engine._dag._conn.commit()
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "_conn", None)
    if lifecycle_conn is not None:
        lifecycle_conn.commit()


def _backup_database(engine) -> dict[str, Any]:
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_dir = engine.backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}-{timestamp}.sqlite3"

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        _flush_engine_connections(engine)

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store._conn.backup(dest)
        finally:
            dest.close()
    except (OSError, sqlite3.Error) as exc:
        return {
            "ok": False,
            "db_path": db_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


def _rotate_backup_database(engine) -> dict[str, Any]:
    """Write a rolling rotate-latest SQLite snapshot of the LCM store.

    Atomic via tmp-then-rename so the slot is never half-written. Unlike
    ``_backup_database`` which produces timestamped files, this overwrites a
    single rolling slot so disk usage stays bounded across repeated rotates.
    """
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_path = engine.rotate_backup_path()
    backup_dir = backup_path.parent
    tmp_path = backup_path.with_name(backup_path.name + ".tmp")

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        _flush_engine_connections(engine)

        if tmp_path.exists():
            tmp_path.unlink()
        dest = sqlite3.connect(str(tmp_path))
        try:
            engine._store._conn.backup(dest)
        finally:
            dest.close()
        # Atomic replace so the rolling slot is never half-written.
        tmp_path.replace(backup_path)
    except (OSError, sqlite3.Error) as exc:
        # Best-effort cleanup of the tmp file if something failed midway.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return {
            "ok": False,
            "db_path": db_path,
            "backup_path": backup_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


def _rotate_text(engine) -> str:
    preview = engine.rotate_active_session(apply=False)
    if not preview.get("ok"):
        reason = preview.get("reason", "unknown")
        lines = [
            "LCM rotate",
            "status: refused",
            f"reason: {reason}",
        ]
        session_id = preview.get("session_id")
        if session_id:
            lines.append(f"session_id: {session_id}")
        lines.append("note: read-only preview — no changes were made")
        return "\n".join(lines)

    backup_path = engine.rotate_backup_path()
    lines = [
        "LCM rotate",
        f"status: {'noop' if preview.get('noop') else 'preview'}",
        f"session_id: {preview['session_id']}",
        f"conversation_id: {preview['conversation_id']}",
        f"total_message_count: {preview['total_message_count']}",
        f"fresh_tail_count: {preview['fresh_tail_count']}",
        f"pre_tail_message_count: {preview.get('pre_tail_message_count', 0)}",
        f"current_frontier_store_id: {preview['current_frontier_store_id']}",
        f"new_frontier_store_id: {preview['new_frontier_store_id']}",
        f"rotate_backup_path: {backup_path}",
    ]
    if preview.get("noop"):
        lines.append(f"reason: {preview.get('reason', 'no_change')}")
        lines.append("note: read-only preview — rotate apply would be a no-op for this session")
    else:
        lines.append("note: read-only preview — use `/lcm rotate apply` to advance the frontier (backup-first)")
        lines.append("note: pre-tail raw messages remain in the store and recoverable via lcm_load_session")
    return "\n".join(lines)


def _rotate_apply_text(engine) -> str:
    # Pre-flight refusal AND noop check before touching disk. This avoids
    # both writing a backup for a session that would refuse and overwriting
    # the previous known-good rolling backup when the apply would be a no-op
    # (e.g., idempotent rerun on an already-rotated session).
    pre = engine.rotate_active_session(apply=False)
    if not pre.get("ok"):
        reason = pre.get("reason", "unknown")
        lines = [
            "LCM rotate apply",
            "status: refused",
            f"reason: {reason}",
        ]
        session_id = pre.get("session_id")
        if session_id:
            lines.append(f"session_id: {session_id}")
        lines.append("note: rotate apply refused; no backup was created and no lifecycle state was changed")
        return "\n".join(lines)

    if pre.get("noop"):
        # Surface the same shape as a successful apply but with status:noop so
        # operators get the standard fields without a fresh backup write
        # destroying the previous known-good snapshot.
        lines = [
            "LCM rotate apply",
            "status: noop",
            f"session_id: {pre['session_id']}",
            f"conversation_id: {pre['conversation_id']}",
            f"total_message_count: {pre['total_message_count']}",
            f"fresh_tail_count: {pre['fresh_tail_count']}",
            f"pre_tail_message_count: {pre.get('pre_tail_message_count', 0)}",
            f"previous_frontier_store_id: {pre['current_frontier_store_id']}",
            f"new_frontier_store_id: {pre['new_frontier_store_id']}",
            f"reason: {pre.get('reason', 'no_change')}",
            "note: rotate is a no-op; rolling backup was not written so the previous rotate-latest snapshot is preserved",
        ]
        return "\n".join(lines)

    backup = _rotate_backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM rotate apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: rotate apply aborted before any lifecycle mutation",
        ])

    result = engine.rotate_active_session(apply=True)
    if not result.get("ok"):
        return "\n".join([
            "LCM rotate apply",
            "status: refused",
            f"reason: {result.get('reason', 'unknown')}",
            f"rotate_backup_path: {backup['backup_path']}",
            f"rotate_backup_size: {_fmt_size(int(backup['backup_size']))}",
            "note: backup was created before rotate refused; lifecycle state unchanged",
        ])

    is_noop = bool(result.get("noop"))
    lines = [
        "LCM rotate apply",
        f"status: {'noop' if is_noop else 'ok'}",
        f"session_id: {result['session_id']}",
        f"conversation_id: {result['conversation_id']}",
        f"rotate_backup_path: {backup['backup_path']}",
        f"rotate_backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"total_message_count: {result['total_message_count']}",
        f"fresh_tail_count: {result['fresh_tail_count']}",
        f"pre_tail_message_count: {result.get('pre_tail_message_count', 0)}",
        f"previous_frontier_store_id: {result['current_frontier_store_id']}",
        f"new_frontier_store_id: {result.get('applied_frontier_store_id', result['new_frontier_store_id'])}",
    ]
    if is_noop:
        lines.append(f"reason: {result.get('reason', 'no_change')}")
        lines.append("note: lifecycle state already at or ahead of the target frontier")
    else:
        lines.append("note: pre-tail raw messages remain in the store and recoverable via lcm_load_session")
        lines.append("note: rolling backup overwrites the previous rotate-latest slot")
    return "\n".join(lines)


def _scan_fts_repair(engine) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    specs = {
        "messages_fts": build_message_fts_spec(),
        "nodes_fts": build_nodes_fts_spec(),
    }
    conn = engine._store._conn
    for label, spec in specs.items():
        try:
            needs_repair = external_content_fts_needs_repair(conn, spec)
            content_count = int(conn.execute(
                f"SELECT COUNT(*) FROM {spec.content_table}"
            ).fetchone()[0])
            try:
                fts_count = int(conn.execute(f"SELECT COUNT(*) FROM {spec.table_name}").fetchone()[0])
            except sqlite3.Error:
                fts_count = None
            checks[label] = {
                "ok": not needs_repair,
                "needs_repair": needs_repair,
                "content_rows": content_count,
                "fts_rows": fts_count,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - defensive
            checks[label] = {
                "ok": False,
                "needs_repair": True,
                "content_rows": None,
                "fts_rows": None,
                "error": str(exc),
            }
    return {
        "checks": checks,
        "needs_repair": any(item["needs_repair"] for item in checks.values()),
    }


def _doctor_repair_text(engine) -> str:
    scan = _scan_fts_repair(engine)
    lines = [
        "LCM doctor repair",
        f"status: {'repair-needed' if scan['needs_repair'] else 'ok'}",
    ]
    for label, item in scan["checks"].items():
        state = "repair-needed" if item["needs_repair"] else "ok"
        lines.append(f"{label}: {state}")
        if item["error"]:
            lines.append(f"{label}_error: {item['error']}")
        else:
            lines.append(f"{label}_content_rows: {item['content_rows']}")
            lines.append(f"{label}_fts_rows: {item['fts_rows']}")
    lines.append("note: read-only scan only — no FTS tables were repaired")
    if scan["needs_repair"]:
        lines.append("note: use `/lcm doctor repair apply` to create a backup and repair FTS indexes")
    return "\n".join(lines)


def _doctor_repair_apply_text(engine) -> str:
    backup = _backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor repair apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: repair apply aborted before any FTS tables were repaired",
        ])

    conn = engine._store._conn
    try:
        messages_result = repair_external_content_fts(conn, build_message_fts_spec())
        nodes_result = repair_external_content_fts(conn, build_nodes_fts_spec())
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM doctor repair apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"backup_path: {backup['backup_path']}",
            f"backup_size: {_fmt_size(int(backup['backup_size']))}",
            f"error: FTS repair failed: {exc}",
            "note: backup was created before repair apply",
        ])

    return "\n".join([
        "LCM doctor repair apply",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"messages_fts_rebuilt: {_fmt_bool(messages_result['rebuilt'])}",
        f"messages_fts_triggers_recreated: {_fmt_bool(messages_result['triggers_recreated'])}",
        f"messages_fts_degraded: {_fmt_bool(messages_result['degraded'])}",
        f"nodes_fts_rebuilt: {_fmt_bool(nodes_result['rebuilt'])}",
        f"nodes_fts_triggers_recreated: {_fmt_bool(nodes_result['triggers_recreated'])}",
        f"nodes_fts_degraded: {_fmt_bool(nodes_result['degraded'])}",
        "note: backup created before repair apply",
    ])


def _doctor_source_text(engine) -> str:
    try:
        plan = engine._store.get_source_normalization_plan()
    except Exception as exc:  # pragma: no cover - defensive
        return "\n".join([
            "LCM doctor source",
            "status: error",
            f"error: source-lineage scan failed: {exc}",
            "note: read-only scan only — no source rows were updated",
        ])

    stats = plan["stats_before"]
    would_update = int(plan["would_update_messages"])
    lines = [
        "LCM doctor source",
        f"status: {'normalization-needed' if would_update else 'ok'}",
        f"messages_total: {stats['messages_total']}",
        f"attributed_messages: {stats['attributed_messages']}",
        f"unknown_messages: {stats['normalized_unknown_messages']}",
        f"legacy_blank_messages: {stats['legacy_blank_source_messages']}",
        f"effective_unknown_messages: {stats['effective_unknown_messages']}",
        f"target_source: {plan['target_source']}",
        f"would_update_messages: {would_update}",
        f"affected_sessions: {plan['affected_sessions']}",
        "note: read-only scan only — no source rows were updated",
    ]
    if would_update:
        lines.append(
            "note: use `/lcm doctor source apply` to create a backup and normalize legacy blank-source rows"
        )
    else:
        lines.append("note: no legacy blank-source rows need normalization")
    return "\n".join(lines)


def _doctor_source_apply_text(engine) -> str:
    try:
        plan = engine._store.get_source_normalization_plan()
    except Exception as exc:  # pragma: no cover - defensive
        return "\n".join([
            "LCM doctor source apply",
            "status: error",
            f"error: source-lineage scan failed: {exc}",
            "note: source normalization apply aborted before any rows were updated",
        ])

    if int(plan["would_update_messages"]) == 0:
        stats = plan["stats_before"]
        return "\n".join([
            "LCM doctor source apply",
            "status: ok",
            f"target_source: {plan['target_source']}",
            "updated_messages: 0",
            f"legacy_blank_before: {stats['legacy_blank_source_messages']}",
            f"legacy_blank_after: {stats['legacy_blank_source_messages']}",
            "note: no legacy blank-source rows needed normalization",
        ])

    backup = _backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor source apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: source normalization apply aborted before any rows were updated",
        ])

    try:
        result = engine._store.normalize_legacy_blank_sources()
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM doctor source apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"backup_path: {backup['backup_path']}",
            f"backup_size: {_fmt_size(int(backup['backup_size']))}",
            f"error: source normalization failed: {exc}",
            "note: backup was created before source normalization apply",
        ])

    before = result["stats_before"]
    after = result["stats_after"]
    return "\n".join([
        "LCM doctor source apply",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"target_source: {result['target_source']}",
        f"updated_messages: {result['updated_messages']}",
        f"legacy_blank_before: {before['legacy_blank_source_messages']}",
        f"legacy_blank_after: {after['legacy_blank_source_messages']}",
        f"unknown_before: {before['normalized_unknown_messages']}",
        f"unknown_after: {after['normalized_unknown_messages']}",
        "note: backup created before source normalization apply",
    ])


def _doctor_text(engine) -> str:
    db_path = Path(engine._store.db_path)
    runtime_identity = engine.get_runtime_identity()
    store_conn = engine._store._conn
    dag_conn = engine._dag._conn

    issues: list[str] = []
    schema_health = inspect_lcm_schema_health(store_conn, database_path=str(db_path))
    schema_missing_raw = schema_health.get("missing_tables")
    schema_missing_tables = [str(name) for name in schema_missing_raw] if isinstance(schema_missing_raw, list) else []
    schema_existing_raw = schema_health.get("existing_tables")
    schema_existing_tables = [str(name) for name in schema_existing_raw] if isinstance(schema_existing_raw, list) else []
    schema_core_status = "error" if schema_health.get("error") else "missing" if schema_missing_tables else "ok"
    if schema_missing_tables or schema_health.get("error"):
        issues.append("schema_core_tables")

    def _safe_count(conn, query: str, issue_key: str) -> int | str:
        try:
            return int(conn.execute(query).fetchone()[0])
        except Exception as exc:  # pragma: no cover - defensive
            issues.append(issue_key)
            return f"error: {exc}"

    try:
        integrity_row = store_conn.execute("PRAGMA integrity_check").fetchone()
        integrity = str(integrity_row[0]) if integrity_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        integrity = f"error: {exc}"
        issues.append("sqlite_integrity")

    try:
        store_fts_count = int(store_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        store_fts = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        store_fts_count = f"error: {exc}"
        store_fts = f"error: {exc}"
        issues.append("messages_fts")

    try:
        node_fts_count = int(dag_conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0])
        node_fts = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        node_fts_count = f"error: {exc}"
        node_fts = f"error: {exc}"
        issues.append("nodes_fts")

    total_messages = _safe_count(store_conn, "SELECT COUNT(*) FROM messages", "messages_total")
    total_message_sessions = _safe_count(
        store_conn,
        "SELECT COUNT(DISTINCT session_id) FROM messages",
        "message_sessions_total",
    )
    total_nodes = _safe_count(dag_conn, "SELECT COUNT(*) FROM summary_nodes", "summary_nodes_total")
    total_node_sessions = _safe_count(
        dag_conn,
        "SELECT COUNT(DISTINCT session_id) FROM summary_nodes",
        "summary_node_sessions_total",
    )

    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    wal_path = Path(str(db_path) + "-wal")
    wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    try:
        journal_row = store_conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(journal_row[0]) if journal_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        journal_mode = f"error: {exc}"
        issues.append("sqlite_journal_mode")
    try:
        quick_row = store_conn.execute("PRAGMA quick_check").fetchone()
        quick_check = str(quick_row[0]) if quick_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        quick_check = f"error: {exc}"
        issues.append("sqlite_quick_check")
    try:
        payload_risks = scan_sqlite_payload_risks(store_conn)
        externalized_stats = externalized_payload_stats(engine._config, hermes_home=engine._hermes_home)
        externalized_integrity = scan_externalized_payload_integrity(
            store_conn,
            engine._config,
            hermes_home=engine._hermes_home,
        )
    except Exception:  # pragma: no cover - defensive
        payload_risks = {
            "largest_content_rows": [],
            "largest_tool_calls_rows": [],
            "suspicious_data_uri_content_rows": [],
            "suspicious_data_uri_tool_calls_rows": [],
            "suspicious_base64_like_rows": [],
            "quarantined_assistant_rows": [],
            "suspicious_repetitive_assistant_rows": [],
            "heartbeat_noise_rows": [],
        }
        externalized_stats = {
            "externalized_payload_count": 0,
            "externalized_payload_bytes": 0,
            "externalized_payload_chars": 0,
            "externalized_payload_dir": "",
            "latest_externalized_payload_path": "",
            "latest_externalized_payload_mtime": 0,
        }
        externalized_integrity = {
            "externalized_payload_refs_total": 0,
            "externalized_payload_refs_existing": 0,
            "externalized_payload_refs_missing": 0,
            "externalized_payload_files_unreferenced": 0,
            "missing_externalized_payload_refs": [],
            "unreferenced_externalized_payload_files": [],
        }
        issues.append("payload_storage")
    clean_scan = _scan_clean_candidates(engine)

    debt_rows = []
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "_conn", None)
    if lifecycle_conn is not None:
        try:
            debt_rows = lifecycle_conn.execute(
                """
                SELECT conversation_id, debt_kind, debt_size_estimate
                FROM lcm_lifecycle_state
                WHERE debt_kind IS NOT NULL AND debt_size_estimate > 0
                ORDER BY updated_at DESC
                """
            ).fetchall()
        except Exception as exc:  # pragma: no cover - defensive
            issues.append("lifecycle_state")
            debt_rows = [(f"error: {exc}", "error", 0)]

    observations: list[str] = []
    recommended_actions: list[str] = []
    missing_externalized_refs = int(externalized_integrity.get("externalized_payload_refs_missing", 0) or 0)

    if schema_health.get("error"):
        observations.append(f"schema_core_tables: error: {schema_health['error']}")
        recommended_actions.append(
            "verify SQLite can read sqlite_master for the database inspected by Hermes"
        )
    elif schema_missing_tables:
        observations.append(
            "schema_core_tables: missing " + ", ".join(schema_missing_tables)
        )
        recommended_actions.append(
            "verify HERMES_HOME/LCM_DATABASE_PATH point at the database inspected by Hermes"
        )
    else:
        observations.append("schema_core_tables: ok")

    if debt_rows:
        first = debt_rows[0]
        observations.append(
            f"maintenance_debt: {len(debt_rows)} conversation(s) currently carry deferred maintenance debt; first={first[0]} kind={first[1]} size={first[2]}"
        )
        recommended_actions.append(
            "let normal compaction turns reduce maintenance debt before attempting broader cleanup"
        )

    if clean_scan["error"]:
        observations.append(f"cleanup_candidates: scan error: {clean_scan['error']}")
    elif clean_scan["candidates"]:
        observations.append(
            f"cleanup_candidates: {len(clean_scan['candidates'])} pattern-matched junk/noise session candidate(s) detected"
        )
        recommended_actions.append("inspect candidate sessions with `/lcm doctor clean`")
        recommended_actions.append("create a safety snapshot first with `/lcm backup`")
    else:
        observations.append("cleanup_candidates: none")

    if missing_externalized_refs:
        issues.append("payload_storage")
        observations.append(
            f"payload_storage: {missing_externalized_refs} externalized payload ref(s) point to missing JSON files"
        )
        recommended_actions.append(
            "inspect missing externalized payload refs and restore from backups if needed"
        )

    try:
        source_stats = engine._store.get_source_stats()
    except Exception as exc:  # pragma: no cover - defensive
        issues.append("source_lineage")
        source_stats = {
            "messages_total": 0,
            "attributed_messages": 0,
            "normalized_unknown_messages": 0,
            "legacy_blank_source_messages": 0,
            "effective_unknown_messages": 0,
            "error": str(exc),
        }
    observations.append(
        "source_lineage: "
        f"attributed={source_stats['attributed_messages']} "
        f"unknown={source_stats['normalized_unknown_messages']} "
        f"legacy_blank={source_stats['legacy_blank_source_messages']} "
        f"effective_unknown={source_stats['effective_unknown_messages']}"
    )
    if source_stats.get("error"):
        observations.append(f"source_lineage_error: {source_stats['error']}")
    if source_stats["legacy_blank_source_messages"]:
        observations.append(
            "legacy blank-source rows are normalized as `source=unknown` for back-compat filters"
        )

    try:
        lifecycle_stats = engine._lifecycle.get_fragmentation_stats(
            state_db_path=_state_db_path_for_engine(engine)
        )
    except Exception as exc:  # pragma: no cover - defensive
        issues.append("lifecycle_fragmentation")
        lifecycle_stats = {"error": str(exc)}
    else:
        observations.append(
            "lifecycle_fragmentation: "
            f"lifecycle_rows={lifecycle_stats['lifecycle_rows']} "
            f"message_sessions={lifecycle_stats['distinct_message_sessions']} "
            f"node_sessions={lifecycle_stats['distinct_node_sessions']} "
            f"current_missing_in_lcm_any={lifecycle_stats['lifecycle_current_missing_in_lcm_any']} "
            f"last_finalized_missing_in_lcm_any={lifecycle_stats['lifecycle_last_finalized_missing_in_lcm_any']} "
            f"current_missing_in_state={lifecycle_stats['lifecycle_current_missing_in_state']} "
            f"last_finalized_missing_in_state={lifecycle_stats['lifecycle_last_finalized_missing_in_state']} "
            f"message_sessions_missing_in_state={lifecycle_stats['lcm_message_sessions_missing_in_state']} "
            f"node_sessions_missing_in_state={lifecycle_stats['lcm_node_sessions_missing_in_state']} "
            f"message_sessions_without_lifecycle_current={lifecycle_stats['message_sessions_without_lifecycle_current']} "
            f"message_sessions_without_lifecycle_reference={lifecycle_stats['message_sessions_without_lifecycle_reference']} "
            f"node_sessions_without_lifecycle_reference={lifecycle_stats['node_sessions_without_lifecycle_reference']} "
            f"state_sessions_missing_in_lcm_any={lifecycle_stats['state_sessions_missing_in_lcm_any']}"
        )
        if lifecycle_stats.get("state_db_error"):
            observations.append(f"lifecycle_fragmentation_state_db_error: {lifecycle_stats['state_db_error']}")
        classification = lifecycle_stats.get("classification") or {}
        categories = classification.get("categories") or []
        if classification:
            observations.append(
                "lifecycle_fragmentation_classification: "
                f"{classification.get('status', 'unknown')}; {len(categories)} categories need review"
            )
            for category in categories:
                sample = ",".join(category.get("sample_session_ids") or []) or "(none)"
                observations.append(
                    "lifecycle_category "
                    f"{category.get('name')}: count={category.get('count', 0)} sample={sample}"
                )
        if _has_lifecycle_fragmentation(lifecycle_stats):
            recommended_actions.append(
                "inspect lifecycle fragmentation before any cleanup/repair behavior mutates state"
            )
            recommended_actions.append(
                "treat this as read-only evidence; do not infer every mismatch is harmful"
            )

    if clean_scan.get("protected_count"):
        observations.append(
            f"protected_sessions: skipped {clean_scan['protected_count']} currently bound session(s) from cleanup candidates"
        )

    protection = sensitive_pattern_status(engine._config)
    if protection["enabled"] and protection["active_patterns"]:
        observations.append(
            "sensitive_pattern_handling: enabled; matching raw secret values are replaced before SQLite, FTS, summaries, active replay, and externalized payloads"
        )
    elif protection["enabled"]:
        observations.append("sensitive_pattern_handling: enabled but no active known patterns are configured")
        recommended_actions.append("set LCM_SENSITIVE_PATTERNS to one or more known names, or disable sensitive handling")
    else:
        observations.append("sensitive_pattern_handling: disabled")
    if protection["unknown_patterns"]:
        issues.append("sensitive_pattern_config")
        recommended_actions.append(
            "remove unknown LCM_SENSITIVE_PATTERNS entries or replace them with supported names"
        )

    doctor_status = "issues-found" if integrity != "ok" or issues else (
        "action-recommended" if recommended_actions else "ok"
    )
    lines = [
        "LCM doctor",
        f"status: {doctor_status}",
        f"plugin_name: {runtime_identity.get('plugin_name', '(unknown)')}",
        f"plugin_version: {runtime_identity.get('plugin_version', '(unknown)')}",
        f"plugin_path: {runtime_identity.get('plugin_path', '(unknown)')}",
        f"module_path: {runtime_identity.get('module_path', '(unknown)')}",
        f"plugin_git_commit: {runtime_identity.get('plugin_git_commit') or '(unavailable)'}",
        f"plugin_git_branch: {runtime_identity.get('plugin_git_branch') or '(unavailable)'}",
        f"plugin_git_dirty: {runtime_identity.get('plugin_git_dirty') if runtime_identity.get('plugin_git_dirty') is not None else '(unavailable)'}",
        f"database_path: {db_path}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"wal_size: {_fmt_size(wal_size)}",
        f"schema_core_tables: {schema_core_status}",
        f"schema_missing_tables: {', '.join(schema_missing_tables) or '(none)'}",
        f"schema_existing_tables: {', '.join(schema_existing_tables) or '(none)'}",
        f"journal_mode: {journal_mode}",
        f"quick_check: {quick_check}",
        f"sqlite_integrity: {integrity}",
        f"messages_total: {total_messages}",
        f"message_sessions_total: {total_message_sessions}",
        f"summary_nodes_total: {total_nodes}",
        f"summary_node_sessions_total: {total_node_sessions}",
        f"messages_fts: {store_fts}",
        f"messages_fts_rows: {store_fts_count}",
        f"nodes_fts: {node_fts}",
        f"nodes_fts_rows: {node_fts_count}",
        f"largest_content_rows: {payload_risks['largest_content_rows']}",
        f"largest_tool_calls_rows: {payload_risks['largest_tool_calls_rows']}",
        f"suspicious_data_uri_content_rows: {payload_risks['suspicious_data_uri_content_rows']}",
        f"suspicious_data_uri_tool_calls_rows: {payload_risks['suspicious_data_uri_tool_calls_rows']}",
        f"suspicious_base64_like_rows: {payload_risks['suspicious_base64_like_rows']}",
        f"quarantined_assistant_rows: {payload_risks['quarantined_assistant_rows']}",
        f"suspicious_repetitive_assistant_rows: {payload_risks['suspicious_repetitive_assistant_rows']}",
        f"heartbeat_noise_rows: {payload_risks['heartbeat_noise_rows']}",
        f"sensitive_patterns_enabled: {_fmt_bool(protection.get('enabled'))}",
        f"sensitive_patterns: {', '.join(protection.get('patterns') or []) or '(none)'}",
        f"sensitive_patterns_source: {protection.get('source', 'default')}",
        f"sensitive_patterns_unknown: {', '.join(protection.get('unknown_patterns') or []) or '(none)'}",
        f"externalized_payload_dir: {externalized_stats['externalized_payload_dir']}",
        f"externalized_payload_count: {externalized_stats['externalized_payload_count']}",
        f"externalized_payload_bytes: {externalized_stats['externalized_payload_bytes']}",
        f"externalized_payload_chars: {externalized_stats['externalized_payload_chars']}",
        f"latest_externalized_payload_path: {externalized_stats['latest_externalized_payload_path'] or '(none)'}",
        f"externalized_payload_refs_total: {externalized_integrity['externalized_payload_refs_total']}",
        f"externalized_payload_refs_existing: {externalized_integrity['externalized_payload_refs_existing']}",
        f"externalized_payload_refs_missing: {externalized_integrity['externalized_payload_refs_missing']}",
        f"externalized_payload_files_unreferenced: {externalized_integrity['externalized_payload_files_unreferenced']}",
        f"missing_externalized_payload_refs: {externalized_integrity['missing_externalized_payload_refs']}",
        f"unreferenced_externalized_payload_files: {externalized_integrity['unreferenced_externalized_payload_files']}",
    ]
    if issues:
        lines.append(f"issues: {', '.join(issues)}")
    else:
        lines.append("issues: none")
    lines.append("observations:")
    for item in observations:
        lines.append(f"- {item}")
    lines.append("recommended_actions:")
    if recommended_actions:
        for item in recommended_actions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _doctor_clean_text(engine) -> str:
    scan = _scan_clean_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor clean",
            "status: error",
            f"error: {scan['error']}",
            "note: read-only scan only — no rows were deleted",
        ])

    candidates = scan["candidates"]
    lines = [
        "LCM doctor clean",
        f"status: {'candidates-found' if candidates else 'ok'}",
        f"candidate_sessions: {len(candidates)}",
        f"ignored_pattern_matches: {scan['ignored_count']}",
        f"stateless_pattern_matches: {scan['stateless_count']}",
    ]
    if scan["protected_count"]:
        lines.append(f"protected_sessions_skipped: {scan['protected_count']}")

    if not candidates:
        lines.append("result: no obvious junk/noise session candidates detected")
        return "\n".join(lines)

    lines.append("candidates:")
    for item in candidates[:20]:
        classes = ", ".join(item["classes"])
        lines.append(
            "- "
            f"{item['session_id']} | class={classes} | messages={item['message_count']} | "
            f"nodes={item['node_count']} | tokens={item['token_total']}"
        )
    if len(candidates) > 20:
        lines.append(f"... {len(candidates) - 20} more candidate session(s) omitted")
    lines.append("note: best-effort stored-session scan only — platform-only matches may not be reconstructable from the SQLite state")
    lines.append("note: read-only scan only — no rows were deleted")
    lines.append("note: use `/lcm doctor clean apply` only after a backup-first review of these safe candidates")
    return "\n".join(lines)


def _doctor_retention_text(engine) -> str:
    scan = _scan_retention_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor retention",
            "status: error",
            f"error: {scan['error']}",
            "note: read-only analysis only — no rows were deleted",
        ])

    sessions = scan["sessions"]
    lines = [
        "LCM doctor retention",
        f"status: {'analysis-ready' if sessions else 'ok'}",
        f"sessions_analyzed: {scan['sessions_analyzed']}",
        f"stale_sessions_30d: {scan['stale_sessions_30d']}",
        f"stale_sessions_90d: {scan['stale_sessions_90d']}",
        f"retained_tokens_30d: {scan['retained_tokens_30d']}",
        f"retained_tokens_90d: {scan['retained_tokens_90d']}",
    ]
    if scan["protected_count"]:
        lines.append(f"protected_sessions: {scan['protected_count']}")

    if not sessions:
        lines.append("result: no stored sessions found for retention analysis")
        lines.append("note: read-only analysis only — no rows were deleted")
        return "\n".join(lines)

    lines.append("retention_candidates:")
    for item in sessions[:20]:
        lines.append(
            "- "
            f"{item['session_id']} | protected={'yes' if item['protected'] else 'no'} | "
            f"messages={item['message_count']} | nodes={item['node_count']} | "
            f"tokens={item['token_total']} | age_days={item['age_days']:.1f}"
        )
    if len(sessions) > 20:
        lines.append(f"... {len(sessions) - 20} more session(s) omitted")
    lines.append("note: retention analysis is scoped to the active session only")
    lines.append("note: stale sessions are listed before fresh ones; within each bucket, candidates are sorted by footprint (tokens/nodes/messages), with protected current-session entries listed after non-protected ones")
    lines.append("note: read-only analysis only — no rows were deleted")
    lines.append("note: if you prune later, create a safety snapshot first with `/lcm backup`")
    return "\n".join(lines)


def _delete_clean_candidates_atomically(engine, session_ids: set[str]) -> dict[str, int]:
    """Delete cleanup candidates in one SQLite transaction.

    All LCM tables live in the same SQLite database, but the store, DAG, and
    lifecycle helpers use separate connections and commit internally. Cleanup
    apply is destructive, so do the coordinated deletes on one connection to
    avoid half-cleaned state if a later table delete fails.
    """
    conn = engine._store._conn
    # Protect the actively-bound session id, not current_session_id. While a
    # cron tick has rebound the engine, _session_id is the row the engine is
    # actively writing to via lifecycle hooks; deleting it during cleanup
    # would race with that ingest.
    protected_session_ids = {getattr(engine, "_session_id", "")}
    protected_session_ids = {str(s) for s in protected_session_ids if s}
    session_ids = {str(s) for s in session_ids if s and str(s) not in protected_session_ids}
    if not session_ids:
        return {
            "messages_deleted": 0,
            "nodes_deleted": 0,
            "lifecycle_deleted": 0,
            "lifecycle_skipped": 0,
        }

    placeholders = ",".join("?" for _ in session_ids)
    params = tuple(sorted(session_ids))
    lifecycle_rows = conn.execute(
        """
        SELECT conversation_id, current_session_id, last_finalized_session_id
        FROM lcm_lifecycle_state
        """
    ).fetchall()
    lifecycle_delete_conversation_ids: list[str] = []
    lifecycle_skipped = 0
    for conversation_id, current_session_id, last_finalized_session_id in lifecycle_rows:
        refs = {
            str(value)
            for value in (current_session_id, last_finalized_session_id)
            if value
        }
        if not refs or not (refs & session_ids):
            continue
        if refs & protected_session_ids:
            lifecycle_skipped += 1
            continue
        if refs <= session_ids:
            lifecycle_delete_conversation_ids.append(str(conversation_id))
            continue
        lifecycle_skipped += 1

    try:
        conn.execute("BEGIN IMMEDIATE")
        msg_cur = conn.execute(f"DELETE FROM messages WHERE session_id IN ({placeholders})", params)
        node_cur = conn.execute(f"DELETE FROM summary_nodes WHERE session_id IN ({placeholders})", params)
        lifecycle_deleted = 0
        for conversation_id in lifecycle_delete_conversation_ids:
            cur = conn.execute(
                "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?",
                (conversation_id,),
            )
            lifecycle_deleted += cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "messages_deleted": msg_cur.rowcount if msg_cur.rowcount is not None else 0,
        "nodes_deleted": node_cur.rowcount if node_cur.rowcount is not None else 0,
        "lifecycle_deleted": lifecycle_deleted,
        "lifecycle_skipped": lifecycle_skipped,
    }


def _doctor_clean_apply_text(engine) -> str:
    if not getattr(getattr(engine, "_config", None), "doctor_clean_apply_enabled", False):
        return "\n".join([
            "LCM doctor clean apply",
            "status: denied",
            "error: destructive cleanup is disabled by default",
            "note: set LCM_DOCTOR_CLEAN_APPLY_ENABLED=true only in trusted operator environments",
            "note: no rows were deleted",
        ])

    scan = _scan_clean_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor clean apply",
            "status: error",
            f"error: {scan['error']}",
            "note: cleanup apply aborted before any rows were deleted",
        ])

    candidates = scan["candidates"]
    if not candidates:
        return "\n".join([
            "LCM doctor clean apply",
            "status: ok",
            "candidate_sessions: 0",
            "result: no safe cleanup candidates detected",
            "note: nothing was deleted",
        ])

    backup = _backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor clean apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: cleanup apply aborted before any rows were deleted",
        ])

    session_ids = {item["session_id"] for item in candidates}
    try:
        deleted = _delete_clean_candidates_atomically(engine, session_ids)
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM doctor clean apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"backup_path: {backup['backup_path']}",
            f"backup_size: {_fmt_size(int(backup['backup_size']))}",
            f"error: cleanup apply failed: {exc}",
            "note: cleanup apply rolled back; restore from the backup if you need to inspect pre-apply state",
        ])

    return "\n".join([
        "LCM doctor clean apply",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"candidate_sessions: {len(candidates)}",
        f"messages_deleted: {deleted['messages_deleted']}",
        f"nodes_deleted: {deleted['nodes_deleted']}",
        f"lifecycle_rows_deleted: {deleted['lifecycle_deleted']}",
        f"lifecycle_rows_skipped: {deleted['lifecycle_skipped']}",
        "note: backup created before cleanup apply",
    ])


def _doctor_clean_lifecycle_text(engine) -> str:
    count = engine._lifecycle.row_count()
    protected = {str(getattr(engine, "_session_id", "") or "")}
    protected = {s for s in protected if s}

    conn = engine._lifecycle._conn
    sessions_with_data: set[str] = set()
    for row in conn.execute("SELECT DISTINCT session_id FROM messages").fetchall():
        sessions_with_data.add(str(row[0]))
    for row in conn.execute("SELECT DISTINCT session_id FROM summary_nodes").fetchall():
        sessions_with_data.add(str(row[0]))

    empty_current = 0
    empty_finalized = 0
    empty_protected = 0
    rows = conn.execute("SELECT * FROM lcm_lifecycle_state").fetchall()
    for row in rows:
        cur = str(row["current_session_id"] or "")
        fin = str(row["last_finalized_session_id"] or "")
        if ((cur and cur in sessions_with_data)
                or (fin and fin in sessions_with_data)):
            continue
        refs = {r for r in (cur, fin) if r}
        if refs & protected:
            empty_protected += 1
            continue
        if cur and not fin:
            empty_current += 1
        else:
            empty_finalized += 1

    total_empty = empty_current + empty_finalized
    if total_empty == 0:
        return "\n".join([
            "LCM doctor clean lifecycle",
            "status: ok",
            f"lifecycle_rows: {count}",
            "empty_rows: 0",
            "note: no empty lifecycle rows to prune",
        ])

    return "\n".join([
        "LCM doctor clean lifecycle",
        "status: candidates-found",
        f"lifecycle_rows: {count}",
        f"empty_rows: {total_empty}",
        f"  empty_current: {empty_current}",
        f"  empty_finalized: {empty_finalized}",
        f"  empty_protected: {empty_protected}",
        "note: read-only scan — no rows were deleted",
        "note: empty rows reference sessions with zero messages and zero nodes",
        "note: use `/lcm doctor clean lifecycle apply` to delete empty rows",
    ])


def _doctor_clean_lifecycle_apply_text(engine) -> str:
    if not getattr(getattr(engine, "_config", None), "doctor_clean_apply_enabled", False):
        return "\n".join([
            "LCM doctor clean lifecycle apply",
            "status: denied",
            "error: destructive cleanup is disabled by default",
            "note: set LCM_DOCTOR_CLEAN_APPLY_ENABLED=true only in trusted operator environments",
            "note: no rows were deleted",
        ])

    backup = _backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor clean lifecycle apply",
            "status: error",
            "error: failed to create backup before destructive cleanup",
            f"database_path: {backup['db_path']}",
            f"backup_error: {backup['error']}",
            "note: no rows were deleted",
        ])

    before = engine._lifecycle.row_count()
    protected = {str(getattr(engine, "_session_id", "") or "")}
    protected = {s for s in protected if s}

    try:
        deleted = engine._lifecycle.prune_empty_sessions(
            protected_session_ids=protected,
        )
    except Exception as exc:
        return "\n".join([
            "LCM doctor clean lifecycle apply",
            "status: error",
            "error: failed to prune empty sessions",
            f"backup_path: {backup['backup_path']}",
            f"prune_error: {exc}",
            "note: no rows were deleted",
        ])

    after = engine._lifecycle.row_count()
    return "\n".join([
        "LCM doctor clean lifecycle apply",
        "status: ok",
        f"lifecycle_rows_before: {before}",
        f"lifecycle_rows_deleted: {deleted}",
        f"lifecycle_rows_remaining: {after}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size_bytes: {backup['backup_size']}",
        "note: only empty lifecycle rows were deleted — messages and nodes untouched",
    ])


def _backup_text(engine) -> str:
    backup = _backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM backup",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: {backup['error']}",
        ])

    return "\n".join([
        "LCM backup",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        "note: backup created before any future cleanup/apply workflow",
    ])


def _unknown_preset_text(name: str) -> str:
    available = ", ".join(preset.name for preset in shipped_presets()) or "(none)"
    return "\n".join([
        "LCM preset",
        "status: error",
        f"error: unknown preset {name}",
        f"available_presets: {available}",
    ])


def _preset_show_text(tokens: list[str], engine) -> str:
    if len(tokens) > 1:
        return _help_text("`/lcm preset show` accepts at most one preset name.")
    preset = get_preset(tokens[0] if tokens else None)
    if preset is None:
        return _unknown_preset_text(tokens[0])
    provenance = dict(preset.provenance)
    metric_summary = dict(provenance.get("metric_summary") or {})
    fixture_suite = ", ".join(str(item) for item in provenance.get("fixture_suite") or []) or "(unknown)"
    applies_to = ", ".join(preset.applies_to) if preset.applies_to else "(unspecified)"
    lines = [
        "LCM preset show",
        f"preset: {preset.name}",
        f"family: {preset.family}",
        f"description: {preset.description}",
        f"policy_version: {preset.policy_version}",
        f"policy_path: {preset.policy_path}",
        f"benchmark_version: {provenance.get('benchmark_version', '(unknown)')}",
        f"fixture_suite: {fixture_suite}",
        f"score: {metric_summary.get('score', '(unknown)')}",
        f"baseline_score: {metric_summary.get('baseline_score', '(unknown)')}",
        f"retrieval_canary_recall: {metric_summary.get('retrieval_canary_recall', '(unknown)')}",
        f"applies_to: {applies_to}",
        "runtime_env:",
    ]
    for item in preset_env_diff(preset, engine._config):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "operator_config_precedence: explicit preset-managed LCM_* overrides win",
        "runtime_mutation: no",
        f"notes: {preset.notes}",
    ])
    return "\n".join(lines)


def _preset_suggest_text(engine) -> str:
    preset, reason = suggest_preset_for_engine(engine)
    lines = ["LCM preset suggest"]
    if preset is None:
        lines.extend([
            "suggested_preset: (none)",
            f"reason: {reason}",
            "note: run deterministic benchmarks before promoting a runtime preset",
            "note: suggestion only; no live config was changed",
        ])
        return "\n".join(lines)

    explicit = explicit_operator_overrides()
    invalid = invalid_operator_overrides()
    invalid_text = ", ".join(
        f"{env_var}={os.environ.get(env_var, '')}" for env_var in sorted(invalid.values())
    ) if invalid else "(none)"
    lines.extend([
        f"suggested_preset: {preset.name}",
        f"reason: {reason}",
        "match_confidence: context-only",
        f"policy_version: {preset.policy_version}",
        f"benchmark_version: {preset.provenance.get('benchmark_version', '(unknown)')}",
        "explicit_overrides: " + (", ".join(sorted(explicit.values())) if explicit else "(none)"),
        f"invalid_overrides: {invalid_text}",
        "preview:",
    ])
    for item in preset_env_diff(preset, engine._config):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "note: suggestion only; no live config was changed",
    ])
    return "\n".join(lines)


def _preset_apply_text(tokens: list[str], engine) -> str:
    if not tokens:
        return _help_text("`/lcm preset apply` requires a preset name and `--dry-run`.")
    dry_run = "--dry-run" in tokens
    selected = [token for token in tokens if token != "--dry-run"]
    if len(selected) != 1:
        return _help_text("`/lcm preset apply` accepts exactly one preset name and optional `--dry-run`.")
    preset_name = selected[0]
    preset = get_preset(preset_name)
    if preset is None:
        return _unknown_preset_text(preset_name)
    if not dry_run:
        return "\n".join([
            "LCM preset apply",
            "status: denied",
            "error: preset apply is preview-only for now; pass --dry-run",
            "note: no live config was changed",
        ])

    lines = [
        "LCM preset apply",
        "status: dry-run",
        f"preset: {preset.name}",
        "would_set:",
    ]
    for item in preset_env_diff(preset, engine._config):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "operator_config_precedence: explicit preset-managed LCM_* overrides win",
        "note: no live config was changed",
    ])
    return "\n".join(lines)


def _preset_text(tokens: list[str], engine) -> str:
    if not tokens:
        return _help_text("`/lcm preset` requires `show`, `suggest`, or `apply`.")
    subcommand = tokens[0].lower()
    rest = tokens[1:]
    if subcommand == "show":
        return _preset_show_text(rest, engine)
    if subcommand == "suggest":
        if rest:
            return _help_text("`/lcm preset suggest` does not accept extra arguments.")
        return _preset_suggest_text(engine)
    if subcommand == "apply":
        return _preset_apply_text(rest, engine)
    return _help_text("`/lcm preset` supports `show`, `suggest`, and `apply`.")


def handle_lcm_command(raw_args: str | None, engine) -> str:
    tokens = [part.strip() for part in (raw_args or "").strip().split() if part.strip()]
    if not tokens:
        return _status_text(engine)

    head = tokens[0].lower()
    rest = tokens[1:]

    if head == "status":
        if rest:
            return _help_text("`/lcm status` does not accept extra arguments.")
        return _status_text(engine)

    if head == "doctor":
        if not rest:
            return _doctor_text(engine)
        if len(rest) == 1 and rest[0].lower() == "clean":
            return _doctor_clean_text(engine)
        if len(rest) == 1 and rest[0].lower() == "repair":
            return _doctor_repair_text(engine)
        if len(rest) == 1 and rest[0].lower() == "source":
            return _doctor_source_text(engine)
        if len(rest) == 1 and rest[0].lower() == "retention":
            return _doctor_retention_text(engine)
        if len(rest) == 2 and rest[0].lower() == "clean" and rest[1].lower() == "apply":
            return _doctor_clean_apply_text(engine)
        if len(rest) == 2 and rest[0].lower() == "clean" and rest[1].lower() == "lifecycle":
            return _doctor_clean_lifecycle_text(engine)
        if len(rest) == 3 and rest[0].lower() == "clean" and rest[1].lower() == "lifecycle" and rest[2].lower() == "apply":
            return _doctor_clean_lifecycle_apply_text(engine)
        if len(rest) == 2 and rest[0].lower() == "repair" and rest[1].lower() == "apply":
            return _doctor_repair_apply_text(engine)
        if len(rest) == 2 and rest[0].lower() == "source" and rest[1].lower() == "apply":
            return _doctor_source_apply_text(engine)
        return _help_text("`/lcm doctor` currently supports `clean`, `clean apply`, `clean lifecycle`, `clean lifecycle apply`, `repair`, `repair apply`, `source`, `source apply`, and `retention` as extra subcommands.")

    if head == "backup":
        if rest:
            return _help_text("`/lcm backup` does not accept extra arguments.")
        return _backup_text(engine)

    if head == "rotate":
        if not rest:
            return _rotate_text(engine)
        if len(rest) == 1 and rest[0].lower() == "apply":
            return _rotate_apply_text(engine)
        return _help_text("`/lcm rotate` accepts an optional `apply` subcommand.")

    if head == "preset":
        return _preset_text(rest, engine)

    if head == "help":
        return _help_text()

    return _help_text(f"Unknown subcommand: {tokens[0]}")
