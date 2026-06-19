from __future__ import annotations

"""Immutable-first message store — the source of truth.

Every message is persisted durably in SQLite. The normal model is append-only,
with one narrow opt-in exception: already-externalized summarized tool-result
rows may be rewritten to compact GC tombstones while preserving the original
row identity (`store_id`) for DAG/source lookup.
"""


import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import (
    ExternalContentFtsSpec,
    configure_connection,
    ensure_external_content_fts,
    run_versioned_migrations,
)
from .config import LCMConfig
from .ingest_protection import (
    protect_message_for_ingest,
    protect_messages_for_ingest,
    redact_sensitive_value,
)
from .search_query import (
    build_snippet,
    compute_search_candidate_cap,
    compute_directness_rank_bonus_upper_bound,
    compute_directness_score,
    compute_like_fallback_fetch_limit,
    compute_search_fetch_limit,
    contains_risky_fts_ascii,
    count_term_matches,
    escape_like,
    extract_quoted_phrases,
    extract_search_terms,
    normalize_search_sort,
    requires_like_fallback,
    sanitize_fts5_query,
    AGE_DECAY_RATE,
    should_apply_directness_rank_adjustment,
)
from .message_content import normalize_content_value as _normalize_content_value
from .storage_security import RowCipher, ensure_lcm_file_permissions
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)


_MESSAGE_ROLE_BIAS_SQL = "CASE m.role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END"
_MESSAGE_SELECT_COLUMNS = (
    "store_id, session_id, source, role, content, tool_call_id, "
    "tool_calls, tool_name, timestamp, token_estimate, pinned"
)
_UNKNOWN_SOURCE = "unknown"


def _legacy_blank_source_clause(column: str) -> str:
    # SQLite TRIM() only strips spaces unless given an explicit character set.
    # Match Python's write-time `str.strip()` behavior for common ASCII whitespace
    # so legacy tabs/newlines do not become a fake attributed source bucket.
    whitespace_chars = "char(9) || char(10) || char(11) || char(12) || char(13) || char(32)"
    return f"({column} IS NULL OR TRIM({column}, {whitespace_chars}) = '')"


def _normalize_source_value(source: str | None) -> str:
    normalized = (source or "").strip()
    return normalized or _UNKNOWN_SOURCE


def _index_safe_text(value: Any, config: LCMConfig) -> str | None:
    redacted = redact_sensitive_value(value, config, parse_json_strings=False)
    return _normalize_content_value(redacted)


def _encrypted_or_plain(cipher: RowCipher, value: str | None, *, field: str) -> str | None:
    return cipher.encrypt_text(value, field=field)


def _source_filter_clause(column: str, source: str | None) -> tuple[str | None, list[str]]:
    normalized = _normalize_source_value(source) if source is not None else ""
    if not normalized:
        return None, []
    if normalized == _UNKNOWN_SOURCE:
        return f"({column} = ? OR {_legacy_blank_source_clause(column)})", [_UNKNOWN_SOURCE]
    return f"{column} = ?", [normalized]


def _message_role_bias(role: str | None) -> float:
    if role == "user":
        return 0.0
    if role == "assistant":
        return 1.0
    if role == "tool":
        return 2.0
    return 1.0


def _message_directness_score(role: str | None, content: str | None, terms: List[str], phrases: List[str] | None = None) -> float:
    score = compute_directness_score(content or "", terms, phrases)
    if role == "tool":
        stripped = (content or "").lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            score -= 4.0
    return score


def _build_search_order_by(
    sort: str | None,
    timestamp_expr: str,
    role_penalty_expr: str | None = None,
) -> str:
    normalized = normalize_search_sort(sort)
    order_parts: list[str] = []
    if normalized == "relevance":
        if role_penalty_expr:
            order_parts.extend(["rank ASC", f"{role_penalty_expr} ASC", f"{timestamp_expr} DESC"])
        else:
            order_parts.extend(["rank ASC", f"{timestamp_expr} DESC"])
        return ", ".join(order_parts)
    if normalized == "hybrid":
        blended = f"(rank / (1 + (MAX(0.0, ((strftime('%s','now') - {timestamp_expr}) / 3600.0)) * {AGE_DECAY_RATE})))"
        if role_penalty_expr:
            order_parts.extend([f"{blended} ASC", f"{role_penalty_expr} ASC", f"{timestamp_expr} DESC"])
        else:
            order_parts.extend([f"{blended} ASC", f"{timestamp_expr} DESC"])
        return ", ".join(order_parts)
    order_parts.append(f"{timestamp_expr} DESC")
    if role_penalty_expr:
        order_parts.append(f"{role_penalty_expr} ASC")
    order_parts.append("rank ASC")
    return ", ".join(order_parts)


def _fallback_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float, float]:
    normalized = normalize_search_sort(sort)
    score = float(result.get("_fallback_score") or 0.0)
    directness = float(result.get("_directness_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    role_bias = _message_role_bias(result.get("role"))

    if normalized == "relevance":
        return (-score, -directness, role_bias, -timestamp)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = score / (1 + (age_hours * AGE_DECAY_RATE))
        return (-blended, -directness, role_bias, -timestamp)
    return (-timestamp, role_bias, -score, -directness)


def _fts_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float, float]:
    normalized = normalize_search_sort(sort)
    rank = result.get("search_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_directness_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    role_bias = _message_role_bias(result.get("role"))

    if normalized == "relevance":
        return (rank_value, -directness, role_bias, -timestamp)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
        return (blended, -directness, role_bias, -timestamp)
    return (-timestamp, role_bias, rank_value, 0.0)


def _fts_primary_value(result: Dict[str, Any], sort: str | None) -> float:
    normalized = normalize_search_sort(sort)
    rank = result.get("search_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    if normalized == "hybrid":
        timestamp = float(result.get("timestamp") or 0.0)
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        return rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
    return rank_value


def build_message_fts_spec() -> ExternalContentFtsSpec:
    return ExternalContentFtsSpec(
        table_name="messages_fts",
        content_table="messages",
        content_rowid="store_id",
        indexed_column="search_content",
        trigger_sqls=(
            """
            CREATE TRIGGER IF NOT EXISTS msg_fts_insert
                AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, search_content)
                    VALUES (new.store_id, new.search_content);
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS msg_fts_delete
                AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, search_content)
                    VALUES('delete', old.store_id, old.search_content);
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS msg_fts_update
                AFTER UPDATE OF search_content ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, search_content)
                    VALUES('delete', old.store_id, old.search_content);
                INSERT INTO messages_fts(rowid, search_content)
                    VALUES (new.store_id, new.search_content);
            END;
            """,
        ),
    )


class MessageStore:
    """SQLite-backed immutable message store."""

    def __init__(self, db_path: str | Path, *, ingest_protection_config=None, hermes_home: str = ""):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        ensure_lcm_file_permissions(self.db_path)
        self._ingest_protection_config = ingest_protection_config or LCMConfig(database_path=str(self.db_path))
        self._hermes_home = hermes_home or str(self.db_path.parent)
        self._cipher = RowCipher.from_config(
            self._ingest_protection_config,
            hermes_home=self._hermes_home,
            db_path=self.db_path,
        )
        self._conn: Optional[sqlite3.Connection] = None
        # ``self._conn`` is shared across threads (the connection is opened with
        # ``check_same_thread=False``). SQLite's own C-level mutex serializes
        # statements at the engine layer, but the Python ``sqlite3`` module
        # releases the GIL while the C call runs. Under heavy thread contention
        # with concurrent HTTPS clients in the same process, downstream
        # operators have observed on-disk corruption that is consistent with
        # external bytes landing inside SQLite's write path (e.g. the first
        # 28 bytes of the database file replaced with a TLS record header +
        # ciphertext while the "SQLit" magic remains intact).
        #
        # This re-entrant lock is defense-in-depth: it forces all write call
        # sites that use ``self._conn`` to be serialized at the Python layer,
        # eliminating any window where Python-side buffer reuse or memory
        # aliasing could intersect SQLite's flush of a write. It does not
        # change semantics for single-threaded callers and adds only a single
        # uncontended ``RLock.acquire``/``release`` pair per operation.
        self._write_lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        configure_connection(self._conn)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                source TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT,
                search_content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_msg_session
                ON messages(session_id, store_id);
            CREATE INDEX IF NOT EXISTS idx_msg_session_ts
                ON messages(session_id, timestamp);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self._ensure_storage_columns()
        ensure_external_content_fts(
            self._conn,
            build_message_fts_spec(),
        )
        run_versioned_migrations(self._conn)
        self._ensure_source_column()
        self._conn.commit()
        ensure_lcm_file_permissions(self.db_path)

    def _ensure_storage_columns(self) -> None:
        assert self._conn is not None
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "search_content" not in columns:
            self._conn.execute("ALTER TABLE messages ADD COLUMN search_content TEXT")
        rows = self._conn.execute(
            "SELECT store_id, content FROM messages WHERE search_content IS NULL"
        ).fetchall()
        for store_id, content in rows:
            try:
                plain_content = self._cipher.decrypt_text(content, field="content")
                search_content = _index_safe_text(plain_content, self._ingest_protection_config)
            except RuntimeError:
                search_content = None
            self._conn.execute(
                "UPDATE messages SET search_content = ? WHERE store_id = ?",
                (search_content, store_id),
            )

    def _ensure_source_column(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "source" not in columns:
            self._conn.execute("ALTER TABLE messages ADD COLUMN source TEXT DEFAULT ''")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_source_session ON messages(source, session_id, store_id)"
        )

    # -- Write operations ---------------------------------------------------

    def append(self, session_id: str, msg: Dict[str, Any],
               token_estimate: int = 0, source: str = "") -> int:
        """Persist a message and return its store_id."""
        msg = protect_message_for_ingest(
            msg,
            config=self._ingest_protection_config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )
        tool_calls = msg.get("tool_calls")
        tc_json = json.dumps(tool_calls) if tool_calls else None
        content_plain = _normalize_content_value(msg.get("content"))
        search_content = _index_safe_text(msg.get("content"), self._ingest_protection_config)
        content_stored = _encrypted_or_plain(self._cipher, content_plain, field="content")
        tool_calls_stored = _encrypted_or_plain(self._cipher, tc_json, field="tool_calls")

        with self._write_lock:
            cur = self._conn.execute(
                """INSERT INTO messages
                   (session_id, source, role, content, search_content, tool_call_id, tool_calls,
                    tool_name, timestamp, token_estimate, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    _normalize_source_value(source),
                    msg.get("role", "unknown"),
                    content_stored,
                    search_content,
                    msg.get("tool_call_id"),
                    tool_calls_stored,
                    msg.get("tool_name"),
                    time.time(),
                    token_estimate,
                    0,
                ),
            )
            self._conn.commit()
            ensure_lcm_file_permissions(self.db_path)
            return cur.lastrowid

    def append_batch(self, session_id: str,
                     messages: List[Dict[str, Any]],
                     token_estimates: List[int] | None = None,
                     source: str = "") -> List[int]:
        """Persist multiple messages in one transaction. Returns store_ids."""
        if token_estimates is None:
            token_estimates = [0] * len(messages)

        protected_messages = protect_messages_for_ingest(
            messages,
            config=self._ingest_protection_config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )

        ids = []
        ts = time.time()
        with self._write_lock, self._conn:
            for msg, est in zip(protected_messages, token_estimates):
                tc = msg.get("tool_calls")
                tc_json = json.dumps(tc) if tc else None
                content_plain = _normalize_content_value(msg.get("content"))
                search_content = _index_safe_text(msg.get("content"), self._ingest_protection_config)
                content_stored = _encrypted_or_plain(self._cipher, content_plain, field="content")
                tool_calls_stored = _encrypted_or_plain(self._cipher, tc_json, field="tool_calls")
                cur = self._conn.execute(
                    """INSERT INTO messages
                       (session_id, source, role, content, search_content, tool_call_id, tool_calls,
                        tool_name, timestamp, token_estimate, pinned)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        _normalize_source_value(source),
                        msg.get("role", "unknown"),
                        content_stored,
                        search_content,
                        msg.get("tool_call_id"),
                        tool_calls_stored,
                        msg.get("tool_name"),
                        ts,
                        est,
                        0,
                    ),
                )
                ids.append(cur.lastrowid)
        ensure_lcm_file_permissions(self.db_path)
        return ids

    def reassign_session_messages(self, old_session_id: str, new_session_id: str) -> int:
        """Move all persisted messages from one session_id to another."""
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return 0
        with self._write_lock:
            cur = self._conn.execute(
                "UPDATE messages SET session_id = ? WHERE session_id = ?",
                (new_session_id, old_session_id),
            )
            self._conn.commit()
            return cur.rowcount if cur.rowcount is not None else 0

    def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session. Returns count deleted."""
        with self._write_lock:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            deleted = cur.rowcount if cur.rowcount is not None else 0
            return deleted

    def gc_externalized_tool_result(self, store_id: int, placeholder: str) -> bool:
        """Rewrite one unpinned tool-result row to a compact GC placeholder."""
        with self._write_lock:
            row = self._conn.execute(
                "SELECT role, pinned, content, tool_call_id FROM messages WHERE store_id = ?",
                (store_id,),
            ).fetchone()
            if row is None:
                return False
            role, pinned, current_content, tool_call_id = row
            try:
                current_plain = self._cipher.decrypt_text(current_content, field="content")
            except RuntimeError:
                current_plain = current_content
            if role != "tool" or bool(pinned) or current_plain == placeholder:
                return False
            placeholder_tokens = count_message_tokens(
                {
                    "role": "tool",
                    "content": placeholder,
                    "tool_call_id": tool_call_id,
                }
            )
            self._conn.execute(
                "UPDATE messages SET content = ?, search_content = ?, token_estimate = ? WHERE store_id = ?",
                (
                    _encrypted_or_plain(self._cipher, placeholder, field="content"),
                    _index_safe_text(placeholder, self._ingest_protection_config),
                    placeholder_tokens,
                    store_id,
                ),
            )
            self._conn.commit()
            ensure_lcm_file_permissions(self.db_path)
            return True

    def pin(self, store_id: int) -> None:

        """Mark a message as pinned (protected from pruning)."""
        with self._write_lock:
            self._conn.execute(
                "UPDATE messages SET pinned = 1 WHERE store_id = ?", (store_id,)
            )
            self._conn.commit()

    def unpin(self, store_id: int) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE messages SET pinned = 0 WHERE store_id = ?", (store_id,)
            )
            self._conn.commit()

    # -- Read operations ----------------------------------------------------

    def get(self, store_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single message by store_id."""
        row = self._conn.execute(
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id = ?", (store_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_batch(self, store_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieve multiple messages by store_id in a single query.

        Returns a dict mapping store_id → message dict.
        """
        if not store_ids:
            return {}
        placeholders = ",".join("?" for _ in store_ids)
        rows = self._conn.execute(
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id IN ({placeholders})",
            store_ids,
        ).fetchall()
        return {row[0]: self._row_to_dict(row) for row in rows}

    def get_range(self, session_id: str, start_id: int = 0,
                  end_id: int | None = None,
                  limit: int = 1000) -> List[Dict[str, Any]]:
        """Get messages in a store_id range for a session."""
        if end_id is not None:
            rows = self._conn.execute(
                f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
                   WHERE session_id = ? AND store_id >= ? AND store_id <= ?
                   ORDER BY store_id LIMIT ?""",
                (session_id, start_id, end_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
                   WHERE session_id = ? AND store_id >= ?
                   ORDER BY store_id LIMIT ?""",
                (session_id, start_id, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _session_load_where(
        self,
        session_id: str,
        *,
        roles: list[str] | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
    ) -> tuple[list[str], list[Any]]:
        where = ["session_id = ?"]
        args: list[Any] = [session_id]
        if roles:
            placeholders = ",".join("?" for _ in roles)
            where.append(f"role IN ({placeholders})")
            args.extend(roles)
        if time_from is not None:
            where.append("timestamp >= ?")
            args.append(time_from)
        if time_to is not None:
            where.append("timestamp <= ?")
            args.append(time_to)
        return where, args

    def count_session_load_messages(
        self,
        session_id: str,
        *,
        roles: list[str] | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
    ) -> int:
        """Count messages matching the lcm_load_session filter contract."""
        where, args = self._session_load_where(
            session_id,
            roles=roles,
            time_from=time_from,
            time_to=time_to,
        )
        return int(
            self._conn.execute(
                f"SELECT COUNT(*) FROM messages WHERE {' AND '.join(where)}",
                args,
            ).fetchone()[0]
        )

    def load_session_page(
        self,
        session_id: str,
        *,
        after_store_id: int = 0,
        limit: int = 100,
        roles: list[str] | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Load one ordered raw-message page for a session.

        ``after_store_id`` is exclusive so callers can use the previous page's
        ``next_cursor`` without duplicating the cursor row.
        """
        where, args = self._session_load_where(
            session_id,
            roles=roles,
            time_from=time_from,
            time_to=time_to,
        )
        where.append("store_id > ?")
        args.extend([after_store_id, limit])
        rows = self._conn.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
               WHERE {' AND '.join(where)}
               ORDER BY store_id LIMIT ?""",
            args,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_messages(self, session_id: str,
                             limit: int = 10000) -> List[Dict[str, Any]]:
        """Get all messages for a session, ordered by store_id."""
        rows = self._conn.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
               WHERE session_id = ?
               ORDER BY store_id LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_tail(self, session_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get the latest messages for a session, returned in store order."""
        if limit <= 0:
            return []
        rows = self._conn.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS}
               FROM (
                   SELECT {_MESSAGE_SELECT_COLUMNS}
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY store_id DESC
                   LIMIT ?
               )
               ORDER BY store_id""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_count(self, session_id: str) -> int:
        """Count messages in a session."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_session_token_total(self, session_id: str) -> int:
        """Sum of token estimates for a session."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_source_stats(self, session_id: str | None = None) -> Dict[str, int]:
        """Return raw source-bucket counts for diagnostics."""
        where = ""
        args: list[Any] = []
        if session_id is not None:
            where = "WHERE session_id = ?"
            args.append(session_id)

        legacy_blank_clause = _legacy_blank_source_clause("source")
        query = f"""
            SELECT COUNT(*) AS messages_total,
                   COALESCE(SUM(CASE WHEN source = ? THEN 1 ELSE 0 END), 0) AS normalized_unknown_messages,
                   COALESCE(SUM(CASE WHEN {legacy_blank_clause} THEN 1 ELSE 0 END), 0) AS legacy_blank_source_messages,
                   COALESCE(SUM(CASE WHEN NOT {legacy_blank_clause} AND source != ? THEN 1 ELSE 0 END), 0) AS attributed_messages
            FROM messages
            {where}
            """
        query_args: list[Any] = [_UNKNOWN_SOURCE, _UNKNOWN_SOURCE, *args]
        row = self._conn.execute(query, query_args).fetchone()

        messages_total = int(row[0] or 0) if row else 0
        normalized_unknown = int(row[1] or 0) if row else 0
        legacy_blank = int(row[2] or 0) if row else 0
        attributed = int(row[3] or 0) if row else 0
        return {
            "messages_total": messages_total,
            "attributed_messages": attributed,
            "normalized_unknown_messages": normalized_unknown,
            "legacy_blank_source_messages": legacy_blank,
            "effective_unknown_messages": normalized_unknown + legacy_blank,
        }

    def get_source_normalization_plan(self) -> Dict[str, Any]:
        """Return a dry-run plan for normalizing legacy blank source values."""
        stats_before = self.get_source_stats()
        blank_clause = _legacy_blank_source_clause("source")
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) AS would_update_messages,
                   COUNT(DISTINCT session_id) AS affected_sessions
            FROM messages
            WHERE {blank_clause}
            """
        ).fetchone()
        would_update = int(row[0] or 0) if row else 0
        affected_sessions = int(row[1] or 0) if row else 0
        return {
            "target_source": _UNKNOWN_SOURCE,
            "would_update_messages": would_update,
            "affected_sessions": affected_sessions,
            "stats_before": stats_before,
        }

    def normalize_legacy_blank_sources(self) -> Dict[str, Any]:
        """Normalize legacy NULL/blank source rows to the explicit unknown bucket."""
        stats_before = self.get_source_stats()
        blank_clause = _legacy_blank_source_clause("source")
        with self._write_lock, self._conn:
            cur = self._conn.execute(
                f"UPDATE messages SET source = ? WHERE {blank_clause}",
                (_UNKNOWN_SOURCE,),
            )
        updated = cur.rowcount if cur.rowcount is not None else 0
        stats_after = self.get_source_stats()
        return {
            "target_source": _UNKNOWN_SOURCE,
            "updated_messages": int(updated),
            "stats_before": stats_before,
            "stats_after": stats_after,
        }

    def get_time_bounds(self, store_ids: List[int]) -> tuple[float | None, float | None]:
        if not store_ids:
            return None, None
        placeholders = ",".join("?" * len(store_ids))
        row = self._conn.execute(
            f"SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE store_id IN ({placeholders})",
            store_ids,
        ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    # -- Search -------------------------------------------------------------

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20, sort: str | None = None,
               source: str | None = None,
               role: str | None = None,
               time_from: float | None = None,
               time_to: float | None = None) -> List[Dict[str, Any]]:
        """FTS5 search across raw messages.

        Retrieval contract:
        - ``session_id`` limits which sessions are eligible
        - ``session_id=None`` means all sessions; an empty string is treated as
          a literal session id
        - ``source`` limits which raw rows inside those sessions are eligible
        - ``source='unknown'`` means the explicit unknown-source bucket, with
          legacy blank-source rows treated as equivalent for back-compat
        """
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if requires_like_fallback(query):
            return self._search_like(
                query,
                session_id=session_id,
                limit=limit,
                sort=sort,
                source=source,
                role=role,
                time_from=time_from,
                time_to=time_to,
            )

        order_by = _build_search_order_by(
            sort,
            "m.timestamp",
            _MESSAGE_ROLE_BIAS_SQL,
        )
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)
        candidate_cap = compute_search_candidate_cap(limit)
        apply_directness_adjustment = should_apply_directness_rank_adjustment(terms, phrases)
        max_rank_bonus = compute_directness_rank_bonus_upper_bound(terms, phrases) * 3e-7
        source_clause, source_args = _source_filter_clause("m.source", source)
        offset = 0
        scanned_rows = 0
        results: list[Dict[str, Any]] = []
        while True:
            try:
                where = ["messages_fts MATCH ?"]
                args: list[Any] = [safe_query]
                if session_id is not None:
                    where.append("m.session_id = ?")
                    args.append(session_id)
                if source_clause:
                    where.append(source_clause)
                    args.extend(source_args)
                if role is not None:
                    where.append("m.role = ?")
                    args.append(role)
                if time_from is not None:
                    where.append("m.timestamp >= ?")
                    args.append(time_from)
                if time_to is not None:
                    where.append("m.timestamp <= ?")
                    args.append(time_to)
                args.extend([fetch_limit, offset])
                rows = self._conn.execute(
                    f"""SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id,
                              m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, m.pinned,
                              rank as search_rank,
                              snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                       FROM messages_fts fts
                       JOIN messages m ON m.store_id = fts.rowid
                       WHERE {' AND '.join(where)}
                       ORDER BY {order_by} LIMIT ? OFFSET ?""",
                    args,
                ).fetchall()
                scanned_rows += len(rows)
            except sqlite3.Error as exc:
                logger.warning("FTS message search failed, falling back to LIKE: %s", exc)
                return self._search_like(
                    query,
                    session_id=session_id,
                    limit=limit,
                    sort=sort,
                    source=source,
                    role=role,
                    time_from=time_from,
                    time_to=time_to,
                )

            raw_primary_values: list[float] = []
            for r in rows:
                d = self._row_to_dict(r)
                base_columns = 11
                d["search_rank"] = r[base_columns] if len(r) > base_columns else None
                d["snippet"] = r[base_columns + 1] if len(r) > (base_columns + 1) else ""
                d["_directness_score"] = _message_directness_score(d.get("role"), d.get("content"), terms, phrases)
                if apply_directness_adjustment and d["search_rank"] is not None:
                    rank_adjustment = max(float(d["_directness_score"]), 0.0)
                    d["search_rank"] = float(d["search_rank"]) - (rank_adjustment * 3e-7)
                raw_primary_values.append(_fts_primary_value(d, sort))
                results.append(d)
            results.sort(key=lambda result: _fts_result_sort_key(result, sort))

            if not apply_directness_adjustment or len(rows) < fetch_limit or len(results) <= limit:
                return results[:limit]

            worst_visible_primary = _fts_primary_value(results[min(limit, len(results)) - 1], sort)
            last_fetched_primary = raw_primary_values[-1]
            best_unseen_primary = last_fetched_primary - max_rank_bonus
            if best_unseen_primary > worst_visible_primary:
                return results[:limit]

            if scanned_rows >= candidate_cap:
                return results[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return results[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None,
                     source: str | None = None,
                     role: str | None = None,
                     time_from: float | None = None,
                     time_to: float | None = None) -> List[Dict[str, Any]]:
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms:
            return []
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)

        where: list[str] = ["search_content IS NOT NULL"]
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        source_clause, source_args = _source_filter_clause("source", source)
        if source_clause:
            where.append(source_clause)
            args.extend(source_args)
        if role is not None:
            where.append("role = ?")
            args.append(role)
        if time_from is not None:
            where.append("timestamp >= ?")
            args.append(time_from)
        if time_to is not None:
            where.append("timestamp <= ?")
            args.append(time_to)
        like_clauses = []
        for term in terms:
            like_clauses.append("search_content LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")
        fetch_limit = compute_like_fallback_fetch_limit(limit, terms, phrases)
        base_args = list(args)
        normalized_sort = normalize_search_sort(sort)
        results: List[Dict[str, Any]] = []
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        order_by = ""
        order_args: list[Any] = []
        if normalized_sort == "recency":
            role_bias = "CASE role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END"

            def count_expr(term: str) -> tuple[str, list[Any]]:
                return (
                    "((LENGTH(LOWER(search_content)) - LENGTH(REPLACE(LOWER(search_content), LOWER(?), ''))) "
                    "/ NULLIF(LENGTH(?), 0))",
                    [term, term],
                )

            score_exprs: list[str] = []
            for term in terms:
                if collapse_risky_repeats:
                    score_exprs.append("CASE WHEN search_content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END")
                    order_args.append(f"%{escape_like(term)}%")
                else:
                    expr, expr_args = count_expr(term)
                    score_exprs.append(expr)
                    order_args.extend(expr_args)
            score_expr = " + ".join(score_exprs) if score_exprs else "0"

            def build_unique_exprs(selected_terms: list[str]) -> tuple[str, list[Any]]:
                parts: list[str] = []
                expr_args: list[Any] = []
                for selected_term in selected_terms:
                    expr, args_for_expr = count_expr(selected_term)
                    parts.append(f"CASE WHEN ({expr}) > 0 THEN 1 ELSE 0 END")
                    expr_args.extend(args_for_expr)
                return (" + ".join(parts) if parts else "0", expr_args)

            def build_total_exprs(selected_terms: list[str]) -> tuple[str, list[Any]]:
                parts: list[str] = []
                expr_args: list[Any] = []
                for selected_term in selected_terms:
                    expr, args_for_expr = count_expr(selected_term)
                    parts.append(expr)
                    expr_args.extend(args_for_expr)
                return (" + ".join(parts) if parts else "0", expr_args)

            directness_args: list[Any] = []
            unique_score_expr, expr_args = build_unique_exprs(terms)
            directness_args.extend(expr_args)
            normalized_phrases = {(phrase or "").strip().lower() for phrase in phrases if (phrase or "").strip()}
            if phrases:
                phrase_hit_exprs: list[str] = []
                for phrase in phrases:
                    phrase_hit_exprs.append("CASE WHEN INSTR(LOWER(search_content), LOWER(?)) > 0 THEN 1 ELSE 0 END")
                    directness_args.append(phrase)
                phrase_hit_expr = " + ".join(phrase_hit_exprs) if phrase_hit_exprs else "0"
                non_phrase_terms = [term for term in terms if term.strip().lower() not in normalized_phrases]
                non_phrase_total_expr, expr_args = build_total_exprs(non_phrase_terms)
                directness_args.extend(expr_args)
                non_phrase_unique_expr, expr_args = build_unique_exprs(non_phrase_terms)
                directness_args.extend(expr_args)
                repetition_expr = f"MAX(({non_phrase_total_expr}) - ({non_phrase_unique_expr}), 0)"
                directness_expr = f"(({unique_score_expr}) * 5.0) + (({phrase_hit_expr}) * 8.0) - MIN(({repetition_expr}), 6)"
            else:
                total_repetition_expr, expr_args = build_total_exprs(terms)
                directness_args.extend(expr_args)
                unique_repetition_expr, expr_args = build_unique_exprs(terms)
                directness_args.extend(expr_args)
                repetition_expr = f"MAX(({total_repetition_expr}) - ({unique_repetition_expr}), 0)"
                directness_expr = f"(({unique_score_expr}) * 5.0) - MIN(({repetition_expr}), 6)"
            order_args.extend(directness_args)
            order_by = (
                f"ORDER BY timestamp DESC, {role_bias} ASC, ({score_expr}) DESC, "
                f"({directness_expr}) DESC, store_id DESC"
            )

        def add_rows(rows: list[sqlite3.Row]) -> None:
            for row in rows:
                result = self._row_to_dict(row)
                search_text = row[11] if len(row) > 11 else result.get("content")
                content = str(search_text or "")
                score = sum(
                    min(count_term_matches(content, term), 1) if collapse_risky_repeats else count_term_matches(content, term)
                    for term in terms
                )
                if score <= 0:
                    continue
                result["search_rank"] = -float(score)
                result["snippet"] = build_snippet(content, terms)
                result["_fallback_score"] = float(score)
                result["_directness_score"] = _message_directness_score(result.get("role"), content, terms, phrases)
                results.append(result)

        if normalized_sort == "recency":
            candidate_cap = compute_search_candidate_cap(limit)
            offset = 0
            scanned_rows = 0
            while True:
                batch_limit = min(fetch_limit, candidate_cap - scanned_rows)
                if batch_limit <= 0:
                    break
                rows = self._conn.execute(
                    f"""SELECT {_MESSAGE_SELECT_COLUMNS}, search_content
                        FROM messages
                        WHERE {' AND '.join(where)}
                        {order_by}
                        LIMIT ? OFFSET ?""",
                    [*base_args, *order_args, batch_limit, offset],
                ).fetchall()
                scanned_rows += len(rows)
                add_rows(rows)
                offset += len(rows)
                if len(rows) < batch_limit:
                    break
                if scanned_rows >= candidate_cap:
                    boundary_timestamp = rows[-1][8]
                    boundary_role_bias = _message_role_bias(rows[-1][3])
                    while True:
                        tie_rows = self._conn.execute(
                            f"""SELECT {_MESSAGE_SELECT_COLUMNS}, search_content
                                FROM messages
                                WHERE {' AND '.join(where)}
                                {order_by}
                                LIMIT ? OFFSET ?""",
                            [*base_args, *order_args, fetch_limit, offset],
                        ).fetchall()
                        if not tie_rows:
                            break
                        matching_tie_rows = []
                        reached_next_primary_group = False
                        for tie_row in tie_rows:
                            if tie_row[8] == boundary_timestamp and _message_role_bias(tie_row[3]) == boundary_role_bias:
                                matching_tie_rows.append(tie_row)
                            else:
                                reached_next_primary_group = True
                                break
                        add_rows(matching_tie_rows)
                        if reached_next_primary_group or len(tie_rows) < fetch_limit:
                            break
                        offset += len(tie_rows)
                    break
        else:
            rows = self._conn.execute(
                f"""SELECT {_MESSAGE_SELECT_COLUMNS}, search_content
                    FROM messages
                    WHERE {' AND '.join(where)}
                    LIMIT ?""",
                [*base_args, fetch_limit],
            ).fetchall()
            add_rows(rows)

        results.sort(key=lambda result: _fallback_result_sort_key(result, sort))
        for result in results:
            result.pop("_fallback_score", None)
        return results[:limit]

    # -- Helpers ------------------------------------------------------------

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a sqlite3 row to a dict."""
        if row is None:
            return {}
        cols = [
            "store_id", "session_id", "source", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "timestamp", "token_estimate", "pinned",
        ]
        d = dict(zip(cols, row[:len(cols)]))
        d["source"] = _normalize_source_value(d.get("source"))
        d["content"] = self._cipher.decrypt_text(d.get("content"), field="content")
        d["tool_calls"] = self._cipher.decrypt_text(d.get("tool_calls"), field="tool_calls")
        # Deserialize tool_calls JSON
        tool_calls_value = d.get("tool_calls")
        if isinstance(tool_calls_value, str) and tool_calls_value:
            try:
                d["tool_calls"] = json.loads(tool_calls_value)
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def to_openai_msg(self, stored: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a stored message back to OpenAI format."""
        msg: Dict[str, Any] = {"role": stored["role"]}
        if stored.get("content") is not None:
            msg["content"] = stored["content"]
        if stored.get("tool_calls"):
            msg["tool_calls"] = stored["tool_calls"]
        if stored.get("tool_call_id"):
            msg["tool_call_id"] = stored["tool_call_id"]
        if stored.get("tool_name"):
            msg["name"] = stored["tool_name"]
        return msg

    # -- Lifecycle ----------------------------------------------------------

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn:
            # Graceful shutdown hygiene: checkpoint committed WAL frames before
            # releasing the connection.  This does not run on crash/kill, and
            # PASSIVE can leave frames behind when another reader is active.
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass  # best-effort only; don't let this mask the real close()
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
