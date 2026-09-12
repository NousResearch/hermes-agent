"""Durable-row import helpers for the SQLite-to-PostgreSQL state migration."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from typing import Any


_MIGRATION_COLUMNS_CACHE: dict[str, list[str]] | None = None
_COLUMN_DEFAULTS: dict[tuple[str, str], Any] = {}
_NOT_NULL_DEFAULTS: dict[tuple[str, str], Any] = {}
_COLUMN_TYPES: dict[tuple[str, str], str] = {}
_PRIMARY_KEYS: dict[str, list[str]] = {}


def _derive_migration_columns(table: str) -> list[str]:
    """Derive columns, defaults and keys from the defining SQLite schemas.

    Keeping the defaults beside the column derivation prevents old backups from
    binding NULL into newly added NOT NULL columns, such as _compressed_summary.
    Optional topic tables are described here but are created on the destination
    only when they exist in the source.
    """
    global _MIGRATION_COLUMNS_CACHE
    if _MIGRATION_COLUMNS_CACHE is None:
        from hermes_state_common import SCHEMA_SQL
        from hermes_state_telegram import _TOPIC_TABLES

        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(SCHEMA_SQL)
            for name, _, ddl in _TOPIC_TABLES:
                ref.execute(f'CREATE TABLE "{name}" ({ddl})')
            cache = {}
            _COLUMN_DEFAULTS.clear()
            _NOT_NULL_DEFAULTS.clear()
            _COLUMN_TYPES.clear()
            _PRIMARY_KEYS.clear()
            for (name,) in ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                info = ref.execute(f'PRAGMA table_info("{name}")').fetchall()
                cache[name] = [col[1] for col in info]
                _PRIMARY_KEYS[name] = [col[1] for col in sorted(info, key=lambda col: col[5]) if col[5]]
                for _, col, sql_type, not_null, default_sql, _ in info:
                    _COLUMN_TYPES[name, col] = sql_type.upper()
                    if default_sql is not None:
                        default = ref.execute(f"SELECT {default_sql}").fetchone()[0]
                        _COLUMN_DEFAULTS[name, col] = default
                        if not_null:
                            _NOT_NULL_DEFAULTS[name, col] = default
        finally:
            ref.close()
        _MIGRATION_COLUMNS_CACHE = cache
    return _MIGRATION_COLUMNS_CACHE.get(table, [])


def migration_primary_key(table: str) -> list[str]:
    _derive_migration_columns(table)
    return _PRIMARY_KEYS[table]


def normalize_migration_row(table: str, row: Mapping[str, Any], decode_content, encode_content) -> dict:
    """Preserve stored values, filling missing legacy fields from schema defaults.

    Raw JSON strings pass through unchanged. Structured TEXT values are
    serialized before binding; message content normalizes the legacy sentinel.
    A changed content representation invalidates its derived display identity.
    """
    normalized = {}
    for col in _derive_migration_columns(table):
        value = row[col] if col in row else _COLUMN_DEFAULTS.get((table, col))
        if value is None:
            value = _NOT_NULL_DEFAULTS.get((table, col))
        if table == "messages" and col == "content":
            value = encode_content(decode_content(value))
        elif _COLUMN_TYPES[table, col] == "TEXT" and isinstance(value, (dict, list)):
            value = json.dumps(value)
        normalized[col] = value
    if table == "messages" and normalized["content"] != row.get("content"):
        # An existing display_order does not prove the hash matches these bytes.
        # NULL identity lets the shared display reader rebuild the session index.
        normalized["display_identity"] = None
    return normalized


def import_rows(conn: Any, table: str, rows: Iterable[Mapping[str, Any]], decode_content, encode_content) -> int:
    """Insert a batch without overwriting existing history; the caller owns the transaction."""
    cols = _derive_migration_columns(table)
    if not cols:
        raise ValueError(f"No migration schema for table {table!r}")
    names = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    conflict = "ON CONFLICT DO NOTHING"
    if table == "conversation_generations":
        # A peer's generation must never be reused, including across a retry
        # against a target whose counter has already advanced further.
        conflict = (
            "ON CONFLICT (source, session_key) DO UPDATE SET generation = "
            "CASE WHEN conversation_generations.generation > excluded.generation "
            "THEN conversation_generations.generation ELSE excluded.generation END"
        )
    # Names come from the trusted schema. Keep the standard INSERT spelling
    # so the adapter recognizes message rows and populates their search index.
    sql = f"INSERT INTO {table} ({names}) VALUES ({placeholders}) {conflict}"
    count = 0
    for row in rows:
        normalized = normalize_migration_row(table, row, decode_content, encode_content)
        conn.execute(sql, tuple(normalized[col] for col in cols))
        count += 1
    return count


def order_session_lineage(sessions: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Yield parents before children without mutating or dropping lineage."""
    remaining = {session["id"]: session for session in sessions}
    children = defaultdict(list)
    ready = deque()
    for sid, session in remaining.items():
        parent = session.get("parent_session_id")
        if parent in remaining:
            children[parent].append(sid)
        else:
            # A parent outside this import may already be on the target; its
            # foreign key will reject the insert if it is truly absent.
            ready.append(sid)
    while ready:
        sid = ready.popleft()
        yield remaining.pop(sid)
        ready.extend(children.pop(sid, ()))
    if remaining:
        raise ValueError("Cannot migrate sessions with cyclic parent_session_id references")


def advance_message_sequence(conn: Any) -> None:
    """Advance past imported ids without recycling previously allocated ids.

    The caller must hold the messages table's SHARE ROW EXCLUSIVE lock through
    commit. Sequence allocation and setval survive rollback, and cached ranges
    may extend beyond MAX(id), so the sequence's own high-water mark is part of
    the target even when no corresponding message survived.
    """
    conn.execute(
        "SELECT setval(pg_get_serial_sequence('messages', 'id'), "
        "GREATEST((SELECT COALESCE(MAX(id), 1) FROM messages), "
        "COALESCE(pg_sequence_last_value(pg_get_serial_sequence('messages', 'id')), 1)), true)"
    )


# Derive defaults with the columns so callers never observe an incomplete map.
_derive_migration_columns("sessions")
