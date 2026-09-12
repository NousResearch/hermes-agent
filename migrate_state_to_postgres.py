"""Copy durable SQLite state into the optional PostgreSQL backend.

The source is opened read-only and held in one read transaction through copy
and verification. Raw rows preserve fields that presentation APIs decode, hide,
or filter, including inactive history, prompt references and display metadata.
Message content alone is decoded and re-encoded to remove the legacy NUL sentinel.

Rows keep their original keys. Existing target history is never overwritten;
a retry fills missing rows and reports differing values as incomplete. Peer
conversation generations may only advance, so a retry cannot reuse an old
prompt-cache identity. Use a fresh target and switch backends only after the
verification succeeds. The source remains intact for recovery.

SQLite-local outboxes, process liveness/leases and derived search metadata are
not transferred. Their ownership belongs to the source process or SQLite file,
not to the conversations being migrated.

Usage::

    python -m migrate_state_to_postgres --dsn postgresql://.../db [--sqlite-path PATH]

The DSN may also be supplied via HERMES_STATE_DATABASE_URL or
HERMES_STATE_POSTGRES_DSN.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from itertools import islice
from pathlib import Path
from typing import Any


_PAGE_SIZE = 500
_MAX_REPORTED_MISMATCHES = 100
_DURABLE_TABLES = (
    "system_prompts", "sessions", "messages", "session_model_usage",
    "gateway_routing", "gateway_hygiene_state", "conversation_generations",
)
_TOPIC_SCHEMA_KEY = "telegram_dm_topic_schema_version"


def _source_batches(source: sqlite3.Connection, table: str, *, has_topics: bool):
    """Read stored rows, excluding only metadata owned by the SQLite file."""
    from hermes_state_errors import _STATE_DB_GENERATION_KEY

    local_meta = {_STATE_DB_GENERATION_KEY, "store_instance_id", "store_created_at_utc", "last_vacuum"}
    cursor = source.execute(f'SELECT * FROM "{table}"')
    while batch := cursor.fetchmany(_PAGE_SIZE):
        rows = [dict(row) for row in batch]
        if table == "state_meta":
            rows = [row for row in rows if row["key"] not in local_meta
                    and not str(row["key"]).startswith("fts_")
                    and (has_topics or row["key"] != _TOPIC_SCHEMA_KEY)]
        if rows:
            yield rows


def _session_batches(source: sqlite3.Connection):
    """Keep only lineage ids in memory while streaming parents before children."""
    from hermes_state_pg_import import order_session_lineage

    columns = {row[1] for row in source.execute('PRAGMA table_info("sessions")')}
    parent = "parent_session_id" if "parent_session_id" in columns else "NULL AS parent_session_id"
    ordered = order_session_lineage(
        dict(row) for row in source.execute(f"SELECT id, {parent} FROM sessions")
    )
    while batch := list(islice(ordered, _PAGE_SIZE)):
        ids = [row["id"] for row in batch]
        placeholders = ", ".join("?" for _ in ids)
        rows = {row["id"]: dict(row) for row in source.execute(
            f"SELECT * FROM sessions WHERE id IN ({placeholders})", ids
        )}
        yield [rows[sid] for sid in ids]


def _verify_field_values(source: sqlite3.Connection, target: Any, tables: list[str],
                         *, has_topics: bool, decode_content, encode_content) -> dict:
    """Verify every source key and durable field, including normalized content.

    Queries are bounded by the migration page size, including composite keys.
    A pre-existing same-id row can satisfy a count check while holding a different
    conversation, so success requires values to match as well as keys to exist.
    """
    from hermes_state_pg_import import (
        _derive_migration_columns, migration_primary_key, normalize_migration_row,
    )

    mismatches = []
    mismatch_count = 0
    table_counts = {}

    def record_mismatch(message):
        nonlocal mismatch_count
        mismatch_count += 1
        if len(mismatches) < _MAX_REPORTED_MISMATCHES:
            mismatches.append(message)

    for table in tables:
        cols = _derive_migration_columns(table)
        keys = migration_primary_key(table)
        names = ", ".join(f'"{col}"' for col in cols)
        key_names = ", ".join(f'"{col}"' for col in keys)
        key_expr = f"({key_names})" if len(keys) > 1 else key_names
        row_params = ", ".join("?" for _ in keys)
        key_placeholder = f"({row_params})" if len(keys) > 1 else row_params
        checked = matched = 0
        for batch in _source_batches(source, table, has_topics=has_topics):
            expected_rows = [normalize_migration_row(table, row, decode_content, encode_content) for row in batch]
            placeholders = ", ".join(key_placeholder for _ in expected_rows)
            params = tuple(row[key] for row in expected_rows for key in keys)
            found = target.execute(
                f'SELECT {names} FROM "{table}" WHERE {key_expr} IN ({placeholders})', params
            ).fetchall()
            actual_rows = {tuple(row[key] for key in keys): dict(row) for row in found}
            checked += len(expected_rows)
            for expected in expected_rows:
                key = tuple(expected[col] for col in keys)
                label = f"{table}[{', '.join(map(str, key))}]"
                actual = actual_rows.get(key)
                if actual is None:
                    record_mismatch(f"{label}: absent from PostgreSQL")
                    continue
                matched += 1
                for col in cols:
                    if table == "conversation_generations" and col == "generation":
                        equal = actual[col] >= expected[col]
                    else:
                        equal = actual[col] == expected[col]
                    if not equal:
                        record_mismatch(f"{label}.{col}: source and target differ")
        table_counts[table] = {"source_rows": checked, "matched_rows": matched}

    return {
        "sessions_checked": table_counts["sessions"]["source_rows"],
        "messages_checked": table_counts["messages"]["source_rows"],
        "field_mismatches": mismatches,
        "mismatch_count": mismatch_count,
        "tables": table_counts,
        "clean": mismatch_count == 0,
    }


def _resolve_sqlite_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state.db"


def _resolve_dsn(explicit: str | None) -> str:
    if explicit:
        return explicit
    for key in ("HERMES_STATE_DATABASE_URL", "HERMES_STATE_POSTGRES_DSN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    raise SystemExit(
        "No PostgreSQL DSN provided. Pass --dsn or set HERMES_STATE_DATABASE_URL "
        "/ HERMES_STATE_POSTGRES_DSN."
    )


def migrate(sqlite_path: Path, dsn: str) -> dict:
    """Copy durable state and verify it against one untouched SQLite snapshot."""
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite state database not found: {sqlite_path}")

    # --help remains available without loading the optional backend or its driver.
    from hermes_state import SessionDB
    from hermes_state_common import SCHEMA_VERSION
    from hermes_state_pg_import import advance_message_sequence, import_rows
    from hermes_state_pg_schema import init_postgres_schema, init_postgres_topic_schema
    from hermes_state_postgres import connect_postgres
    from hermes_state_telegram import _TOPIC_TABLES

    decode_content, encode_content = SessionDB._decode_content, SessionDB._encode_content
    source = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        source.execute("BEGIN")
        present = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if not {"sessions", "messages"} <= present:
            raise ValueError("SQLite source does not contain the sessions and messages tables")
        if "schema_version" in present:
            version = source.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            if version is not None and version > SCHEMA_VERSION:
                raise ValueError("SQLite source uses a newer schema; update Hermes before migrating")
        topic_tables = [name for name, _, _ in _TOPIC_TABLES if name in present]
        tables = [name for name in (*_DURABLE_TABLES, *topic_tables, "state_meta") if name in present]
        has_topics = bool(topic_tables)

        target = connect_postgres(dsn)
        try:
            init_postgres_schema(target, SCHEMA_VERSION)
            if has_topics:
                init_postgres_topic_schema(target)
            target.execute("BEGIN")
            # Serialize with ordinary message writers before importing anything.
            # Taking this lock after INSERTs can deadlock two imports upgrading
            # their ROW EXCLUSIVE locks. It also protects the sequence high-water
            # read/set from writers allocating ids while the copy is uncommitted.
            target.execute("LOCK TABLE messages IN SHARE ROW EXCLUSIVE MODE")
            imported_sessions = 0
            for table in tables:
                batches = (_session_batches(source) if table == "sessions"
                           else _source_batches(source, table, has_topics=has_topics))
                for batch in batches:
                    count = import_rows(target, table, batch, decode_content, encode_content)
                    if table == "sessions":
                        imported_sessions += count
            advance_message_sequence(target)
            target.commit()

            field_check = _verify_field_values(
                source, target, tables, has_topics=has_topics,
                decode_content=decode_content, encode_content=encode_content,
            )
            counts = field_check["tables"]
            return {
                "sqlite_path": str(sqlite_path),
                "source_sessions": counts["sessions"]["source_rows"],
                "source_messages": counts["messages"]["source_rows"],
                "imported_sessions": imported_sessions,
                "migrated_sessions": counts["sessions"]["matched_rows"],
                "migrated_messages": counts["messages"]["matched_rows"],
                "target_sessions": target.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
                "target_messages": target.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                # PostgreSQL rejects NUL text at INSERT, so a successful copy
                # proves no legacy NUL sentinel reached stored target content.
                "nul_rows": 0,
                "field_check": field_check,
                "complete": field_check["clean"],
            }
        finally:
            target.close()
    finally:
        source.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="migrate_state_to_postgres",
        description="Copy SQLite session/state data into a PostgreSQL backend "
        "(read-only on the SQLite source).",
    )
    parser.add_argument(
        "--dsn",
        help="PostgreSQL DSN. Defaults to HERMES_STATE_DATABASE_URL / "
        "HERMES_STATE_POSTGRES_DSN.",
    )
    parser.add_argument(
        "--sqlite-path",
        help="Source SQLite state.db path (default: <hermes home>/state.db).",
    )
    args = parser.parse_args(argv)

    sqlite_path = _resolve_sqlite_path(args.sqlite_path)
    dsn = _resolve_dsn(args.dsn)
    summary = migrate(sqlite_path, dsn)

    fc = summary["field_check"]
    ok = summary["complete"] and summary["nul_rows"] == 0
    status = "OK" if ok else "MISMATCH"
    print(
        f"{status} migrated {summary['migrated_sessions']}/"
        f"{summary['source_sessions']} sessions and "
        f"{summary['migrated_messages']}/{summary['source_messages']} messages "
        f"-> PostgreSQL (target now holds {summary['target_sessions']} sessions "
        f"/ {summary['target_messages']} messages in total). "
        f"Field check: {fc['sessions_checked']} sessions / "
        f"{fc['messages_checked']} messages checked, "
        f"{fc['mismatch_count']} field mismatch(es). "
        f"SQLite source left untouched: {summary['sqlite_path']}"
    )
    if not ok:
        print("Target verification failed. Migrate into an empty database before switching backends.", file=sys.stderr)
        for mismatch in fc["field_mismatches"][:10]:
            print(f"  {mismatch}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
