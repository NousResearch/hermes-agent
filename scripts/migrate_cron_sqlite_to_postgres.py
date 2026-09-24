#!/usr/bin/env python3
"""Copy a stopped Hermes profile's cron databases into PostgreSQL."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import ExitStack, closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_constants import get_hermes_home
from hermes_state import SessionDB


_SOURCE_TABLES = {
    "executions.db": ("executions", "cron_incidents"),
    "deliveries.db": ("deliveries", "delivery_tombstones"),
    "notepad.db": ("cron_notepad",),
}
_TARGET_TABLES = tuple(
    table for tables in _SOURCE_TABLES.values() for table in tables
)
_STORE_LOCKS = ("executions", "deliveries", "notepad")


def _open_sources(source_dir: Path, stack: ExitStack) -> dict[str, sqlite3.Connection]:
    sources: dict[str, sqlite3.Connection] = {}
    for filename in _SOURCE_TABLES:
        path = source_dir / filename
        if not path.is_file():
            continue
        connection = stack.enter_context(
            closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True))
        )
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError(f"SQLite integrity check failed for {filename}")
        sources[filename] = connection
    if not sources:
        raise ValueError(f"No cron SQLite databases found in {source_dir}")
    return sources


def migrate_cron_sqlite_to_postgres(
    source_dir: Path,
    target: SessionDB,
    *,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Copy all cron tables in one PostgreSQL transaction; never alter SQLite."""
    if target.backend != "postgres" or target.read_only:
        raise ValueError("Migration requires a writable PostgreSQL session database")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from psycopg import sql

    source_dir = source_dir.expanduser().resolve()
    with ExitStack() as stack:
        sources = _open_sources(source_dir, stack)

        def copy_tables(connection):
            raw_connection = connection._conn
            for store in _STORE_LOCKS:
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                    (f"{target.schema}:cron.{store}",),
                )
            for table in _TARGET_TABLES:
                query = sql.SQL("SELECT 1 FROM {} LIMIT 1").format(sql.Identifier(table))
                if raw_connection.execute(query).fetchone() is not None:
                    raise ValueError(
                        f"PostgreSQL table {table} is not empty; migration refused"
                    )

            counts: dict[str, int] = {}
            for filename, expected_tables in _SOURCE_TABLES.items():
                source = sources.get(filename)
                if source is None:
                    continue
                available = {
                    row[0]
                    for row in source.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                for table in expected_tables:
                    if table not in available:
                        continue
                    columns = [
                        row[1]
                        for row in source.execute(f'PRAGMA table_info("{table}")')
                    ]
                    target_columns = {
                        row[0]
                        for row in raw_connection.execute(
                            """SELECT column_name FROM information_schema.columns
                               WHERE table_schema = %s AND table_name = %s
                                 AND is_generated = 'NEVER'""",
                            (target.schema, table),
                        )
                    }
                    unsupported = set(columns) - target_columns
                    if unsupported:
                        names = ", ".join(sorted(unsupported))
                        raise ValueError(f"Unsupported {table} columns: {names}")

                    rows = source.execute(f'SELECT * FROM "{table}"')
                    copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
                        sql.Identifier(table),
                        sql.SQL(", ").join(map(sql.Identifier, columns)),
                    )
                    count = 0
                    with raw_connection.cursor() as cursor, cursor.copy(copy_sql) as copier:
                        while batch := rows.fetchmany(batch_size):
                            for row in batch:
                                copier.write_row(tuple(row[column] for column in columns))
                                count += 1
                    actual = raw_connection.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
                    ).fetchone()[0]
                    if actual != count:
                        raise RuntimeError(
                            f"Row count mismatch for {table}: {count} source, {actual} target"
                        )
                    counts[table] = count
            return counts

        with target._write_ctx(lock_scope="cron.migration") as connection:
            return copy_tables(connection)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Directory containing executions.db, deliveries.db, and notepad.db",
    )
    parser.add_argument(
        "--profile-home",
        type=Path,
        default=None,
        help="Profile configured for PostgreSQL",
    )
    args = parser.parse_args()
    home = args.profile_home or get_hermes_home()
    with SessionDB(home / "state.db") as target:
        counts = migrate_cron_sqlite_to_postgres(args.source, target)
    for table, count in counts.items():
        print(f"{table}: {count} rows copied")
    print("Migration committed. SQLite sources were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
