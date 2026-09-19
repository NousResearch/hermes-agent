#!/usr/bin/env python3
"""Copy a stopped Hermes profile's conversation state into an empty PostgreSQL schema."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_constants import get_hermes_home
from hermes_state import SessionDB


_TABLES = (
    'system_prompts', 'sessions', 'messages', 'session_model_usage',
    'state_meta', 'gateway_routing', 'gateway_hygiene_state', 'conversation_generations',
    'telegram_dm_topic_mode', 'telegram_dm_topic_bindings',
)


def migrate_sqlite_to_postgres(source_path: Path, target: SessionDB, *, batch_size: int = 1000) -> dict[str, int]:
    """Keep a read-only SQLite snapshot; commit every copied table together or none."""
    if target.backend != 'postgres' or target.read_only:
        raise ValueError('Migration requires a writable PostgreSQL session database')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    from psycopg import sql

    with closing(sqlite3.connect(source_path.resolve().as_uri() + '?mode=ro', uri=True)) as source:
        source.row_factory = sqlite3.Row
        source.execute('BEGIN')
        if source.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity check failed; source was not changed')
        if source.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise ValueError('SQLite foreign-key check failed; source was not changed')
        source_tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if not {'sessions', 'messages'} <= source_tables:
            raise ValueError('Source is not a Hermes session database')
        tables = [table for table in _TABLES if table in source_tables]

        def copy_tables(connection):
            conn = connection._conn
            if 'telegram_dm_topic_mode' in source_tables:
                from hermes_state_postgres_schema import TELEGRAM_SCHEMA_SQL
                conn.execute(TELEGRAM_SCHEMA_SQL)
            for table in tables:
                if conn.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(sql.Identifier(table))).fetchone():
                    if table == 'state_meta':
                        if not conn.execute("SELECT 1 FROM state_meta WHERE key <> 'telegram_dm_topic_schema_version' LIMIT 1").fetchone():
                            conn.execute("DELETE FROM state_meta WHERE key = 'telegram_dm_topic_schema_version'")
                            continue
                    raise ValueError(f'PostgreSQL table {table} is not empty; migration refused')
            counts = {}
            for table in tables:
                columns = [row[1] for row in source.execute(f'PRAGMA table_info("{table}")')]
                target_columns = {row[0] for row in conn.execute(
                    'SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s '
                    "AND is_generated = 'NEVER'", (target.schema, table))}
                if unsupported := set(columns) - target_columns:
                    raise ValueError(f'Unsupported {table} columns: {", ".join(sorted(unsupported))}')
                source_rows = source.execute(f'SELECT * FROM "{table}"')
                copy_sql = sql.SQL('COPY {} ({}) FROM STDIN').format(
                    sql.Identifier(table), sql.SQL(', ').join(map(sql.Identifier, columns)))
                count = 0
                with conn.cursor() as cursor, cursor.copy(copy_sql) as copier:
                    while rows := source_rows.fetchmany(batch_size):
                        for row in rows:
                            values = dict(row)
                            if table == 'messages':
                                values['content'] = target._encode_content(SessionDB._decode_content(values['content']))
                                for column in ('display_identity', 'display_order'):
                                    if column in values:
                                        values[column] = None
                            copier.write_row(tuple(values[column] for column in columns))
                            count += 1
                actual = conn.execute(sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]
                if actual != count:
                    raise RuntimeError(f'Row count mismatch for {table}: {count} source, {actual} target')
                counts[table] = count
            conn.execute("SELECT setval(pg_get_serial_sequence('messages', 'id'), "
                         "COALESCE((SELECT MAX(id) FROM messages), 1), EXISTS(SELECT 1 FROM messages))")
            return counts

        return target._execute_write(copy_tables)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='SQLite state.db; stop its Hermes writers first')
    parser.add_argument('--profile-home', type=Path, default=None, help='Profile configured for PostgreSQL')
    args = parser.parse_args()
    home = args.profile_home or get_hermes_home()
    with SessionDB(home / 'state.db') as target:
        counts = migrate_sqlite_to_postgres(args.source, target)
    for table, count in counts.items():
        print(f'{table}: {count} rows copied')
    print('Migration committed. SQLite source was not changed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
