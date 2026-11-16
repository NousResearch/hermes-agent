"""Shared trigger DDL installation stays atomic without taking settled writer locks."""
from contextlib import closing
import sqlite3

import pytest

from hermes_state_schema import SessionSchemaMixin


_DDL = """DROP TRIGGER IF EXISTS observe_insert;
CREATE TRIGGER IF NOT EXISTS observe_insert AFTER INSERT ON items
BEGIN
    INSERT INTO audit(value) VALUES (new.value);
END;
"""


def _schema_connection(path):
    conn = sqlite3.connect(path, isolation_level=None, timeout=0.1)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript('CREATE TABLE items(value); CREATE TABLE audit(value);')
    return conn


def test_settled_trigger_open_does_not_request_a_writer_lock(tmp_path):
    with closing(_schema_connection(tmp_path / 'state.db')) as conn:
        SessionSchemaMixin._execute_ddl_skipping_settled_triggers(conn.cursor(), _DDL)
        with closing(sqlite3.connect(tmp_path / 'state.db', isolation_level=None)) as writer:
            writer.execute('BEGIN IMMEDIATE')
            statements = []
            conn.set_trace_callback(statements.append)
            SessionSchemaMixin._execute_ddl_skipping_settled_triggers(conn.cursor(), _DDL)
            assert not any(sql.startswith('BEGIN') or sql.startswith('DROP') for sql in statements)
            writer.rollback()


@pytest.mark.parametrize('outer', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_trigger_ddl_respects_transaction_owner(tmp_path, outer, fail):
    with closing(_schema_connection(tmp_path / 'state.db')) as conn:
        if outer:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute("INSERT INTO items VALUES ('outer write')")
        ddl = _DDL + ('SELECT * FROM absent_table;\n' if fail else '')
        if fail:
            with pytest.raises(sqlite3.OperationalError, match='absent_table'):
                SessionSchemaMixin._execute_ddl_skipping_settled_triggers(conn.cursor(), ddl)
        else:
            SessionSchemaMixin._execute_ddl_skipping_settled_triggers(conn.cursor(), ddl)
        assert conn.in_transaction is outer
        installed = conn.execute("SELECT count(*) FROM sqlite_master WHERE name='observe_insert'").fetchone()[0]
        assert installed == int(outer or not fail)
        if outer:
            assert conn.execute('SELECT value FROM items').fetchall() == [('outer write',)]
            conn.rollback()
            assert conn.execute('SELECT value FROM items').fetchall() == []
            assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='observe_insert'").fetchone()[0] == 0


def test_deferred_snapshot_conflict_propagates(tmp_path):
    path = tmp_path / 'state.db'
    with closing(_schema_connection(path)) as conn, closing(sqlite3.connect(path)) as writer:
        conn.execute('BEGIN')
        conn.execute('SELECT * FROM items').fetchall()
        writer.execute("INSERT INTO items VALUES ('committed')")
        writer.commit()
        with pytest.raises(sqlite3.OperationalError) as raised:
            SessionSchemaMixin._execute_ddl_skipping_settled_triggers(conn.cursor(), _DDL)
        assert raised.value.sqlite_errorcode == sqlite3.SQLITE_BUSY_SNAPSHOT
        assert conn.in_transaction
        conn.rollback()


def test_writer_cannot_commit_inside_trigger_replacement(tmp_path):
    path = tmp_path / 'state.db'
    with closing(_schema_connection(path)) as conn:
        SessionSchemaMixin._execute_ddl_skipping_settled_triggers(conn.cursor(), _DDL)
        attempts = []

        class Cursor(sqlite3.Cursor):
            def execute(self, sql, *args):
                result = super().execute(sql, *args)
                if sql.strip().startswith('DROP TRIGGER IF EXISTS observe_insert'):
                    with closing(sqlite3.connect(path, timeout=0.01)) as writer:
                        with pytest.raises(sqlite3.OperationalError) as raised:
                            writer.execute("INSERT INTO items VALUES ('must not slip')")
                        attempts.append(raised.value.sqlite_errorcode)
                return result

        changed = _DDL.replace('VALUES (new.value)', "VALUES ('replacement')")
        SessionSchemaMixin._execute_ddl_skipping_settled_triggers(conn.cursor(factory=Cursor), changed)
        assert attempts == [sqlite3.SQLITE_BUSY]
        conn.execute("INSERT INTO items VALUES ('after')")
        assert conn.execute('SELECT value FROM audit').fetchall() == [('replacement',)]
