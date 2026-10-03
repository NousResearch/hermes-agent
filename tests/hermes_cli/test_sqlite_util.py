"""Tests for hermes_cli/sqlite_util.py — add_column_if_missing."""

import sqlite3


def test_adds_column_when_missing():
    from hermes_cli.sqlite_util import add_column_if_missing
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a)")
    assert add_column_if_missing(conn, "t", "b", "b TEXT") is True
    rows = conn.execute("SELECT b FROM t").fetchall()
    assert rows == [(None,)]


def test_idempotent_when_column_already_exists():
    from hermes_cli.sqlite_util import add_column_if_missing
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a, b TEXT)")
    assert add_column_if_missing(conn, "t", "b", "b TEXT") is False


def test_raises_on_other_operational_errors():
    from hermes_cli.sqlite_util import add_column_if_missing
    conn = sqlite3.connect(":memory:")
    with pytest.raises(sqlite3.OperationalError):
        add_column_if_missing(conn, "no_such_table", "c", "c INT")
