from __future__ import annotations

import logging
import sqlite3

import pytest

from hermes_cli import kanban_sql_trace as trace


def test_trace_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_SQL_TRACE", raising=False)
    assert trace._enabled() is True


def test_trace_can_be_explicitly_disabled(monkeypatch):
    for value in ("0", "false", "no", "off", " FALSE "):
        monkeypatch.setenv("HERMES_KANBAN_SQL_TRACE", value)
        assert trace._enabled() is False


def test_install_if_enabled_patches_connect_and_traces_task_writes(
    tmp_path, monkeypatch, caplog
):
    original_connect = trace._ORIGINAL_CONNECT
    monkeypatch.delenv("HERMES_KANBAN_SQL_TRACE", raising=False)
    monkeypatch.setattr(trace, "_INSTALLED", False)
    monkeypatch.setattr(sqlite3, "connect", original_connect)
    monkeypatch.setattr(sqlite3.dbapi2, "connect", original_connect)
    caplog.set_level(logging.WARNING, logger=trace.logger.name)

    assert trace.install_if_enabled() is True
    assert sqlite3.connect is trace._traced_connect
    assert sqlite3.dbapi2.connect is trace._traced_connect

    conn = sqlite3.connect(tmp_path / "installed.db")
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, body TEXT, status TEXT)"
    )
    caplog.clear()
    conn.execute(
        "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
        ("t_install_test", "private title", "private body", "running"),
    )

    text = caplog.text
    assert "[kanban-sql-trace] tasks write" in text
    assert "INSERT INTO tasks" in text
    assert "t_install_test" not in text
    assert "private title" not in text
    assert "private body" not in text
    conn.close()


def _db(tmp_path):
    conn = sqlite3.connect(tmp_path / "trace.db")
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, body TEXT, status TEXT)"
    )
    conn.execute("CREATE TABLE other (id TEXT, value TEXT)")
    return conn


def test_task_write_trace_logs_shape_stack_and_context_without_values(
    tmp_path, monkeypatch, caplog
):
    conn = _db(tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "mara")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_context_secret")
    trace._install_connection_trace(conn, tmp_path / "trace.db")

    caplog.set_level(logging.WARNING, logger=trace.logger.name)
    conn.execute(
        "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
        ("t_deadbeef", "secret title", "very secret body", "running"),
    )

    text = caplog.text
    assert "[kanban-sql-trace] tasks write" in text
    assert "INSERT INTO tasks" in text
    assert "profile_env='mara'" in text
    assert "board_env='default'" in text
    assert "task_env_set=True" in text
    assert "test_kanban_sql_trace.py" in text
    assert "t_context_secret" not in text
    assert "t_deadbeef" not in text
    assert "secret title" not in text
    assert "very secret body" not in text
    conn.close()


def test_task_write_trace_redacts_sqlite_double_quoted_values(tmp_path, caplog):
    conn = _db(tmp_path)
    trace._install_connection_trace(conn, tmp_path / "trace.db")
    caplog.set_level(logging.WARNING, logger=trace.logger.name)

    conn.execute(
        'UPDATE tasks SET title = "PRIVATE TITLE" WHERE id = "t_deadbeef"'
    )

    text = caplog.text
    assert "[kanban-sql-trace] tasks write" in text
    assert "UPDATE tasks" in text
    assert "PRIVATE TITLE" not in text
    assert "t_deadbeef" not in text
    conn.close()


def test_quoted_task_identifier_is_detected_before_sql_redaction(tmp_path, caplog):
    conn = _db(tmp_path)
    trace._install_connection_trace(conn, tmp_path / "trace.db")
    caplog.set_level(logging.WARNING, logger=trace.logger.name)

    conn.execute(
        'UPDATE "main"."tasks" SET "title" = \'PRIVATE TITLE\' '
        'WHERE "id" = \'t_deadbeef\''
    )

    text = caplog.text
    assert "[kanban-sql-trace] tasks write op=UPDATE" in text
    assert "PRIVATE TITLE" not in text
    assert "t_deadbeef" not in text
    conn.close()


def test_trace_ignores_reads_and_other_tables(tmp_path, caplog):
    conn = _db(tmp_path)
    trace._install_connection_trace(conn, tmp_path / "trace.db")
    caplog.set_level(logging.WARNING, logger=trace.logger.name)

    conn.execute("SELECT * FROM tasks").fetchall()
    conn.execute("INSERT INTO other (id, value) VALUES (?, ?)", ("x", "hidden"))

    assert "[kanban-sql-trace]" not in caplog.text
    conn.close()


@pytest.mark.parametrize(
    ("statement", "secrets"),
    [
        (
            "UPDATE tasks SET body='don''t leak' WHERE id='t_deadbeef'",
            ("don''t leak", "t_deadbeef"),
        ),
        (
            'UPDATE tasks SET title="PRIVATE TITLE" WHERE id="t_deadbeef"',
            ("PRIVATE TITLE", "t_deadbeef"),
        ),
        (
            'UPDATE tasks SET title="PRIVATE ""TITLE""" WHERE id=1',
            ("PRIVATE", "TITLE"),
        ),
        (
            "UPDATE tasks SET result=X'CAFE' WHERE id=1",
            ("CAFE",),
        ),
        (
            "UPDATE tasks SET priority=-42.5, score=6.02e23 WHERE id=1",
            ("-42.5", "6.02e23"),
        ),
        (
            "UPDATE tasks SET priority=0xCAFE WHERE id=1",
            ("CAFE",),
        ),
        (
            "UPDATE tasks SET priority=1 -- private line\nWHERE id=2",
            ("private line",),
        ),
        (
            "UPDATE tasks SET priority=1 /* private block */ WHERE id=2",
            ("private block",),
        ),
    ],
)
def test_redaction_removes_private_sql_tokens(statement, secrets):
    shaped = trace._redacted_sql_shape(statement)

    for secret in secrets:
        assert secret not in shaped
    assert shaped.startswith("UPDATE tasks SET")


def test_runtime_context_exposes_task_presence_not_task_id(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_context_secret")

    context = trace._runtime_context()

    assert context["task_env_set"] is True
    assert "task_env" not in context
    assert "t_context_secret" not in context.values()
