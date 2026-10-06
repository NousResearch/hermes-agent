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
    caplog.set_level(logging.INFO, logger=trace.logger.name)

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
    assert "[kanban-sql-trace] tasks write op=INSERT" in text
    assert "t_install_test" not in text
    assert "private title" not in text
    assert "private body" not in text
    conn.close()


def _db(tmp_path):
    conn = sqlite3.connect(tmp_path / "trace.db", isolation_level=None)
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, body TEXT, status TEXT)"
    )
    conn.execute("CREATE TABLE other (id TEXT, value TEXT)")
    conn.execute("INSERT INTO tasks VALUES ('t_deadbeef', 'PRIVATE TITLE', 'body', 'ready')")
    trace._install_connection_trace(conn, tmp_path / "trace.db")
    return conn


def _records(caplog):
    return [r for r in caplog.records if "[kanban-sql-trace] tasks write" in r.getMessage()]


def test_task_write_trace_logs_op_stack_and_context_without_values(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_PROFILE", "mara")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_context_secret")
    conn = _db(tmp_path)
    caplog.set_level(logging.INFO, logger=trace.logger.name)

    conn.execute(
        "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
        ("t_cafe0001", "secret title", "very secret body", "running"),
    )

    text = caplog.text
    assert "op=INSERT via_trigger=''" in text
    assert "profile_env='mara'" in text
    assert "board_env='default'" in text
    assert "task_env_set=True" in text
    assert "test_kanban_sql_trace.py" in text
    for secret in ("t_context_secret", "t_cafe0001", "secret title", "very secret body"):
        assert secret not in text
    conn.close()


@pytest.mark.parametrize(
    ("statement", "op"),
    [
        ("DELETE FROM tasks WHERE id = 't_deadbeef'", "DELETE"),
        ('DELETE FROM "main"."tasks"', "DELETE"),
        ("UPDATE tasks SET id = 't_running' WHERE id = 't_deadbeef'", "UPDATE id"),
        ("UPDATE \"TASKS\" SET \"ID\" = 't_running'", "UPDATE id"),
        ("DROP TABLE tasks", "DROP TABLE"),
    ],
)
def test_identity_changing_writes_are_warnings(tmp_path, caplog, statement, op):
    conn = _db(tmp_path)
    caplog.set_level(logging.INFO, logger=trace.logger.name)

    conn.execute(statement)

    records = [r for r in _records(caplog) if f"op={op} " in r.getMessage()]
    assert records and all(r.levelno == logging.WARNING for r in records)
    assert "PRIVATE TITLE" not in caplog.text and "t_running" not in caplog.text
    conn.close()


def test_routine_updates_reads_and_other_tables_are_not_logged(tmp_path, caplog):
    conn = _db(tmp_path)
    caplog.set_level(logging.DEBUG, logger=trace.logger.name)

    conn.execute("UPDATE tasks SET status = 'running', title = 'x' WHERE id = 't_deadbeef'")
    conn.execute("SELECT * FROM tasks").fetchall()
    # Statement text that merely mentions a tasks write is not a tasks write.
    conn.execute(
        "INSERT INTO other (id, value) VALUES (?, ?)",
        ("x", "please run: DELETE FROM tasks WHERE id = 't_deadbeef'"),
    )

    assert _records(caplog) == []
    conn.close()


def test_write_issued_by_a_trigger_names_the_trigger(tmp_path, caplog):
    conn = _db(tmp_path)
    conn.execute(
        "CREATE TRIGGER rogue AFTER INSERT ON other BEGIN "
        "DELETE FROM tasks; INSERT INTO tasks (id, title, status) VALUES ('t_running', '', 'running'); END"
    )
    caplog.set_level(logging.INFO, logger=trace.logger.name)

    conn.execute("INSERT INTO other (id, value) VALUES ('1', '2')")

    records = _records(caplog)
    assert {r.getMessage().split(" via_trigger=")[1].split(" ")[0] for r in records} == {"'rogue'"}
    # Even the INSERT is a warning when a trigger issues it.
    assert all(r.levelno == logging.WARNING for r in records)
    conn.close()


def test_trace_failure_never_denies_the_statement(tmp_path, monkeypatch):
    conn = _db(tmp_path)

    def _boom():
        raise RuntimeError("context lookup failed")

    monkeypatch.setattr(trace, "_runtime_context", _boom)
    conn.execute("DELETE FROM tasks WHERE id = 't_deadbeef'")

    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    conn.close()


def test_runtime_context_exposes_task_presence_not_task_id(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_context_secret")

    context = trace._runtime_context()

    assert context["task_env_set"] is True
    assert "task_env" not in context
    assert "t_context_secret" not in context.values()


def test_insert_or_replace_is_reported_as_a_plain_insert(tmp_path, caplog):
    """Pins the documented blind spot: the authorizer sees ``INSERT OR
    REPLACE`` as one SQLITE_INSERT (no SQLITE_DELETE for the implicit drop), so
    the trace cannot tell it from a create. The schema guard, not the trace,
    is what refuses it on a real board (test_kanban_task_id_guard)."""
    conn = _db(tmp_path)
    caplog.set_level(logging.INFO, logger=trace.logger.name)

    conn.execute(
        "INSERT OR REPLACE INTO tasks (id, title, body, status) VALUES ('t_deadbeef', 'x', 'y', 'running')"
    )

    assert [r.getMessage().split(" via_trigger")[0] for r in _records(caplog)] == [
        "[kanban-sql-trace] tasks write op=INSERT"
    ]
    conn.close()
