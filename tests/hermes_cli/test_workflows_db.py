import sqlite3

import pytest

from hermes_cli import workflows_db as wfdb
from hermes_cli.workflows_spec import WorkflowSpec


def _demo_spec(*, version: int = 1) -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": version,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {"start": {"type": "pass"}},
    })


def test_init_db_creates_tables(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    wfdb.init_db()
    with wfdb.connect() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        node_run_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(workflow_node_runs)")
        }
        node_run_indexes = {
            row["name"] for row in conn.execute("PRAGMA index_list(workflow_node_runs)")
        }
    assert "workflow_definitions" in tables
    assert "workflow_executions" in tables
    assert "workflow_node_runs" in tables
    assert "workflow_events" in tables
    assert "workflow_schedules" in tables
    assert "kanban_task_id" in node_run_columns
    assert "idx_workflow_node_runs_kanban_task" in node_run_indexes


def test_init_db_migrates_old_node_runs_kanban_task_id(tmp_path):
    db_path = tmp_path / "workflows.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE workflow_node_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_id  TEXT NOT NULL,
            node_id       TEXT NOT NULL,
            status        TEXT NOT NULL,
            input_json    TEXT,
            output_json   TEXT,
            error         TEXT,
            started_at    INTEGER,
            completed_at  INTEGER,
            wait_until    INTEGER
        );
        """)

    wfdb.init_db(db_path)

    with wfdb.connect(db_path) as conn:
        node_run_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(workflow_node_runs)")
        }
        node_run_indexes = {
            row["name"] for row in conn.execute("PRAGMA index_list(workflow_node_runs)")
        }
    assert "kanban_task_id" in node_run_columns
    assert "idx_workflow_node_runs_kanban_task" in node_run_indexes


def test_deploy_definition_and_get_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    spec = _demo_spec(version=1)
    latest = _demo_spec(version=2)
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        wfdb.deploy_definition(conn, latest, created_by="test")
        loaded = wfdb.get_definition(conn, "demo")
    assert loaded.id == "demo"
    assert loaded.version == 2


def test_connect_uses_wal_fallback_before_foreign_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    calls = []

    def fake_apply_wal(conn, *, db_label="state.db"):
        calls.append((db_label, conn.execute("PRAGMA foreign_keys").fetchone()[0]))
        return "wal"

    monkeypatch.setattr("hermes_state.apply_wal_with_fallback", fake_apply_wal)
    with wfdb.connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert calls == [("workflows.db", 0)]


def test_start_execution_persists_input_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        exec_id = wfdb.start_execution(conn, "demo", input_data={"x": 1}, trigger_type="manual")
        execution = wfdb.get_execution(conn, exec_id)
    assert execution.workflow_id == "demo"
    assert execution.status == "queued"
    assert execution.input == {"x": 1}
    assert execution.context == {"input": {"x": 1}, "node": {}}


def test_list_definitions_and_append_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {"start": {"type": "pass"}},
    })
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        records = wfdb.list_definitions(conn)
        exec_id = wfdb.start_execution(conn, "demo", input_data={"x": 1}, trigger_type="manual")
        wfdb.append_event(conn, exec_id, "queued", {"b": 2, "a": 1})
        event = conn.execute(
            "SELECT kind, payload_json FROM workflow_events WHERE execution_id = ?",
            (exec_id,),
        ).fetchone()
    assert [record.workflow_id for record in records] == ["demo"]
    assert records[0].spec.id == "demo"
    assert event["kind"] == "queued"
    assert event["payload_json"] == '{"a":1,"b":2}'


def test_public_writers_compose_inside_write_txn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        with wfdb.write_txn(conn):
            wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
            exec_id = wfdb.start_execution(
                conn, "demo", input_data={"x": 1}, trigger_type="manual"
            )
            wfdb.append_event(conn, exec_id, "queued", {"ok": True})
        assert conn.execute("SELECT count(*) FROM workflow_events").fetchone()[0] == 1


def test_append_event_rejects_missing_node_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
        exec_id = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual")
        with pytest.raises(KeyError, match="workflow node run not found"):
            wfdb.append_event(conn, exec_id, "node.started", node_run_id=9999)


def test_append_event_rejects_node_run_from_other_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
        exec_a = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual")
        exec_b = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual")
        node_run_id = conn.execute(
            """
            INSERT INTO workflow_node_runs (execution_id, node_id, status)
            VALUES (?, 'start', 'running')
            """,
            (exec_b,),
        ).lastrowid

        with pytest.raises(ValueError, match="node_run_id does not belong to execution"):
            wfdb.append_event(conn, exec_a, "node.started", node_run_id=node_run_id)


def test_missing_definition_and_execution_raise_keyerror(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        with pytest.raises(KeyError, match="workflow definition not found"):
            wfdb.get_definition(conn, "missing")
        with pytest.raises(KeyError, match="workflow definition not found"):
            wfdb.start_execution(conn, "missing", input_data={}, trigger_type="manual")
        with pytest.raises(KeyError, match="workflow execution not found"):
            wfdb.get_execution(conn, "missing")
