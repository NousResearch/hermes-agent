import json
import sqlite3
from argparse import Namespace

import pytest

from hermes_cli import workflows
from hermes_cli import workflows_db as wfdb
from hermes_cli.workflows_spec import WorkflowSpec


def _spec(output=True, *, version=1):
    return WorkflowSpec.model_validate({
        "id": "immutable_demo",
        "name": "Immutable Demo",
        "version": version,
        "nodes": {"start": {"type": "pass", "output": {"ok": output}}},
    })


def _schedule_spec(*, version=1):
    return WorkflowSpec.model_validate({
        "id": "immutable_schedule_demo",
        "name": "Immutable Schedule Demo",
        "version": version,
        "triggers": [{"type": "schedule", "id": "hourly", "cron": "0 * * * *"}],
        "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
    })


def test_init_db_runs_schema_once_per_resolved_path(tmp_path, monkeypatch):
    calls = {"executescript": 0}

    class FakeConn:
        def executescript(self, _sql):
            calls["executescript"] += 1

        def execute(self, sql):
            if "PRAGMA table_info(workflow_node_runs)" in sql:
                return [{"name": "wait_until"}, {"name": "kanban_task_id"}]
            return []

        def close(self):
            pass

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "workflows.db").touch()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(wfdb, "connect", lambda db_path=None: FakeConn())
    wfdb._INITIALIZED_DB_PATHS.clear()
    try:
        wfdb.init_db()
        wfdb.init_db()
        assert calls["executescript"] == 1

        custom_db = tmp_path / "custom-workflows.db"
        custom_db.touch()
        wfdb.init_db(custom_db)
        wfdb.init_db(custom_db)
        assert calls["executescript"] == 2
    finally:
        wfdb._INITIALIZED_DB_PATHS.clear()


def test_init_db_upgrades_pre_continuous_input_database_without_losing_rows(tmp_path):
    db_path = tmp_path / "legacy-workflows.db"
    spec = _spec(True)
    spec_json = json.dumps(spec.model_dump(mode="json", by_alias=True), sort_keys=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE workflow_definitions (
                workflow_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                name TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                spec_json TEXT NOT NULL,
                checksum TEXT NOT NULL,
                created_by TEXT,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (workflow_id, version)
            );
            CREATE TABLE workflow_executions (
                execution_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                context_json TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                trigger_id TEXT,
                claim_lock TEXT,
                claim_expires INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE workflow_node_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT,
                output_json TEXT,
                error TEXT,
                started_at INTEGER,
                completed_at INTEGER
            );
            CREATE TABLE workflow_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                node_run_id INTEGER,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE workflow_schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                version INTEGER,
                trigger_id TEXT,
                schedule TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                next_run_at INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO workflow_definitions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (spec.id, spec.version, spec.name, 1, spec_json, "legacy-checksum", "legacy", 10),
        )
        conn.execute(
            "INSERT INTO workflow_executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("exec-legacy", spec.id, spec.version, "succeeded", '{"ok":true}', '{"node":{}}', "manual", None, None, None, 11, 12),
        )
        conn.execute(
            "INSERT INTO workflow_node_runs (execution_id, node_id, status, started_at, completed_at) VALUES (?, ?, ?, ?, ?)",
            ("exec-legacy", "start", "succeeded", 11, 12),
        )

    wfdb._INITIALIZED_DB_PATHS.clear()
    wfdb.init_db(db_path)

    with wfdb.connect(db_path) as conn:
        definitions = wfdb.list_definitions(conn)
        execution = wfdb.get_execution(conn, "exec-legacy")
        node_columns = {row["name"] for row in conn.execute("PRAGMA table_info(workflow_node_runs)")}
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    assert [(record.workflow_id, record.version) for record in definitions] == [("immutable_demo", 1)]
    assert execution.workflow_id == "immutable_demo"
    assert execution.input == {"ok": True}
    assert {"wait_until", "kanban_task_id", "kanban_board"}.issubset(node_columns)
    assert {"workflow_input_feeds", "workflow_input_items"}.issubset(tables)


def test_redeploy_same_version_same_checksum_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _spec(True), created_by="first")
        first = wfdb.list_definitions(conn)[0]
        wfdb.deploy_definition(conn, _spec(True), created_by="second")
        second = wfdb.list_definitions(conn)[0]

    assert first.workflow_id == "immutable_demo"
    assert first.version == 1
    assert second.workflow_id == first.workflow_id
    assert second.version == first.version
    assert second.checksum == first.checksum
    assert second.created_by == "first", "idempotent redeploy must not overwrite created_by"
    assert second.created_at == first.created_at, "idempotent redeploy must not bump created_at"
    assert second.name == first.name
    assert second.enabled == first.enabled
    assert second.spec.model_dump(mode="json") == first.spec.model_dump(mode="json")


def test_redeploy_same_version_different_checksum_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _spec(True), created_by="first")
        with pytest.raises(ValueError) as exc:
            wfdb.deploy_definition(conn, _spec(False), created_by="second")

    assert "already exists with different checksum" in str(exc.value)


def test_deploy_new_version_with_different_checksum_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _spec(True, version=1), created_by="v1")
        wfdb.deploy_definition(conn, _spec(False, version=2), created_by="v2")
        records = wfdb.list_definitions(conn)

    assert {(r.workflow_id, r.version) for r in records} == {("immutable_demo", 1), ("immutable_demo", 2)}


def test_cli_deploy_json_reports_exact_deployed_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    v1 = tmp_path / "v1.json"
    v2 = tmp_path / "v2.json"
    v1.write_text(json.dumps(_spec(True, version=1).model_dump(mode="json")), encoding="utf-8")
    v2.write_text(json.dumps(_spec(False, version=2).model_dump(mode="json")), encoding="utf-8")

    assert workflows._cmd_deploy(Namespace(file=str(v2), json=True)) == 0
    capsys.readouterr()
    assert workflows._cmd_deploy(Namespace(file=str(v1), json=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow_id"] == "immutable_demo"
    assert payload["version"] == 1


def test_redeploy_same_schedule_definition_preserves_schedule_row(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _schedule_spec(), created_by="first")
        first_rows = conn.execute(
            "SELECT id, workflow_id, version, trigger_id, schedule, enabled, "
            "next_run_at, created_at, updated_at FROM workflow_schedules"
        ).fetchall()
        first_row = dict(first_rows[0])

        wfdb.deploy_definition(conn, _schedule_spec(), created_by="second")
        second_rows = conn.execute(
            "SELECT id, workflow_id, version, trigger_id, schedule, enabled, "
            "next_run_at, created_at, updated_at FROM workflow_schedules"
        ).fetchall()

    assert len(first_rows) == 1
    assert len(second_rows) == 1
    second_row = dict(second_rows[0])
    assert second_row == first_row, (
        f"schedule row mutated on idempotent redeploy: {first_row} -> {second_row}"
    )