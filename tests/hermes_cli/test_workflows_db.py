import hashlib
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


def _continuous_demo_spec(*, version: int = 1) -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": version,
        "triggers": [{"type": "manual", "id": "manual", "intake": {"mode": "continuous"}}],
        "nodes": {"start": {"type": "pass"}},
    })


def _required_manual_spec(*, version: int = 1) -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "manual_required",
        "name": "Manual Required",
        "version": version,
        "triggers": [{
            "type": "manual",
            "id": "manual",
            "input_schema": {"brief": {"kind": "long_text", "required": True, "min_length": 3}},
            "intake": {"ready_when": {"op": "exists", "path": "$.input.brief"}},
        }],
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
    assert "workflow_input_feeds" in tables
    assert "workflow_input_items" in tables
    assert "kanban_task_id" in node_run_columns
    assert "idx_workflow_node_runs_kanban_task" in node_run_indexes


def test_input_feed_enqueue_evaluates_and_claims_ready_items(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "intake_demo",
        "name": "Intake Demo",
        "version": 1,
        "triggers": [{
            "type": "manual",
            "id": "kickoff",
            "input_schema": {"brief": {"kind": "long_text", "required": True, "min_length": 5}},
            "intake": {"mode": "continuous", "dedupe_key": "$.input.source_id"},
        }],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        feed = wfdb.open_input_feed(conn, "intake_demo", trigger_id="kickoff")
        item = wfdb.enqueue_input_item(conn, feed.feed_id, {"source_id": "a", "brief": "bad"})
        duplicate = wfdb.enqueue_input_item(conn, feed.feed_id, {"source_id": "a", "brief": "a valid brief"})

        assert item.status == "needs_input"
        assert "at least 5 characters" in item.criteria["messages"][0]
        assert duplicate.item_id == item.item_id

        updated = wfdb.update_input_item(conn, item.item_id, {"source_id": "a", "brief": "a valid brief"})
        with wfdb.write_txn(conn):
            claimed = wfdb.claim_next_ready_input_item(conn)

    assert updated.status == "queued"
    assert claimed.item_id == item.item_id
    assert claimed.input == {"source_id": "a", "brief": "a valid brief"}


def test_input_feed_materializes_static_input_and_schema_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "defaults_demo",
        "name": "Defaults Demo",
        "version": 1,
        "triggers": [{
            "type": "manual",
            "id": "kickoff",
            "input": {"mode": "review"},
            "input_schema": {
                "repo_path": {"kind": "repo_path", "required": True, "default": "/tmp/repo"},
                "criteria": {"kind": "criteria", "default": "match README"},
            },
            "intake": {"mode": "continuous", "dedupe_key": "$.input.repo_path"},
        }],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        feed = wfdb.open_input_feed(conn, "defaults_demo", trigger_id="kickoff")
        item = wfdb.enqueue_input_item(conn, feed.feed_id, {})
        with wfdb.write_txn(conn):
            claimed = wfdb.claim_next_ready_input_item(conn)

    expected_dedupe = "sha256:" + hashlib.sha256(b"/tmp/repo").hexdigest()
    assert item.status == "queued"
    assert item.input == {"mode": "review", "repo_path": "/tmp/repo", "criteria": "match README"}
    assert item.dedupe_value is not None
    assert item.dedupe_value == expected_dedupe
    assert "/tmp/repo" not in item.dedupe_value
    assert claimed is not None
    assert claimed.input == item.input


def test_input_feed_dedupe_is_a_database_invariant(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        conn.execute(
            """
            INSERT INTO workflow_input_items (
                item_id, feed_id, workflow_id, version, trigger_id, status,
                input_json, criteria_json, dedupe_value, created_at, updated_at
            ) VALUES (?, ?, 'demo', 1, 'manual', 'queued', '{}', '{}', 'same', 1, 1)
            """,
            ("wfitem_a", feed.feed_id),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO workflow_input_items (
                    item_id, feed_id, workflow_id, version, trigger_id, status,
                    input_json, criteria_json, dedupe_value, created_at, updated_at
                ) VALUES (?, ?, 'demo', 1, 'manual', 'queued', '{}', '{}', 'same', 2, 2)
                """,
                ("wfitem_b", feed.feed_id),
            )


def test_update_input_item_reports_dedupe_conflict_without_returning_other_item(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "dedupe_update_demo",
        "name": "Dedupe Update Demo",
        "version": 1,
        "triggers": [{
            "type": "manual",
            "id": "manual",
            "intake": {"mode": "continuous", "dedupe_key": "$.input.source_id"},
        }],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        feed = wfdb.open_input_feed(conn, "dedupe_update_demo", trigger_id="manual")
        first = wfdb.enqueue_input_item(conn, feed.feed_id, {"source_id": "a", "body": "first"})
        second = wfdb.enqueue_input_item(conn, feed.feed_id, {"source_id": "b", "body": "second"})

        with pytest.raises(ValueError, match="dedupe conflict"):
            wfdb.update_input_item(conn, second.item_id, {"source_id": "a", "body": "changed"})

        assert wfdb.get_input_item(conn, first.item_id).input == {"source_id": "a", "body": "first"}
        assert wfdb.get_input_item(conn, second.item_id).input == {"source_id": "b", "body": "second"}


def test_sync_terminal_input_items_marks_blocked_executions_terminal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        item = wfdb.enqueue_input_item(conn, feed.feed_id, {"source_id": "a"})
        exec_id = wfdb.start_execution(conn, "demo", input_data=item.input, trigger_type="input_feed")
        wfdb.mark_input_item_running(conn, item.item_id, exec_id)
        conn.execute(
            "UPDATE workflow_executions SET status = 'blocked' WHERE execution_id = ?",
            (exec_id,),
        )

        assert wfdb.sync_terminal_input_items(conn) == 1

        synced = wfdb.get_input_item(conn, item.item_id)
        assert synced.status == "blocked"
        assert synced.execution_id == exec_id


def test_claim_next_ready_input_item_requires_write_transaction(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        with pytest.raises(RuntimeError, match="write_txn"):
            wfdb.claim_next_ready_input_item(conn)


def test_update_input_item_rejects_running_items(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        item = wfdb.enqueue_input_item(conn, feed.feed_id, {"source_id": "a"})
        wfdb.mark_input_item_running(conn, item.item_id, "wfexec_test")

        with pytest.raises(ValueError, match="not mutable"):
            wfdb.update_input_item(conn, item.item_id, {"source_id": "b"})

        unchanged = wfdb.get_input_item(conn, item.item_id)
        assert unchanged.status == "running"
        assert unchanged.execution_id == "wfexec_test"
        assert unchanged.input == {"source_id": "a"}


def test_update_input_item_rejects_terminal_items(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        item = wfdb.enqueue_input_item(conn, feed.feed_id, {"source_id": "a"})
        wfdb.mark_input_item_terminal(conn, item.item_id, "succeeded")

        with pytest.raises(ValueError, match="not mutable"):
            wfdb.update_input_item(conn, item.item_id, {"source_id": "b"})

        unchanged = wfdb.get_input_item(conn, item.item_id)
        assert unchanged.status == "succeeded"
        assert unchanged.input == {"source_id": "a"}


def test_set_input_feed_status_rejects_unknown_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")

        with pytest.raises(ValueError, match="invalid feed status"):
            wfdb.set_input_feed_status(conn, feed.feed_id, "foreverish")


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


@pytest.mark.parametrize(
    ("input_data", "message"),
    [
        ({}, "brief is required"),
        ({"brief": "ab"}, "brief must be at least 3 characters"),
    ],
)
def test_start_manual_execution_rejects_invalid_trigger_input_before_insert(
    tmp_path,
    monkeypatch,
    input_data,
    message,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _required_manual_spec(), created_by="test")

        with pytest.raises(ValueError, match=message):
            wfdb.start_manual_execution(conn, "manual_required", input_data=input_data)

        assert conn.execute("SELECT count(*) FROM workflow_executions").fetchone()[0] == 0


def test_start_manual_execution_rejects_false_ready_when_before_insert(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "manual_ready",
        "name": "Manual Ready",
        "version": 1,
        "triggers": [{
            "type": "manual",
            "id": "manual",
            "intake": {"ready_when": {"op": "exists", "path": "$.input.confirmed"}},
        }],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")

        with pytest.raises(ValueError, match="ready_when is false"):
            wfdb.start_manual_execution(conn, "manual_ready", input_data={})

        assert conn.execute("SELECT count(*) FROM workflow_executions").fetchone()[0] == 0


def test_start_manual_execution_materializes_static_input_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "manual_defaults",
        "name": "Manual Defaults",
        "version": 1,
        "triggers": [{
            "type": "manual",
            "id": "manual",
            "input": {"mode": "review"},
            "input_schema": {"brief": {"kind": "long_text", "default": "check it"}},
        }],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        exec_id = wfdb.start_manual_execution(conn, "manual_defaults", input_data={})
        execution = wfdb.get_execution(conn, exec_id)

    assert execution.input == {"mode": "review", "brief": "check it"}


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


def test_deploy_same_version_different_checksum_requires_bump(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    changed = _demo_spec(version=1).model_copy(update={"name": "Demo Changed"})
    with wfdb.connect() as conn:
        assert wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test") == 1
        with pytest.raises(ValueError, match="different checksum; bump version"):
            wfdb.deploy_definition(conn, changed, created_by="test")


def test_deploy_auto_bump_redeploys_as_next_version(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    changed = _demo_spec(version=1).model_copy(update={"name": "Demo Changed"})
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test")
        wfdb.deploy_definition(conn, _demo_spec(version=3), created_by="test")

        deployed = wfdb.deploy_definition(conn, changed, created_by="test", auto_bump=True)

        assert deployed == 4
        record = wfdb.get_definition_record(conn, "demo", 4)
        assert record.name == "Demo Changed"
        # Idempotent no-op path still returns the version unchanged.
        assert wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test") == 1


def test_disabled_definition_blocks_new_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    disabled = _demo_spec(version=1).model_copy(update={"enabled": False})
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, disabled, created_by="test")

        with pytest.raises(ValueError, match="is disabled"):
            wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual")


def test_set_definition_enabled_toggles_runs_and_schedules(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "sched_demo", "name": "Sched Demo", "version": 1,
        "triggers": [
            {"type": "manual", "id": "manual"},
            {"type": "schedule", "id": "daily", "cron": "0 9 * * *"},
        ],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")

        def schedule_count():
            return conn.execute(
                "SELECT count(*) FROM workflow_schedules WHERE workflow_id = 'sched_demo'"
            ).fetchone()[0]

        assert schedule_count() == 1

        record = wfdb.set_definition_enabled(conn, "sched_demo", False)
        assert record.enabled is False
        assert schedule_count() == 0
        with pytest.raises(ValueError, match="is disabled"):
            wfdb.start_execution(conn, "sched_demo", input_data={}, trigger_type="manual")

        record = wfdb.set_definition_enabled(conn, "sched_demo", True)
        assert record.enabled is True
        assert schedule_count() == 1
        exec_id = wfdb.start_execution(conn, "sched_demo", input_data={}, trigger_type="manual")
        assert exec_id.startswith("wfexec_")


def test_cancel_terminalizes_inflight_node_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
        exec_id = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual")
        conn.execute(
            "UPDATE workflow_executions SET status = 'waiting' WHERE execution_id = ?",
            (exec_id,),
        )
        conn.execute(
            """
            INSERT INTO workflow_node_runs (execution_id, node_id, status, started_at, wait_until)
            VALUES (?, 'start', 'waiting', 1, 9999999999)
            """,
            (exec_id,),
        )
        conn.execute(
            """
            INSERT INTO workflow_node_runs (execution_id, node_id, status, started_at)
            VALUES (?, 'start', 'queued', 1)
            """,
            (exec_id,),
        )

        execution, cancelled = wfdb.cancel_execution(conn, exec_id, source="test")

        assert cancelled is True
        assert execution.status == "cancelled"
        rows = conn.execute(
            "SELECT status, completed_at, wait_until FROM workflow_node_runs WHERE execution_id = ?",
            (exec_id,),
        ).fetchall()
        assert {row["status"] for row in rows} == {"cancelled"}
        assert all(row["completed_at"] is not None for row in rows)
        assert all(row["wait_until"] is None for row in rows)


def test_list_executions_newest_first_with_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
        ids = [
            wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual", now=100 + i)
            for i in range(3)
        ]

        newest_first = wfdb.list_executions(conn)
        assert [e.execution_id for e in newest_first] == list(reversed(ids))

        limited = wfdb.list_executions(conn, "demo", limit=2)
        assert [e.execution_id for e in limited] == list(reversed(ids))[:2]

        assert wfdb.list_executions(conn, "other") == []


def test_list_events_returns_timeline_and_raises_on_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
        exec_id = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual")
        wfdb.append_event(conn, exec_id, "execution_started", {})
        wfdb.append_event(conn, exec_id, "node_succeeded", {"node_id": "start"})

        events = wfdb.list_events(conn, exec_id)

        assert [event["kind"] for event in events] == ["execution_started", "node_succeeded"]
        assert events[1]["payload"] == {"node_id": "start"}

        with pytest.raises(KeyError, match="workflow execution not found"):
            wfdb.list_events(conn, "missing")
