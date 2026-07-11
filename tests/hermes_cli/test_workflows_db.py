import hashlib
import sqlite3

import pytest

from hermes_cli import workflows_db as wfdb
from hermes_cli import workflows_dispatcher
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


# --- Task 3: drafts, immutable publish, archive, feed lifecycle ---


def test_save_draft_round_trips_spec_and_overwrites_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = _demo_spec(version=1)
    with wfdb.connect() as conn:
        wfdb.save_draft(conn, spec, base_version=None)
        loaded = wfdb.get_draft(conn, spec.id)
        assert loaded.spec.id == "demo"
        assert loaded.base_version is None

        changed = spec.model_copy(update={"name": "Changed"})
        wfdb.save_draft(conn, changed, base_version=1)
        loaded = wfdb.get_draft(conn, spec.id)
        assert loaded.spec.name == "Changed"
        assert loaded.base_version == 1


def test_delete_draft_returns_true_when_present_and_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = _demo_spec(version=1)
    with wfdb.connect() as conn:
        assert wfdb.delete_draft(conn, spec.id) is False
        wfdb.save_draft(conn, spec, base_version=None)
        assert wfdb.delete_draft(conn, spec.id) is True
        assert wfdb.delete_draft(conn, spec.id) is False


def test_draft_publish_is_immutable_and_conflict_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = _demo_spec().model_copy(update={"version": 1})
    with wfdb.connect() as conn:
        wfdb.save_draft(conn, spec, base_version=None)
        published = wfdb.publish_draft(
            conn, spec.id, expected_latest_version=None, created_by="test"
        )
        assert published.version == 1
        changed = spec.model_copy(update={"name": "Changed"})
        wfdb.save_draft(conn, changed, base_version=1)
        with pytest.raises(wfdb.WorkflowVersionConflict):
            wfdb.publish_draft(
                conn, spec.id, expected_latest_version=0, created_by="test"
            )
        # Original immutable record still exists with the original name.
        assert wfdb.get_definition(conn, spec.id, 1).name == spec.name


def test_draft_publish_stale_expected_version_raises_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test")
        wfdb.deploy_definition(conn, _demo_spec(version=2), created_by="test")
        draft_spec = _demo_spec(version=2).model_copy(update={"name": "Draft v3"})
        wfdb.save_draft(conn, draft_spec, base_version=1)
        with pytest.raises(wfdb.WorkflowVersionConflict):
            wfdb.publish_draft(
                conn, draft_spec.id, expected_latest_version=1, created_by="test"
            )


def test_publish_draft_clears_draft_after_successful_publish(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = _demo_spec(version=1)
    with wfdb.connect() as conn:
        wfdb.save_draft(conn, spec, base_version=None)
        wfdb.publish_draft(conn, spec.id, expected_latest_version=None, created_by="test")
        assert wfdb.get_draft(conn, spec.id) is None


def test_publish_draft_rolls_back_definition_when_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test")
        draft_spec = _demo_spec(version=2).model_copy(update={"name": "Draft v2"})
        wfdb.save_draft(conn, draft_spec, base_version=1)
        with pytest.raises(wfdb.WorkflowVersionConflict):
            wfdb.publish_draft(
                conn, draft_spec.id, expected_latest_version=0, created_by="test"
            )
        # Conflict means no new definition row exists.
        assert (
            conn.execute(
                "SELECT count(*) FROM workflow_definitions WHERE workflow_id = ?",
                (draft_spec.id,),
            ).fetchone()[0]
            == 1
        )
        # Draft is preserved so the operator can rebaseline.
        assert wfdb.get_draft(conn, draft_spec.id).spec.name == "Draft v2"


def test_set_workflow_archived_hides_definition_from_summary_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test")
        wfdb.set_workflow_archived(conn, "demo", True)
        summaries = wfdb.list_workflow_summaries(conn)
        assert summaries == []
        all_summaries = wfdb.list_workflow_summaries(conn, include_archived=True)
        assert len(all_summaries) == 1
        assert all_summaries[0]["archived"] is True


def test_list_workflow_summaries_exposes_draft_archive_and_exec_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual", "intake": {"mode": "continuous"}}],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        exec_id = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual")
        wfdb.append_event(conn, exec_id, "node_succeeded", {"node_id": "start"})
        conn.execute(
            "UPDATE workflow_executions SET status = 'succeeded', updated_at = ? WHERE execution_id = ?",
            (1, exec_id),
        )
        wfdb.save_draft(conn, spec.model_copy(update={"name": "Draft v2"}), base_version=1)
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        wfdb.enqueue_input_item(conn, feed.feed_id, {})
        summaries = wfdb.list_workflow_summaries(conn)
        assert len(summaries) == 1
        row = summaries[0]
        assert row["workflow_id"] == "demo"
        assert row["has_draft"] is True
        assert row["latest_version"] == 1
        assert row["enabled"] is True
        assert row["archived"] is False
        assert row["latest_execution_status"] == "succeeded"
        assert row["open_feed_count"] == 1


def test_delete_definition_without_history_succeeds_and_removes_row(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test")
        assert wfdb.delete_definition(conn, "demo", purge=False) is True
        assert wfdb.delete_definition(conn, "missing", purge=False) is False


def test_delete_definition_with_history_raises_conflict_unless_purge(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test")
        wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual")
        with pytest.raises(wfdb.WorkflowHistoryExists):
            wfdb.delete_definition(conn, "demo", purge=False)
        assert wfdb.delete_definition(conn, "demo", purge=True) is True


# --- Feed lifecycle (Task 3) ---


def test_open_feed_pause_resume_close_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        # 1) enqueue while open.
        item = wfdb.enqueue_input_item(conn, feed.feed_id, {"repo_path": "/repo"})
        assert item.status == "queued"
        # 2) pause -> no new claims.
        wfdb.set_input_feed_status(conn, feed.feed_id, "paused")
        assert workflows_dispatcher.tick(limit=1) == 0
        # 3) resume -> admission resumes.
        wfdb.set_input_feed_status(conn, feed.feed_id, "open")
        assert workflows_dispatcher.tick(limit=1) == 1
        # 4) close -> terminal.
        wfdb.set_input_feed_status(conn, feed.feed_id, "closed")
        with pytest.raises(ValueError, match="closed feed cannot transition"):
            wfdb.set_input_feed_status(conn, feed.feed_id, "open")
        # 5) opening a new feed on the same workflow yields a different feed_id.
        next_feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        assert next_feed.feed_id != feed.feed_id


def test_closed_feed_is_terminal_and_rejects_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        wfdb.set_input_feed_status(conn, feed.feed_id, "closed")
        with pytest.raises(ValueError, match="closed feed cannot transition"):
            wfdb.set_input_feed_status(conn, feed.feed_id, "open")
        with pytest.raises(ValueError, match="feed is closed"):
            wfdb.enqueue_input_item(conn, feed.feed_id, {"repo_path": "/repo"})


def test_paused_feed_rejects_item_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        wfdb.set_input_feed_status(conn, feed.feed_id, "paused")
        with pytest.raises(ValueError, match="feed is paused"):
            wfdb.enqueue_input_item(conn, feed.feed_id, {"repo_path": "/repo"})


def test_paused_feed_rejects_item_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        item = wfdb.enqueue_input_item(conn, feed.feed_id, {"repo_path": "/repo"})
        wfdb.set_input_feed_status(conn, feed.feed_id, "paused")
        with pytest.raises(ValueError, match="feed is paused"):
            wfdb.update_input_item(conn, item.item_id, {"repo_path": "/repo"})


def test_set_input_feed_status_idempotent_noop_returns_feed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        same = wfdb.set_input_feed_status(conn, feed.feed_id, "open")
        assert same.feed_id == feed.feed_id
        assert same.status == "open"


# --- Task 8: workflow-filtered history, detail, cancel, rerun ---


def _second_spec(*, version: int = 1) -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "other", "name": "Other", "version": version,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {"start": {"type": "pass"}},
    })


def test_list_executions_filters_by_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
        e1 = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual", now=100)
        e2 = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual", now=101)
        conn.execute("UPDATE workflow_executions SET status = 'succeeded' WHERE execution_id = ?", (e1,))
        conn.execute("UPDATE workflow_executions SET status = 'running' WHERE execution_id = ?", (e2,))

        queued = wfdb.list_executions(conn, status="queued")
        assert queued == []

        succeeded = wfdb.list_executions(conn, status="succeeded")
        assert len(succeeded) == 1
        assert succeeded[0].execution_id == e1

        running = wfdb.list_executions(conn, status="running")
        assert len(running) == 1
        assert running[0].execution_id == e2


def test_list_executions_filters_by_version(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test")
        wfdb.deploy_definition(conn, _demo_spec(version=2), created_by="test")
        e1 = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual", version=1, now=100)
        e2 = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual", version=2, now=101)

        v1 = wfdb.list_executions(conn, version=1)
        assert len(v1) == 1
        assert v1[0].execution_id == e1

        v2 = wfdb.list_executions(conn, version=2)
        assert len(v2) == 1
        assert v2[0].execution_id == e2


def test_list_executions_filters_by_trigger_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "multi", "name": "Multi", "version": 1,
        "triggers": [
            {"type": "manual", "id": "t1"},
            {"type": "manual", "id": "t2"},
        ],
        "nodes": {"start": {"type": "pass"}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        e1 = wfdb.start_execution(conn, "multi", input_data={}, trigger_type="manual", trigger_id="t1", now=100)
        e2 = wfdb.start_execution(conn, "multi", input_data={}, trigger_type="manual", trigger_id="t2", now=101)

        filtered = wfdb.list_executions(conn, trigger_id="t1")
        assert len(filtered) == 1
        assert filtered[0].execution_id == e1


def test_list_executions_before_cursor_pages_newest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
        ids = [
            wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual", now=100 + i)
            for i in range(5)
        ]
        # ids[4] is newest (created_at=104). Before cursor on ids[3] (created_at=103)
        # should return ids[4] only.
        page = wfdb.list_executions(conn, before=(103, ids[3]))
        assert [e.execution_id for e in page] == [ids[4]]


def test_list_executions_combined_filters(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(version=1), created_by="test")
        wfdb.deploy_definition(conn, _second_spec(version=1), created_by="test")
        e1 = wfdb.start_execution(conn, "demo", input_data={}, trigger_type="manual", now=100)
        e2 = wfdb.start_execution(conn, "other", input_data={}, trigger_type="manual", now=101)
        conn.execute("UPDATE workflow_executions SET status = 'succeeded' WHERE execution_id = ?", (e1,))
        conn.execute("UPDATE workflow_executions SET status = 'succeeded' WHERE execution_id = ?", (e2,))

        # workflow_id + status
        result = wfdb.list_executions(conn, "demo", status="succeeded")
        assert len(result) == 1
        assert result[0].workflow_id == "demo"


def test_get_execution_detail_returns_execution_node_runs_and_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _demo_spec(), created_by="test")
        exec_id = wfdb.start_execution(conn, "demo", input_data={"x": 1}, trigger_type="manual")
        conn.execute(
            "INSERT INTO workflow_node_runs (execution_id, node_id, status) VALUES (?, 'start', 'succeeded')",
            (exec_id,),
        )
        wfdb.append_event(conn, exec_id, "execution_started", {"ok": True})

        detail = wfdb.get_execution_detail(conn, exec_id)
        assert detail["execution"].execution_id == exec_id
        assert detail["definition"].workflow_id == "demo"
        assert len(detail["node_runs"]) == 1
        assert len(detail["events"]) == 1
        assert detail["events"][0]["kind"] == "execution_started"
