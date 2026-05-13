from __future__ import annotations

import sqlite3

import pytest

from hermes_cli.workflow.store import add_event, connect, create_workflow, get_workflow, list_events, list_workflows


def test_connect_initializes_schema_version_and_tables(tmp_path):
    db_path = tmp_path / "workflow.db"

    with connect(db_path) as conn:
        schema_version = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0]
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'workflow%'"
            ).fetchall()
        }

    assert schema_version == "1"
    assert table_names == {
        "workflow_artifacts",
        "workflow_edges",
        "workflow_events",
        "workflow_gates",
        "workflow_kanban_mappings",
        "workflow_nodes",
        "workflows",
    }


def test_create_get_and_serialize_workflow_preserves_policy_and_routing_metadata(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        workflow = create_workflow(
            conn,
            workflow_id="wf_test",
            title="  Build workflow store  ",
            description="Persist workflow state",
            workspace_path=tmp_path / "workspace",
            board="hermes-core",
            scale="large",
            status="planning",
            current_gate="prd",
            policy_path=tmp_path / "workspace" / ".hermes" / "workflow.yaml",
            policy_snapshot={"roles": {"engineer": "fixer"}},
            created_by="planner",
            metadata={"profileSuite": "oh-my-hermes-agent"},
            now=123.5,
        )

        fetched = get_workflow(conn, "wf_test")

    assert fetched == workflow
    assert workflow.id == "wf_test"
    assert workflow.title == "Build workflow store"
    assert workflow.workspace_path == str(tmp_path / "workspace")
    assert workflow.board == "hermes-core"
    assert workflow.scale == "large"
    assert workflow.current_gate == "prd"
    assert workflow.policy_snapshot == {"roles": {"engineer": "fixer"}}
    assert workflow.metadata == {"profileSuite": "oh-my-hermes-agent"}
    assert workflow.to_dict()["workspacePath"] == str(tmp_path / "workspace")
    assert workflow.to_dict()["policySnapshot"] == {"roles": {"engineer": "fixer"}}


def test_list_workflows_filters_by_board_and_status_and_orders_by_updated_time(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_old", title="Old", board="core", status="running", now=1.0)
        create_workflow(conn, workflow_id="wf_other_board", title="Other board", board="webui", status="running", now=3.0)
        create_workflow(conn, workflow_id="wf_done", title="Done", board="core", status="done", now=4.0)
        create_workflow(conn, workflow_id="wf_new", title="New", board="core", status="running", now=5.0)

        workflows = list_workflows(conn, board="core", status="running")

    assert [workflow.id for workflow in workflows] == ["wf_new", "wf_old"]


def test_add_and_list_events_preserves_actor_node_and_structured_data(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_events", title="Events")

        add_event(
            conn,
            workflow_id="wf_events",
            event_id="evt_created",
            event_type="workflow_created",
            actor_type="system",
            message="Workflow created",
            data={"source": "test"},
            now=1.0,
        )
        reviewed = add_event(
            conn,
            workflow_id="wf_events",
            event_id="evt_reviewed",
            node_id="review-spec",
            event_type="gate_reviewed",
            actor_type="profile",
            actor_id="oracle",
            message="Spec accepted",
            data={"verdict": "approved", "evidence": ["tests"]},
            now=2.0,
        )

        events = list_events(conn, "wf_events")

    assert reviewed.to_dict() == {
        "id": "evt_reviewed",
        "workflowId": "wf_events",
        "nodeId": "review-spec",
        "eventType": "gate_reviewed",
        "actorType": "profile",
        "actorId": "oracle",
        "message": "Spec accepted",
        "data": {"verdict": "approved", "evidence": ["tests"]},
        "createdAt": 2.0,
    }
    assert [event.id for event in events] == ["evt_created", "evt_reviewed"]


def test_add_event_rejects_unknown_workflow(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(ValueError, match="workflow not found: wf_missing"):
            add_event(conn, workflow_id="wf_missing", event_type="test", actor_type="system")


def test_create_workflow_validates_required_title_and_scale(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(ValueError, match="workflow title must be non-empty"):
            create_workflow(conn, title="   ")

        with pytest.raises(ValueError, match="invalid workflow scale"):
            create_workflow(conn, title="Bad scale", scale="epic")


def test_delete_workflow_cascades_events(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_delete", title="Delete me")
        add_event(conn, workflow_id="wf_delete", event_type="created", actor_type="system")

        conn.execute("DELETE FROM workflows WHERE id = ?", ("wf_delete",))
        conn.commit()

        events = list_events(conn, "wf_delete")

    assert events == []


def test_foreign_keys_are_enabled(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO workflow_events (id, workflow_id, event_type, actor_type, data_json, created_at)
                VALUES ('evt_orphan', 'wf_missing', 'created', 'system', '{}', 1.0)
                """
            )
