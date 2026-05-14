from __future__ import annotations

import sqlite3

import pytest

from hermes_cli.workflow import WorkflowArtifact, WorkflowGate, WorkflowInboxItem
from hermes_cli.workflow.store import (
    add_artifact,
    add_event,
    add_gate,
    connect,
    create_inbox_item,
    create_workflow,
    get_inbox_item,
    get_workflow,
    list_artifacts,
    list_events,
    list_gates,
    list_inbox_items,
    list_workflows,
    resolve_gate,
    save_dag,
)


def test_public_workflow_package_exports_artifact_record():
    assert WorkflowArtifact.__name__ == "WorkflowArtifact"


def test_public_workflow_package_exports_gate_record():
    assert WorkflowGate.__name__ == "WorkflowGate"


def test_public_workflow_package_exports_inbox_item_record():
    assert WorkflowInboxItem.__name__ == "WorkflowInboxItem"


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
        "workflow_inbox_items",
        "workflow_kanban_mappings",
        "workflow_nodes",
        "workflows",
    }


def test_create_get_and_list_inbox_items_preserves_intake_classification_and_metadata(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        older = create_inbox_item(
            conn,
            inbox_item_id="inbox_old",
            title="  Improve workflow DAG  ",
            body="Make the DAG operational.",
            source="webui_chat",
            status="new",
            classification="needs_shaping",
            workspace_path=tmp_path / "webui",
            created_by="user",
            metadata={"labels": ["workflow-system"]},
            now=1.0,
        )
        create_inbox_item(
            conn,
            inbox_item_id="inbox_other",
            title="Other",
            source="github_issue",
            status="new",
            classification="one_off",
            now=3.0,
        )
        newer = create_inbox_item(
            conn,
            inbox_item_id="inbox_new",
            title="Build inbox",
            source="webui_chat",
            status="triaged",
            classification="decomposition_worthy",
            assigned_workflow_id="wf_future",
            now=5.0,
        )

        fetched = get_inbox_item(conn, "inbox_old")
        webui_items = list_inbox_items(conn, source="webui_chat")
        new_items = list_inbox_items(conn, status="new")

    assert fetched == older
    assert older.title == "Improve workflow DAG"
    assert older.workspace_path == str(tmp_path / "webui")
    assert older.metadata == {"labels": ["workflow-system"]}
    assert older.to_dict() == {
        "id": "inbox_old",
        "title": "Improve workflow DAG",
        "body": "Make the DAG operational.",
        "source": "webui_chat",
        "status": "new",
        "classification": "needs_shaping",
        "workspacePath": str(tmp_path / "webui"),
        "assignedWorkflowId": None,
        "createdAt": 1.0,
        "updatedAt": 1.0,
        "createdBy": "user",
        "metadata": {"labels": ["workflow-system"]},
    }
    assert newer.assigned_workflow_id == "wf_future"
    assert [item.id for item in webui_items] == ["inbox_new", "inbox_old"]
    assert [item.id for item in new_items] == ["inbox_other", "inbox_old"]


def test_create_inbox_item_requires_non_empty_title(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(ValueError, match="inbox item title must be non-empty"):
            create_inbox_item(conn, title="   ", source="webui_chat")


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


def test_add_and_resolve_gates_preserves_required_actor_verdict_and_metadata(tmp_path):
    artifact_path = tmp_path / "spec.md"
    artifact_path.write_text("# Spec\n", encoding="utf-8")
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_gates", title="Gates")
        add_artifact(conn, workflow_id="wf_gates", artifact_id="art_spec", kind="spec", path=artifact_path)
        gate = add_gate(
            conn,
            gate_id="gate_spec",
            workflow_id="wf_gates",
            node_id="spec-review",
            gate_type="spec_review",
            level=2,
            status="pending",
            required_actor="reviewer",
            artifact_id="art_spec",
            reason="Spec needs approval",
            metadata={"policy": "required"},
        )
        resolved = resolve_gate(
            conn,
            gate_id="gate_spec",
            status="approved",
            verdict="approved",
            resolved_by="reviewer-profile",
            reason="Looks good",
            metadata={"evidence": ["tests"]},
            now=7.0,
        )
        gates = list_gates(conn, "wf_gates")

    assert gate.status == "pending"
    assert resolved.to_dict() == {
        "id": "gate_spec",
        "workflowId": "wf_gates",
        "nodeId": "spec-review",
        "gateType": "spec_review",
        "level": 2,
        "status": "approved",
        "verdict": "approved",
        "requiredActor": "reviewer",
        "resolvedBy": "reviewer-profile",
        "resolvedAt": 7.0,
        "artifactId": "art_spec",
        "reason": "Looks good",
        "metadata": {"policy": "required", "evidence": ["tests"]},
    }
    assert [item.id for item in gates] == ["gate_spec"]


def test_resolve_gate_merges_metadata_and_preserves_reason_when_omitted(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_gate_merge", title="Gate merge")
        add_gate(
            conn,
            gate_id="gate_merge",
            workflow_id="wf_gate_merge",
            gate_type="dag_review",
            level=1,
            required_actor="reviewer",
            reason="Initial gate reason",
            metadata={"policy": "required", "attempts": 1},
        )

        resolved = resolve_gate(
            conn,
            gate_id="gate_merge",
            status="approved",
            verdict="approved",
            resolved_by="reviewer-profile",
            metadata={"evidence": ["tests"], "attempts": 2},
            now=8.0,
        )

    assert resolved.reason == "Initial gate reason"
    assert resolved.metadata == {"policy": "required", "attempts": 2, "evidence": ["tests"]}


def test_add_gate_rejects_unknown_artifact_reference(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_gate_artifact", title="Gate artifact")

        with pytest.raises(ValueError, match="artifact not found: art_missing"):
            add_gate(
                conn,
                workflow_id="wf_gate_artifact",
                gate_type="dag_review",
                level=1,
                required_actor="human",
                artifact_id="art_missing",
            )


def test_gate_helpers_reject_unknown_workflow_and_gate(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(ValueError, match="workflow not found: wf_missing"):
            add_gate(conn, workflow_id="wf_missing", gate_type="dag_review", level=1, required_actor="human")
        with pytest.raises(ValueError, match="gate not found: gate_missing"):
            resolve_gate(conn, gate_id="gate_missing", status="approved", verdict="approved", resolved_by="human")


def test_delete_workflow_cascades_gates(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_delete_gates", title="Delete gates")
        add_gate(conn, workflow_id="wf_delete_gates", gate_type="dag_review", level=1, required_actor="human")
        conn.execute("DELETE FROM workflows WHERE id = ?", ("wf_delete_gates",))
        conn.commit()

        gates = list_gates(conn, "wf_delete_gates")

    assert gates == []


def test_add_and_list_artifacts_records_auditable_file_metadata(tmp_path):
    artifact_path = tmp_path / "artifacts" / "dag.yaml"
    artifact_path.parent.mkdir()
    artifact_path.write_text("schema_version: 1\nkind: dag\n", encoding="utf-8")

    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_artifacts", title="Artifacts")

        artifact = add_artifact(
            conn,
            workflow_id="wf_artifacts",
            artifact_id="art_dag",
            kind="dag",
            path=artifact_path,
            mime_type="application/yaml",
            schema_version=1,
            created_by="decomposer",
            metadata={"source": "test"},
            now=3.5,
        )
        artifacts = list_artifacts(conn, "wf_artifacts")

    assert artifact.id == "art_dag"
    assert artifact.workflow_id == "wf_artifacts"
    assert artifact.kind == "dag"
    assert artifact.path == str(artifact_path)
    assert artifact.sha256 == "fece7043be67c89251cb4088c07bd29e861debce78320cc6360881c631c87d51"
    assert artifact.mime_type == "application/yaml"
    assert artifact.created_by == "decomposer"
    assert artifact.metadata == {"source": "test"}
    assert artifact.to_dict()["createdAt"] == 3.5
    assert [item.id for item in artifacts] == ["art_dag"]


def test_save_dag_replaces_nodes_and_edges_transactionally(tmp_path):
    normalized = {
        "nodes": [
            {
                "id": "backend-api",
                "title": "Implement backend API",
                "role": "engineer",
                "profile": "engineer",
                "status": "waiting",
                "parents": [],
                "gate_level": 1,
                "definition_of_done": ["Tests pass."],
                "scope": {"summary": "Build API."},
                "workspace": {"kind": "worktree", "base_ref": "origin/main"},
            },
            {
                "id": "integration",
                "title": "Integrate outputs",
                "role": "integrator",
                "profile": "integrator",
                "status": "waiting",
                "parents": ["backend-api"],
                "gate_level": 2,
                "gate": {"type": "integration_review"},
                "definition_of_done": ["Integration tests pass."],
                "scope": {"summary": "Integrate API."},
            },
        ],
        "edges": [{"source": "backend-api", "target": "integration", "kind": "depends_on"}],
    }

    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_dag", title="DAG")
        save_dag(conn, workflow_id="wf_dag", normalized_dag=normalized, now=10.0)
        save_dag(
            conn,
            workflow_id="wf_dag",
            normalized_dag={**normalized, "nodes": normalized["nodes"][:1], "edges": []},
            now=11.0,
        )

        node_rows = conn.execute("SELECT * FROM workflow_nodes WHERE workflow_id = ?", ("wf_dag",)).fetchall()
        edge_rows = conn.execute("SELECT * FROM workflow_edges WHERE workflow_id = ?", ("wf_dag",)).fetchall()
        workflow = get_workflow(conn, "wf_dag")

    assert workflow is not None
    assert workflow.updated_at == 11.0
    assert len(node_rows) == 1
    assert node_rows[0]["node_id"] == "backend-api"
    assert node_rows[0]["role"] == "engineer"
    assert node_rows[0]["base_ref"] == "origin/main"
    assert node_rows[0]["definition_of_done_json"] == '["Tests pass."]'
    assert node_rows[0]["scope_json"] == '{"summary":"Build API."}'
    assert node_rows[0]["created_at"] == 11.0
    assert node_rows[0]["updated_at"] == 11.0
    assert edge_rows == []


def test_save_dag_rejects_workflow_id_mismatch(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_dag", title="DAG")
        with pytest.raises(ValueError, match="normalized DAG workflow_id does not match wf_dag"):
            save_dag(conn, workflow_id="wf_dag", normalized_dag={"workflow_id": "wf_other", "nodes": [], "edges": []})


def test_save_dag_rejects_unknown_workflow(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(ValueError, match="workflow not found: wf_missing"):
            save_dag(conn, workflow_id="wf_missing", normalized_dag={"nodes": [], "edges": []})


def test_add_artifact_rejects_unknown_workflow_and_missing_file(tmp_path):
    missing_path = tmp_path / "missing.md"
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(ValueError, match="workflow not found: wf_missing"):
            add_artifact(conn, workflow_id="wf_missing", kind="dag", path=missing_path)

        create_workflow(conn, workflow_id="wf_artifact_missing_file", title="Artifact missing file")
        with pytest.raises(FileNotFoundError):
            add_artifact(conn, workflow_id="wf_artifact_missing_file", kind="dag", path=missing_path)


def test_delete_workflow_cascades_artifacts(tmp_path):
    artifact_path = tmp_path / "dag.yaml"
    artifact_path.write_text("kind: dag\n", encoding="utf-8")

    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_delete_artifacts", title="Delete artifacts")
        add_artifact(conn, workflow_id="wf_delete_artifacts", kind="dag", path=artifact_path)

        conn.execute("DELETE FROM workflows WHERE id = ?", ("wf_delete_artifacts",))
        conn.commit()

        artifacts = list_artifacts(conn, "wf_delete_artifacts")

    assert artifacts == []


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
