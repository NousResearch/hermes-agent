from __future__ import annotations

import pytest

from hermes_cli import kanban_db
from hermes_cli.workflow import (
    create_inbox_item,
    get_inbox_item_detail,
    get_workflow_artifacts,
    get_workflow_dag,
    get_workflow_events,
    get_workflow_node,
    list_inbox_item_summaries,
    list_workflow_summaries,
    approve_workflow_for_materialization,
    materialize_workflow_to_kanban,
    promote_inbox_item_to_workflow,
    resolve_workflow_gate_control,
    shape_inbox_item_as_draft_workflow,
    update_inbox_item_triage,
)
from hermes_cli.workflow.materialize import materialize_workflow
from hermes_cli.workflow.store import add_artifact, add_event, add_gate, connect, create_workflow, resolve_gate, save_dag


def _dag() -> dict:
    return {
        "workflow_id": "wf_api",
        "nodes": [
            {
                "id": "backend-api",
                "title": "Implement backend API",
                "role": "engineer",
                "profile": "engineer",
                "status": "waiting",
                "gate_level": 1,
                "definition_of_done": ["API tests pass."],
                "scope": {"summary": "Build API."},
                "evidence": {"tests": ["api"]},
                "workspace": {
                    "kind": "worktree",
                    "branch": "workflow/wf_api/backend-api",
                    "worktree_path": "/tmp/wf_api-backend-api",
                    "base_ref": "origin/main",
                },
            },
            {
                "id": "integration",
                "title": "Integrate API",
                "role": "integrator",
                "profile": "integrator",
                "status": "waiting",
                "gate_level": 2,
                "parents": ["backend-api"],
                "definition_of_done": ["Integration passes."],
                "scope": {"summary": "Wire API into system."},
            },
        ],
        "edges": [{"source": "backend-api", "target": "integration", "kind": "depends_on"}],
    }


def _create_workflow(conn):
    create_workflow(conn, workflow_id="wf_api", title="API Read Model", board="core", scale="medium", status="dag_approved", now=1.0)
    save_dag(conn, workflow_id="wf_api", normalized_dag=_dag(), now=2.0)


def test_public_workflow_package_exports_read_model_api():
    assert callable(list_workflow_summaries)
    assert callable(get_workflow_dag)
    assert callable(get_workflow_node)
    assert callable(get_workflow_events)
    assert callable(get_workflow_artifacts)
    assert callable(list_inbox_item_summaries)
    assert callable(get_inbox_item_detail)
    assert callable(approve_workflow_for_materialization)
    assert callable(resolve_workflow_gate_control)
    assert callable(materialize_workflow_to_kanban)
    assert callable(promote_inbox_item_to_workflow)
    assert callable(shape_inbox_item_as_draft_workflow)
    assert callable(update_inbox_item_triage)


def test_shape_inbox_item_as_draft_workflow_returns_core_authored_dag_without_mutating_inbox(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_inbox_item(
            conn,
            inbox_item_id="inbox_shape",
            title="Shape this workflow idea",
            body="Build a durable workflow system path.",
            source="webui_chat",
            status="triaged",
            classification="decomposition_worthy",
            workspace_path=str(tmp_path),
            metadata={"labels": ["workflow-system"]},
            now=2.0,
        )

        payload = shape_inbox_item_as_draft_workflow(
            conn,
            "inbox_shape",
            workflow_id="wf_shaped",
            title="Shaped workflow",
            board="core",
            scale="medium",
        )
        item_after = get_inbox_item_detail(conn, "inbox_shape")["facts"]["inboxItem"]
        workflows = list_workflow_summaries(conn)["facts"]["workflows"]

    facts = payload["facts"]
    assert payload["insights"] is None
    assert facts["inboxItem"]["id"] == "inbox_shape"
    assert facts["draftWorkflow"] == {
        "id": "wf_shaped",
        "title": "Shaped workflow",
        "description": "Build a durable workflow system path.",
        "workspacePath": str(tmp_path),
        "board": "core",
        "scale": "medium",
        "sourceInboxItemId": "inbox_shape",
    }
    assert facts["draftDag"]["workflow_id"] == "wf_shaped"
    assert facts["draftDag"]["name"] == "Shaped workflow"
    assert facts["draftDag"]["scale"] == "medium"
    assert [node["id"] for node in facts["draftDag"]["nodes"]] == ["shape-plan", "build-slice"]
    assert facts["draftDag"]["edges"] == [{"source": "shape-plan", "target": "build-slice", "kind": "depends_on"}]
    assert "Build a durable workflow system path." in facts["draftDag"]["nodes"][0]["scope"]["summary"]
    assert item_after["status"] == "triaged"
    assert item_after["assignedWorkflowId"] is None
    assert workflows == []


def test_shape_inbox_item_as_draft_workflow_raises_for_missing_item(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(ValueError, match="workflow inbox item not found: missing"):
            shape_inbox_item_as_draft_workflow(conn, "missing", workflow_id="wf_missing")

def test_list_inbox_item_summaries_returns_intake_facts_shape_and_filters(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_inbox_item(conn, inbox_item_id="inbox_old", title="Old", source="webui_chat", status="new", now=1.0)
        create_inbox_item(conn, inbox_item_id="inbox_other", title="Other", source="github_issue", status="new", now=3.0)
        create_inbox_item(conn, inbox_item_id="inbox_new", title="New", source="webui_chat", status="triaged", classification="decomposition_worthy", now=5.0)

        payload = list_inbox_item_summaries(conn, source="webui_chat")

    assert payload["insights"] is None
    assert [item["id"] for item in payload["facts"]["inboxItems"]] == ["inbox_new", "inbox_old"]
    assert payload["facts"]["count"] == 2
    assert payload["facts"]["inboxItems"][0]["classification"] == "decomposition_worthy"


def test_get_inbox_item_detail_returns_single_intake_record(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_inbox_item(
            conn,
            inbox_item_id="inbox_detail",
            title="Shape workflow",
            source="webui_chat",
            status="new",
            metadata={"labels": ["workflow-system"]},
            now=4.0,
        )

        payload = get_inbox_item_detail(conn, "inbox_detail")

    assert payload == {
        "facts": {
            "inboxItem": {
                "id": "inbox_detail",
                "title": "Shape workflow",
                "body": "",
                "source": "webui_chat",
                "status": "new",
                "classification": None,
                "workspacePath": None,
                "assignedWorkflowId": None,
                "createdAt": 4.0,
                "updatedAt": 4.0,
                "createdBy": None,
                "metadata": {"labels": ["workflow-system"]},
            }
        },
        "insights": None,
    }


def test_get_inbox_item_detail_raises_for_missing_item(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        with pytest.raises(ValueError, match="workflow inbox item not found: missing"):
            get_inbox_item_detail(conn, "missing")


def test_update_inbox_item_triage_returns_updated_item_payload(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_inbox_item(
            conn,
            inbox_item_id="inbox_update",
            title="Rough idea",
            source="webui_chat",
            status="new",
            metadata={"labels": ["workflow-system"]},
            now=2.0,
        )

        payload = update_inbox_item_triage(
            conn,
            "inbox_update",
            status="triaged",
            classification="decomposition_worthy",
            workspace_path="/tmp/workspace",
            assigned_workflow_id="wf_update",
            metadata={"triagedBy": "planner"},
            now=8.0,
        )

    item = payload["facts"]["inboxItem"]
    assert payload["insights"] is None
    assert item["status"] == "triaged"
    assert item["classification"] == "decomposition_worthy"
    assert item["workspacePath"] == "/tmp/workspace"
    assert item["assignedWorkflowId"] == "wf_update"
    assert item["updatedAt"] == 8.0
    assert item["metadata"] == {"labels": ["workflow-system"], "triagedBy": "planner"}


def test_promote_inbox_item_to_workflow_creates_draft_workflow_and_assigns_intake(tmp_path):
    draft_dag = {
        "schema_version": 1,
        "workflow_id": "wf_promoted",
        "name": "Promoted workflow",
        "scale": "medium",
        "nodes": [
            {
                "id": "shape-plan",
                "title": "Shape the implementation plan",
                "role": "planner",
                "profile": "planner",
                "scope": {"summary": "Turn the rough request into a concrete workflow plan."},
            },
            {
                "id": "build-slice",
                "title": "Build the first slice",
                "role": "engineer",
                "profile": "engineer",
                "parents": ["shape-plan"],
                "definition_of_done": ["Targeted tests pass."],
                "scope": {"summary": "Implement the first independently useful slice."},
            },
        ],
    }
    with connect(tmp_path / "workflow.db") as conn:
        create_inbox_item(
            conn,
            inbox_item_id="inbox_promote",
            title="Rough workflow idea",
            body="Build the workflow system inbox promotion path.",
            source="webui_chat",
            status="triaged",
            classification="decomposition_worthy",
            workspace_path=str(tmp_path),
            metadata={"labels": ["workflow-system"]},
            now=2.0,
        )

        payload = promote_inbox_item_to_workflow(
            conn,
            "inbox_promote",
            workflow_id="wf_promoted",
            title="Promoted workflow",
            description="Drafted from inbox item inbox_promote.",
            board="core",
            scale="medium",
            draft_dag=draft_dag,
            actor_id="webui",
            now=8.0,
        )
        dag_payload = get_workflow_dag(conn, "wf_promoted")
        events_payload = get_workflow_events(conn, "wf_promoted")

    workflow = payload["facts"]["workflow"]
    inbox_item = payload["facts"]["inboxItem"]
    assert payload["insights"] is None
    assert workflow["id"] == "wf_promoted"
    assert workflow["status"] == "dag_draft"
    assert workflow["title"] == "Promoted workflow"
    assert workflow["workspacePath"] == str(tmp_path)
    assert workflow["metadata"] == {"sourceInboxItemId": "inbox_promote"}
    assert inbox_item["status"] == "promoted"
    assert inbox_item["assignedWorkflowId"] == "wf_promoted"
    assert inbox_item["metadata"] == {"labels": ["workflow-system"], "promotedBy": "webui"}
    assert {node["id"] for node in dag_payload["facts"]["nodes"]} == {"shape-plan", "build-slice"}
    assert dag_payload["facts"]["edges"] == [{"source": "shape-plan", "target": "build-slice", "kind": "depends_on"}]
    assert events_payload["facts"]["events"][0]["eventType"] == "inbox_promoted"
    assert events_payload["facts"]["events"][0]["data"] == {"inboxItemId": "inbox_promote"}


def test_promote_inbox_item_to_workflow_rejects_invalid_draft_dag_without_assigning_inbox(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_inbox_item(conn, inbox_item_id="inbox_bad_dag", title="Bad DAG", status="triaged", now=1.0)

        with pytest.raises(ValueError, match="workflow draft DAG is invalid"):
            promote_inbox_item_to_workflow(
                conn,
                "inbox_bad_dag",
                workflow_id="wf_bad_dag",
                title="Bad DAG workflow",
                draft_dag={"schema_version": 1, "workflow_id": "wf_bad_dag", "scale": "medium", "nodes": []},
            )
        item = get_inbox_item_detail(conn, "inbox_bad_dag")["facts"]["inboxItem"]

    assert item["status"] == "triaged"
    assert item["assignedWorkflowId"] is None


def test_promote_inbox_item_to_workflow_rejects_stale_already_promoted_item_without_creating_workflow(tmp_path):
    draft_dag = {
        "schema_version": 1,
        "workflow_id": "wf_second",
        "scale": "medium",
        "nodes": [
            {"id": "shape-plan", "title": "Shape", "role": "planner", "profile": "planner", "scope": {"summary": "Shape the stale request."}},
            {"id": "build-slice", "title": "Build", "role": "engineer", "profile": "engineer", "parents": ["shape-plan"], "definition_of_done": ["Tests pass."], "scope": {"summary": "Build the stale request."}},
        ],
    }
    with connect(tmp_path / "workflow.db") as conn:
        create_inbox_item(
            conn,
            inbox_item_id="inbox_stale_promote",
            title="Already promoted",
            status="promoted",
            assigned_workflow_id="wf_first",
            now=1.0,
        )

        with pytest.raises(ValueError, match="workflow inbox item already promoted: inbox_stale_promote"):
            promote_inbox_item_to_workflow(
                conn,
                "inbox_stale_promote",
                workflow_id="wf_second",
                title="Second workflow",
                draft_dag=draft_dag,
                now=2.0,
            )
        item = get_inbox_item_detail(conn, "inbox_stale_promote")["facts"]["inboxItem"]
        workflows = list_workflow_summaries(conn)["facts"]["workflows"]

    assert item["status"] == "promoted"
    assert item["assignedWorkflowId"] == "wf_first"
    assert workflows == []


def test_list_workflow_summaries_returns_facts_insights_shape_and_filters(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_old", title="Old", board="core", status="running", now=1.0)
        create_workflow(conn, workflow_id="wf_other", title="Other", board="webui", status="running", now=3.0)
        create_workflow(conn, workflow_id="wf_done", title="Done", board="core", status="done", now=4.0)
        create_workflow(conn, workflow_id="wf_new", title="New", board="core", status="running", now=5.0)

        payload = list_workflow_summaries(conn, board="core", status="running")

    assert payload["insights"] is None
    assert [workflow["id"] for workflow in payload["facts"]["workflows"]] == ["wf_new", "wf_old"]
    assert payload["facts"]["count"] == 2


def test_get_workflow_dag_reconstructs_nodes_edges_and_related_records(tmp_path):
    artifact_path = tmp_path / "spec.md"
    artifact_path.write_text("# Spec\n", encoding="utf-8")
    with connect(tmp_path / "workflow.db") as conn:
        kanban_conn = kanban_db.connect(tmp_path / "kanban.db")
        _create_workflow(conn)
        add_event(conn, workflow_id="wf_api", event_type="reviewed", node_id="backend-api", actor_type="agent", data={"ok": True}, now=3.0)
        artifact = add_artifact(conn, workflow_id="wf_api", kind="spec", path=artifact_path, created_by="architect", metadata={"format": "md"}, now=4.0)
        gate = add_gate(conn, workflow_id="wf_api", node_id="backend-api", gate_type="review", level=1, required_actor="reviewer", artifact_id=artifact.id, metadata={"severity": "high"})
        resolve_gate(conn, gate_id=gate.id, status="approved", verdict="pass", resolved_by="reviewer", now=6.0)
        materialize_workflow(conn, "wf_api", kanban_conn=kanban_conn, now=7.0)

        payload = get_workflow_dag(conn, "wf_api")

    facts = payload["facts"]
    node_by_id = {node["id"]: node for node in facts["nodes"]}
    assert payload["insights"] is None
    assert facts["workflow"]["id"] == "wf_api"
    assert facts["workflow"]["status"] == "materialized"
    assert facts["edges"] == [{"source": "backend-api", "target": "integration", "kind": "depends_on"}]
    assert node_by_id["backend-api"]["children"] == ["integration"]
    assert node_by_id["integration"]["parents"] == ["backend-api"]
    assert node_by_id["backend-api"]["workspace"] == {
        "kind": "worktree",
        "branch": "workflow/wf_api/backend-api",
        "worktreePath": "/tmp/wf_api-backend-api",
        "baseRef": "origin/main",
    }
    assert node_by_id["backend-api"]["kanbanTaskId"]
    assert facts["gates"][0]["status"] == "approved"
    assert facts["artifacts"][0]["kind"] == "spec"
    assert facts["controlActions"] == []


def test_workflow_dag_exposes_core_control_actions_for_pending_gates_and_approval(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_controls", title="Controls", board="core", status="dag_validated", now=1.0)
        save_dag(conn, workflow_id="wf_controls", normalized_dag={**_dag(), "workflow_id": "wf_controls"}, now=2.0)
        gate = add_gate(conn, workflow_id="wf_controls", gate_id="gate_review", gate_type="dag_review", level=1, required_actor="human", reason="needs review")

        pending_payload = get_workflow_dag(conn, "wf_controls")
        resolved_payload = resolve_workflow_gate_control(
            conn,
            "wf_controls",
            "gate_review",
            status="approved",
            verdict="approved",
            resolved_by="webui",
            reason="Looks good",
            now=3.0,
        )
        approved_payload = approve_workflow_for_materialization(conn, "wf_controls", actor_id="webui", now=4.0)

    pending_actions = pending_payload["facts"]["controlActions"]
    assert [action["type"] for action in pending_actions] == ["resolve_gate", "resolve_gate", "resolve_gate"]
    assert pending_actions[0] == {
        "id": "approve-gate:gate_review",
        "type": "resolve_gate",
        "label": "Approve dag_review gate",
        "method": "POST",
        "endpoint": "/api/workflows/wf_controls/gates/gate_review/resolve",
        "gateId": "gate_review",
        "status": "approved",
        "verdict": "approved",
        "enabled": True,
    }
    assert resolved_payload["facts"]["gate"]["status"] == "approved"
    assert [action["type"] for action in resolved_payload["facts"]["controlActions"]] == ["approve_workflow"]
    assert approved_payload["facts"]["workflow"]["status"] == "dag_approved"
    assert [action["type"] for action in approved_payload["facts"]["controlActions"]] == ["materialize_workflow"]


def test_get_workflow_node_returns_detail_drawer_payload(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        _create_workflow(conn)
        add_event(conn, workflow_id="wf_api", event_type="node_started", node_id="backend-api", actor_type="workflow", now=3.0)
        add_gate(
            conn,
            workflow_id="wf_api",
            node_id="backend-api",
            gate_type="review",
            level=1,
            required_actor="reviewer",
            status="pending",
            reason="needs human review",
            metadata={"severity": "medium"},
        )

        payload = get_workflow_node(conn, "wf_api", "backend-api")

    facts = payload["facts"]
    assert payload["insights"] is None
    assert facts["workflowId"] == "wf_api"
    assert facts["node"]["id"] == "backend-api"
    assert facts["node"]["children"] == ["integration"]
    assert facts["node"]["definitionOfDone"] == ["API tests pass."]
    assert facts["node"]["scope"] == {"summary": "Build API."}
    assert facts["gates"][0]["gateType"] == "review"
    assert facts["gates"][0]["status"] == "pending"
    assert facts["gates"][0]["reason"] == "needs human review"
    assert facts["gates"][0]["metadata"] == {"severity": "medium"}
    assert facts["events"][0]["eventType"] == "node_started"


def test_read_model_helpers_filter_events_and_artifacts(tmp_path):
    spec_path = tmp_path / "spec.md"
    log_path = tmp_path / "log.txt"
    spec_path.write_text("spec", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    with connect(tmp_path / "workflow.db") as conn:
        _create_workflow(conn)
        add_event(conn, workflow_id="wf_api", event_type="old", actor_type="workflow", now=1.0)
        add_event(conn, workflow_id="wf_api", event_type="new", actor_type="workflow", now=2.0)
        add_artifact(conn, workflow_id="wf_api", kind="spec", path=spec_path, now=1.0)
        add_artifact(conn, workflow_id="wf_api", kind="log", path=log_path, now=2.0)

        events = get_workflow_events(conn, "wf_api", limit=1)
        artifacts = get_workflow_artifacts(conn, "wf_api", kind="spec")

    assert [event["eventType"] for event in events["facts"]["events"]] == ["new"]
    assert events["facts"]["count"] == 1
    assert [artifact["kind"] for artifact in artifacts["facts"]["artifacts"]] == ["spec"]
    assert artifacts["facts"]["count"] == 1


def test_materialize_workflow_to_kanban_returns_control_action_payload(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        kanban_conn = kanban_db.connect(tmp_path / "kanban.db")
        _create_workflow(conn)

        payload = materialize_workflow_to_kanban(conn, "wf_api", kanban_conn=kanban_conn, actor_id="webui", now=9.0)

    facts = payload["facts"]
    assert payload["insights"] is None
    assert facts["workflowId"] == "wf_api"
    assert facts["board"] == "core"
    assert facts["status"] == "materialized"
    assert facts["alreadyMaterialized"] is False
    assert [task["nodeId"] for task in facts["tasks"]] == ["backend-api", "integration"]
    assert facts["tasks"][0]["taskId"]
    assert facts["linksCreated"] == [
        {"parentTaskId": facts["tasks"][0]["taskId"], "childTaskId": facts["tasks"][1]["taskId"]}
    ]


def test_get_workflow_node_uses_direct_related_record_queries_not_prefiltered_defaults(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        _create_workflow(conn)
        for index in range(100):
            add_event(conn, workflow_id="wf_api", event_type=f"workflow_{index}", actor_type="workflow", now=float(index))
        add_event(conn, workflow_id="wf_api", event_type="node_late", node_id="backend-api", actor_type="workflow", now=101.0)

        payload = get_workflow_node(conn, "wf_api", "backend-api")

    assert [event["eventType"] for event in payload["facts"]["events"]] == ["node_late"]


def test_read_model_rejects_invalid_limits(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        _create_workflow(conn)
        with pytest.raises(ValueError, match="limit must be positive"):
            list_workflow_summaries(conn, limit=0)
        with pytest.raises(ValueError, match="limit must be positive"):
            get_workflow_events(conn, "wf_api", limit=-1)
        with pytest.raises(ValueError, match="limit must be positive"):
            get_workflow_artifacts(conn, "wf_api", limit=0)


def test_read_model_rejects_missing_workflow_or_node(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        _create_workflow(conn)
        with pytest.raises(ValueError, match="workflow not found: wf_missing"):
            get_workflow_dag(conn, "wf_missing")
        with pytest.raises(ValueError, match="workflow node not found: missing"):
            get_workflow_node(conn, "wf_api", "missing")
