from __future__ import annotations

from hermes_cli.workflow import render_workflow_status_report, workflow_status_report
from hermes_cli.workflow.store import add_gate, connect, create_workflow, save_dag


def _seed_report_workflow(conn):
    create_workflow(
        conn,
        workflow_id="wf_report",
        title="Reportable Workflow",
        board="core",
        scale="large",
        status="running",
        current_gate="gate_review",
        now=1.0,
    )
    save_dag(
        conn,
        workflow_id="wf_report",
        normalized_dag={
            "schema_version": 1,
            "workflow_id": "wf_report",
            "nodes": [
                {"id": "shape", "title": "Shape", "role": "planner", "profile": "planner", "status": "done"},
                {"id": "build", "title": "Build", "role": "engineer", "profile": "engineer", "status": "running", "parents": ["shape"]},
                {"id": "review", "title": "Review", "role": "reviewer", "profile": "reviewer", "status": "blocked", "parents": ["build"]},
                {"id": "publish", "title": "Publish", "role": "publisher", "profile": "publisher", "status": "waiting", "parents": ["review"]},
            ],
            "edges": [
                {"source": "shape", "target": "build", "kind": "depends_on"},
                {"source": "build", "target": "review", "kind": "depends_on"},
                {"source": "review", "target": "publish", "kind": "depends_on"},
            ],
        },
        now=2.0,
    )
    add_gate(
        conn,
        gate_id="gate_review",
        workflow_id="wf_report",
        node_id="review",
        gate_type="review",
        level=2,
        status="pending",
        required_actor="human",
        reason="Needs human review.",
    )


def test_workflow_status_report_returns_deterministic_facts_for_no_agent_use(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        _seed_report_workflow(conn)

        report = workflow_status_report(conn, "wf_report")

    assert report["facts"]["workflow"]["id"] == "wf_report"
    assert report["facts"]["nodeCounts"] == {
        "total": 4,
        "done": 1,
        "running": 1,
        "blocked": 1,
        "waiting": 1,
    }
    assert report["facts"]["criticalPath"] == ["shape", "build", "review", "publish"]
    assert report["facts"]["humanActionRequired"] == [
        {
            "gateId": "gate_review",
            "nodeId": "review",
            "gateType": "review",
            "requiredActor": "human",
            "reason": "Needs human review.",
        }
    ]
    assert report["facts"]["controlActions"][0]["id"] == "approve-gate:gate_review"
    assert report["insights"] is None


def test_render_workflow_status_report_outputs_stable_plain_text(tmp_path):
    with connect(tmp_path / "workflow.db") as conn:
        _seed_report_workflow(conn)
        report = workflow_status_report(conn, "wf_report")

    text = render_workflow_status_report(report)

    assert text.splitlines() == [
        "Workflow: wf_report Reportable Workflow",
        "State: running",
        "Nodes: 4 total, 1 done, 1 running, 1 blocked, 1 waiting",
        "Critical path: shape → build → review → publish",
        "Human action required: gate_review on review (human) — Needs human review.",
    ]
