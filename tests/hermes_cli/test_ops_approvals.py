from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest


def _valid_request(**overrides):
    data = {
        "title": "Restart dashboard proxy",
        "project": "Hermes Ops",
        "profile": "default",
        "risk_label": "Live-service",
        "proposed_action": "Restart dashboard proxy only, not the messaging gateway",
        "target": "hermes-dashboard.service",
        "preview": "systemctl --user restart hermes-dashboard.service",
        "reason": "Dashboard proxy is down while gateway is healthy",
        "rollback_or_verification": "Verify /api/status returns 200 after restart",
        "created_by": "test",
    }
    data.update(overrides)
    return data


def test_approval_store_creates_request_and_appends_audit(_isolate_hermes_home):
    from hermes_cli.ops_approvals import ApprovalStore

    store = ApprovalStore()
    approval = store.create(_valid_request())

    assert approval["id"].startswith("appr_")
    assert approval["status"] == "pending"
    assert approval["execution_allowed"] is False
    assert approval["blocked_until_approved"] is True
    assert approval["risk_label"] == "Live-service"
    assert approval["generated_command"] is None

    current = store.list()
    assert [item["id"] for item in current] == [approval["id"]]

    audit_path = store.audit_path
    lines = audit_path.read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "created"
    assert event["approval_id"] == approval["id"]


def test_approval_decision_records_approve_without_execution(_isolate_hermes_home):
    from hermes_cli.ops_approvals import ApprovalStore

    store = ApprovalStore()
    approval = store.create(_valid_request())
    decided = store.decide(approval["id"], "approved", decided_by="Travis", decision_note="Approved for dashboard only")

    assert decided["status"] == "approved"
    assert decided["decided_by"] == "Travis"
    assert decided["execution_allowed"] is False
    assert "approved:" in decided["generated_command"].lower()
    assert "Restart dashboard proxy" in decided["generated_command"]

    events = [json.loads(line) for line in store.audit_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["created", "approved"]


def test_approval_store_rejects_invalid_transition(_isolate_hermes_home):
    from hermes_cli.ops_approvals import ApprovalStore, ApprovalError

    store = ApprovalStore()
    approval = store.create(_valid_request())
    store.decide(approval["id"], "rejected", decided_by="Travis")

    with pytest.raises(ApprovalError):
        store.decide(approval["id"], "approved", decided_by="Travis")


def test_approval_store_marks_expired_items(_isolate_hermes_home):
    from hermes_cli.ops_approvals import ApprovalStore

    store = ApprovalStore()
    approval = store.create(_valid_request(expires_at="2000-01-01T00:00:00+00:00"))

    listed = store.list(now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert listed[0]["id"] == approval["id"]
    assert listed[0]["status"] == "expired"


def test_draft_proposal_packet_validates_without_writing_state(_isolate_hermes_home):
    from hermes_cli.ops_approvals import ApprovalStore, draft_proposal_packet

    packet = draft_proposal_packet(_valid_request(
        created_by="Jenny",
        source_surface="discord",
        source_ref="thread:ops message:456",
        conversation_excerpt="A gated restart was requested.",
        related_paths=["/home/jenny/.hermes/hermes-agent/hermes_cli/ops_approvals.py"],
    ))
    store = ApprovalStore()

    assert packet["created_by"] == "Jenny"
    assert packet["risk_label"] == "Live-service"
    assert packet["source_surface"] == "discord"
    assert packet["related_paths"] == ["/home/jenny/.hermes/hermes-agent/hermes_cli/ops_approvals.py"]
    assert store.list() == []
    assert not store.inbox_path.exists()


def test_proposal_ingestion_records_chat_context_and_stays_pending(_isolate_hermes_home):
    from hermes_cli.ops_approvals import ApprovalStore

    store = ApprovalStore()
    proposal = store.propose_from_context({
        "project": "Hermes Ops",
        "profile": "default",
        "risk_label": "Live-service",
        "title": "Restart dashboard only",
        "proposed_action": "Reload the dashboard process so a code change is active",
        "target": "Hermes dashboard process",
        "preview": "Dashboard reload only; messaging gateway remains untouched",
        "reason": "New read-only API route needs process reload",
        "rollback_or_verification": "GET /api/status and confirm gateway PID unchanged",
        "created_by": "Jenny",
        "source_surface": "discord",
        "source_ref": "thread:1508516222545956966 message:abc123",
        "conversation_excerpt": "Travis: proceed with the safe dashboard-only reload",
        "related_paths": ["/home/jenny/.hermes/hermes-agent/hermes_cli/web_server.py"],
    })

    assert proposal["status"] == "pending"
    assert proposal["proposal_kind"] == "gated_action"
    assert proposal["source_surface"] == "discord"
    assert proposal["source_ref"] == "thread:1508516222545956966 message:abc123"
    assert proposal["conversation_excerpt"].startswith("Travis: proceed")
    assert proposal["related_paths"] == ["/home/jenny/.hermes/hermes-agent/hermes_cli/web_server.py"]
    assert proposal["execution_allowed"] is False


def test_ops_approvals_module_cli_can_draft_json_proposal_without_creating_record(_isolate_hermes_home, tmp_path):
    import json
    import subprocess
    import sys

    from hermes_cli.ops_approvals import ApprovalStore

    payload_path = tmp_path / "draft-input.json"
    output_path = tmp_path / "draft-output.json"
    payload_path.write_text(json.dumps(_valid_request(
        title="Create gated proposal packet",
        risk_label="Draft-only",
        source_surface="discord",
        source_ref="thread:test message:789",
        related_paths=["/tmp/example.txt"],
    )))

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.ops_approvals", "draft", "--json-file", str(payload_path), "--output", str(output_path)],
        check=True,
        cwd=str(payload_path.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    packet = json.loads(output_path.read_text())

    assert str(output_path) in result.stdout
    assert packet["title"] == "Create gated proposal packet"
    assert packet["created_by"] == "test"
    assert packet["source_ref"] == "thread:test message:789"
    assert packet["related_paths"] == ["/tmp/example.txt"]
    assert ApprovalStore().list() == []


def test_ops_approvals_module_cli_can_ingest_json_proposal(_isolate_hermes_home, tmp_path):
    import json
    import subprocess
    import sys

    payload_path = tmp_path / "proposal.json"
    payload_path.write_text(json.dumps({
        "project": "Tool & Tally",
        "profile": "no-call-estimateready",
        "risk_label": "Money/customer",
        "title": "Send customer report",
        "proposed_action": "Send prepared internal report to customer after Travis review",
        "target": "hello@toolandtally.com outbound email",
        "preview": "Email body and report path are ready for review",
        "reason": "Customer-facing delivery requires explicit approval",
        "rollback_or_verification": "Confirm sent email in tooltally mailbox; do not send before approval",
        "created_by": "Jenny",
        "source_surface": "discord",
        "source_ref": "thread:test",
        "conversation_excerpt": "Jenny identified a gated customer-facing action.",
    }))

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.ops_approvals", "propose", "--json-file", str(payload_path), "--json"],
        check=True,
        cwd=str(payload_path.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    created = json.loads(result.stdout)

    assert created["status"] == "pending"
    assert created["project"] == "Tool & Tally"
    assert created["source_surface"] == "discord"
    assert created["execution_allowed"] is False


def test_approval_api_can_ingest_context_proposal(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    payload = _valid_request(
        title="Rotate API key",
        risk_label="Credential/auth",
        source_surface="discord",
        source_ref="thread:ops message:123",
        conversation_excerpt="Jenny identified a credential-changing action.",
        related_paths=["/home/jenny/.hermes/.env"],
    )

    resp = client.post("/api/ops/approvals/propose", json=payload)

    assert resp.status_code == 200
    created = resp.json()
    assert created["proposal_kind"] == "gated_action"
    assert created["source_surface"] == "discord"
    assert created["related_paths"] == ["/home/jenny/.hermes/.env"]
    assert created["execution_allowed"] is False


def test_approval_audit_listing_filters_by_approval_id(_isolate_hermes_home):
    from hermes_cli.ops_approvals import ApprovalStore

    store = ApprovalStore()
    first = store.create(_valid_request(title="First approval"))
    second = store.create(_valid_request(title="Second approval"))
    store.decide(first["id"], "approved", decided_by="Travis", decision_note="ok")

    first_events = store.audit_events(first["id"])
    all_events = store.audit_events()

    assert [event["event"] for event in first_events] == ["created", "approved"]
    assert all(event["approval_id"] == first["id"] for event in first_events)
    assert {event["approval_id"] for event in all_events} == {first["id"], second["id"]}
    assert all("timestamp" in event for event in all_events)


def test_approval_api_lists_audit_events(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    created = client.post("/api/ops/approvals", json=_valid_request()).json()
    client.post(
        f"/api/ops/approvals/{created['id']}/reject",
        json={"decided_by": "Travis", "decision_note": "not now"},
    )

    resp = client.get(f"/api/ops/approvals/audit?approval_id={created['id']}")

    assert resp.status_code == 200
    events = resp.json()
    assert [event["event"] for event in events] == ["created", "rejected"]
    assert all(event["approval_id"] == created["id"] for event in events)


def test_approval_summary_counts_pending_and_generates_review_text(_isolate_hermes_home):
    from hermes_cli.ops_approvals import ApprovalStore

    store = ApprovalStore()
    pending = store.propose_from_context(_valid_request(
        title="Reload dashboard",
        risk_label="Live-service",
        source_surface="discord",
        source_ref="thread:ops",
    ))
    rejected = store.propose_from_context(_valid_request(title="Old request", risk_label="Draft-only"))
    store.decide(rejected["id"], "rejected", decided_by="Travis")

    summary = store.summary()

    assert summary["pending_count"] == 1
    assert summary["total_count"] == 2
    assert summary["blocked_execution"] is True
    assert summary["pending"][0]["id"] == pending["id"]
    assert "Reload dashboard" in summary["review_text"]
    assert "No action has been executed" in summary["review_text"]


def test_approval_api_summary_endpoint(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    created = client.post("/api/ops/approvals/propose", json=_valid_request(title="Approve dashboard reload")).json()

    resp = client.get("/api/ops/approvals/summary")

    assert resp.status_code == 200
    summary = resp.json()
    assert summary["pending_count"] == 1
    assert summary["pending"][0]["id"] == created["id"]
    assert "Approve dashboard reload" in summary["review_text"]
    assert summary["blocked_execution"] is True


def test_approval_api_records_decision_but_has_no_execute_route(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    create_resp = client.post("/api/ops/approvals", json=_valid_request())
    assert create_resp.status_code == 200
    approval = create_resp.json()
    assert approval["status"] == "pending"
    assert approval["execution_allowed"] is False

    approve_resp = client.post(
        f"/api/ops/approvals/{approval['id']}/approve",
        json={"decided_by": "Travis", "decision_note": "Approved for dashboard proxy only"},
    )
    assert approve_resp.status_code == 200
    decided = approve_resp.json()
    assert decided["status"] == "approved"
    assert decided["execution_allowed"] is False
    assert decided["generated_command"]

    route_paths = {getattr(route, "path", "") for route in app.routes}
    assert f"/api/ops/approvals/{{approval_id}}/execute" not in route_paths
    assert "/api/ops/approvals/{approval_id}/actions/{action_name}/execute" not in route_paths
    execute_resp = client.post(f"/api/ops/approvals/{approval['id']}/execute")
    assert execute_resp.status_code in {404, 405}


def test_approval_action_dry_run_route_preflights_without_execution(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    created = client.post("/api/ops/approvals", json=_valid_request(
        title="Probe status only",
        risk_label="Read-only",
        proposed_action="read_only_status_probe",
        target="read_only_status_probe",
        preview="Read status metadata only; no restart or write.",
    )).json()
    approved = client.post(
        f"/api/ops/approvals/{created['id']}/approve",
        json={"decided_by": "Travis", "decision_note": "Approved dry-run check only"},
    ).json()

    resp = client.post(f"/api/ops/approvals/{approved['id']}/actions/read_only_status_probe/dry-run")

    assert resp.status_code == 200
    body = resp.json()
    assert body["approval_id"] == approved["id"]
    assert body["approval"]["approval_ok"] is True
    assert body["would_execute"] is False
    assert body["execution_allowed"] is False
    assert body["message"] == "Dry run only — no action executed"
    assert body["config"]["allowed"] is False  # default config remains disabled


def test_approval_action_dry_run_route_rejects_unknown_action(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    created = client.post("/api/ops/approvals", json=_valid_request()).json()

    resp = client.post(f"/api/ops/approvals/{created['id']}/actions/shell/dry-run")

    assert resp.status_code == 400
