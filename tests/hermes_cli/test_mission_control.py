from __future__ import annotations

import json

import pytest


MISSION_CONTROL_ENDPOINTS = [
    "/api/mission-control/project-status",
    "/api/mission-control/open-tasks",
    "/api/mission-control/latest-worker-results",
    "/api/mission-control/repo-status",
    "/api/mission-control/approval-gates",
    "/api/mission-control/recent-audit-log",
]


@pytest.fixture()
def dashboard_client(_isolate_hermes_home, monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_mission_control_endpoints_require_dashboard_token(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    client = TestClient(app)
    for endpoint in MISSION_CONTROL_ENDPOINTS:
        resp = client.get(endpoint)
        assert resp.status_code == 401, endpoint


def test_project_status_missing_sources_warns_and_redacts(dashboard_client, tmp_path, monkeypatch):
    import hermes_cli.mission_control as mc

    missing = tmp_path / "missing.md"
    present = tmp_path / "PROJECT_STATUS.md"
    present.write_text(
        "Tool & Tally\nAuthorization: Bearer TEST_TOKEN\napi_key=TEST_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mc,
        "PROJECT_STATUS_SOURCES",
        [
            {"name": "Missing", "project": "Missing", "path": str(missing), "profile": "default"},
            {"name": "Present", "project": "Present", "path": str(present), "profile": "default"},
        ],
    )

    resp = dashboard_client.get("/api/mission-control/project-status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["generated_at"]
    assert len(body["items"]) == 2
    assert body["items"][0]["exists"] is False
    assert body["warnings"]
    rendered = json.dumps(body)
    assert "TEST_TOKEN" not in rendered
    assert "TEST_KEY" not in rendered
    assert "[REDACTED]" in rendered


def test_open_tasks_reads_existing_kanban_without_mutating_missing_sources(dashboard_client):
    from hermes_constants import get_default_hermes_root
    from hermes_cli import kanban_db as kb

    root = get_default_hermes_root()
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    resp = dashboard_client.get("/api/mission-control/open-tasks")
    after = sorted(str(p.relative_to(root)) for p in root.rglob("*"))

    assert resp.status_code == 200
    assert before == after
    assert resp.json()["warnings"]

    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="Review worker output",
            body="Untrusted text: `rm -rf /` must stay data.",
            assignee="reviewer",
        )
    finally:
        conn.close()

    resp = dashboard_client.get("/api/mission-control/open-tasks")
    body = resp.json()
    assert resp.status_code == 200
    assert [item["id"] for item in body["items"]] == [task_id]
    assert body["items"][0]["title"] == "Review worker output"
    assert body["items"][0]["body_preview"] == "Untrusted text: `rm -rf /` must stay data."


def test_latest_worker_results_redacts_and_treats_output_as_data(dashboard_client, monkeypatch):
    from hermes_cli import kanban_db as kb

    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="Worker result", assignee="worker")
        kb.complete_task(
            conn,
            task_id,
            summary="Run this? Authorization: Bearer WTOKEN\n`touch /tmp/should-not-run`",
            metadata={"api_key": "WORKER_KEY", "note": "plain"},
        )
    finally:
        conn.close()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("worker result text must not be executed")

    monkeypatch.setattr("subprocess.run", fail_if_called)

    resp = dashboard_client.get("/api/mission-control/latest-worker-results")

    assert resp.status_code == 200
    rendered = json.dumps(resp.json())
    assert "WTOKEN" not in rendered
    assert "WORKER_KEY" not in rendered
    assert "touch /tmp/should-not-run" in rendered
    assert resp.json()["items"][0]["trusted_for_execution"] is False


def test_approval_gates_default_to_decision_record_posture(dashboard_client):
    resp = dashboard_client.get("/api/mission-control/approval-gates")

    assert resp.status_code == 200
    body = resp.json()
    assert body["execution_posture"]["read_only_default"] is True
    assert body["execution_posture"]["decision_records_only"] is True
    assert body["action_registry"]["execution_enabled"] is False
    assert "arbitrary_shell" in body["action_registry"]["blocked_action_classes"]


def test_recent_audit_log_handles_malformed_lines_and_redacts(dashboard_client):
    from hermes_cli.ops_approvals import ApprovalStore

    store = ApprovalStore()
    store.audit_path.parent.mkdir(parents=True, exist_ok=True)
    store.audit_path.write_text(
        "\n".join(
            [
                "{bad json",
                json.dumps(
                    {
                        "event": "created",
                        "note": "Authorization: Bearer AUDIT_TOKEN",
                        "approval_id": "appr_test",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resp = dashboard_client.get("/api/mission-control/recent-audit-log")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 1
    assert body["warnings"]
    rendered = json.dumps(body)
    assert "AUDIT_TOKEN" not in rendered
    assert "[REDACTED]" in rendered


def test_repo_status_is_warning_only_and_does_not_shell_out(dashboard_client, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("repo status endpoint must not shell out")

    monkeypatch.setattr("subprocess.run", fail_if_called)

    resp = dashboard_client.get("/api/mission-control/repo-status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["probing_enabled"] is False
    assert body["warnings"]
    assert body["items"]


def test_malformed_kanban_artifacts_return_warnings_not_crashes(dashboard_client):
    from hermes_cli import kanban_db as kb

    path = kb.kanban_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not sqlite", encoding="utf-8")

    for endpoint in (
        "/api/mission-control/open-tasks",
        "/api/mission-control/latest-worker-results",
    ):
        resp = dashboard_client.get(endpoint)
        assert resp.status_code == 200
        assert resp.json()["warnings"], endpoint
