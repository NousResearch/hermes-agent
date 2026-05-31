from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

from hermes_cli import agents_os


def _setup(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "agent", "add", "doni-local", "--capabilities", "code,research,qa,ops", "--json"]) == 0
    capsys.readouterr()
    return vault


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def test_web_payload_routes_are_local_only_and_wrapped(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))

    status = app.handle_json("GET", "/api/status")
    assert status["ok"] is True
    assert status["data"]["bind_host"] == "127.0.0.1"
    assert status["data"]["schema_version"] == "3"
    assert status["data"]["safety"]["network_side_effects"] is False

    dashboard = app.handle_json("GET", "/api/dashboard")
    assert dashboard["ok"] is True
    assert "queue_summary" in dashboard["data"]

    missing = app.handle_json("GET", "/api/not-real")
    assert missing["ok"] is False
    assert missing["error"]["code"] == "not_found"


def test_web_action_routes_create_route_execute_close_and_approval_flow(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))

    created = app.handle_json("POST", "/api/tasks", {"title": "UI task", "workflow": "code-task", "notes": "local only"})
    assert created["ok"] is True
    task_id = created["data"]["id"]
    assert created["data"]["status"] == "pending"

    routed = app.handle_json("POST", f"/api/tasks/{task_id}/route")
    assert routed["ok"] is True
    assert routed["data"]["execution_allowed"] is True

    executed = app.handle_json("POST", f"/api/tasks/{task_id}/execute")
    assert executed["ok"] is True
    assert executed["data"]["status"] == "succeeded"

    closed = app.handle_json("POST", f"/api/tasks/{task_id}/close", {"evidence": "verified through web action test"})
    assert closed["ok"] is True
    assert closed["data"]["status"] == "completed"

    approval_task = app.handle_json("POST", "/api/workflows/external-action-draft/run", {"input": "send outbound payload", "title": "Needs approval"})
    assert approval_task["ok"] is True
    approval_id = approval_task["data"]["approval_id"]

    approvals = app.handle_json("GET", "/api/approvals")
    assert any(row["id"] == approval_id for row in approvals["data"])

    denied = app.handle_json("POST", f"/api/approvals/{approval_id}/deny", {"notes": "not safe"})
    assert denied["ok"] is True
    assert denied["data"]["status"] == "rejected"


def test_artifact_preview_is_restricted_to_agents_os_and_vault_roots(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp

    assert agents_os.main(["--vault-root", str(vault), "artifact", "create", "Preview me", "--kind", "note", "--body", "hello artifact", "--task-id", "task-x"]) == 0
    capsys.readouterr()
    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))

    artifacts = app.handle_json("GET", "/api/artifacts")
    assert artifacts["ok"] is True
    artifact_id = artifacts["data"][0]["id"]
    preview = app.handle_json("GET", f"/api/artifacts/{artifact_id}")
    assert preview["ok"] is True
    assert preview["data"]["preview_type"] in {"markdown", "text"}
    assert "hello artifact" in preview["data"]["content"]

    blocked = app.preview_path("/home/goran/.hermes-doni-clean/.env")
    assert blocked["ok"] is False
    assert blocked["error"]["code"] == "path_not_allowed"


def test_cli_web_json_and_http_server_smoke(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    assert agents_os.main(["--vault-root", str(vault), "web", "--json"]) == 0
    payload = _json(capsys)
    assert payload["status"] == "ok"
    assert payload["bind_host"] == "127.0.0.1"
    assert "/api/status" in payload["routes"]

    from hermes_cli.agents_os_web import MissionControlWebApp, create_server

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))
    server = create_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert data["ok"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            html = resp.read().decode("utf-8")
        assert "Agents OS Mission Control" in html
    finally:
        server.shutdown()
        thread.join(timeout=5)
