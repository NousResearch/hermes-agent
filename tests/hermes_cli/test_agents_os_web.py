from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

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


def _seed_session_db(home):
    db_path = home / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, user_id TEXT, model TEXT, started_at REAL NOT NULL, ended_at REAL, message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0, title TEXT)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT, timestamp REAL NOT NULL, tool_name TEXT)")
        conn.execute("INSERT INTO sessions(id,source,user_id,model,started_at,ended_at,message_count,tool_call_count,title) VALUES(?,?,?,?,?,?,?,?,?)", ("s1", "telegram", "user-1", "gpt-test", 1710000000.0, 1710000300.0, 2, 1, "Morning ops"))
        conn.execute("INSERT INTO messages(session_id,role,content,timestamp,tool_name) VALUES(?,?,?,?,?)", ("s1", "user", "tajni token SECRET_TOKEN=abc should not leak", 1710000001.0, None))
        conn.execute("INSERT INTO messages(session_id,role,content,timestamp,tool_name) VALUES(?,?,?,?,?)", ("s1", "assistant", "Kratak siguran preview odgovora", 1710000002.0, None))


def _seed_skills(home):
    skill_dir = home / "skills" / "devops" / "sample-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: sample-skill\ndescription: Test skill\n---\n\n# Sample\n", encoding="utf-8")


def _seed_cron(home):
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True)
    (cron_dir / "jobs.json").write_text(json.dumps({"jobs": [{"id": "job-1", "name": "Daily check", "schedule_display": "0 9 * * *", "enabled": True, "last_run_at": "2026-06-01T07:00:00Z", "deliver": "local", "no_agent": True}]}), encoding="utf-8")


def test_static_shell_uses_upstream_neutral_copy():
    index = Path(__file__).resolve().parents[2] / "hermes_cli" / "agents_os_web_static" / "index.html"
    text = index.read_text(encoding="utf-8")
    assert "Mikac" not in text
    assert "Doni" not in text
    assert "Local control-plane" in text


def test_default_agent_roster_is_profile_neutral(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))
    agents = app.handle_json("GET", "/api/agents")
    dumped = json.dumps(agents["data"])
    assert "Marija" not in dumped
    assert "OpenClaw" not in dumped
    assert "/home/goran" not in dumped
    assert any(a["id"] == "external-code-agent" for a in agents["data"])


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


def test_v21_visibility_routes_are_read_only_and_sanitized(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    home = tmp_path / "home"
    _seed_session_db(home)
    _seed_skills(home)
    _seed_cron(home)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))

    sessions = app.handle_json("GET", "/api/sessions")
    assert sessions["ok"] is True
    assert sessions["data"][0]["id"] == "s1"
    assert sessions["data"][0]["title"] == "Morning ops"
    assert "SECRET_TOKEN" not in json.dumps(sessions["data"])
    assert sessions["data"][0]["last_message_preview"] == "Kratak siguran preview odgovora"

    skills = app.handle_json("GET", "/api/skills")
    assert skills["ok"] is True
    assert any(s["name"] == "sample-skill" and s["category"] == "devops" for s in skills["data"])
    assert "SKILL.md" not in skills["data"][0].get("content", "")

    cron = app.handle_json("GET", "/api/cron")
    assert cron["ok"] is True
    assert cron["data"][0]["id"] == "job-1"
    assert cron["data"][0]["no_agent"] is True
    assert "prompt" not in cron["data"][0]


def test_v21_run_detail_route_links_events_and_artifacts(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))
    created = app.handle_json("POST", "/api/workflows/research-brief/run", {"input": "local research", "title": "Run detail smoke"})
    assert created["ok"] is True
    run_id = created["data"]["run_id"]

    detail = app.handle_json("GET", f"/api/runs/{run_id}")
    assert detail["ok"] is True
    assert detail["data"]["run"]["id"] == run_id
    assert detail["data"]["task"]["id"] == created["data"]["task_id"]
    assert detail["data"]["artifacts"][0]["id"] == created["data"]["artifact_id"]
    assert any(e["event_type"] == "task_created" for e in detail["data"]["events"])


def _seed_v22_parent_child(paths):
    now = agents_os.utc_now()
    parent_id = "task-parent"
    child_id = "task-child"
    run_id = "run-child"
    artifact_id = "artifact-parent"
    artifact_path = paths.artifacts / "handoff" / "parent-evidence.md"
    agents_os.write_markdown(artifact_path, "Parent evidence", "Parent result with token SECRET_TOKEN=abc123", {"id": artifact_id})
    with agents_os.connect(paths) as conn:
        conn.execute("INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,approval_required) VALUES(?,?,?,?,?,?,?,?,?)", (parent_id, "Parent research", "completed", "research-brief", 2, now, now, "{\"evidence\":\"Parent closed with api_key=raw123\"}", 0))
        conn.execute("INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,approval_required) VALUES(?,?,?,?,?,?,?,?,?)", (child_id, "Child implementation", "blocked", "code-task", 2, now, now, "{\"parent_id\":\"task-parent\"}", 0))
        conn.execute("INSERT INTO runs(id,task_id,workflow,status,input,created_at,completed_at) VALUES(?,?,?,?,?,?,?)", (run_id, child_id, "code-task", "blocked", "child input", now, None))
        conn.execute("INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at,run_id) VALUES(?,?,?,?,?,?,?,?)", (artifact_id, "note", "Parent evidence", str(artifact_path), parent_id, "research-brief", now, None))
        conn.execute("INSERT INTO approvals(id,title,status,risk,task_id,payload,created_at) VALUES(?,?,?,?,?,?,?)", ("approval-child", "Child approval", "pending", "external-action", child_id, "payload password=raw", now))
        conn.execute("INSERT INTO events(id,task_id,run_id,event_type,payload,created_at) VALUES(?,?,?,?,?,?)", ("event-parent-closed", parent_id, None, "task_closed", json.dumps({"evidence": "Parent closed with api_key=raw123", "child_task_id": child_id}), now))
        conn.execute("INSERT INTO events(id,task_id,run_id,event_type,payload,created_at) VALUES(?,?,?,?,?,?)", ("event-child-linked", child_id, run_id, "dependency_linked", json.dumps({"parent_id": parent_id, "handoff_artifact_id": artifact_id, "token": "SECRET_TOKEN=abc"}), now))
        conn.commit()
    return parent_id, child_id, run_id, artifact_id


def test_task_detail_route_returns_connected_operational_context(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    paths = agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault)))
    parent_id, child_id, run_id, artifact_id = _seed_v22_parent_child(paths)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(paths)
    detail = app.handle_json("GET", f"/api/tasks/{child_id}")

    assert detail["ok"] is True
    data = detail["data"]
    assert data["task"]["id"] == child_id
    assert data["parent"]["id"] == parent_id
    assert isinstance(data["children"], list)
    assert data["approvals"][0]["id"] == "approval-child"
    assert data["runs"][0]["id"] == run_id
    assert any(e["event_type"] == "dependency_linked" for e in data["events"])
    assert data["artifacts"][0]["id"] == artifact_id
    assert data["dependency_status"]["state"] in {"blocked", "waiting", "ready"}
    assert "safe_next_actions" in data


def test_task_detail_redacts_credential_like_previews(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    paths = agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault)))
    _, child_id, _, _ = _seed_v22_parent_child(paths)
    from hermes_cli.agents_os_web import MissionControlWebApp

    detail = MissionControlWebApp(paths).handle_json("GET", f"/api/tasks/{child_id}")
    dumped = json.dumps(detail["data"])
    assert "SECRET_TOKEN=abc" not in dumped
    assert "api_key=raw123" not in dumped
    assert "password=raw" not in dumped
    assert "[redacted]" in dumped


def test_approval_gated_task_detail_blocks_execute_and_close(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))
    created = app.handle_json("POST", "/api/workflows/external-action-draft/run", {"input": "send outbound", "title": "Approval gated"})
    task_id = created["data"]["task_id"]
    detail = app.handle_json("GET", f"/api/tasks/{task_id}")
    allowed = {a["id"]: a["allowed"] for a in detail["data"]["safe_next_actions"]}
    assert allowed["execute"] is False
    assert allowed["close"] is False
    assert app.handle_json("POST", f"/api/tasks/{task_id}/execute")["error"]["code"] == "approval_required"
    assert app.handle_json("POST", f"/api/tasks/{task_id}/close", {"evidence": "done"})["error"]["code"] == "approval_required"


def test_child_dependency_handoff_preview_uses_parent_evidence_or_artifact(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    paths = agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault)))
    parent_id, child_id, _, artifact_id = _seed_v22_parent_child(paths)
    from hermes_cli.agents_os_web import MissionControlWebApp

    detail = MissionControlWebApp(paths).handle_json("GET", f"/api/tasks/{child_id}")
    handoff = detail["data"]["handoff_preview"]
    assert handoff["parent_id"] == parent_id
    assert handoff["artifact_id"] == artifact_id
    assert "Parent" in handoff["preview"]
    assert len(handoff["preview"]) < 500
    assert "SECRET_TOKEN" not in handoff["preview"]


def test_web_json_exposes_task_detail_route(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    assert agents_os.main(["--vault-root", str(vault), "web", "--json"]) == 0
    payload = _json(capsys)
    assert "/api/tasks/{id}" in payload["routes"]


def _seed_v23_approval(paths, *, approval_id="approval-risk", task_id="task-risk", payload="send public request with API_KEY=raw-secret", risk="external-action", status="pending", created_at=None, workflow="external-action-draft", title="Risk approval"):
    now = created_at or agents_os.utc_now()
    with agents_os.connect(paths) as conn:
        conn.execute("INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,approval_required) VALUES(?,?,?,?,?,?,?,?,?)", (task_id, title, "needs_approval" if status == "pending" else "blocked", workflow, 1, now, now, payload, 1 if status == "pending" else 0))
        conn.execute("INSERT INTO approvals(id,title,status,risk,task_id,payload,created_at,resolved_at) VALUES(?,?,?,?,?,?,?,?)", (approval_id, title, status, risk, task_id, payload, now, None if status == "pending" else agents_os.utc_now()))
        agents_os.log_event(conn, "approval_requested", task_id=task_id, payload={"approval_id": approval_id, "source": "test", "payload": payload})
        conn.commit()
    return approval_id, task_id


def test_approval_list_enriches_risk_taxonomy_and_redacts_payload(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    paths = agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault)))
    approval_id, task_id = _seed_v23_approval(paths, payload="send to public api with token=raw-token and password=raw-pass")
    from hermes_cli.agents_os_web import MissionControlWebApp

    approvals = MissionControlWebApp(paths).handle_json("GET", "/api/approvals")
    row = next(a for a in approvals["data"] if a["id"] == approval_id)
    dumped = json.dumps(row)
    assert row["risk_category"] == "external_action"
    assert row["risk_level"] in {"high", "critical"}
    assert "public_side_effect" in row["risk_flags"]
    assert "credential_sensitive" in row["risk_flags"]
    assert row["required_decision"]
    assert row["minimum_input_needed"]
    assert row["source"]
    assert row["actor"]
    assert row["created_from_workflow"] == "external-action-draft"
    assert row["task_id"] == task_id
    assert "payload_preview" in row
    assert "raw-token" not in dumped
    assert "raw-pass" not in dumped
    assert "[redacted]" in dumped


def test_approval_detail_route_returns_task_events_and_safe_actions(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    paths = agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault)))
    approval_id, task_id = _seed_v23_approval(paths, payload="restart gateway after approval")
    from hermes_cli.agents_os_web import MissionControlWebApp

    detail = MissionControlWebApp(paths).handle_json("GET", f"/api/approvals/{approval_id}")
    assert detail["ok"] is True
    data = detail["data"]
    assert data["approval"]["id"] == approval_id
    assert data["task"]["id"] == task_id
    assert data["events"]
    assert "risk_taxonomy" in data
    actions = {a["id"]: a for a in data["safe_next_actions"]}
    assert actions["approve"]["allowed"] is True
    assert actions["deny"]["allowed"] is True
    assert actions["refresh"]["allowed"] is True
    assert "execute" not in actions


def test_approval_risk_flags_detect_sensitive_categories(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    paths = agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault)))
    cases = [
        ("approval-public", "send public network webhook to external url", "public_side_effect"),
        ("approval-cred", "use api_key=abc123 token=raw", "credential_sensitive"),
        ("approval-destroy", "delete production database destructive action", "destructive"),
        ("approval-fin", "place live trade payment for crypto", "financial"),
        ("approval-gw", "restart gateway runtime config", "gateway_or_runtime_change"),
        ("approval-memory", "modify memory profile for Marija", "profile_or_memory_mutation"),
        ("approval-sec", "run security scan exploit payload", "security_sensitive"),
    ]
    for approval_id, payload, _flag in cases:
        _seed_v23_approval(paths, approval_id=approval_id, task_id=f"task-{approval_id}", payload=payload, title=approval_id)
    from hermes_cli.agents_os_web import MissionControlWebApp

    rows = {a["id"]: a for a in MissionControlWebApp(paths).handle_json("GET", "/api/approvals")["data"]}
    for approval_id, _payload, flag in cases:
        assert flag in rows[approval_id]["risk_flags"]
        assert rows[approval_id]["risk_level"] in {"high", "critical"}


def test_stale_pending_approval_is_flagged(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    paths = agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault)))
    _seed_v23_approval(paths, approval_id="approval-stale", task_id="task-stale", created_at="2000-01-01T00:00:00Z", payload="underspecified external action")
    _seed_v23_approval(paths, approval_id="approval-resolved", task_id="task-resolved", created_at="2000-01-01T00:00:00Z", payload="old but resolved", status="approved")
    from hermes_cli.agents_os_web import MissionControlWebApp

    rows = {a["id"]: a for a in MissionControlWebApp(paths).handle_json("GET", "/api/approvals")["data"]}
    assert rows["approval-stale"]["stale"] is True
    assert rows["approval-stale"]["stale_reason"]
    assert rows["approval-resolved"]["stale"] is False


def test_safety_payload_aggregates_approval_risk_signals(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    paths = agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault)))
    _seed_v23_approval(paths, approval_id="approval-high", task_id="task-high", payload="send external network token=raw to public api", created_at="2000-01-01T00:00:00Z")
    from hermes_cli.agents_os_web import MissionControlWebApp

    safety = MissionControlWebApp(paths).handle_json("GET", "/api/safety")
    agg = safety["data"]["approval_risk"]
    assert agg["high_risk_pending_approvals"] >= 1
    assert agg["stale_approvals"] >= 1
    assert agg["approval_blocked_tasks"] >= 1
    assert agg["credential_sensitive_pending"] >= 1
    assert agg["external_action_pending"] >= 1
    assert agg["status"] == "attention"


def test_approval_gated_execute_close_still_blocked_after_risk_taxonomy(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))
    created = app.handle_json("POST", "/api/workflows/external-action-draft/run", {"input": "send outbound", "title": "Approval gated V23"})
    task_id = created["data"]["task_id"]
    approvals = app.handle_json("GET", "/api/approvals")["data"]
    assert any(a["task_id"] == task_id and a["risk_flags"] for a in approvals)
    assert app.handle_json("POST", f"/api/tasks/{task_id}/execute")["error"]["code"] == "approval_required"
    assert app.handle_json("POST", f"/api/tasks/{task_id}/close", {"evidence": "done"})["error"]["code"] == "approval_required"


def test_web_json_exposes_approval_detail_route(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    assert agents_os.main(["--vault-root", str(vault), "web", "--json"]) == 0
    payload = _json(capsys)
    assert "/api/approvals/{id}" in payload["routes"]


def test_cli_web_json_exposes_v21_routes(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    assert agents_os.main(["--vault-root", str(vault), "web", "--json"]) == 0
    payload = _json(capsys)
    assert "/api/sessions" in payload["routes"]
    assert "/api/skills" in payload["routes"]
    assert "/api/cron" in payload["routes"]
    assert "/api/runs/{id}" in payload["routes"]


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


def test_web_json_reports_launcher_health_and_windows_command(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    assert agents_os.main(["--vault-root", str(vault), "web", "--port", "59999", "--json"]) == 0
    payload = _json(capsys)

    assert payload["launcher"]["mode"] == "status"
    assert payload["launcher"]["local_only"] is True
    assert payload["launcher"]["health_url"] == "http://127.0.0.1:59999/api/status"
    assert payload["launcher"]["ui_url"] == "http://127.0.0.1:59999/"
    assert payload["launcher"]["existing_server"]["running"] is False
    assert "HERMES_HOME=" in payload["launcher"]["start_command"]
    assert "agents-os" in payload["launcher"]["start_command"]
    assert " web " in payload["launcher"]["start_command"]
    assert payload["launcher"]["windows_launcher"]["path"].endswith("Launch-Agents-OS-Mission-Control.bat")
    assert "127.0.0.1:59999" in payload["launcher"]["windows_launcher"]["command"]


def test_web_json_detects_existing_local_server_for_reuse(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp, create_server

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))
    server = create_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = str(server.server_address[1])
        assert agents_os.main(["--vault-root", str(vault), "web", "--port", port, "--json"]) == 0
        payload = _json(capsys)
        existing = payload["launcher"]["existing_server"]
        assert existing["running"] is True
        assert existing["reusable"] is True
        assert existing["status_url"].endswith(f":{port}/api/status")
        assert existing["status"]["ok"] is True
        assert existing["status"]["data"]["bind_host"] == "127.0.0.1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_status_payload_exposes_operator_ui_health_contract(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    from hermes_cli.agents_os_web import MissionControlWebApp

    app = MissionControlWebApp(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))
    status = app.handle_json("GET", "/api/status")

    ui = status["data"]["operator_ui"]
    assert ui["product"] == "Agents OS Mission Control"
    assert ui["local_only"] is True
    assert ui["launcher_hardened"] is True
    assert ui["safe_stop"] == "Ctrl+C on the Mission Control web process only"
    assert ui["gateway_restart"] is False
    assert ui["required_panels"] == ["home", "tasks", "approvals", "runs", "sessions", "skills", "cron", "agents", "artifacts", "workflows", "safety"]
