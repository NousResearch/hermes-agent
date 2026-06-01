from __future__ import annotations

import argparse
import json
import sqlite3
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
