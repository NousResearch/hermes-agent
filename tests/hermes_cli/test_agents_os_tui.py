from __future__ import annotations

import argparse
import json

from hermes_cli import agents_os
from hermes_cli.agents_os_tui import MissionControlCore, MissionControlState, apply_key, render_screen


def _setup(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()
    return vault


def _json_out(capsys):
    return json.loads(capsys.readouterr().out)


def test_mission_control_core_navigation_and_rendering_is_terminal_free():
    state = MissionControlState(view="tasks", selected=0)
    rows = [{"id": "task-a", "title": "A", "status": "ready", "priority": 1, "workflow": "code-task", "notes": "note"}]
    text = render_screen(state, {"tasks": rows, "counts": {"tasks": {"ready": 1}}, "detail": rows[0]}, width=90, height=24)
    assert "Agents OS Mission Control" in text
    assert "Tasks" in text
    assert "task-a" in text
    assert "Detail" in text
    assert "Commands:" in text

    state = apply_key(state, "2", item_count=1)
    assert state.view == "next"
    state = apply_key(state, "j", item_count=3)
    assert state.selected == 1
    state = apply_key(state, "down", item_count=3)
    assert state.selected == 2
    state = apply_key(state, "k", item_count=3)
    assert state.selected == 1
    state = apply_key(state, "g", item_count=3)
    assert state.selected == 0
    state = apply_key(state, "G", item_count=3)
    assert state.selected == 2
    state = apply_key(state, "6", item_count=0)
    assert state.view == "doctor"
    assert state.selected == 0
    assert apply_key(state, "q", item_count=0).quit is True


def test_mission_control_loads_required_views_and_detail(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    assert agents_os.main(["--vault-root", str(vault), "agent", "add", "local-agent", "--capabilities", "code", "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "run", "code-task", "local work", "--task-id", "task-ready", "--title", "Ready task"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "route", "task-ready", "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "run", "external-action-draft", "approval", "--task-id", "task-gate", "--title", "Gate"]) == 0
    capsys.readouterr()

    core = MissionControlCore(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))
    assert core.items_for_view("tasks")[0]["id"] in {"task-gate", "task-ready"}
    assert core.items_for_view("next")[0]["id"] == "task-ready"
    assert core.items_for_view("approvals")[0]["task_id"] == "task-gate"
    assert core.items_for_view("runs")
    assert core.items_for_view("events")
    doctor = core.items_for_view("doctor")
    names = {row["id"] for row in doctor}
    assert "doctor" in names
    assert "mirror" in names

    state = MissionControlState(view="approvals", selected=0)
    detail = core.detail_for_state(state)
    assert detail["id"].startswith("approval-")
    rendered = render_screen(state, {"approvals": core.items_for_view("approvals"), "detail": detail}, width=100, height=30)
    assert "Approvals" in rendered
    assert "approve" in rendered
    assert "deny" in rendered


def test_mission_control_safe_actions_create_close_approve_and_deny(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    core = MissionControlCore(agents_os.resolve_paths(argparse.Namespace(vault_root=str(vault))))

    created = core.create_task(title="Created from TUI", workflow="code-task", priority=2, notes="local only")
    assert created["id"].startswith("task-")
    assert created["status"] == "pending"
    assert created["title"] == "Created from TUI"

    closed = core.close_task(created["id"], evidence="verified in local TUI test")
    assert closed["status"] == "completed"
    assert closed["evidence"] == "verified in local TUI test"

    assert agents_os.main(["--vault-root", str(vault), "run", "external-action-draft", "payload", "--task-id", "task-approve", "--title", "Needs approval"]) == 0
    approval_result = _json_out(capsys)
    approved = core.resolve_approval(approval_result["approval_id"], "approved")
    assert approved["status"] == "approved"
    assert approved["task_id"] == "task-approve"
    tasks = {row["id"]: row for row in core.items_for_view("tasks")}
    assert tasks["task-approve"]["approval_required"] == 0
    assert tasks["task-approve"]["status"] == "ready"

    assert agents_os.main(["--vault-root", str(vault), "run", "external-action-draft", "payload", "--task-id", "task-deny", "--title", "Deny me"]) == 0
    deny_result = _json_out(capsys)
    denied = core.resolve_approval(deny_result["approval_id"], "rejected", notes="unsafe")
    assert denied["status"] == "rejected"
    tasks = {row["id"]: row for row in core.items_for_view("tasks")}
    assert tasks["task-deny"]["status"] == "blocked"


def test_agents_os_tui_cli_supports_json_status_and_scripted_keys(tmp_path, monkeypatch, capsys):
    vault = _setup(tmp_path, monkeypatch, capsys)
    assert agents_os.main(["--vault-root", str(vault), "tui", "--json"]) == 0
    payload = _json_out(capsys)
    assert payload["status"] == "ok"
    assert payload["views"] == ["tasks", "next", "approvals", "runs", "events", "doctor"]
    assert payload["safety"]["network_side_effects"] is False

    assert agents_os.main(["--vault-root", str(vault), "tui", "--script", "2j6q"]) == 0
    scripted = capsys.readouterr().out
    assert "Agents OS Mission Control" in scripted
    assert "Doctor / Mirror" in scripted
