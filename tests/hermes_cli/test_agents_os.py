from __future__ import annotations

import json
from pathlib import Path

from hermes_cli import agents_os


def test_agents_os_local_control_plane_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()

    assert agents_os.main(["--vault-root", str(vault), "init"]) == 0
    assert (tmp_path / "home" / "agents_os" / "state.sqlite").exists()
    assert (vault / "00-command-center" / "RUNTIME-CONTROL-PLANE.md").exists()
    capsys.readouterr()

    assert agents_os.main([
        "--vault-root",
        str(vault),
        "run",
        "external-action-draft",
        "send nothing; draft only",
        "--title",
        "Draft gate",
        "--task-id",
        "task-gate",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["task_id"] == "task-gate"
    assert result["approval_id"].startswith("approval-")
    assert Path(result["artifact_path"]).exists()

    assert agents_os.main(["--vault-root", str(vault), "approval", "list", "--json"]) == 0
    approvals = json.loads(capsys.readouterr().out)
    assert approvals[0]["status"] == "pending"
    assert approvals[0]["risk"] == "external-action"

    assert agents_os.main(["--vault-root", str(vault), "task", "set", "task-gate", "completed"]) == 2
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "task", "list", "--status", "needs_approval", "--json"]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert tasks[0]["id"] == "task-gate"

    assert agents_os.main(["--vault-root", str(vault), "status", "--json"]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["status"] == "ok"
    assert status_payload["counts"]["approvals"]["pending"] == 1

    assert agents_os.main(["--vault-root", str(vault), "doctor", "--json"]) == 0
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["ok"] is True
    assert doctor_payload["checks"]["network_side_effects"] is False
    assert doctor_payload["checks"]["runtime_config_changed"] is False


def test_agents_os_sprint1_schema_routing_next_and_dashboard(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()

    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()

    assert agents_os.main(["--vault-root", str(vault), "doctor", "--json"]) == 0
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["checks"]["schema_version"] == "3"
    assert doctor_payload["checks"]["required_tables_present"] is True
    assert {"events", "runs", "agents", "workflows", "routing_rules", "reviews"}.issubset(
        set(doctor_payload["checks"]["tables"])
    )

    assert agents_os.main([
        "--vault-root", str(vault), "run", "code-task", "implement deterministic router", "--title", "Code lane", "--task-id", "task-code", "--priority", "1"
    ]) == 0
    capsys.readouterr()
    assert agents_os.main([
        "--vault-root", str(vault), "run", "external-action-draft", "publish public post", "--title", "Public post", "--task-id", "task-public", "--priority", "0"
    ]) == 0
    capsys.readouterr()

    assert agents_os.main(["--vault-root", str(vault), "route", "task-code", "--json"]) == 0
    code_route = json.loads(capsys.readouterr().out)
    assert code_route["route"] == "skill:test-driven-development"
    assert code_route["execution_allowed"] is True
    assert code_route["new_status"] == "ready"

    assert agents_os.main(["--vault-root", str(vault), "route", "task-public", "--json"]) == 0
    public_route = json.loads(capsys.readouterr().out)
    assert public_route["route"] == "approval_gate"
    assert public_route["execution_allowed"] is False
    assert public_route["new_status"] == "needs_approval"

    assert agents_os.main(["--vault-root", str(vault), "next", "--json"]) == 0
    next_payload = json.loads(capsys.readouterr().out)
    assert next_payload["task"]["id"] == "task-code"
    assert next_payload["task"]["status"] == "ready"

    assert agents_os.main(["--vault-root", str(vault), "dashboard", "--markdown"]) == 0
    dashboard_stdout = capsys.readouterr().out
    dashboard_path = vault / "00-command-center" / "RUNTIME-DASHBOARD.md"
    assert dashboard_path.exists()
    dashboard_text = dashboard_path.read_text(encoding="utf-8")
    assert "# Agents OS Runtime Dashboard" in dashboard_stdout
    assert "task-code" in dashboard_text
    assert "task-public" in dashboard_text
    assert "Approval gated" in dashboard_text
