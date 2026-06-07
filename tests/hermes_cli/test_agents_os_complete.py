from __future__ import annotations

import json
from pathlib import Path

from hermes_cli import agents_os


def _json_out(capsys):
    return json.loads(capsys.readouterr().out)


def test_agents_os_complete_runtime_sprints(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()

    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()

    assert agents_os.main(["--vault-root", str(vault), "doctor", "--json"]) == 0
    doctor = _json_out(capsys)
    assert doctor["checks"]["schema_version"] == "3"
    assert doctor["checks"]["orphan_records"] == 0
    assert doctor["checks"]["policy_home_isolated"] is True

    assert agents_os.main(["--vault-root", str(vault), "snapshot", "create", "baseline", "--json"]) == 0
    snap = _json_out(capsys)
    assert snap["snapshot_id"].startswith("snapshot-")
    assert Path(snap["export_path"]).exists()

    assert agents_os.main(["--vault-root", str(vault), "agent", "add", "local-agent", "--name", "Local Agent", "--capabilities", "code,research,qa", "--json"]) == 0
    agent = _json_out(capsys)
    assert agent["id"] == "local-agent"
    assert "code" in agent["capabilities"]

    assert agents_os.main(["--vault-root", str(vault), "agent", "list", "--json"]) == 0
    agents = _json_out(capsys)
    assert agents[0]["status"] == "available"

    wf_path = tmp_path / "workflow.json"
    wf_path.write_text(json.dumps({
        "id": "local-proof",
        "kind": "implementation",
        "requires_approval": False,
        "template": "Local proof workflow",
        "route": "local:direct",
        "capabilities": ["code"],
        "allowed_paths": [str(tmp_path)],
        "blocked_paths": [],
    }), encoding="utf-8")
    assert agents_os.main(["--vault-root", str(vault), "workflow", "validate", str(wf_path), "--json"]) == 0
    wf_valid = _json_out(capsys)
    assert wf_valid["valid"] is True
    assert agents_os.main(["--vault-root", str(vault), "workflow", "import", str(wf_path), "--json"]) == 0
    wf_import = _json_out(capsys)
    assert wf_import["workflow_id"] == "local-proof"

    assert agents_os.main(["--vault-root", str(vault), "run", "code-task", "implement local proof", "--title", "Proof", "--task-id", "task-proof"]) == 0
    run_result = _json_out(capsys)
    assert run_result["task_id"] == "task-proof"

    assert agents_os.main(["--vault-root", str(vault), "route", "task-proof", "--json"]) == 0
    route = _json_out(capsys)
    assert route["execution_allowed"] is True
    assert route["assigned_agent"] == "local-agent"

    assert agents_os.main(["--vault-root", str(vault), "execute", "task-proof", "--json"]) == 0
    executed = _json_out(capsys)
    assert executed["status"] == "succeeded"
    assert Path(executed["log_path"]).exists()

    assert agents_os.main(["--vault-root", str(vault), "review", "request", "task-proof", "--kind", "spec", "--json"]) == 0
    review = _json_out(capsys)
    assert review["review_id"].startswith("review-")
    assert agents_os.main(["--vault-root", str(vault), "review", "set", review["review_id"], "approved", "--notes", "ok", "--json"]) == 0
    review_done = _json_out(capsys)
    assert review_done["status"] == "approved"

    assert agents_os.main(["--vault-root", str(vault), "task", "set", "task-proof", "completed"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "task", "add", "Blocked task", "--id", "task-blocked", "--workflow", "code-task"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "task", "set", "task-blocked", "blocked"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "run", "external-action-draft", "approval payload", "--task-id", "task-approval", "--title", "Approval draft"]) == 0
    capsys.readouterr()

    assert agents_os.main(["--vault-root", str(vault), "dashboard", "--json"]) == 0
    dashboard = _json_out(capsys)
    assert dashboard["health"]["ok"] is True
    assert dashboard["queue_summary"] == {
        "open_tasks": 1,
        "blocked_tasks": 1,
        "review_tasks": 0,
        "completed_tasks": 1,
        "pending_approvals": 1,
        "failed_executions": 0,
        "stale_drafts": 1,
        "action_required": 2,
    }
    assert dashboard["tasks"][0]["id"] == "task-blocked"
    assert dashboard["agents"][0]["id"] == "local-agent"
    assert dashboard["reviews"][0]["status"] == "approved"
    assert dashboard["snapshots"][0]["label"] == "baseline"
    run_kinds = {(run["task_id"], run["status"]): run["kind"] for run in dashboard["runs"]}
    assert run_kinds[("task-proof", "created")] == "draft_superseded"
    assert run_kinds[("task-proof", "succeeded")] == "execution"
    assert run_kinds[("task-approval", "created")] == "draft"
    dashboard_text = Path(dashboard["dashboard_path"]).read_text(encoding="utf-8")
    assert "## Queue summary" in dashboard_text
    assert "action_required: 2" in dashboard_text
    assert "pending_approvals: 1" in dashboard_text
    assert "stale_drafts: 1" in dashboard_text
    assert "kind=draft_superseded" in dashboard_text
    assert "kind=draft" in dashboard_text
    assert "kind=execution" in dashboard_text
    assert "## Agent registry" in dashboard_text
    assert "## Review gateovi" in dashboard_text
    assert "## Snapshoti" in dashboard_text

    assert agents_os.main(["--vault-root", str(vault), "maintenance", "--json"]) == 0
    maintenance = _json_out(capsys)
    assert maintenance["status"] == "ok"
    assert Path(maintenance["report_path"]).exists()

    service = agents_os.AgentsOSService(agents_os.resolve_paths(None))
    status = service.status_payload()
    assert status["status"] == "ok"
    assert status["schema_version"] == "3"

    assert agents_os.main(["--vault-root", str(vault), "service", "status", "--json"]) == 0
    service_cli = _json_out(capsys)
    assert service_cli["status"] == "ok"
    assert service_cli["schema_version"] == "3"

    assert agents_os.main(["--vault-root", str(vault), "docs", "--json"]) == 0
    docs = _json_out(capsys)
    assert Path(docs["docs_path"]).exists()
    assert set(docs["docs"].keys()) == {"runtime", "command_reference", "recovery_runbook", "safety_policy"}
    text = Path(docs["docs_path"]).read_text(encoding="utf-8")
    assert "Agents OS" in text
    assert "schema_version: 3" in text
    for path in docs["docs"].values():
        assert Path(path).exists()


def test_agents_os_close_requires_evidence_or_approved_review(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()

    assert agents_os.main(["--vault-root", str(vault), "run", "code-task", "close proof", "--task-id", "task-close"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "route", "task-close", "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "execute", "task-close", "--json"]) == 0
    capsys.readouterr()

    assert agents_os.main(["--vault-root", str(vault), "close", "task-close", "--json"]) == 2
    rejected = _json_out(capsys)
    assert rejected["status"] == "error"
    assert rejected["reason"] == "evidence_or_approved_review_required"

    assert agents_os.main(["--vault-root", str(vault), "review", "request", "task-close", "--kind", "qa", "--json"]) == 0
    review = _json_out(capsys)
    assert agents_os.main(["--vault-root", str(vault), "review", "set", review["review_id"], "approved", "--notes", "ok", "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "close", "task-close", "--review-id", review["review_id"], "--json"]) == 0
    closed = _json_out(capsys)
    assert closed["status"] == "completed"
    assert closed["review_id"] == review["review_id"]

    assert agents_os.main(["--vault-root", str(vault), "dashboard", "--json"]) == 0
    dashboard = _json_out(capsys)
    assert dashboard["recent_completions"][0]["task_id"] == "task-close"
    assert dashboard["recent_completions"][0]["review_id"] == review["review_id"]

    assert agents_os.main(["--vault-root", str(vault), "run", "code-task", "evidence proof", "--task-id", "task-evidence"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "close", "task-evidence", "--evidence", "local proof text", "--json"]) == 0
    evidence_closed = _json_out(capsys)
    assert evidence_closed["status"] == "completed"
    assert evidence_closed["evidence"] == "local proof text"



def test_agents_os_agent_crud_and_routing_policy(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()

    wf_path = tmp_path / "code-workflow.json"
    wf_path.write_text(json.dumps({
        "id": "needs-code",
        "kind": "implementation",
        "requires_approval": False,
        "template": "Needs code capability",
        "route": "local:direct",
        "capabilities": ["code"],
        "allowed_paths": [str(tmp_path)],
        "blocked_paths": [],
    }), encoding="utf-8")
    assert agents_os.main(["--vault-root", str(vault), "workflow", "import", str(wf_path), "--json"]) == 0
    capsys.readouterr()

    assert agents_os.main(["--vault-root", str(vault), "agent", "add", "researcher", "--capabilities", "research", "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "task", "add", "Needs code", "--id", "task-needs-code", "--workflow", "needs-code"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "route", "task-needs-code", "--json"]) == 0
    no_agent_route = _json_out(capsys)
    assert no_agent_route["assigned_agent"] is None
    assert no_agent_route["execution_allowed"] is False
    assert no_agent_route["new_status"] == "blocked"

    assert agents_os.main(["--vault-root", str(vault), "agent", "add", "coder", "--capabilities", "code", "--status", "disabled", "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "agent", "set", "coder", "--status", "available", "--json"]) == 0
    updated = _json_out(capsys)
    assert updated["status"] == "available"
    assert agents_os.main(["--vault-root", str(vault), "agent", "show", "coder", "--json"]) == 0
    shown = _json_out(capsys)
    assert shown["id"] == "coder"
    assert shown["capabilities"] == ["code"]

    assert agents_os.main(["--vault-root", str(vault), "route", "task-needs-code", "--json"]) == 0
    assigned_route = _json_out(capsys)
    assert assigned_route["assigned_agent"] == "coder"
    assert assigned_route["execution_allowed"] is True

    assert agents_os.main(["--vault-root", str(vault), "agent", "remove", "coder", "--json"]) == 0
    removed = _json_out(capsys)
    assert removed["removed"] is True
    assert agents_os.main(["--vault-root", str(vault), "agent", "show", "coder", "--json"]) == 2
    missing = _json_out(capsys)
    assert missing["reason"] == "agent_not_found"



def test_agents_os_workflow_schema_v1_show_persists_contract(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()

    wf_path = tmp_path / "schema-v1.json"
    contract = {
        "id": "schema-v1-proof",
        "kind": "implementation",
        "requires_approval": False,
        "template": "Schema v1 proof",
        "route": "local:direct",
        "capabilities": ["code"],
        "allowed_paths": [str(tmp_path)],
        "blocked_paths": [],
        "approval_risks": ["runtime-config-change"],
        "precheck": ["doctor"],
        "execute": ["local-only"],
        "verify": ["pytest"],
        "review": ["qa"],
        "close": ["evidence-required"],
    }
    wf_path.write_text(json.dumps(contract), encoding="utf-8")
    assert agents_os.main(["--vault-root", str(vault), "workflow", "validate", str(wf_path), "--json"]) == 0
    validated = _json_out(capsys)
    assert validated["valid"] is True
    assert agents_os.main(["--vault-root", str(vault), "workflow", "import", str(wf_path), "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "workflow", "show", "schema-v1-proof", "--json"]) == 0
    shown = _json_out(capsys)
    assert shown["id"] == "schema-v1-proof"
    assert shown["approval_risks"] == ["runtime-config-change"]
    assert shown["precheck"] == ["doctor"]
    assert shown["execute"] == ["local-only"]
    assert shown["verify"] == ["pytest"]
    assert shown["review"] == ["qa"]
    assert shown["close"] == ["evidence-required"]



def test_agents_os_policy_blocks_credential_paths_and_bad_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()

    wf_path = tmp_path / "unsafe-workflow.json"
    wf_path.write_text(json.dumps({
        "id": "unsafe",
        "kind": "implementation",
        "requires_approval": False,
        "template": "Unsafe workflow",
        "route": "local:direct",
        "capabilities": ["code"],
        "allowed_paths": [str(tmp_path / ".env")],
        "blocked_paths": [],
    }), encoding="utf-8")
    assert agents_os.main(["--vault-root", str(vault), "workflow", "validate", str(wf_path), "--json"]) == 2
    invalid = _json_out(capsys)
    assert invalid["valid"] is False
    assert "credential_path:allowed_paths" in invalid["errors"]

    monkeypatch.setenv("AGENTS_OS_HOME", str(tmp_path / ".openclaw" / "agents_os"))
    assert agents_os.main(["--vault-root", str(vault), "doctor", "--json"]) == 1
    doctor = _json_out(capsys)
    assert doctor["checks"]["policy_home_isolated"] is False
    monkeypatch.delenv("AGENTS_OS_HOME", raising=False)



def test_agents_os_mirror_validate_detects_missing_dashboard(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "dashboard", "--json"]) == 0
    dashboard = _json_out(capsys)
    Path(dashboard["dashboard_path"]).unlink()

    assert agents_os.main(["--vault-root", str(vault), "mirror", "validate", "--json"]) == 1
    invalid = _json_out(capsys)
    assert invalid["status"] == "attention"
    assert "missing_dashboard" in invalid["issues"]

    assert agents_os.main(["--vault-root", str(vault), "mirror", "rebuild", "--json"]) == 0
    rebuilt = _json_out(capsys)
    assert Path(rebuilt["dashboard_path"]).exists()
    assert agents_os.main(["--vault-root", str(vault), "mirror", "validate", "--json"]) == 0
    valid = _json_out(capsys)
    assert valid["status"] == "ok"



def test_agents_os_execute_dry_run_does_not_mutate_task_to_review(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "agent", "add", "local-agent", "--capabilities", "code", "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "run", "code-task", "dry run", "--task-id", "task-dry"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "route", "task-dry", "--json"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "execute", "task-dry", "--dry-run", "--json"]) == 0
    dry = _json_out(capsys)
    assert dry["status"] == "dry_run"
    assert agents_os.main(["--vault-root", str(vault), "task", "list", "--status", "ready", "--json"]) == 0
    ready = _json_out(capsys)
    assert ready[0]["id"] == "task-dry"



def test_agents_os_service_adapter_exposes_core_payloads(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()
    service = agents_os.AgentsOSService(agents_os.resolve_paths(None))
    assert service.status_payload()["status"] == "ok"
    assert service.doctor_payload()["ok"] is True
    assert service.dashboard_payload()["health"]["ok"] is True
    assert service.maintenance_payload()["status"] == "ok"



def test_agents_os_execute_blocks_approval_gated_task(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    vault = tmp_path / "vault"
    vault.mkdir()
    assert agents_os.main(["--vault-root", str(vault), "init", "--no-vault"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "run", "external-action-draft", "publish", "--task-id", "task-public"]) == 0
    capsys.readouterr()
    assert agents_os.main(["--vault-root", str(vault), "execute", "task-public", "--json"]) == 2
    blocked = _json_out(capsys)
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "approval_required"
