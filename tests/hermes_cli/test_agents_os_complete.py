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

    assert agents_os.main(["--vault-root", str(vault), "agent", "add", "doni-local", "--name", "Doni Local", "--capabilities", "code,research,qa", "--json"]) == 0
    agent = _json_out(capsys)
    assert agent["id"] == "doni-local"
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
        "route": "doni:direct",
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
    assert route["assigned_agent"] == "doni-local"

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

    assert agents_os.main(["--vault-root", str(vault), "dashboard", "--json"]) == 0
    dashboard = _json_out(capsys)
    assert dashboard["health"]["ok"] is True
    assert dashboard["tasks"][0]["id"] == "task-proof"

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
    text = Path(docs["docs_path"]).read_text(encoding="utf-8")
    assert "Agents OS" in text
    assert "schema_version: 3" in text


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
