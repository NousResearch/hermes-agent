"""Full SYNTHETIC HTTP -> real App/MCP -> native acceptance -> SQLite proof."""
import json
import sys
from pathlib import Path
import pytest

from hermes_cli import kanban_db as kb, kanban_pr_acceptance_store as store
from hermes_cli.kanban_db_connect import connect
from tests.hermes_cli.test_kanban_pr_acceptance_mcp import _write_profile, _receipts, PR_URL

ENTRY = Path(__file__).parent / "fixtures" / "kanban_acceptance_full_pipeline.py"

@pytest.mark.parametrize("scenario,accepted", [
    ("success", True), ("failed", False), ("head_moved", False), ("none_required", False)])
def test_actual_evidence_server_drives_native_sqlite_gate(tmp_path, monkeypatch, scenario, accepted):
    home = tmp_path / "home"
    own = home / "profiles" / "a"
    _write_profile(own, 11)
    cfg = {"kanban": {"pr_acceptance": {"transport": "mcp"}},
           "mcp_servers": {"github_acceptance": {}}}
    server = cfg["mcp_servers"]["github_acceptance"]
    server["command"] = sys.executable
    server["args"] = [str(ENTRY), scenario]
    server["env"] = {"GITHUB_APP_ID": "${APP_ID}", "GITHUB_APP_INSTALLATION_ID": "${INSTALL_ID}",
                     "GITHUB_APP_PRIVATE_KEY": "${APP_KEY}", "PIPELINE_TRACE": str(own / "pipeline.json")}
    # JSON is a YAML subset; use no dependency outside the supported PM lock.
    (own / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-ambient-must-not-leak")
    monkeypatch.setenv("GH_TOKEN", "synthetic-ambient-must-not-leak")
    def forbid_gh(*args, **kwargs):
        raise AssertionError("MCP must never delegate to gh")
    monkeypatch.setattr(store, "collect_acceptance", forbid_gh)
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="synthetic composition", completion_contract="acme/repo", assignee="a")
        original_status = kb.get_task(conn, tid).status
        assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR_URL}) is accepted
        receipt = _receipts(conn, tid)[-1]
        assert receipt["ok"] is accepted
        assert kb.get_task(conn, tid).status == ("done" if accepted else original_status)
        assert kb.get_task(conn, tid).completion_contract == PR_URL
        if accepted:
            assert receipt["head_sha"] == "a" * 40
            assert receipt["required"] == [{"context": "required", "app_id": 11},
                                             {"context": "status-required", "app_id": None}]
    trace = json.loads((own / "pipeline.json").read_text(encoding="utf-8"))
    assert trace["synthetic"] and not trace["real_network"] and not trace["ambient_token_forwarded"]
    assert "POST /graphql" in trace["call_paths"]
    assert "GET /repos/acme/repo/rules/branches/main?per_page=100&page=1" in trace["call_paths"]
    assert any("filter=latest" in p for p in trace["call_paths"])
