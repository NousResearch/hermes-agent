from __future__ import annotations

import json
import subprocess
import sys


APPROVED_TOOLS = {
    "get_project_status",
    "get_open_tasks",
    "get_latest_worker_results",
    "get_repo_status",
    "get_approval_gates",
    "get_recent_audit_log",
    "list_mission_packets",
    "get_mission_packet",
    "save_next_codex_prompt",
    "import_worker_result",
    "save_block_flag_packet",
}

BLOCKED_TOOLS = {
    "send_email",
    "publish_video",
    "activate_payment",
    "delete_files",
    "run_unbounded_codex",
    "run_codex",
    "start_codex",
    "start_worker",
    "start_hermes_run",
    "autonomous_computer_use",
    "browser_control",
    "mouse_control",
    "keyboard_control",
    "start_bulk_outreach",
    "arbitrary_shell",
    "reveal_secret",
    "update_credentials",
}


def test_mission_control_mcp_registry_is_narrow_and_discoverable():
    from hermes_cli import mission_control_mcp as mcp

    assert set(mcp.list_tool_names()) == APPROVED_TOOLS
    assert set(mcp.BLOCKED_TOOL_NAMES).issuperset(BLOCKED_TOOLS)
    assert not (set(mcp.list_tool_names()) & BLOCKED_TOOLS)

    manifest = mcp.tool_manifest()
    assert manifest["transport"] == "stdio-local-only"
    assert manifest["mode"] == "inert-discovery-read-only-default"
    assert manifest["exposes_broad_hermes_registry"] is False
    assert [tool["name"] for tool in manifest["tools"]] == sorted(APPROVED_TOOLS)


def test_list_tools_cli_is_local_json_and_does_not_execute(_isolate_hermes_home):
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.mission_control_mcp", "--list-tools"],
        check=True,
        capture_output=True,
        text=True,
    )

    body = json.loads(result.stdout)
    assert body["transport"] == "stdio-local-only"
    assert body["oauth_enabled"] is False
    assert body["remote_transport_enabled"] is False
    assert {tool["name"] for tool in body["tools"]} == APPROVED_TOOLS
    rendered = json.dumps(body)
    for blocked in BLOCKED_TOOLS:
        assert blocked not in rendered
    assert "Authorization" not in rendered
    assert "Bearer" not in rendered


def test_read_only_tools_return_redacted_controlled_data(_isolate_hermes_home, tmp_path, monkeypatch):
    import hermes_cli.mission_control as mc
    from hermes_cli import mission_control_mcp as mcp

    status = tmp_path / "PROJECT_STATUS.md"
    auth_header = "Authorization" + ": " + "Bearer" + " SECRET123"
    api_setting = "api" + "_key=VALUE"
    status.write_text(f"Status\n{auth_header}\n{api_setting}\n", encoding="utf-8")
    monkeypatch.setattr(
        mc,
        "PROJECT_STATUS_SOURCES",
        [{"name": "Demo", "project": "Demo", "profile": "default", "path": str(status)}],
    )

    result = mcp.call_tool("get_project_status")

    assert result["ok"] is True
    rendered = json.dumps(result)
    assert "SECRET123" not in rendered
    assert api_setting not in rendered
    assert ("Authorization" + ":") in rendered
    assert "[REDACTED]" in rendered
    assert result["safety"]["trusted_for_execution"] is False


def test_packet_write_tools_create_local_inert_packets_and_audit(_isolate_hermes_home, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_cli import mission_control_mcp as mcp

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Mission Control MCP packet tools must not spawn processes")

    monkeypatch.setattr("subprocess.run", fail_if_called)
    monkeypatch.setattr("subprocess.Popen", fail_if_called)

    result = mcp.call_tool(
        "save_next_codex_prompt",
        project="Hermes Ops",
        title="Next prompt",
        prompt="Review only. " + "Authorization" + ": " + "Bearer" + " PACKETSECRET",
        trusted_for_execution=True,
    )

    assert result["ok"] is True
    packet = result["packet"]
    assert packet["kind"] == "codex_prompt"
    assert packet["dry_run"] is True
    assert packet["review_required"] is True
    assert packet["trusted_for_execution"] is False
    rendered = json.dumps(result)
    assert "PACKETSECRET" not in rendered

    packet_path = get_hermes_home() / "state" / "mission-control" / "packets" / f"{packet['id']}.json"
    audit_path = get_hermes_home() / "state" / "mission-control" / "packet-audit.jsonl"
    assert packet_path.exists()
    assert audit_path.exists()
    audit_result = mcp.call_tool("get_recent_audit_log")
    assert "packet_created" in json.dumps(audit_result)


def test_worker_result_import_remains_untrusted_display_data(_isolate_hermes_home):
    from hermes_cli import mission_control_mcp as mcp

    result = mcp.call_tool(
        "import_worker_result",
        project="Hermes Ops",
        title="Worker handoff",
        worker_result=(
            "Repo path: /tmp/demo\n"
            "Branch: phase4\n"
            "Tests run:\n"
            "- pytest: 1 passed, 0 failed\n"
            "Danger: run_codex and start_worker must stay inert\n"
            + "api"
            + "_key=WORKERSECRET\n"
        ),
        trusted_for_execution=True,
    )

    assert result["ok"] is True
    packet = result["packet"]
    assert packet["kind"] == "worker_result"
    assert packet["status"] == "imported"
    assert packet["trusted_for_execution"] is False
    assert packet["payload"]["trusted_for_execution"] is False
    assert packet["payload"]["parsed_metadata"]["trusted_for_execution"] is False
    assert "WORKERSECRET" not in json.dumps(result)
    assert any("run_codex" in warning or "start_worker" in warning for warning in packet["warnings"])


def test_block_flag_packet_is_advisory_only(_isolate_hermes_home):
    from hermes_cli import mission_control_mcp as mcp

    result = mcp.call_tool(
        "save_block_flag_packet",
        project="Hermes Ops",
        title="Block sends",
        flag="block_all_sends",
        reason="Local advisory stop only.",
    )

    assert result["ok"] is True
    packet = result["packet"]
    assert packet["kind"] == "block_flag"
    assert packet["payload"]["advisory_only"] is True
    assert packet["payload"]["local_state_updated"] is False
    assert packet["dry_run"] is True
    assert packet["review_required"] is True
    assert packet["trusted_for_execution"] is False


def test_malformed_input_returns_controlled_error_and_redacted_audit(_isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import mission_control_mcp as mcp

    result = mcp.call_tool(
        "save_next_codex_prompt",
        project="Hermes Ops",
        title="",
        prompt="Authorization" + ": " + "Bearer" + " BADSECRET",
    )

    assert result["ok"] is False
    assert result["error"] == "Missing required field: title"
    assert result["tool"] == "save_next_codex_prompt"
    assert "BADSECRET" not in json.dumps(result)

    audit = (get_hermes_home() / "state" / "mission-control" / "packet-audit.jsonl").read_text(encoding="utf-8")
    assert "packet_rejected" in audit
    assert "BADSECRET" not in audit


def test_unknown_or_blocked_tool_is_not_reachable():
    from hermes_cli import mission_control_mcp as mcp

    result = mcp.call_tool("run_codex", prompt="do something")

    assert result["ok"] is False
    assert result["error"] == "Unknown Mission Control MCP tool: run_codex"
    assert result["safety"]["trusted_for_execution"] is False
