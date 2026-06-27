import json
from pathlib import Path

import pytest

from governance.policy import (
    ActionClass,
    Decision,
    PolicyGate,
    PolicyGateRequest,
    classify_command,
    classify_tool_call,
    default_capabilities_for_profile,
)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestCommandClassification:
    def test_command_semantics_not_name_only(self):
        assert classify_command("python -c 'import shutil; shutil.rmtree(\"/tmp/x\")'") == ActionClass.DESTRUCTIVE
        assert classify_command("cat ~/.env") == ActionClass.READ_ONLY_SECRET_ADJACENT
        assert classify_command("ps eww") == ActionClass.PROCESS_ENV_READ
        assert classify_command("pip install requests") == ActionClass.DEPENDENCY_CHANGE
        assert classify_command("python -m pytest tests/governance -q") == ActionClass.TEST_ONLY
        assert classify_command("true") == ActionClass.READ_ONLY_LOCAL_SAFE

    def test_tool_call_classification_covers_raw_bypass_surfaces(self):
        terminal_req = classify_tool_call("terminal", {"command": "systemctl --user restart hermes-gateway.service"}, profile="omega")
        assert terminal_req.action_class == ActionClass.SERVICE_RUNTIME_CHANGE
        assert terminal_req.affected_services == ["hermes-gateway.service"]
        assert terminal_req.requested_action == "tool:terminal"

        write_req = classify_tool_call("write_file", {"path": "/tmp/demo.txt", "content": "x"}, profile="omega")
        assert write_req.action_class == ActionClass.REVERSIBLE_EDIT
        assert write_req.affected_paths == ["/tmp/demo.txt"]

        memory_req = classify_tool_call("memory", {"action": "add", "target": "memory", "content": "User prefers concise replies"}, profile="omega")
        assert memory_req.action_class == ActionClass.MEMORY_WRITE
        assert memory_req.affected_memory_stores == ["memory"]

    def test_unknown_and_mutation_looking_tools_fail_closed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        unknown_req = classify_tool_call("unregistered_experimental_tool", {}, profile="omega")
        assert unknown_req.action_class == ActionClass.UNKNOWN
        unknown_decision = PolicyGate(profile="omega").evaluate(unknown_req)
        assert unknown_decision.decision == Decision.REQUIRE_APPROVAL

        assert classify_tool_call("mcp_example_delete_record", {}, profile="omega").action_class == ActionClass.REMOTE_WRITE
        assert classify_tool_call("mcp_example_update_record", {}, profile="omega").action_class == ActionClass.REMOTE_WRITE
        assert classify_tool_call("mcp_example_write_file", {}, profile="omega").action_class == ActionClass.REMOTE_WRITE
        assert classify_tool_call("mcp_example_apply_patch", {}, profile="omega").action_class == ActionClass.REMOTE_WRITE
        assert classify_tool_call("mcp_example_sql_execute", {}, profile="omega").action_class == ActionClass.LIVE_DATA_MIGRATION
        assert classify_tool_call("mcp_example_migration_apply", {}, profile="omega").action_class == ActionClass.LIVE_DATA_MIGRATION


class TestPolicyGate:
    def test_denies_unapproved_tier4_and_logs_hash_chained_decision(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        gate = PolicyGate(profile="omega")
        req = classify_tool_call("terminal", {"command": "rm -rf /"}, profile="omega")
        decision = gate.evaluate(req)

        assert decision.decision == Decision.DENY
        assert decision.logged is True
        assert "destructive" in decision.reason.lower()

        log_path = tmp_path / "governance" / "policy_decisions.jsonl"
        rows = _read_jsonl(log_path)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "policy_decision"
        assert rows[0]["decision"] == "deny"
        assert rows[0]["previous_entry_hash"] == "GENESIS"
        assert len(rows[0]["entry_hash"]) == 64

    def test_fail_closed_on_unknown_action_or_missing_capability(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        capabilities = default_capabilities_for_profile("omega")
        capabilities.allowed_tools = ["read_file"]
        gate = PolicyGate(profile="omega", capabilities=capabilities)
        req = classify_tool_call("terminal", {"command": "pwd"}, profile="omega")
        decision = gate.evaluate(req)
        assert decision.decision == Decision.DENY
        assert decision.capability_check == "failed"

    def test_reversible_edit_requires_backup_status_but_stays_under_standing_approval(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        target = tmp_path / "note.txt"
        target.write_text("old", encoding="utf-8")
        gate = PolicyGate(profile="omega")
        req = classify_tool_call("write_file", {"path": str(target), "content": "new"}, profile="omega")
        decision = gate.evaluate(req)
        assert decision.decision == Decision.ALLOW_AFTER_BACKUP
        assert "verified backup" in decision.reason.lower()
        assert decision.backup_check == "failed"

    def test_safe_read_allows_and_logs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        gate = PolicyGate(profile="omega")
        req = classify_tool_call("read_file", {"path": str(tmp_path / "note.txt")}, profile="omega")
        decision = gate.evaluate(req)
        assert decision.decision == Decision.ALLOW
        assert decision.capability_check == "passed"
        rows = _read_jsonl(tmp_path / "governance" / "policy_decisions.jsonl")
        assert rows[-1]["decision"] == "allow"
