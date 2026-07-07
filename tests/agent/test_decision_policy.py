import json

from agent.decision_packet import DecisionPacket, decision_packet_tool_result
from agent.decision_policy import evaluate_terminal_command, evaluate_tool_call
from tools.approval import detect_hardline_command


def _needs_chad(result) -> DecisionPacket:
    assert result.needs_chad
    packet = result.packet
    assert packet is not None
    assert packet.status == "NEEDS_CHAD"
    text = packet.to_text()
    for field in (
        "status: NEEDS_CHAD",
        "reason:",
        "proposed action:",
        "why this is a fork:",
        "safest default if no answer:",
        "exact approve/deny/narrow options:",
        "evidence summary:",
    ):
        assert field in text
    return packet


def test_read_only_terminal_command_continues():
    assert not evaluate_terminal_command("git status --short").needs_chad
    assert not evaluate_terminal_command("ls -la && pwd").needs_chad
    assert not evaluate_terminal_command("rg secret docs").needs_chad


def test_local_bounded_file_edit_inside_workspace_continues(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = evaluate_tool_call(
        "write_file",
        {"path": "src/app.py", "content": "print('ok')\n"},
        cwd=str(tmp_path),
    )

    assert not result.needs_chad


def test_local_file_edit_outside_workspace_stops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.txt"

    packet = _needs_chad(
        evaluate_tool_call(
            "write_file",
            {"path": str(outside), "content": "x"},
            cwd=str(tmp_path),
        )
    )

    assert "outside the approved repo/worktree" in packet.reason


def test_git_commit_push_merge_rebase_reset_stop():
    commands = [
        "git commit -m test",
        "git push origin main",
        "git merge feature",
        "git rebase main",
        "git reset --soft HEAD~1",
    ]

    for command in commands:
        packet = _needs_chad(evaluate_terminal_command(command))
        assert command in packet.proposed_action


def test_branch_strategy_changes_stop_but_status_continues():
    assert not evaluate_terminal_command("git branch --show-current").needs_chad

    packet = _needs_chad(evaluate_terminal_command("git switch -c feature/doctrine"))

    assert "branch" in packet.reason.lower() or "switch" in packet.reason.lower()


def test_runtime_service_and_config_changes_stop():
    service_packet = _needs_chad(evaluate_terminal_command("systemctl restart hermes-gateway"))
    config_packet = _needs_chad(evaluate_terminal_command("sed -i 's/a/b/' config.yaml"))

    assert "Runtime" in service_packet.reason
    assert "Runtime" in config_packet.reason


def test_rm_destructive_filesystem_commands_stop():
    commands = [
        "rm /tmp/pilot-test-file",
        "rm -r /tmp/pilot-test",
        "rm -rf /tmp/pilot-test",
        "rm -fr /tmp/pilot-test",
        "rm --recursive /tmp/pilot-test",
        "rm --force --recursive /tmp/pilot-test",
        "rm -rf '/tmp/pilot test'",
        'rm -rf "./local dir"',
        "rm -rf ./local-dir",
    ]

    for command in commands:
        packet = _needs_chad(evaluate_terminal_command(command))
        assert "Destructive filesystem" in packet.reason
        assert command in packet.proposed_action


def test_catastrophic_rm_root_deferred_to_hardline_floor():
    result = evaluate_terminal_command("rm -rf /")
    is_hardline, reason = detect_hardline_command("rm -rf /")

    assert not result.needs_chad
    assert is_hardline
    assert "root" in reason.lower()


def test_credential_secret_handling_stops():
    for command in ("cat .env", "gh auth token", "security find-generic-password -a chad"):
        packet = _needs_chad(evaluate_terminal_command(command))
        assert "Credential" in packet.reason


def test_phi_client_case_sensitive_material_stops_without_blocking_generic_client_code():
    assert not evaluate_terminal_command("rg client src").needs_chad

    terminal_packet = _needs_chad(evaluate_terminal_command("cat client-data/case-123.txt"))
    file_packet = _needs_chad(
        evaluate_tool_call("write_file", {"path": "case-files/summary.txt", "content": "x"})
    )

    assert "PHI" in terminal_packet.reason
    assert "case-sensitive" in file_packet.reason


def test_external_and_autonomous_non_terminal_tools_stop():
    send_packet = _needs_chad(
        evaluate_tool_call("send_message", {"action": "send", "message": "hello"})
    )
    cron_packet = _needs_chad(
        evaluate_tool_call("cronjob", {"action": "create", "name": "daily", "schedule": "0 9 * * *"})
    )

    assert "External" in send_packet.reason
    assert "cron" in cron_packet.reason.lower()
    assert not evaluate_tool_call("send_message", {"action": "list"}).needs_chad
    assert not evaluate_tool_call("cronjob", {"action": "list"}).needs_chad


def test_decision_packet_tool_result_is_structured_json():
    packet = DecisionPacket(
        reason="reason",
        proposed_action="action",
        why_this_is_a_fork="why",
        safest_default="default",
        options=["approve: yes", "deny: no", "narrow: smaller"],
        evidence_summary="evidence",
    )

    data = json.loads(decision_packet_tool_result(packet))

    assert data["status"] == "needs_chad"
    assert data["decision_packet"]["status"] == "NEEDS_CHAD"
    assert data["decision_packet"]["reason"] == "reason"
