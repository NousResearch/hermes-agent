from gateway.approval_brief import build_exec_approval_brief, format_exec_approval_brief_text


def test_launchctl_kickstart_gets_human_intent_not_detector_category():
    command = "launchctl kickstart -k gui/501/com.crucible.agent"

    brief = build_exec_approval_brief(command, "launchctl kickstart/bootstrap")

    assert brief["Intent"] == "Restart the local LaunchAgent com.crucible.agent"
    assert brief["Scope"] == "local user LaunchAgent"
    assert "restarts a macOS launchd job" in brief["What it does"]
    assert "Command category" not in brief


def test_explicit_approval_description_wins_over_heuristics():
    command = "rm -rf build/cache"

    brief = build_exec_approval_brief(
        command,
        "destructive rm",
        {"approval_description": "Remove stale build cache before rerunning tests"},
    )

    assert brief["Intent"] == "Remove stale build cache before rerunning tests"
    assert "Remove files or directories" in brief["What it does"]


def test_common_command_summary_includes_command_scope_and_risk():
    text = format_exec_approval_brief_text(
        "git push origin fix/approval-intent-prompts",
        "git repository mutation",
    )

    assert "Intent: Push local commits to the remote git repository" in text
    assert "Command summary: git push origin fix/approval-intent-prompts" in text
    assert "Scope: project git repository and configured remote" in text
    assert "Risks:" in text
