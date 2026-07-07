import pytest
from unittest.mock import MagicMock, patch

from agent.skill_commands import scan_skill_commands
from cli import HermesCLI
from hermes_cli.commands import resolve_command


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "sess-123"
    cli_obj._pending_input = MagicMock()
    cli_obj._agent_running = False
    return cli_obj


def _make_skill(skills_dir, name, description="Imported command wrapper"):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
---

# {name}

Use this imported command wrapper.
"""
    )


def test_registry_resolves_new_native_commands():
    assert resolve_command("codex-review").name == "codex-review"
    assert resolve_command("codex-status").name == "codex-status"
    assert resolve_command("codex-result").name == "codex-result"
    assert resolve_command("codex-setup").name == "codex-setup"
    assert resolve_command("codex-cancel").name == "codex-cancel"
    assert resolve_command("codex-rescue").name == "codex-rescue"
    assert resolve_command("codex-adversarial-review").name == "codex-adversarial-review"
    assert resolve_command("ralph-loop").name == "ralph-loop"
    assert resolve_command("review-pr").name == "review-pr"
    assert resolve_command("playwright-test").name == "playwright-test"
    assert resolve_command("triage").name == "triage"
    assert resolve_command("report-bug").name == "report-bug"
    assert resolve_command("reproduce-bug").name == "reproduce-bug"
    assert resolve_command("plan-review").name == "plan-review"
    assert resolve_command("commit").name == "commit"
    assert resolve_command("commit-push-pr").name == "commit-push-pr"
    assert resolve_command("code-review").name == "code-review"
    assert resolve_command("changelog").name == "changelog"
    assert resolve_command("create-agent-skill").name == "create-agent-skill"
    assert resolve_command("deepen-plan").name == "deepen-plan"
    assert resolve_command("generate-command").name == "generate-command"
    assert resolve_command("heal-skill").name == "heal-skill"
    assert resolve_command("hookify").name == "hookify"
    assert resolve_command("hookify-configure").name == "hookify-configure"
    assert resolve_command("hookify-list").name == "hookify-list"
    assert resolve_command("hookify-help").name == "hookify-help"
    assert resolve_command("new-sdk-app").name == "new-sdk-app"
    assert resolve_command("revise-claude-md").name == "revise-claude-md"
    assert resolve_command("quiz-me").name == "quiz-me"
    assert resolve_command("sync-tutorials").name == "sync-tutorials"
    assert resolve_command("teach-me").name == "teach-me"
    assert resolve_command("clean-gone").name == "clean-gone"
    assert resolve_command("deploy-docs").name == "deploy-docs"
    assert resolve_command("feature-video").name == "feature-video"
    assert resolve_command("release-docs").name == "release-docs"
    assert resolve_command("resolve-parallel").name == "resolve-parallel"
    assert resolve_command("resolve-pr-parallel").name == "resolve-pr-parallel"
    assert resolve_command("resolve-todo-parallel").name == "resolve-todo-parallel"
    assert resolve_command("xcode-test").name == "xcode-test"
    assert resolve_command("create-plugin").name == "create-plugin"
    assert resolve_command("cancel-ralph").name == "cancel-ralph"
    assert resolve_command("ralph-help").name == "ralph-help"
    assert resolve_command("example-plugin-example-command").name == "example-plugin-example-command"
    assert resolve_command("feature-dev-feature-dev").name == "feature-dev-feature-dev"


def test_native_codex_review_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "codex-review")
        scan_skill_commands()
        result = cli_obj.process_command("/codex-review --wait --scope working-tree")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "Imported command wrapper" in queued
    assert "--wait --scope working-tree" in queued


def test_native_ralph_loop_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "ralph-loop-ralph-loop")
        scan_skill_commands()
        result = cli_obj.process_command("/ralph-loop Ship the feature")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "Ship the feature" in queued
    assert "Imported command wrapper" in queued


def test_native_review_pr_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "pr-review-toolkit-review-pr")
        scan_skill_commands()
        result = cli_obj.process_command("/review-pr 123")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "123" in queued
    assert "Imported command wrapper" in queued


def test_native_codex_setup_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "codex-setup")
        scan_skill_commands()
        result = cli_obj.process_command("/codex-setup project bootstrap")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "project bootstrap" in queued


def test_native_playwright_test_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "compound-engineering-playwright-test")
        scan_skill_commands()
        result = cli_obj.process_command("/playwright-test login flow")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "login flow" in queued


def test_native_triage_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "compound-engineering-triage")
        scan_skill_commands()
        result = cli_obj.process_command("/triage flaky auth failures")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "flaky auth failures" in queued


def test_native_commit_push_pr_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "commit-commands-commit-push-pr")
        scan_skill_commands()
        result = cli_obj.process_command("/commit-push-pr release notes")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "release notes" in queued


def test_native_code_review_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "code-review-code-review")
        scan_skill_commands()
        result = cli_obj.process_command("/code-review auth diff")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "auth diff" in queued


def test_native_hookify_command_queues_wrapper_skill(tmp_path):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "hookify-hookify")
        scan_skill_commands()
        result = cli_obj.process_command("/hookify add a write guard")

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert "add a write guard" in queued


@pytest.mark.parametrize(
    "command_text,skill_name,needle",
    [
        ("/hookify-help examples", "hookify-help", "examples"),
        ("/new-sdk-app starter", "agent-sdk-dev-new-sdk-app", "starter"),
        ("/revise-claude-md tighten rules", "claude-md-management-revise-claude-md", "tighten rules"),
        ("/quiz-me sorting", "coding-tutor-quiz-me", "sorting"),
        ("/sync-tutorials python", "coding-tutor-sync-tutorials", "python"),
        ("/teach-me graphs", "coding-tutor-teach-me", "graphs"),
        ("/clean-gone prune old branches", "commit-commands-clean-gone", "prune old branches"),
        ("/deploy-docs site build", "compound-engineering-deploy-docs", "site build"),
        ("/feature-video launch demo", "compound-engineering-feature-video", "launch demo"),
        ("/release-docs v1.2", "compound-engineering-release-docs", "v1.2"),
        ("/resolve-parallel bug queue", "compound-engineering-resolve-parallel", "bug queue"),
        ("/resolve-pr-parallel pr 42", "compound-engineering-resolve-pr-parallel", "pr 42"),
        ("/resolve-todo-parallel backlog", "compound-engineering-resolve-todo-parallel", "backlog"),
        ("/xcode-test ios smoke", "compound-engineering-xcode-test", "ios smoke"),
        ("/create-plugin hook bundle", "plugin-dev-create-plugin", "hook bundle"),
        ("/cancel-ralph stop loop", "ralph-loop-cancel-ralph", "stop loop"),
        ("/ralph-help usage", "ralph-loop-help", "usage"),
        ("/example-plugin-example-command demo", "example-plugin-example-command", "demo"),
        ("/feature-dev-feature-dev ship it", "feature-dev-feature-dev", "ship it"),
    ],
)
def test_remaining_native_commands_queue_wrapper_skills(tmp_path, command_text, skill_name, needle):
    cli_obj = _make_cli()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, skill_name)
        scan_skill_commands()
        result = cli_obj.process_command(command_text)

    assert result is True
    cli_obj._pending_input.put.assert_called_once()
    queued = cli_obj._pending_input.put.call_args[0][0]
    assert needle in queued
