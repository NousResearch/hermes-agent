"""Tests for Hermes long-task policy prompt/config behavior."""

from unittest.mock import MagicMock, patch

from hermes_cli.config import DEFAULT_CONFIG
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent(long_task_policy: dict | None = None, tools: tuple[str, ...] = ("terminal", "process")) -> AIAgent:
    agent_section = {"tool_use_enforcement": "off"}
    if long_task_policy is not None:
        agent_section["long_task_policy"] = long_task_policy

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tools)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value={"agent": agent_section}),
    ):
        agent = AIAgent(
            model="anthropic/claude-sonnet-4",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def test_default_config_includes_long_task_policy_defaults():
    policy = DEFAULT_CONFIG["agent"]["long_task_policy"]

    assert policy["enabled"] is True
    assert policy["foreground_soft_limit_seconds"] == 120
    assert policy["staged_task_soft_limit_seconds"] == 600
    assert policy["require_background_after_seconds"] == 600
    assert policy["require_progress_artifacts"] is True
    assert policy["default_log_dir"] == "reports/_run_logs"
    assert policy["status_filename"] == "status.json"
    assert policy["manifest_filename"] == "manifest.json"
    assert policy["notify_on_completion"] is True
    assert "FATAL" in policy["fail_fast_markers"]


def test_enabled_long_task_policy_injects_concise_prompt_guidance():
    agent = _make_agent(
        {
            "enabled": True,
            "foreground_soft_limit_seconds": 90,
            "staged_task_soft_limit_seconds": 240,
            "require_background_after_seconds": 480,
            "default_log_dir": "custom/logs",
            "status_filename": "state.json",
            "manifest_filename": "files.json",
        }
    )

    prompt = agent._build_system_prompt()

    assert "Hermes long-task policy" in prompt
    assert "longer than 90 seconds" in prompt
    assert "before 240 seconds" in prompt
    assert "longer than 480 seconds" in prompt
    assert "terminal(background=true, notify_on_complete=true)" in prompt
    assert "custom/logs" in prompt
    assert "state.json" in prompt
    assert "files.json" in prompt


def test_disabled_long_task_policy_suppresses_prompt_guidance():
    agent = _make_agent({"enabled": False})

    prompt = agent._build_system_prompt()

    assert "Hermes long-task policy" not in prompt
    assert "terminal(background=true, notify_on_complete=true)" not in prompt


def test_string_false_disables_long_task_policy_guidance():
    agent = _make_agent({"enabled": "false"})

    prompt = agent._build_system_prompt()

    assert "Hermes long-task policy" not in prompt


def test_long_task_policy_avoids_terminal_instruction_when_terminal_unavailable():
    agent = _make_agent(
        {
            "enabled": True,
            "foreground_soft_limit_seconds": 120,
            "require_background_after_seconds": 600,
        },
        tools=("web_search",),
    )

    prompt = agent._build_system_prompt()

    assert "Hermes long-task policy" in prompt
    assert "terminal(background=true" not in prompt
    assert "unavailable background tool calls" in prompt
    assert "save logs/status/manifest" not in prompt
    assert "artifact-writing tools are unavailable" in prompt
    assert "avoid inventing artifact paths" in prompt


def test_long_task_policy_honors_notification_and_artifact_knobs():
    agent = _make_agent(
        {
            "enabled": True,
            "notify_on_completion": False,
            "require_progress_artifacts": False,
        }
    )

    prompt = agent._build_system_prompt()

    assert "terminal(background=true)" in prompt
    assert "notify_on_complete=true" not in prompt
    assert "save logs/status/manifest" not in prompt
    assert "verify artifacts" not in prompt
    assert "recoverable progress notes" in prompt
    assert "verify the progress state" in prompt
