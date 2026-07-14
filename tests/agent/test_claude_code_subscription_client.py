"""Behavior tests for the Claude Code subscription-backed provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agent.claude_code_subscription_client import (
    CLAUDE_CODE_SUBSCRIPTION_BASE_URL,
    ClaudeCodeSubscriptionClient,
    ClaudeCodeSubscriptionError,
    ClaudeCodeSubscriptionExhaustedError,
    _build_claude_command,
    _format_prompt,
    _parse_claude_result,
)


def _success_payload(result: str = "hello") -> dict:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result,
        "session_id": "session-123",
        "usage": {
            "input_tokens": 11,
            "cache_creation_input_tokens": 13,
            "cache_read_input_tokens": 17,
            "output_tokens": 19,
        },
    }


def test_build_command_runs_one_turn_without_claude_tools() -> None:
    command = _build_claude_command("/opt/homebrew/bin/claude", model="sonnet")

    assert command[:2] == ["/opt/homebrew/bin/claude", "-p"]
    assert command[command.index("--model") + 1] == "sonnet"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--max-turns") + 1] == "1"
    assert "--no-session-persistence" in command
    assert command[command.index("--output-format") + 1] == "json"


def test_parse_success_maps_usage_to_openai_shape() -> None:
    result = _parse_claude_result(json.dumps(_success_payload()))

    assert result.text == "hello"
    assert result.session_id == "session-123"
    assert result.prompt_tokens == 41
    assert result.cached_tokens == 17
    assert result.completion_tokens == 19
    assert result.total_tokens == 60


@pytest.mark.parametrize(
    "message",
    [
        "You've hit your weekly limit · resets Jul 17 at 2pm",
        "Claude usage limit reached",
        "rate_limit_error: plan limit reached",
    ],
)
def test_parse_limit_response_raises_fallback_compatible_error(message: str) -> None:
    payload = {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "result": message,
    }

    with pytest.raises(ClaudeCodeSubscriptionExhaustedError) as exc_info:
        _parse_claude_result(json.dumps(payload))

    assert exc_info.value.status_code == 429


def test_exhaustion_error_activates_existing_fallback_path() -> None:
    from agent.error_classifier import FailoverReason, classify_api_error

    error = ClaudeCodeSubscriptionExhaustedError("You've hit your weekly limit")
    classified = classify_api_error(
        error,
        provider="claude-code-subscription",
        model="sonnet",
    )

    assert classified.reason == FailoverReason.rate_limit
    assert classified.should_fallback is True


def test_parse_invalid_json_raises_provider_error() -> None:
    with pytest.raises(ClaudeCodeSubscriptionError, match="valid JSON"):
        _parse_claude_result("not-json")


def test_prompt_preserves_assistant_tool_calls_and_tool_result_identity() -> None:
    prompt = _format_prompt(
        [
            {"role": "user", "content": "Inspect config"},
            {
                "role": "assistant",
                "content": "I'll inspect it.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"config.yaml"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "read_file",
                "tool_call_id": "call_123",
                "content": "model: sonnet",
            },
        ],
        tools=None,
        tool_choice=None,
    )

    assert '"id": "call_123"' in prompt
    assert '"name": "read_file"' in prompt
    assert "config.yaml" in prompt
    assert "Tool [read_file] result for call_123" in prompt
    assert "model: sonnet" in prompt


def test_client_maps_text_and_tool_calls_to_openai_completion() -> None:
    client = ClaudeCodeSubscriptionClient(
        command="/opt/homebrew/bin/claude",
        config_dir="/tmp/claude-main",
        cwd="/tmp",
    )
    response_text = (
        "I'll inspect that.\n"
        "<tool_call>"
        '{"id":"call_read","type":"function",'
        '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
        "</tool_call>"
    )
    payload = _success_payload(response_text)

    with patch.object(
        client, "_run_prompt", return_value=_parse_claude_result(json.dumps(payload))
    ):
        response = client.chat.completions.create(
            model="sonnet",
            messages=[{"role": "user", "content": "read README.md"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {}},
                }
            ],
        )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "I'll inspect that."
    assert choice.message.tool_calls[0].function.name == "read_file"
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {
        "path": "README.md"
    }
    assert response.usage.prompt_tokens == 41
    assert response.usage.completion_tokens == 19


def test_runtime_factory_selects_subscription_client() -> None:
    from agent.agent_runtime_helpers import create_openai_client

    class FakeAgent:
        provider = "claude-code-subscription"

        @staticmethod
        def _client_log_context() -> str:
            return "test"

    with patch(
        "agent.claude_code_subscription_client.ClaudeCodeSubscriptionClient"
    ) as client_cls:
        client = create_openai_client(
            FakeAgent(),
            {
                "api_key": "external-process",
                "base_url": CLAUDE_CODE_SUBSCRIPTION_BASE_URL,
            },
            reason="test",
            shared=False,
        )

    assert client is client_cls.return_value


def test_provider_profile_registers_external_subscription_runtime() -> None:
    from providers import get_provider_profile

    profile = get_provider_profile("claude-code-subscription")

    assert profile is not None
    assert profile.auth_type == "external_process"
    assert profile.base_url == CLAUDE_CODE_SUBSCRIPTION_BASE_URL
    assert "sonnet" in profile.fallback_models
    assert profile.supports_health_check is False


def test_subprocess_receives_explicit_claude_profile_and_prompt() -> None:
    client = ClaudeCodeSubscriptionClient(
        command="/opt/homebrew/bin/claude",
        config_dir="/tmp/claude-main",
        cwd="/tmp",
    )
    process = Mock()
    process.communicate.return_value = (json.dumps(_success_payload("ok")), "")
    process.returncode = 0

    with patch(
        "agent.claude_code_subscription_client.subprocess.Popen",
        return_value=process,
    ) as popen:
        result = client._run_prompt("hello", model="sonnet", timeout_seconds=30)

    assert result.text == "ok"
    assert process.communicate.call_args.kwargs == {"input": "hello", "timeout": 30}
    assert popen.call_args.kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(
        Path("/tmp/claude-main").resolve()
    )
    assert popen.call_args.kwargs["cwd"] == str(Path("/tmp").resolve())


def test_runtime_provider_resolves_without_api_credentials() -> None:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with patch("hermes_cli.auth.shutil.which", return_value="/opt/homebrew/bin/claude"):
        runtime = resolve_runtime_provider(
            requested="claude-code-subscription",
            target_model="sonnet",
        )

    assert runtime["provider"] == "claude-code-subscription"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["base_url"] == CLAUDE_CODE_SUBSCRIPTION_BASE_URL
    assert runtime["api_key"] == "claude-code-subscription"
    assert runtime["command"] == "/opt/homebrew/bin/claude"


def test_authenticated_provider_listing_includes_claude_subscription() -> None:
    from hermes_cli.model_switch import list_authenticated_providers

    def status(provider_id: str) -> dict:
        return {"logged_in": provider_id == "claude-code-subscription"}

    with patch(
        "hermes_cli.auth.get_external_process_provider_status",
        side_effect=status,
    ):
        providers = list_authenticated_providers(
            user_providers={},
            custom_providers=[],
            probe_custom_providers=False,
        )

    claude = next(row for row in providers if row["slug"] == "claude-code-subscription")
    assert claude["models"][:3] == ["sonnet", "opus", "haiku"]


def test_missing_claude_executable_raises_provider_error() -> None:
    client = ClaudeCodeSubscriptionClient(command="missing-claude", cwd="/tmp")

    with patch(
        "agent.claude_code_subscription_client.subprocess.Popen",
        side_effect=FileNotFoundError,
    ):
        with pytest.raises(ClaudeCodeSubscriptionError, match="Could not start"):
            client._run_prompt("hello", model="sonnet", timeout_seconds=30)


def test_timeout_kills_claude_process() -> None:
    import subprocess

    client = ClaudeCodeSubscriptionClient(command="claude", cwd="/tmp")
    process = Mock()
    process.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="claude", timeout=3),
        ("", ""),
    ]

    with patch(
        "agent.claude_code_subscription_client.subprocess.Popen",
        return_value=process,
    ):
        with pytest.raises(ClaudeCodeSubscriptionError) as exc_info:
            client._run_prompt("hello", model="sonnet", timeout_seconds=3)

    assert exc_info.value.status_code == 504
    process.kill.assert_called_once_with()


def test_unauthenticated_cli_error_maps_to_401() -> None:
    client = ClaudeCodeSubscriptionClient(command="claude", cwd="/tmp")
    process = Mock()
    process.communicate.return_value = ("", "Not logged in. Please run /login")
    process.returncode = 1

    with patch(
        "agent.claude_code_subscription_client.subprocess.Popen",
        return_value=process,
    ):
        with pytest.raises(ClaudeCodeSubscriptionError) as exc_info:
            client._run_prompt("hello", model="sonnet", timeout_seconds=30)

    assert exc_info.value.status_code == 401
