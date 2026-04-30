"""Focused regressions for the stateless Claude CLI transport."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.claude_cli_client import ClaudeCLIClient


class _Proc(SimpleNamespace):
    pass


def _stdin_text(kwargs: dict) -> str:
    payload = json.loads(kwargs["input"])
    return payload["message"]["content"][0]["text"]


def _assert_safe_stateless_cmd(cmd: list[str]) -> None:
    assert cmd[:2] == ["claude", "-p"]
    assert "--input-format" in cmd
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    assert "--output-format" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--tools" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--no-session-persistence" in cmd
    assert "--permission-mode" not in cmd
    assert "bypassPermissions" not in cmd
    assert "--resume" not in cmd
    assert "--session-id" not in cmd


def _stdout(result_text: str = "OK", *, usage: dict | None = None) -> str:
    usage = usage or {
        "input_tokens": 2,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20,
        "output_tokens": 3,
    }
    payloads = [
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": result_text}],
                "usage": usage,
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": result_text,
            "usage": usage,
        },
    ]
    return "\n".join(json.dumps(item) for item in payloads)


def test_full_transcript_is_sent_every_turn():
    client = ClaudeCLIClient(command="claude")
    prompts: list[str] = []

    def _run(cmd, **kwargs):
        _assert_safe_stateless_cmd(cmd)
        prompts.append(_stdin_text(kwargs))
        assert "Original request" not in cmd
        assert kwargs["text"] is True
        return _Proc(returncode=0, stdout=_stdout("Hello"), stderr="")

    with patch("agent.claude_cli_client.subprocess.run", side_effect=_run):
        client.chat.completions.create(
            model="claude-sonnet-4.6",
            messages=[
                {"role": "system", "content": "System guidance"},
                {"role": "user", "content": "Original request"},
            ],
        )
        client.chat.completions.create(
            model="claude-sonnet-4.6",
            messages=[
                {"role": "system", "content": "System guidance"},
                {"role": "user", "content": "Original request"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Follow up"},
            ],
        )

    assert len(prompts) == 2
    assert "Original request" in prompts[0]
    assert "Original request" in prompts[1]
    assert "Previous answer" in prompts[1]
    assert "Follow up" in prompts[1]


def test_arbitrary_extra_args_are_not_accepted(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_ARGS", "--dangerous")
    client = ClaudeCLIClient(command="claude")

    def _run(cmd, **kwargs):
        _assert_safe_stateless_cmd(cmd)
        assert "--dangerous" not in cmd
        return _Proc(returncode=0, stdout=_stdout("OK"), stderr="")

    with patch("agent.claude_cli_client.subprocess.run", side_effect=_run):
        client.chat.completions.create(model="sonnet", messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(ValueError, match="does not accept extra CLI args"):
        ClaudeCLIClient(command="claude", args=["--dangerous"])


def test_timeout_object_is_converted_to_subprocess_timeout():
    client = ClaudeCLIClient(command="claude")

    class _Timeout:
        connect = 1.0
        read = 12.5
        write = 3.0
        pool = 2.0

    def _run(cmd, **kwargs):
        _assert_safe_stateless_cmd(cmd)
        assert kwargs["timeout"] == 12.5
        return _Proc(returncode=0, stdout=_stdout("OK"), stderr="")

    with patch("agent.claude_cli_client.subprocess.run", side_effect=_run):
        client.chat.completions.create(
            model="claude-sonnet-4.6",
            messages=[{"role": "user", "content": "hi"}],
            timeout=_Timeout(),
        )


def test_stream_json_result_reasoning_and_usage_are_parsed():
    client = ClaudeCLIClient(command="claude")
    payloads = [
        {"type": "stream_event", "event": {"type": "thinking_delta", "delta": {"thinking": "plan"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"text": "part"}}},
        {
            "type": "result",
            "subtype": "success",
            "result": "final",
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
                "output_tokens": 3,
            },
        },
    ]
    stdout = "\n".join(json.dumps(item) for item in payloads)

    with patch(
        "agent.claude_cli_client.subprocess.run",
    ) as mock_run:
        mock_run.return_value = _Proc(returncode=0, stdout=stdout, stderr="")
        response = client.chat.completions.create(
            model="claude-opus-4.6",
            messages=[{"role": "user", "content": "hi"}],
        )

    _assert_safe_stateless_cmd(mock_run.call_args.args[0])
    message = response.choices[0].message
    assert message.content == "final"
    assert message.reasoning == "plan"
    assert response.usage.prompt_tokens == 32
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 35
    assert response.usage.prompt_tokens_details.cached_tokens == 20


def test_tool_call_text_is_returned_to_hermes_loop():
    client = ClaudeCLIClient(command="claude")
    tool_text = json.dumps(
        {
            "id": "call_1",
            "function": {"name": "shell", "arguments": {"cmd": "pwd"}},
        }
    )
    stdout = _stdout(f"Need a tool.\n<tool_call>{tool_text}</tool_call>")

    with patch(
        "agent.claude_cli_client.subprocess.run",
    ) as mock_run:
        mock_run.return_value = _Proc(returncode=0, stdout=stdout, stderr="")
        response = client.chat.completions.create(
            model="claude-sonnet-4.6",
            messages=[{"role": "user", "content": "run pwd"}],
        )

    _assert_safe_stateless_cmd(mock_run.call_args.args[0])
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "Need a tool."
    assert choice.message.tool_calls[0].id == "call_1"
    assert choice.message.tool_calls[0].function.name == "shell"
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {"cmd": "pwd"}


def test_large_transcript_is_sent_via_stdin_not_argv():
    client = ClaudeCLIClient(command="claude")
    large_content = "large transcript chunk " * 20000
    large_content_core = large_content.strip()

    def _run(cmd, **kwargs):
        _assert_safe_stateless_cmd(cmd)
        argv_text = "\n".join(cmd)
        assert large_content_core not in argv_text
        stdin_text = _stdin_text(kwargs)
        assert large_content_core in stdin_text
        assert len(kwargs["input"]) > len(large_content)
        return _Proc(returncode=0, stdout=_stdout("OK"), stderr="")

    with patch("agent.claude_cli_client.subprocess.run", side_effect=_run):
        response = client.chat.completions.create(
            model="claude-sonnet-4.6",
            messages=[
                {"role": "system", "content": "System guidance"},
                {"role": "user", "content": large_content},
            ],
        )

    assert response.choices[0].message.content == "OK"
