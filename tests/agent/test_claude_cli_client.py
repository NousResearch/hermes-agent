"""Tests for the claude-cli (Claude Code CLI subprocess) provider client."""

import json
from types import SimpleNamespace

import pytest

from agent import claude_cli_client as c


# ---------------------------------------------------------------------------
# Prompt serialization
# ---------------------------------------------------------------------------

def test_format_messages_includes_tools_and_transcript():
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "weather",
                "parameters": {"type": "object"},
            },
        }
    ]
    prompt = c._format_messages_as_prompt(messages, tools=tools)
    assert "get_weather" in prompt
    assert "<tool_call>" in prompt
    assert "User:" in prompt and "System:" in prompt


def test_render_message_content_handles_list_blocks():
    rendered = c._render_message_content(
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    )
    assert rendered == "a\nb"


# ---------------------------------------------------------------------------
# argv construction — model/effort/no-bare/tools-disabled
# ---------------------------------------------------------------------------

def test_argv_carries_model_effort_and_disables_tools():
    client = c.ClaudeCLIClient()
    argv = client._build_argv(model="opus", effort="high")
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "high"
    # --tools "" disables every built-in tool so claude never executes.
    assert argv[argv.index("--tools") + 1] == ""
    assert "-p" in argv
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"


def test_argv_never_uses_bare():
    # --bare would force ANTHROPIC_API_KEY and skip the OAuth session we want.
    client = c.ClaudeCLIClient()
    argv = client._build_argv(model="sonnet", effort="medium")
    assert "--bare" not in argv


def test_extra_args_from_env(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_ARGS", "--add-dir /tmp")
    client = c.ClaudeCLIClient()
    argv = client._build_argv(model="sonnet", effort="low")
    assert argv[-2:] == ["--add-dir", "/tmp"]


def test_command_from_env(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_COMMAND", "/opt/claude")
    client = c.ClaudeCLIClient()
    assert client._command == "/opt/claude"


# ---------------------------------------------------------------------------
# Live model/effort resolution (honors /model and /effort mid-session)
# ---------------------------------------------------------------------------

def test_effort_read_from_extra_body_per_call(monkeypatch):
    captured = {}

    def fake_run_prompt(self, prompt_text, *, model, effort, timeout_seconds):
        captured["model"] = model
        captured["effort"] = effort
        return "ok", "", []

    monkeypatch.setattr(c.ClaudeCLIClient, "_run_prompt", fake_run_prompt)
    client = c.ClaudeCLIClient()
    client.chat.completions.create(
        model="opus",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"_hermes_claude_effort": "xhigh"},
    )
    assert captured == {"model": "opus", "effort": "xhigh"}


def test_effort_defaults_when_absent(monkeypatch):
    captured = {}

    def fake_run_prompt(self, prompt_text, *, model, effort, timeout_seconds):
        captured["effort"] = effort
        return "ok", "", []

    monkeypatch.setattr(c.ClaudeCLIClient, "_run_prompt", fake_run_prompt)
    client = c.ClaudeCLIClient()
    client.chat.completions.create(
        model="sonnet", messages=[{"role": "user", "content": "hi"}]
    )
    assert captured["effort"] == "medium"


def test_invalid_effort_falls_back_to_medium():
    assert c._normalize_effort("bogus") == "medium"
    assert c._normalize_effort("high") == "high"
    assert c._normalize_effort(None) == "medium"


# ---------------------------------------------------------------------------
# Stream parsing — native tool_use, text, thinking, error handling
# ---------------------------------------------------------------------------

def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def test_parse_stream_captures_native_tool_use():
    stdout = _stream(
        {"type": "system", "subtype": "init"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "I'll check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ]
            },
        },
        {"type": "result", "subtype": "error_max_turns", "is_error": True},
    )
    text, reasoning, tool_calls, fatal = c._parse_stream(stdout)
    assert text == "I'll check."
    assert fatal == ""  # error_max_turns is benign
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"city": "Tokyo"}


def test_parse_stream_plain_text_success():
    stdout = _stream(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "PONG"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "PONG"},
    )
    text, reasoning, tool_calls, fatal = c._parse_stream(stdout)
    assert text == "PONG"
    assert tool_calls == []
    assert fatal == ""


def test_parse_stream_records_fatal_error_without_content():
    stdout = _stream(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "errors": ["auth failed"],
        }
    )
    text, reasoning, tool_calls, fatal = c._parse_stream(stdout)
    assert text == ""
    assert tool_calls == []
    assert "auth failed" in fatal


def test_parse_stream_captures_thinking_as_reasoning():
    stdout = _stream(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "answer"},
                ]
            },
        },
    )
    text, reasoning, tool_calls, fatal = c._parse_stream(stdout)
    assert text == "answer"
    assert reasoning == "hmm"


# ---------------------------------------------------------------------------
# <tool_call> text fallback (when a model answers in text, not native)
# ---------------------------------------------------------------------------

def test_text_tool_call_fallback():
    blob = (
        'ok <tool_call>{"id":"x","type":"function",'
        '"function":{"name":"do_it","arguments":"{}"}}</tool_call> done'
    )
    calls, cleaned = c._extract_tool_calls_from_text(blob)
    assert len(calls) == 1
    assert calls[0].function.name == "do_it"
    assert "ok" in cleaned and "done" in cleaned
    assert "<tool_call>" not in cleaned


# ---------------------------------------------------------------------------
# Full completion object shape (duck-types openai ChatCompletion)
# ---------------------------------------------------------------------------

def test_completion_shape_for_tool_call(monkeypatch):
    tc = c._build_openai_tool_call(call_id="c1", name="t", arguments="{}")
    monkeypatch.setattr(
        c.ClaudeCLIClient,
        "_run_prompt",
        lambda self, *a, **k: ("preamble", "reasoning", [tc]),
    )
    client = c.ClaudeCLIClient()
    completion = client.chat.completions.create(
        model="sonnet", messages=[{"role": "user", "content": "hi"}]
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls[0].function.name == "t"
    assert choice.message.content == "preamble"
    assert completion.model == "sonnet"
    assert hasattr(completion, "usage")


def test_completion_shape_for_plain_text(monkeypatch):
    monkeypatch.setattr(
        c.ClaudeCLIClient, "_run_prompt", lambda self, *a, **k: ("hello", "", [])
    )
    client = c.ClaudeCLIClient()
    completion = client.chat.completions.create(
        model="haiku", messages=[{"role": "user", "content": "hi"}]
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content == "hello"
    assert not choice.message.tool_calls


def test_run_prompt_raises_only_when_no_content(monkeypatch):
    # A failed subprocess with no usable stream output must raise.
    def fake_subprocess_run(*a, **k):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(c.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(c, "_build_subprocess_env", lambda: {})
    client = c.ClaudeCLIClient()
    with pytest.raises(RuntimeError, match="boom|no output"):
        client._run_prompt("p", model="sonnet", effort="low", timeout_seconds=5)
