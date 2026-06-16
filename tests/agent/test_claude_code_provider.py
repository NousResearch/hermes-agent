"""Tests for the `claude-code` provider — running Hermes inference through the
official `claude` CLI on a personal subscription (no API key, no spoofed
user-agent, no request to api.anthropic.com from Hermes).

The CLI itself is mocked so these tests are hermetic.
"""

from __future__ import annotations

import json
import types

import pytest

import agent.claude_code_client as cc
from agent.transports.anthropic import AnthropicTransport


# --- prompt / system serialization -----------------------------------------

def test_single_user_message_passes_through():
    msgs = [{"role": "user", "content": "Hello there"}]
    assert cc._serialize_messages(msgs) == "Hello there"


def test_multi_turn_history_is_serialized_as_transcript():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "get_weather",
             "input": {"city": "Paris"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "sunny"},
        ]},
    ]
    out = cc._serialize_messages(msgs)
    assert "User: weather?" in out
    assert "called tool `get_weather`" in out
    assert "tool result for t1" in out and "sunny" in out


def test_system_text_flattens_block_list():
    assert cc._system_text([{"type": "text", "text": "You are Hermes."}]) == "You are Hermes."
    assert cc._system_text("plain") == "plain"
    assert cc._system_text(None) == ""


def test_map_model_passthrough_and_default():
    assert cc._map_model("opus") == "opus"
    assert cc._map_model("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert cc._map_model("") == "sonnet"
    assert cc._map_model("auto") == "sonnet"


# --- stream-json parsing -----------------------------------------------------

def _stream(*objs):
    return "\n".join(json.dumps(o) for o in objs)


def test_parse_stream_text_only():
    out = _stream(
        {"type": "assistant", "message": {"model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "HERMES_PROBE_OK"}]}},
        {"type": "result", "subtype": "success", "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 5}},
    )
    msg = cc._parse_stream(out)
    assert msg.stop_reason == "end_turn"
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "HERMES_PROBE_OK"
    assert msg.usage.input_tokens == 3


def test_parse_stream_strips_mcp_prefix_and_sets_tool_use():
    out = _stream(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1",
             "name": "mcp__hermes__get_weather", "input": {"city": "Paris"}},
        ]}},
        # The harness marks the cut as error_max_turns; we must still report
        # tool_use, not treat it as a failure.
        {"type": "result", "subtype": "error_max_turns", "is_error": True,
            "stop_reason": "tool_use"},
    )
    msg = cc._parse_stream(out)
    assert msg.stop_reason == "tool_use"
    tool_blocks = [b for b in msg.content if b.type == "tool_use"]
    assert tool_blocks[0].name == "get_weather"  # mcp__hermes__ prefix stripped
    assert tool_blocks[0].input == {"city": "Paris"}


def test_parse_stream_raises_on_real_error_with_no_content():
    out = _stream(
        {"type": "result", "subtype": "error_during_execution", "is_error": True},
    )
    with pytest.raises(cc.ClaudeCodeError):
        cc._parse_stream(out)


# --- normalization integration ----------------------------------------------

def test_parsed_message_normalizes_through_anthropic_transport():
    out = _stream(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1",
             "name": "mcp__hermes__get_weather", "input": {"city": "Paris"}},
        ]}},
        {"type": "result", "stop_reason": "tool_use"},
    )
    msg = cc._parse_stream(out)
    norm = AnthropicTransport().normalize_response(msg)
    assert norm.content == "Checking."
    assert norm.finish_reason == "tool_calls"
    assert norm.tool_calls[0].name == "get_weather"
    assert json.loads(norm.tool_calls[0].arguments) == {"city": "Paris"}


# --- end-to-end client.create() with a mocked CLI ----------------------------

def test_client_create_invokes_claude_cli(monkeypatch):
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        stdout = _stream(
            {"type": "assistant", "message": {"model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "HERMES_PROBE_OK"}]}},
            {"type": "result", "subtype": "success", "stop_reason": "end_turn"},
        )
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(cc, "resolve_claude_command", lambda: "/usr/bin/claude")
    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    client = cc.ClaudeCodeClient(model="sonnet")
    msg = client.messages.create(
        model="sonnet",
        system="You are Hermes.",
        messages=[{"role": "user", "content": "ping"}],
        tools=[{"name": "get_weather", "description": "w",
                "input_schema": {"type": "object", "properties": {}}}],
    )

    assert msg.content[0].text == "HERMES_PROBE_OK"
    cmd = captured["cmd"]
    # Compliance + correctness of the invocation:
    assert cmd[0] == "/usr/bin/claude"
    assert "-p" in cmd and "--max-turns" in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert "--mcp-config" in cmd  # Hermes tools exposed via the MCP bridge
    assert any(a == "mcp__hermes__get_weather" for a in cmd)
    assert captured["input"] == "ping"  # prompt piped via stdin


def test_client_create_errors_clearly_without_claude_cli(monkeypatch):
    monkeypatch.setattr(cc, "resolve_claude_command", lambda: None)
    client = cc.ClaudeCodeClient()
    with pytest.raises(cc.ClaudeCodeError, match="claude` CLI was not found"):
        client.messages.create(messages=[{"role": "user", "content": "hi"}])


# --- bridge ------------------------------------------------------------------

def test_bridge_loads_anthropic_tools_as_mcp_descriptors(tmp_path):
    import agent.claude_code_bridge as br

    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps([
        {"name": "get_weather", "description": "w",
         "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
        {"name": "noschema"},
    ]), encoding="utf-8")

    loaded = br._load_tools(str(tools_file))
    assert [t["name"] for t in loaded] == ["get_weather", "noschema"]
    assert loaded[0]["inputSchema"]["properties"]["city"]["type"] == "string"
    # missing schema gets a permissive default
    assert loaded[1]["inputSchema"]["type"] == "object"


# --- runtime resolution does NOT fall back to direct-API anthropic ----------

def test_runtime_resolution_stays_on_claude_code(monkeypatch):
    # Even with a Claude OAuth token in env, requesting claude-code must NOT
    # resolve to the direct-API `anthropic` provider (that would hit
    # api.anthropic.com). Auth is owned by the CLI.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-be-ignored")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ignored")
    from hermes_cli.runtime_provider import resolve_runtime_provider

    rt = resolve_runtime_provider(requested="claude-code")
    assert rt["provider"] == "claude-code"
    assert rt["api_mode"] == "anthropic_messages"
    assert rt["base_url"] == ""
    # sentinel key only — never sent anywhere
    assert rt["api_key"] == "claude-code"


def test_provider_alias_no_longer_maps_to_anthropic():
    from hermes_cli.providers import normalize_provider

    assert normalize_provider("claude-code") == "claude-code"
    assert normalize_provider("claude-max") == "claude-code"
    # plain "claude" still means the direct-API Anthropic provider
    assert normalize_provider("claude") == "anthropic"


# --- hardened paths (stress-test-driven) ------------------------------------

def test_parallel_tool_use_preserves_order_and_ids():
    out = _stream(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Two things."},
            {"type": "tool_use", "id": "toolu_A",
             "name": "mcp__hermes__read_file", "input": {"path": "a"}},
            {"type": "tool_use", "id": "toolu_B",
             "name": "mcp__hermes__read_file", "input": {"path": "b"}},
        ]}},
        {"type": "result", "stop_reason": "tool_use"},
    )
    msg = cc._parse_stream(out)
    tools = [b for b in msg.content if b.type == "tool_use"]
    assert [t.id for t in tools] == ["toolu_A", "toolu_B"]  # order preserved
    assert [t.input["path"] for t in tools] == ["a", "b"]
    assert all(t.name == "read_file" for t in tools)  # prefix stripped


def test_parse_stream_tolerates_malformed_and_unknown_lines():
    out = (
        "this is not json\n"
        + json.dumps({"type": "system", "subtype": "init", "tools": []}) + "\n"
        + json.dumps({"type": "stream_event", "event": {"type": "ping"}}) + "\n"
        + json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "ok"}]}}) + "\n"
        + "{ truncated json"
    )
    msg = cc._parse_stream(out)
    assert [b.text for b in msg.content] == ["ok"]
    assert msg.saw_result is False  # no terminal result in this stream


def test_parse_stream_empty_turn_with_result_is_legitimate():
    # An assistant that completed a turn with nothing to add (result present,
    # no content) is valid — must NOT be treated as a failure.
    msg = cc._parse_stream(_stream(
        {"type": "result", "subtype": "success", "stop_reason": "end_turn"}))
    assert msg.content == []
    assert msg.saw_result is True


def test_parse_stream_empty_output_marks_no_result():
    msg = cc._parse_stream("")
    assert msg.content == []
    assert msg.saw_result is False


def _fake_proc(stdout="", stderr="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_client_create_raises_on_silent_empty_output(monkeypatch):
    # exit 0 but no stdout at all — must raise, not return an empty turn.
    monkeypatch.setattr(cc, "resolve_claude_command", lambda: "/usr/bin/claude")
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda *a, **k: _fake_proc(stdout="", returncode=0))
    with pytest.raises(cc.ClaudeCodeError, match="no usable output"):
        cc.ClaudeCodeClient().messages.create(
            messages=[{"role": "user", "content": "hi"}])


def test_client_create_surfaces_stderr_on_failure(monkeypatch):
    monkeypatch.setattr(cc, "resolve_claude_command", lambda: "/usr/bin/claude")
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda *a, **k: _fake_proc(stderr="boom happened", returncode=2))
    with pytest.raises(cc.ClaudeCodeError, match="boom happened"):
        cc.ClaudeCodeClient().messages.create(
            messages=[{"role": "user", "content": "hi"}])


def test_client_create_raises_clear_error_on_timeout(monkeypatch):
    import subprocess as _sp

    def _timeout(*a, **k):
        raise _sp.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr(cc, "resolve_claude_command", lambda: "/usr/bin/claude")
    monkeypatch.setattr(cc.subprocess, "run", _timeout)
    with pytest.raises(cc.ClaudeCodeError, match="timed out"):
        cc.ClaudeCodeClient().messages.create(
            messages=[{"role": "user", "content": "hi"}])


def test_client_create_cleans_tempdir_on_cancel(monkeypatch, tmp_path):
    import glob
    import tempfile as _tf

    def _cancel(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cc, "resolve_claude_command", lambda: "/usr/bin/claude")
    monkeypatch.setattr(cc.subprocess, "run", _cancel)
    before = set(glob.glob(_tf.gettempdir() + "/hermes-claude-code-*"))
    with pytest.raises(KeyboardInterrupt):
        cc.ClaudeCodeClient().messages.create(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "d",
                    "input_schema": {"type": "object", "properties": {}}}],
        )
    after = set(glob.glob(_tf.gettempdir() + "/hermes-claude-code-*"))
    assert after == before  # temp dir cleaned up even on cancellation


def test_long_prompt_is_piped_via_stdin_not_argv(monkeypatch):
    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        captured["input"] = k.get("input")
        return _fake_proc(stdout=_stream(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
            {"type": "result", "stop_reason": "end_turn"}))

    monkeypatch.setattr(cc, "resolve_claude_command", lambda: "/usr/bin/claude")
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    big = "X" * 300_000
    cc.ClaudeCodeClient().messages.create(
        messages=[{"role": "user", "content": big}])
    # The big prompt must travel via stdin; argv must stay small (no OS limit).
    assert captured["input"] == big
    assert all(big not in str(arg) for arg in captured["cmd"])


def test_context_window_pinned_to_subscription_limit(monkeypatch):
    from agent.model_metadata import get_model_context_length as g

    monkeypatch.delenv("HERMES_CLAUDE_CODE_CONTEXT_TOKENS", raising=False)
    # Bare aliases and 4.6 ids would resolve to 256K/1M via name patterns;
    # claude-code must pin to the real `claude -p` window so compaction fires.
    assert g("sonnet", provider="claude-code") == 200_000
    assert g("claude-sonnet-4-6", provider="claude-code") == 200_000
    # Other providers are unaffected.
    assert g("sonnet", provider="anthropic") != 200_000
    # Env override wins.
    monkeypatch.setenv("HERMES_CLAUDE_CODE_CONTEXT_TOKENS", "150000")
    assert g("opus", provider="claude-code") == 150_000
