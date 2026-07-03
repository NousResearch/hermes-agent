"""Tests for the Claude Agent SDK inference/agent backend.

These never spawn the real `claude` CLI: `_get_claude_agent_sdk` is
monkeypatched to a fake module whose `query()` yields canned messages.
The critical property under test is that `create_claude_agent_message`
returns an object the real `AnthropicTransport.normalize_response`
consumes unchanged.
"""

import asyncio
import types

import pytest

from agent import claude_agent_sdk_adapter as adp


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------
class _FakeOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeThinkingBlock:
    def __init__(self, thinking):
        self.type = "thinking"
        self.thinking = thinking


class _FakeToolUseBlock:
    def __init__(self, id, name, input):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class _FakeAssistantMessage:
    def __init__(self, content, parent_tool_use_id=None):
        self.content = content
        self.parent_tool_use_id = parent_tool_use_id


class _FakeHookMatcher:
    def __init__(self, matcher=None, hooks=None):
        self.matcher = matcher
        self.hooks = hooks or []


class _FakeResultMessage:
    def __init__(self, *, result, session_id="sess-1", is_error=False,
                 subtype="success", usage=None, total_cost_usd=0.01, errors=None,
                 stop_reason="end_turn"):
        self.result = result
        self.session_id = session_id
        self.is_error = is_error
        self.subtype = subtype
        self.stop_reason = stop_reason
        self.usage = usage or {"input_tokens": 10, "output_tokens": 5,
                               "cache_read_input_tokens": 2, "cache_creation_input_tokens": 0}
        self.total_cost_usd = total_cost_usd
        self.errors = errors


class _FakeUserMessage:
    def __init__(self, content):
        self.content = content


class _FakeToolResultBlock:
    def __init__(self, tool_use_id, content, is_error=None):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class _FakeStreamEvent:
    def __init__(self, event, parent_tool_use_id=None):
        self.uuid = "ev-1"
        self.session_id = "sess-1"
        self.event = event
        self.parent_tool_use_id = parent_tool_use_id


# Background-task / dynamic-workflow progress messages (Task* system messages).
# Named to match the real SDK classes: the adapter dispatches on
# type(message).__name__, mirroring how it matches the real SDK's Task* types.
class TaskStartedMessage:
    def __init__(self, task_id, description, task_type=None):
        self.task_id = task_id
        self.description = description
        self.task_type = task_type


class TaskProgressMessage:
    def __init__(self, task_id, description="", last_tool_name=None):
        self.task_id = task_id
        self.description = description
        self.last_tool_name = last_tool_name


class TaskUpdatedMessage:
    def __init__(self, task_id, status):
        self.task_id = task_id
        self.status = status


class TaskNotificationMessage:
    def __init__(self, task_id, status, summary="", output_file=""):
        self.task_id = task_id
        self.status = status
        self.summary = summary
        self.output_file = output_file


def _make_fake_sdk(*, assistant_blocks, result_kwargs, capture=None, stream_events=None):
    """Build a fake claude_agent_sdk module."""
    async def _query(*, prompt, options):
        if capture is not None:
            capture["prompt"] = prompt
            capture["options"] = options
        for ev in stream_events or []:
            yield ev
        yield _FakeAssistantMessage(assistant_blocks)
        yield _FakeResultMessage(**result_kwargs)

    fake = types.SimpleNamespace(
        ClaudeAgentOptions=_FakeOptions,
        AssistantMessage=_FakeAssistantMessage,
        ResultMessage=_FakeResultMessage,
        TextBlock=_FakeTextBlock,
        ThinkingBlock=_FakeThinkingBlock,
        ToolUseBlock=_FakeToolUseBlock,
        HookMatcher=_FakeHookMatcher,
        StreamEvent=_FakeStreamEvent,
        UserMessage=_FakeUserMessage,
        ToolResultBlock=_FakeToolResultBlock,
        TaskStartedMessage=TaskStartedMessage,
        TaskProgressMessage=TaskProgressMessage,
        TaskUpdatedMessage=TaskUpdatedMessage,
        TaskNotificationMessage=TaskNotificationMessage,
        query=_query,
        tool=lambda name, desc, schema: (lambda fn: {"name": name, "schema": schema, "fn": fn}),
        create_sdk_mcp_server=lambda *, name, version, tools: {"name": name, "tools": tools},
    )
    return fake


def _agent(**over):
    base = dict(
        _claude_agent_sdk_mode="inference",
        _claude_agent_sdk_settings={"mode": "inference", "permission_mode": "dontAsk",
                                    "max_turns": 1, "allowed_tools": []},
        _claude_sdk_session_id=None,
        model="claude-opus-4-20250514",
        provider="anthropic",
        log_prefix="",
        _anthropic_api_key="sk-ant-api-xxxxx",
        _anthropic_base_url="https://api.anthropic.com",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# resolve_claude_agent_sdk_settings
# ---------------------------------------------------------------------------
def test_settings_disabled_for_non_anthropic():
    assert adp.resolve_claude_agent_sdk_settings("openai") is None
    assert adp.resolve_claude_agent_sdk_settings(None) is None


def test_settings_string_mode(monkeypatch):
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config_readonly",
                        lambda: {"model": {"claude_agent_sdk": "hybrid"}})
    s = adp.resolve_claude_agent_sdk_settings("anthropic")
    assert s is not None and s["mode"] == "hybrid"
    # delegate/hybrid default to bypassPermissions
    assert s["permission_mode"] == "bypassPermissions"


def test_settings_dict_mode(monkeypatch):
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config_readonly",
                        lambda: {"model": {"claude_agent_sdk": {"mode": "delegate", "max_turns": 7}}})
    s = adp.resolve_claude_agent_sdk_settings("anthropic")
    assert s["mode"] == "delegate" and s["max_turns"] == 7


def test_settings_off_values(monkeypatch):
    import hermes_cli.config as cfg
    for val in ["auto", "", "off", {"enabled": False, "mode": "inference"}]:
        monkeypatch.setattr(cfg, "load_config_readonly",
                            lambda v=val: {"model": {"claude_agent_sdk": v}})
        assert adp.resolve_claude_agent_sdk_settings("anthropic") is None

    monkeypatch.setattr(cfg, "load_config_readonly", lambda: {"model": {}})
    assert adp.resolve_claude_agent_sdk_settings("anthropic") is None


# ---------------------------------------------------------------------------
# build_auth_env
# ---------------------------------------------------------------------------
def test_auth_env_oauth_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    env = adp.build_auth_env(_agent(_anthropic_api_key="sk-ant-oat01-abc"))
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-abc"
    assert "ANTHROPIC_API_KEY" not in env


def test_auth_env_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    env = adp.build_auth_env(_agent(_anthropic_api_key="sk-ant-api03-xyz"))
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-api03-xyz"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_auth_env_base_url(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    env = adp.build_auth_env(_agent(_anthropic_base_url="https://gw.example.com/anthropic"))
    assert env["ANTHROPIC_BASE_URL"] == "https://gw.example.com/anthropic"


# ---------------------------------------------------------------------------
# create_claude_agent_message → AnthropicTransport.normalize_response
# ---------------------------------------------------------------------------
def _api_kwargs():
    return {
        "model": "claude-opus-4-20250514",
        "system": "You are Hermes.",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [],
        "max_tokens": 1024,
    }


def test_inference_message_normalizes(monkeypatch):
    fake = _make_fake_sdk(
        assistant_blocks=[_FakeTextBlock("4")],
        result_kwargs={"result": "2 + 2 = 4", "session_id": "s-42"},
    )
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent()

    msg = adp.create_claude_agent_message(agent, _api_kwargs())

    # Shape the loop expects.
    assert msg.stop_reason == "end_turn"
    assert any(getattr(b, "type", None) == "text" and b.text == "2 + 2 = 4" for b in msg.content)
    # Session id persisted for the next turn's resume.
    assert agent._claude_sdk_session_id == "s-42"

    # The real Anthropic transport must consume it unchanged.
    from agent.transports.anthropic import AnthropicTransport
    normalized = AnthropicTransport().normalize_response(msg)
    assert normalized.content == "2 + 2 = 4"
    assert normalized.finish_reason == "stop"
    assert normalized.tool_calls is None

    # Cache stats read from our usage object.
    stats = AnthropicTransport().extract_cache_stats(msg)
    assert stats == {"cached_tokens": 2, "creation_tokens": 0}


def test_inference_passes_last_user_message_and_system(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(
        assistant_blocks=[_FakeTextBlock("ok")],
        result_kwargs={"result": "ok"},
        capture=capture,
    )
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    kw = _api_kwargs()
    kw["messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "prior"},
        {"role": "user", "content": [{"type": "text", "text": "the real question"}]},
    ]
    adp.create_claude_agent_message(_agent(), kw)
    assert capture["prompt"] == "the real question"
    assert capture["options"].system_prompt == "You are Hermes."
    assert capture["options"].max_turns == 1
    assert capture["options"].tools == []


def test_error_result_raises(monkeypatch):
    fake = _make_fake_sdk(
        assistant_blocks=[],
        result_kwargs={"result": None, "is_error": True, "subtype": "error_during_execution",
                       "errors": ["boom"]},
    )
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    with pytest.raises(RuntimeError, match="boom"):
        adp.create_claude_agent_message(_agent(), _api_kwargs())


def test_hybrid_builds_mcp_and_allowed_tools(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(
        assistant_blocks=[_FakeTextBlock("done")],
        result_kwargs={"result": "done"},
        capture=capture,
    )
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(
        _claude_agent_sdk_mode="hybrid",
        _claude_agent_sdk_settings={"mode": "hybrid", "permission_mode": "bypassPermissions",
                                    "max_turns": 5, "allowed_tools": []},
    )
    kw = _api_kwargs()
    kw["tools"] = [{"name": "read_file", "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]
    adp.create_claude_agent_message(agent, kw)
    opts = capture["options"]
    assert "hermes" in opts.mcp_servers
    # Hermes MCP tools + Claude's orchestration built-ins (so workflows work).
    assert opts.allowed_tools == ["mcp__hermes__read_file"] + list(adp._SDK_ORCHESTRATION_TOOLS)
    assert opts.tools == list(adp._SDK_ORCHESTRATION_TOOLS)
    assert "Workflow" in opts.tools and "Agent" in opts.tools
    assert opts.max_turns == 5


def test_hybrid_prefers_agent_tools_and_strips_oauth_prefix(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(
        assistant_blocks=[_FakeTextBlock("done")],
        result_kwargs={"result": "done"},
        capture=capture,
    )
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    # agent.tools is OpenAI-format; the OAuth wire prefixes names with mcp__.
    agent = _agent(
        _claude_agent_sdk_mode="hybrid",
        _claude_agent_sdk_settings={"mode": "hybrid", "permission_mode": "bypassPermissions",
                                    "max_turns": 5, "allowed_tools": []},
        tools=[{"type": "function", "function": {
            "name": "mcp__read_file", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
    )
    adp.create_claude_agent_message(agent, _api_kwargs())
    # Registry name is stripped clean; the fully-qualified MCP name uses it.
    assert capture["options"].allowed_tools == ["mcp__hermes__read_file"] + list(adp._SDK_ORCHESTRATION_TOOLS)
    assert "hermes" in capture["options"].mcp_servers


def test_hybrid_enables_workflow_and_subagent_builtins(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok"}, capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(
        _claude_agent_sdk_mode="hybrid",
        _claude_agent_sdk_settings={"mode": "hybrid", "permission_mode": "bypassPermissions",
                                    "max_turns": 24, "allowed_tools": []},
    )
    adp.create_claude_agent_message(agent, _api_kwargs())
    # Workflow (dynamic workflows) + Agent/Task (subagents) are enabled built-ins.
    assert "Workflow" in capture["options"].tools
    assert "Agent" in capture["options"].tools


def test_hybrid_builtin_tools_override_can_strip(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok"}, capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(
        _claude_agent_sdk_mode="hybrid",
        # builtin_tools: [] restores the legacy strip-everything behaviour.
        _claude_agent_sdk_settings={"mode": "hybrid", "permission_mode": "bypassPermissions",
                                    "max_turns": 24, "allowed_tools": [], "builtin_tools": []},
    )
    adp.create_claude_agent_message(agent, _api_kwargs())
    assert capture["options"].tools == []


def test_delegate_uses_builtin_tools(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(
        assistant_blocks=[_FakeTextBlock("edited")],
        result_kwargs={"result": "edited files"},
        capture=capture,
    )
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(
        _claude_agent_sdk_mode="delegate",
        _claude_agent_sdk_settings={"mode": "delegate", "permission_mode": "bypassPermissions",
                                    "max_turns": 12, "allowed_tools": []},
    )
    msg = adp.create_claude_agent_message(agent, _api_kwargs())
    opts = capture["options"]
    # delegate does NOT strip built-ins (no tools=[] override) and has no MCP.
    assert getattr(opts, "mcp_servers", None) is None
    assert opts.permission_mode == "bypassPermissions"
    assert opts.max_turns == 12
    from agent.transports.anthropic import AnthropicTransport
    assert AnthropicTransport().normalize_response(msg).content == "edited files"


def test_missing_sdk_raises_friendly(monkeypatch):
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: None)
    with pytest.raises(RuntimeError, match="claude-agent-sdk is not installed"):
        adp.create_claude_agent_message(_agent(), _api_kwargs())


# ---------------------------------------------------------------------------
# Deepened hybrid execution, guardrails, budget, refusal mapping
# ---------------------------------------------------------------------------
def test_settings_budget_and_disallowed(monkeypatch):
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config_readonly", lambda: {
        "model": {"claude_agent_sdk": {"mode": "hybrid", "max_budget_usd": 1.5,
                                       "disallowed_tools": ["Bash"]}}})
    s = adp.resolve_claude_agent_sdk_settings("anthropic")
    assert s["max_budget_usd"] == 1.5
    assert s["disallowed_tools"] == ["Bash"]


def test_hybrid_handler_routes_through_invoke_tool(monkeypatch):
    import agent.agent_runtime_helpers as arh
    calls = {}

    def fake_invoke(agent, name, args, task_id, tool_call_id=None):
        calls["name"] = name
        calls["args"] = args
        return "a plain result from a hermes tool, long enough to exceed the wrap threshold"

    monkeypatch.setattr(arh, "invoke_tool", fake_invoke)
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("x")], result_kwargs={"result": "x"})
    # web_search is an untrusted tool → result must be wrapped.
    agent = _agent(tools=[{"type": "function", "function": {
        "name": "web_search", "description": "s", "parameters": {"type": "object"}}}])
    server, allowed = adp._build_hybrid_mcp_server(fake, agent, agent.tools)
    assert allowed == ["mcp__hermes__web_search"]

    handler = server["tools"][0]["fn"]
    out = asyncio.run(handler({"q": "hi"}))
    assert calls["name"] == "web_search"          # routed through invoke_tool
    assert "<untrusted_tool_result" in out["content"][0]["text"]
    assert out["is_error"] is False


def test_hybrid_handler_guardrail_denies(monkeypatch):
    import agent.agent_runtime_helpers as arh
    ran = {"invoked": False}

    def fake_invoke(*a, **k):
        ran["invoked"] = True
        return "should not run"

    monkeypatch.setattr(arh, "invoke_tool", fake_invoke)
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("x")], result_kwargs={"result": "x"})
    guard = types.SimpleNamespace(
        before_call=lambda n, a: types.SimpleNamespace(allows_execution=False, message="blocked by policy"))
    agent = _agent(
        _tool_guardrails=guard,
        tools=[{"type": "function", "function": {"name": "read_file", "description": "r",
                                                 "parameters": {"type": "object"}}}])
    server, _ = adp._build_hybrid_mcp_server(fake, agent, agent.tools)
    out = asyncio.run(server["tools"][0]["fn"]({"path": "x"}))
    assert out["is_error"] is True
    assert "blocked by policy" in out["content"][0]["text"]
    assert ran["invoked"] is False                # guardrail blocked before execution


def test_guardrail_hook_denies_and_allows():
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("x")], result_kwargs={"result": "x"})
    guard = types.SimpleNamespace(
        before_call=lambda n, a: types.SimpleNamespace(allows_execution=(n != "Bash"), message="no bash"))
    hooks = adp._build_guardrail_hook(fake, _agent(_tool_guardrails=guard))
    cb = hooks["PreToolUse"][0].hooks[0]
    denied = asyncio.run(cb({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, "tid", None))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    # hybrid mcp name is stripped to the registry name before the guardrail check
    allowed = asyncio.run(cb({"tool_name": "mcp__hermes__Read", "tool_input": {}}, "tid", None))
    assert allowed == {}


def test_delegate_passes_hooks_and_budget(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("done")],
                          result_kwargs={"result": "done"}, capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(
        _claude_agent_sdk_mode="delegate",
        _claude_agent_sdk_settings={"mode": "delegate", "permission_mode": "bypassPermissions",
                                    "max_turns": 10, "allowed_tools": [],
                                    "disallowed_tools": ["Bash(rm *)"], "max_budget_usd": 2.5})
    adp.create_claude_agent_message(agent, _api_kwargs())
    opts = capture["options"]
    assert "PreToolUse" in opts.hooks
    assert opts.max_budget_usd == 2.5
    assert opts.disallowed_tools == ["Bash(rm *)"]


def test_refusal_maps_to_content_filter(monkeypatch):
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("")],
                          result_kwargs={"result": "I can't help with that", "stop_reason": "refusal"})
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    msg = adp.create_claude_agent_message(_agent(), _api_kwargs())
    assert msg.stop_reason == "refusal"
    from agent.transports.anthropic import AnthropicTransport
    assert AnthropicTransport().normalize_response(msg).finish_reason == "content_filter"


# ---------------------------------------------------------------------------
# Streaming bridge (include_partial_messages → live progress callbacks)
# ---------------------------------------------------------------------------
def _streaming_agent(**over):
    """Agent double with live-progress consumers attached."""
    seen = {"text": [], "tools": [], "gen": [], "reasoning": [], "flushes": 0,
            "start_cb": [], "complete_cb": []}

    def _delta_cb(text):
        if text is None:
            seen["flushes"] += 1
        else:
            seen["text"].append(text)

    def _tool_progress(ev, name, preview, args):
        seen["tools"].append((ev, name, preview, args))

    def _tool_start(tc_id, name, args):
        seen["start_cb"].append((tc_id, name, args))

    def _tool_complete(tc_id, name, args, result):
        seen["complete_cb"].append((tc_id, name, args, result))

    agent = _agent(
        stream_delta_callback=_delta_cb,
        _has_stream_consumers=lambda: True,
        _fire_stream_delta=lambda t: seen["text"].append(t),
        _fire_reasoning_delta=lambda t: seen["reasoning"].append(t),
        _fire_tool_gen_started=lambda name: seen["gen"].append(name),
        tool_progress_callback=_tool_progress,
        tool_start_callback=_tool_start,
        tool_complete_callback=_tool_complete,
        reasoning_callback=lambda t: None,
        **over,
    )
    return agent, seen


def _stream_events():
    return [
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": "Hel"}}),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": "lo"}}),
        _FakeStreamEvent({"type": "content_block_start",
                          "content_block": {"type": "tool_use", "name": "mcp__hermes__terminal",
                                            "id": "tu-99"}}),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "input_json_delta", "partial_json": '{"command": "gh '}}),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "input_json_delta", "partial_json": 'pr list"}'}}),
        _FakeStreamEvent({"type": "content_block_stop"}),
        # SDK executes the tool and echoes its result back into the loop.
        _FakeUserMessage([_FakeToolResultBlock("tu-99", "PR list output")]),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
        # Subagent event — must be ignored (would interleave with top-level).
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": "SUBAGENT"}},
                         parent_tool_use_id="tu-1"),
    ]


def test_stream_bridge_fires_live_callbacks(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("Hello")],
                          result_kwargs={"result": "Hello"},
                          capture=capture, stream_events=_stream_events())
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent, seen = _streaming_agent()

    msg = adp.create_claude_agent_message(agent, _api_kwargs())

    # Live progress reached the callbacks…
    assert seen["text"] == ["Hel", "lo"]
    assert seen["reasoning"] == ["hmm"]
    # …tool start surfaced with the mcp__ prefix stripped, via BOTH channels:
    # the gen spinner at block start, and tool.started at block stop carrying
    # the ACTUAL command in preview + args (not just the tool name).
    assert seen["gen"] == ["terminal"]
    # tool.started carries the ACTUAL command; tool.completed closes the bubble
    # (parity with the native executor, so CLI/messaging surfaces resolve).
    started = [row for row in seen["tools"] if row[0] == "tool.started"]
    completed = [row for row in seen["tools"] if row[0] == "tool.completed"]
    assert len(started) == 1 and len(completed) == 1
    ev, name, preview, args = started[0]
    assert name == "terminal"
    assert args == {"command": "gh pr list"}
    assert "gh pr list" in (preview or "")
    assert completed[0][1] == "terminal"
    # The desktop/TUI gateway channel got the authoritative start (id + args)
    # and the completion resolving the chip.
    assert seen["start_cb"] == [("tu-99", "terminal", {"command": "gh pr list"})]
    assert seen["complete_cb"] == [("tu-99", "terminal", {"command": "gh pr list"}, "PR list output")]
    # The open display segment was flushed before tool chrome.
    assert seen["flushes"] == 1
    # Streaming was requested from the SDK.
    assert capture["options"].include_partial_messages is True
    # …and the final collected message is unaffected by stream events.
    assert msg.content[-1].text == "Hello"


def test_stream_events_ignored_without_consumers(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("Hi")],
                          result_kwargs={"result": "Hi"},
                          capture=capture, stream_events=_stream_events())
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)

    msg = adp.create_claude_agent_message(_agent(), _api_kwargs())

    # No consumers → partial messages not requested, result still clean.
    assert not hasattr(capture["options"], "include_partial_messages")
    assert msg.content[-1].text == "Hi"


# ---------------------------------------------------------------------------
# Working directory + resume gating + thinking/effort passthrough
# ---------------------------------------------------------------------------
def test_cwd_anchors_to_agent_runtime_cwd(monkeypatch, tmp_path):
    from agent import runtime_cwd
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok", "session_id": "s-1"},
                          capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: tmp_path)

    agent = _agent()
    adp.create_claude_agent_message(agent, _api_kwargs())

    # The SDK session is anchored to the agent's logical cwd (project dir),
    # not the backend process cwd — in ALL modes, including inference.
    assert capture["options"].cwd == str(tmp_path)
    # …and the session id is remembered together with its directory.
    assert agent._claude_sdk_session_id == "s-1"
    assert agent._claude_sdk_session_cwd == str(tmp_path)


def test_resume_only_within_same_cwd(monkeypatch, tmp_path):
    from agent import runtime_cwd
    monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: tmp_path)

    # Same dir → resume passed through.
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok"}, capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(_claude_sdk_session_id="sess-prev", _claude_sdk_session_cwd=str(tmp_path))
    adp.create_claude_agent_message(agent, _api_kwargs())
    assert capture["options"].resume == "sess-prev"

    # Different dir → session must NOT resume (transcripts are keyed by cwd).
    capture2 = {}
    fake2 = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                           result_kwargs={"result": "ok"}, capture=capture2)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake2)
    agent2 = _agent(_claude_sdk_session_id="sess-prev", _claude_sdk_session_cwd="/somewhere/else")
    adp.create_claude_agent_message(agent2, _api_kwargs())
    assert not hasattr(capture2["options"], "resume")


def test_thinking_and_effort_passthrough(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok"}, capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)

    kw = _api_kwargs()
    kw["thinking"] = {"type": "adaptive", "display": "summarized"}
    kw["output_config"] = {"effort": "low"}
    adp.create_claude_agent_message(_agent(), kw)

    # Hermes' reasoning settings reach the SDK — the CLI must not pick its
    # own thinking depth ("low" behaving like "max").
    assert capture["options"].thinking == {"type": "adaptive", "display": "summarized"}
    assert capture["options"].effort == "low"


def test_thinking_disabled_when_reasoning_off(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok"}, capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)

    adp.create_claude_agent_message(_agent(), _api_kwargs())  # no thinking kwarg

    # Reasoning off in Hermes → explicitly disabled at the SDK, not left to
    # the CLI default (adaptive).
    assert capture["options"].thinking == {"type": "disabled"}


# ---------------------------------------------------------------------------
# Interrupt + option passthroughs + strict MCP (full SDK-surface coverage)
# ---------------------------------------------------------------------------
def test_interrupt_stops_sdk_turn(monkeypatch):
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("never returned")],
                          result_kwargs={"result": "never"},
                          stream_events=_stream_events())
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(_interrupt_requested=True)

    with pytest.raises(InterruptedError):
        adp.create_claude_agent_message(agent, _api_kwargs())


def test_advanced_option_passthroughs(monkeypatch):
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config_readonly", lambda: {"model": {"claude_agent_sdk": {
        "mode": "delegate",
        "cli_path": "/opt/claude/bin/claude",
        "betas": ["context-1m-2025-08-07"],
        "fallback_model": "claude-sonnet-4-6",
        "add_dirs": ["/data"],
        "extra_args": {"verbose": None},
    }}})
    settings = adp.resolve_claude_agent_sdk_settings("anthropic")
    assert settings["cli_path"] == "/opt/claude/bin/claude"
    assert settings["betas"] == ["context-1m-2025-08-07"]

    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok"}, capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(_claude_agent_sdk_mode="delegate", _claude_agent_sdk_settings=settings)
    adp.create_claude_agent_message(agent, _api_kwargs())

    opts = capture["options"]
    assert opts.cli_path == "/opt/claude/bin/claude"
    assert opts.betas == ["context-1m-2025-08-07"]
    assert opts.fallback_model == "claude-sonnet-4-6"
    assert opts.add_dirs == ["/data"]
    assert opts.extra_args == {"verbose": None}


def test_hybrid_pins_mcp_surface(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok"}, capture=capture)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)
    agent = _agent(
        _claude_agent_sdk_mode="hybrid",
        _claude_agent_sdk_settings={"mode": "hybrid", "permission_mode": "bypassPermissions",
                                    "max_turns": 24, "allowed_tools": []},
        tools=[{"type": "function", "function": {"name": "terminal", "description": "t",
                                                 "parameters": {"type": "object", "properties": {}}}}],
    )
    adp.create_claude_agent_message(agent, _api_kwargs())
    # No user/project MCP config may add tools behind Hermes' back.
    assert capture["options"].strict_mcp_config is True


# ---------------------------------------------------------------------------
# _classify_sdk_error — friendly guidance for CLI-not-found / start failures.
# Class names mirror claude_agent_sdk's real exception hierarchy so the
# name-based matching in the adapter is exercised the same way at runtime.
# ---------------------------------------------------------------------------
class _CLIConnectionError(Exception):
    pass


class _CLINotFoundError(_CLIConnectionError):
    def __init__(self, message="Claude Code not found", cli_path=None):
        super().__init__(message if not cli_path else f"{message}: {cli_path}")


class _ProcessError(Exception):
    def __init__(self, message, exit_code=None, stderr=None):
        self.exit_code = exit_code
        self.stderr = stderr
        suffix = f" (exit code: {exit_code})" if exit_code is not None else ""
        super().__init__(f"{message}{suffix}")


class _CLIJSONDecodeError(Exception):
    pass


def test_classify_cli_not_found():
    msg = adp._classify_sdk_error(_CLINotFoundError(cli_path="/usr/bin/claude"))
    assert msg is not None
    assert "could not find the `claude` CLI" in msg
    assert "PATH" in msg


def test_classify_process_error_surfaces_exit_code_and_stderr():
    exc = _ProcessError("Command failed", exit_code=1, stderr="boom: bad flag --nope")
    msg = adp._classify_sdk_error(exc)
    assert msg is not None
    assert "could not start the `claude` CLI" in msg
    assert "Exit code: 1" in msg
    assert "bad flag --nope" in msg


def test_classify_process_error_detects_auth_failure():
    exc = _ProcessError("Command failed", exit_code=1,
                        stderr="Invalid API key · Please run /login")
    msg = adp._classify_sdk_error(exc)
    assert msg is not None
    assert "authentication problem" in msg
    assert "/login" in msg


def test_classify_process_error_truncates_long_stderr():
    exc = _ProcessError("Command failed", exit_code=2, stderr="x" * 900)
    msg = adp._classify_sdk_error(exc)
    assert "…" in msg
    assert "x" * 900 not in msg


def test_classify_transport_errors():
    assert "transport error" in (adp._classify_sdk_error(_CLIJSONDecodeError("bad json")) or "")
    assert "transport error" in (adp._classify_sdk_error(_CLIConnectionError("no socket")) or "")


def test_classify_unknown_error_returns_none():
    assert adp._classify_sdk_error(ValueError("something else")) is None


# ---------------------------------------------------------------------------
# Subagent visibility — a nested SDK subagent (Task tool) surfaces its TOOL
# activity (marked with the subagent glyph) but never leaks its text/thinking
# into the top-level response.
# ---------------------------------------------------------------------------
def _subagent_stream_events():
    return [
        # Top-level agent invokes the Task tool (spawns a subagent).
        _FakeStreamEvent({"type": "content_block_start",
                          "content_block": {"type": "tool_use", "name": "Task", "id": "task-1"}}),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "input_json_delta",
                                    "partial_json": '{"description": "look it up"}'}}),
        _FakeStreamEvent({"type": "content_block_stop"}),
        # --- Subagent activity (parent_tool_use_id set on every event) ---
        # Subagent's own chatter — must NOT reach the main text stream.
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": "subagent thinking out loud"}},
                         parent_tool_use_id="task-1"),
        # Subagent runs a tool — SHOULD surface, marked as subagent work.
        _FakeStreamEvent({"type": "content_block_start",
                          "content_block": {"type": "tool_use",
                                            "name": "mcp__hermes__read_file", "id": "sub-tu-1"}},
                         parent_tool_use_id="task-1"),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "input_json_delta", "partial_json": '{"path": "a.py"}'}},
                         parent_tool_use_id="task-1"),
        _FakeStreamEvent({"type": "content_block_stop"}, parent_tool_use_id="task-1"),
        # Subagent's final assistant message — its text must stay out of result.
        _FakeAssistantMessage([_FakeTextBlock("subagent conclusion")],
                              parent_tool_use_id="task-1"),
    ]


def test_subagent_tool_activity_surfaced_and_marked(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("top-level answer")],
                          result_kwargs={"result": "top-level answer"},
                          capture=capture, stream_events=_subagent_stream_events())
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)

    # Progress callback that accepts the subagent kwarg (realistic consumer).
    got = {"progress": [], "starts": [], "text": []}
    agent = _agent(
        _has_stream_consumers=lambda: True,
        _fire_stream_delta=lambda t: got["text"].append(t),
        _fire_reasoning_delta=lambda t: got["text"].append(t),
        _fire_tool_gen_started=lambda name: None,
        stream_delta_callback=lambda t: None,
        reasoning_callback=lambda t: None,
        tool_progress_callback=lambda ev, name=None, preview=None, args=None, **kw:
            got["progress"].append((ev, name, kw.get("subagent"))),
        tool_start_callback=lambda tc_id, name, args: got["starts"].append((tc_id, name)),
    )

    msg = adp.create_claude_agent_message(agent, _api_kwargs())

    # Subagent's text/thinking never reached the main response stream…
    assert "subagent thinking out loud" not in "".join(got["text"])
    # …and never leaked into the collected top-level message.
    assert msg.content[-1].text == "top-level answer"
    assert "subagent conclusion" not in msg.content[-1].text

    # The Task invocation (top-level) surfaced unmarked; the subagent's tool
    # surfaced WITH the marker, through both channels.
    start_names = [n for _, n in got["starts"]]
    assert "Task" in start_names
    assert adp._SUBAGENT_TOOL_MARKER + "read_file" in start_names

    # The subagent flag rode along on the progress channel for the nested tool.
    sub_rows = [(name, flag) for ev, name, flag in got["progress"] if flag]
    assert (adp._SUBAGENT_TOOL_MARKER + "read_file", True) in sub_rows


def _interleaved_subagent_stream_events():
    """Two BACKGROUND subagents stream tools concurrently; their content-block
    events interleave under different parent_tool_use_ids (the default since
    Claude Code v2.1.198 runs subagents in the background)."""
    return [
        # Top-level agent invokes the Agent tool (spawns background subagents).
        _FakeStreamEvent({"type": "content_block_start",
                          "content_block": {"type": "tool_use", "name": "Agent", "id": "wf-1"}}),
        _FakeStreamEvent({"type": "content_block_stop"}),
        # Subagent A and B start tools INTERLEAVED.
        _FakeStreamEvent({"type": "content_block_start",
                          "content_block": {"type": "tool_use",
                                            "name": "mcp__hermes__read_file", "id": "a-tu"}},
                         parent_tool_use_id="wf-a"),
        _FakeStreamEvent({"type": "content_block_start",
                          "content_block": {"type": "tool_use",
                                            "name": "mcp__hermes__terminal", "id": "b-tu"}},
                         parent_tool_use_id="wf-b"),
        # Their input JSON arrives interleaved — must not cross-contaminate.
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "input_json_delta", "partial_json": '{"path":'}},
                         parent_tool_use_id="wf-a"),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "input_json_delta", "partial_json": '{"command":'}},
                         parent_tool_use_id="wf-b"),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "input_json_delta", "partial_json": ' "x.py"}'}},
                         parent_tool_use_id="wf-a"),
        _FakeStreamEvent({"type": "content_block_delta",
                          "delta": {"type": "input_json_delta", "partial_json": ' "ls"}'}},
                         parent_tool_use_id="wf-b"),
        _FakeStreamEvent({"type": "content_block_stop"}, parent_tool_use_id="wf-b"),
        _FakeStreamEvent({"type": "content_block_stop"}, parent_tool_use_id="wf-a"),
    ]


def test_concurrent_subagent_tool_streams_do_not_clobber(monkeypatch):
    capture = {}
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("done")],
                          result_kwargs={"result": "done"},
                          capture=capture,
                          stream_events=_interleaved_subagent_stream_events())
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)

    starts = []
    agent = _agent(
        _has_stream_consumers=lambda: True,
        _fire_stream_delta=lambda t: None,
        _fire_tool_gen_started=lambda name: None,
        stream_delta_callback=lambda t: None,
        tool_start_callback=lambda tc_id, name, args: starts.append((tc_id, name, args)),
    )
    adp.create_claude_agent_message(agent, _api_kwargs())

    by_id = {tc_id: (name, args) for tc_id, name, args in starts}
    # Each nested agent's tool resolved with ITS OWN args despite interleaving.
    assert by_id["a-tu"] == (adp._SUBAGENT_TOOL_MARKER + "read_file", {"path": "x.py"})
    assert by_id["b-tu"] == (adp._SUBAGENT_TOOL_MARKER + "terminal", {"command": "ls"})
    # The top-level Agent invocation surfaced unmarked.
    assert ("wf-1", "Agent", {}) in starts


# ---------------------------------------------------------------------------
# Dynamic workflows / background tasks — surfaced via Task* system messages
# (out-of-band from the assistant stream), rendered as a chip lifecycle.
# ---------------------------------------------------------------------------
def test_workflow_task_messages_surface_chip_lifecycle(monkeypatch):
    events = [
        TaskStartedMessage("t-1", "Audit routes for auth", task_type="workflow"),
        TaskProgressMessage("t-1", "Audit routes for auth", last_tool_name="Grep"),
        TaskNotificationMessage("t-1", "completed", summary="12 handlers, 2 missing auth"),
    ]
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("done")],
                          result_kwargs={"result": "done"}, stream_events=events)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)

    starts, completes, progress = [], [], []
    agent = _agent(
        _has_stream_consumers=lambda: True,
        _fire_stream_delta=lambda t: None,
        _fire_tool_gen_started=lambda name: None,
        stream_delta_callback=lambda t: None,
        tool_start_callback=lambda tc_id, name, args: starts.append((tc_id, name)),
        tool_progress_callback=lambda ev, name=None, preview=None, args=None, **kw:
            progress.append((ev, name, kw.get("subagent"))),
        tool_complete_callback=lambda tc_id, name, args, result: completes.append((tc_id, name, result)),
    )
    msg = adp.create_claude_agent_message(agent, _api_kwargs())

    label = adp._TASK_MARKER + "Audit routes for auth"
    # Task started → chip opened on both channels (id == task_id).
    assert ("t-1", label) in starts
    assert ("tool.started", label, True) in progress
    # Task notification → chip resolved with the workflow's summary.
    assert ("t-1", label, "12 handlers, 2 missing auth") in completes
    # The workflow's internal chatter never entered the top-level result.
    assert msg.content[-1].text == "done"


def test_workflow_task_failure_marks_result(monkeypatch):
    events = [
        TaskStartedMessage("t-9", "Migrate components"),
        TaskUpdatedMessage("t-9", "failed"),
    ]
    fake = _make_fake_sdk(assistant_blocks=[_FakeTextBlock("ok")],
                          result_kwargs={"result": "ok"}, stream_events=events)
    monkeypatch.setattr(adp, "_get_claude_agent_sdk", lambda: fake)

    completes = []
    agent = _agent(
        _has_stream_consumers=lambda: True,
        _fire_tool_gen_started=lambda name: None,
        tool_start_callback=lambda tc_id, name, args: None,
        tool_complete_callback=lambda tc_id, name, args, result: completes.append(result),
    )
    adp.create_claude_agent_message(agent, _api_kwargs())
    # A terminal TaskUpdated(failed) resolves the chip and marks the failure.
    assert completes and completes[0].startswith("[failed]")
