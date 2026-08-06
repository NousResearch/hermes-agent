"""Contract for the ``claude_agent_sdk`` whole-turn runtime.

Three things are being pinned down here:

1. The turn refuses to start unless the gate is on, the optional extra is
   installed, and the user is signed in — with an error that names the fix.
2. The SDK options carry Hermes' ownership boundary: the exact Hermes system
   prompt, no SDK built-ins, no second settings load, a pinned MCP toolset.
3. A Claude turn fires the same canonical Hermes callbacks and produces the
   same message shape as every other provider, exactly once.

``claude-agent-sdk`` is an optional extra, so a stand-in module is installed
when it is absent. The SDK message/block classes are re-declared locally
because the runtime dispatches on class name — that is what lets the projector
behave identically with and without the real package.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent import claude_runtime
from agent.claude_runtime import (
    ClaudeEventProjector,
    build_claude_agent_options,
    claude_runtime_preflight,
    display_tool_name,
    run_claude_agent_sdk_turn,
)
from agent.transports.claude_tool_bridge import MCP_SERVER_NAME, bridged_allowed_tools
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# SDK stand-in (used only when the optional extra is not installed)
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolAnnotations:
    readOnlyHint: bool = False


@dataclass
class _FakeSdkTool:
    name: str
    description: str
    input_schema: dict
    handler: Any
    annotations: Any = None


def _fake_tool(name, description, input_schema, annotations=None):
    def _decorate(handler):
        return _FakeSdkTool(name, description, input_schema, handler, annotations)

    return _decorate


def _fake_create_sdk_mcp_server(*, name, version, tools):
    return SimpleNamespace(name=name, version=version, tools=list(tools))


@dataclass
class _FakeClaudeAgentOptions:
    system_prompt: Any = None
    tools: Any = None
    allowed_tools: list = field(default_factory=list)
    disallowed_tools: list = field(default_factory=list)
    mcp_servers: dict = field(default_factory=dict)
    strict_mcp_config: bool = False
    setting_sources: Any = None
    cwd: Any = None
    env: dict = field(default_factory=dict)
    stderr: Any = None
    model: Any = None
    include_partial_messages: bool = False
    resume: Any = None
    session_store: Any = None
    hooks: Any = None


@dataclass
class _FakeHookMatcher:
    matcher: Any = None
    hooks: list = field(default_factory=list)
    timeout: Any = None


@pytest.fixture
def sdk_module(monkeypatch):
    """Yield an importable ``claude_agent_sdk``, faking it when absent."""
    try:  # pragma: no cover - exercised only where the extra is installed
        import claude_agent_sdk  # noqa: F401

        yield sys.modules["claude_agent_sdk"]
        return
    except ImportError:
        pass

    module = types.ModuleType("claude_agent_sdk")
    module.tool = _fake_tool
    module.create_sdk_mcp_server = _fake_create_sdk_mcp_server
    module.ToolAnnotations = _FakeToolAnnotations
    module.ClaudeAgentOptions = _FakeClaudeAgentOptions
    module.HookMatcher = _FakeHookMatcher
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    yield module


# ---------------------------------------------------------------------------
# SDK message shapes (dispatch is by class name)
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: bool | None = None


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-sonnet-4-5"
    stop_reason: str | None = None
    session_id: str | None = None
    error: str | None = None


@dataclass
class UserMessage:
    content: Any


@dataclass
class SystemMessage:
    subtype: str
    data: dict = field(default_factory=dict)


@dataclass
class StreamEvent:
    event: dict
    session_id: str = "sdk-session-1"
    uuid: str = "u1"


@dataclass
class ResultMessage:
    subtype: str = "success"
    session_id: str = "sdk-session-1"
    result: str | None = None
    usage: dict | None = None
    total_cost_usd: float | None = None
    terminal_reason: str | None = None
    is_error: bool = False
    errors: list | None = None


def _text_delta(text: str) -> StreamEvent:
    return StreamEvent(
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
    )


def _thinking_delta(text: str) -> StreamEvent:
    return StreamEvent(
        event={
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": text},
        }
    )


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
        for name in names
    ]


def _make_agent(*tool_names: str, api_mode: str = "claude_agent_sdk") -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.api_mode = api_mode
    agent.client = MagicMock()
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


class _Recorder:
    """Captures the canonical Hermes callbacks a runtime must fire."""

    def __init__(self, agent) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.interim: list[dict] = []
        self.progress: list[tuple] = []
        self.tool_started: list[tuple] = []
        self.tool_completed: list[tuple] = []
        agent._fire_stream_delta = self.text.append
        agent._fire_reasoning_delta = self.reasoning.append
        agent._emit_interim_assistant_message = self.interim.append
        agent.tool_progress_callback = self._progress
        agent.tool_start_callback = lambda *a: self.tool_started.append(a)
        agent.tool_complete_callback = lambda *a: self.tool_completed.append(a)

    def _progress(self, phase, name, *rest, **kwargs):
        self.progress.append((phase, name, kwargs))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def test_preflight_names_the_config_key_when_the_gate_is_off():
    message = claude_runtime_preflight({"claude_subscription": {"enabled": False}})
    assert message is not None
    assert "claude_subscription.enabled" in message


def test_preflight_names_the_extra_when_the_sdk_is_missing():
    with patch(
        "hermes_cli.claude_subscription.claude_agent_sdk_available", return_value=False
    ):
        message = claude_runtime_preflight({"claude_subscription": {"enabled": True}})
    assert message is not None
    assert "claude-code" in message


def test_preflight_names_the_login_command_when_signed_out():
    with (
        patch(
            "hermes_cli.claude_subscription.claude_agent_sdk_available",
            return_value=True,
        ),
        patch(
            "hermes_cli.claude_code.probe_claude_auth_cached",
            return_value={"logged_in": False, "message": ""},
        ),
    ):
        message = claude_runtime_preflight({"claude_subscription": {"enabled": True}})
    assert message is not None
    assert "claude auth login" in message


def test_preflight_passes_when_all_three_gates_are_open():
    with (
        patch(
            "hermes_cli.claude_subscription.claude_agent_sdk_available",
            return_value=True,
        ),
        patch(
            "hermes_cli.claude_code.probe_claude_auth_cached",
            return_value={"logged_in": True, "message": "Signed in."},
        ),
    ):
        assert claude_runtime_preflight({"claude_subscription": {"enabled": True}}) is None


@pytest.mark.parametrize(
    "config,patches",
    [
        ({"claude_subscription": {"enabled": False}}, {}),
        ({"claude_subscription": {"enabled": True}}, {"sdk": False}),
        ({"claude_subscription": {"enabled": True}}, {"sdk": True, "login": False}),
    ],
)
def test_a_refused_turn_never_builds_a_session(config, patches):
    agent = _make_agent("web_search")
    messages: list[dict] = []
    stack = [patch("hermes_cli.config.load_config_readonly", return_value=config)]
    if "sdk" in patches:
        stack.append(
            patch(
                "hermes_cli.claude_subscription.claude_agent_sdk_available",
                return_value=patches["sdk"],
            )
        )
    if "login" in patches:
        stack.append(
            patch(
                "hermes_cli.claude_code.probe_claude_auth_cached",
                return_value={"logged_in": patches["login"], "message": ""},
            )
        )
    built = []
    stack.append(
        patch.object(
            claude_runtime, "_ensure_session", lambda *a, **k: built.append(a)
        )
    )

    from contextlib import ExitStack

    with ExitStack() as es:
        for ctx in stack:
            es.enter_context(ctx)
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )

    assert built == []
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["error"]
    assert messages == []


# ---------------------------------------------------------------------------
# Options — the ownership boundary
# ---------------------------------------------------------------------------


def _options_for(agent, prompt="HERMES SYSTEM PROMPT"):
    return build_claude_agent_options(
        agent,
        system_prompt=prompt,
        effective_task_id=lambda: "task-1",
        cwd="/tmp/workspace",
    )


def test_options_append_hermes_own_identity_and_author_nothing(sdk_module):
    from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

    agent = _make_agent("web_search")
    agent._cached_system_prompt = "SYSTEM PROMPT — byte stable"
    options = _options_for(agent, agent._cached_system_prompt)

    # The append is Hermes' own identity section verbatim — no text authored
    # for this provider. The classifier bills a preset-replacing request, or
    # one carrying the full prompt, to extra usage (decision record §11), so
    # the rest of the prompt rides the first user turn instead.
    assert options.system_prompt == {
        "type": "preset",
        "preset": "claude_code",
        "append": claude_runtime.claude_subscription_append(agent),
    }
    append = options.system_prompt["append"]
    assert append.startswith(DEFAULT_AGENT_IDENTITY.strip())
    # The only non-Hermes text is the factual tool-routing note — no persona.
    assert claude_runtime.CLAUDE_TOOL_ROUTING_NOTE in append
    assert "mcp__hermes__" in append
    # The full prompt must NOT be in the system slot.
    assert agent._cached_system_prompt not in str(options.system_prompt)


def test_identity_anchor_follows_a_customised_soul(sdk_module, monkeypatch):
    # A user who customises SOUL.md gets their customisation in the anchor
    # too, rather than a provider-specific persona overriding it.
    monkeypatch.setattr(
        "agent.prompt_builder.load_soul_md", lambda *a, **k: "You are Ares Agent."
    )
    assert claude_runtime.hermes_identity_anchor() == "You are Ares Agent."


def test_options_keep_builtins_in_context_but_deny_them_via_hook(sdk_module):
    agent = _make_agent("web_search", "terminal")
    options = _options_for(agent)

    # Built-ins stay context-visible (stripping them trips the billing
    # classifier); execution is pinned to the bridge by the PreToolUse hook.
    assert getattr(options, "tools", None) in (None, ()) or options.tools is None
    hooks = options.hooks or {}
    assert "PreToolUse" in hooks

    matcher = hooks["PreToolUse"][0]
    hook = matcher.hooks[0]

    async def _run(name):
        return await hook({"tool_name": name}, "toolu_1", None)

    denied = asyncio.run(_run("Bash"))
    decision = denied["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "mcp__hermes__" in decision["permissionDecisionReason"]
    # Read is auto-allowed by the CLI without a permission prompt, so the hook
    # (not can_use_tool) must be the choke point for it too.
    assert asyncio.run(_run("Read"))["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert asyncio.run(_run("mcp__hermes__web_search")) == {}


def test_options_expose_only_the_hermes_bridge(sdk_module):
    agent = _make_agent("web_search", "terminal")
    options = _options_for(agent)

    assert set(options.mcp_servers) == {MCP_SERVER_NAME}
    assert sorted(options.allowed_tools) == sorted(bridged_allowed_tools(agent))
    assert all(name.startswith("mcp__hermes__") for name in options.allowed_tools)
    # Every allowed name is actually served by the bridge.
    assert len(options.allowed_tools) == len(agent.tools)


def test_options_pin_the_toolset_and_skip_a_second_settings_load(sdk_module):
    agent = _make_agent("web_search")
    options = _options_for(agent)

    assert options.setting_sources == []
    assert options.strict_mcp_config is True


def test_options_carry_cwd_and_model_and_hold_no_credential(sdk_module):
    agent = _make_agent("web_search")
    agent.model = "claude-sonnet-4-5"
    options = _options_for(agent)

    assert options.cwd == "/tmp/workspace"
    assert options.model == "claude-sonnet-4-5"
    # PR7 owns the sanitized environment; PR4 must not smuggle credentials in.
    assert options.env == {}


def test_options_leave_the_pr5_resume_seam_unset(sdk_module):
    agent = _make_agent("web_search")
    options = _options_for(agent)

    assert getattr(options, "resume", None) is None
    assert getattr(options, "session_store", None) is None


# ---------------------------------------------------------------------------
# Event projection
# ---------------------------------------------------------------------------


def test_mcp_namespacing_is_stripped_for_display():
    assert display_tool_name("mcp__hermes__web_search") == "web_search"
    assert display_tool_name("Bash") == "Bash"


def test_a_multi_tool_streaming_turn_fires_the_canonical_callbacks():
    agent = _make_agent("web_search", "terminal")
    rec = _Recorder(agent)
    projector = ClaudeEventProjector(agent)

    for message in [
        SystemMessage("init", {"session_id": "sdk-session-1"}),
        _thinking_delta("planning"),
        _text_delta("Looking that up"),
        AssistantMessage(
            content=[
                ThinkingBlock("planning"),
                TextBlock("Looking that up"),
                ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "hermes"}),
                ToolUseBlock("t2", "mcp__hermes__terminal", {"command": "ls"}),
            ],
            session_id="sdk-session-1",
        ),
        UserMessage(content=[ToolResultBlock("t1", "search results")]),
        UserMessage(content=[ToolResultBlock("t2", "file-a\nfile-b")]),
        _text_delta("All done"),
        AssistantMessage(content=[TextBlock("All done")]),
        ResultMessage(result="All done", usage={"input_tokens": 10, "output_tokens": 4}),
    ]:
        projector(message)
    projector.finalize()

    assert rec.text == ["Looking that up", "All done"]
    assert rec.reasoning == ["planning"]
    started = [p for p in rec.progress if p[0] == "tool.started"]
    completed = [p for p in rec.progress if p[0] == "tool.completed"]
    assert [p[1] for p in started] == ["web_search", "terminal"]
    assert [p[1] for p in completed] == ["web_search", "terminal"]
    assert all(p[2]["is_error"] is False for p in completed)
    assert [c[1] for c in rec.tool_started] == ["web_search", "terminal"]
    assert [c[1] for c in rec.tool_completed] == ["web_search", "terminal"]
    assert projector.tool_iterations == 2


def test_streamed_text_is_not_replayed_when_the_block_completes():
    agent = _make_agent("web_search")
    rec = _Recorder(agent)
    projector = ClaudeEventProjector(agent)

    projector(_text_delta("hel"))
    projector(_text_delta("lo"))
    projector(AssistantMessage(content=[TextBlock("hello")]))
    projector(ResultMessage(result="hello"))

    assert rec.text == ["hel", "lo"]


def test_completed_blocks_still_stream_when_partial_events_are_absent():
    agent = _make_agent("web_search")
    rec = _Recorder(agent)
    projector = ClaudeEventProjector(agent)

    projector(AssistantMessage(content=[TextBlock("hello")]))
    projector(ResultMessage(result="hello"))

    assert rec.text == ["hello"]


def test_projected_messages_preserve_role_alternation():
    agent = _make_agent("web_search", "terminal")
    projector = ClaudeEventProjector(agent)

    for message in [
        AssistantMessage(
            content=[
                TextBlock("working"),
                ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"}),
                ToolUseBlock("t2", "mcp__hermes__terminal", {"command": "ls"}),
            ]
        ),
        # Results land out of order; the transcript must still answer the
        # tool_calls in the order they were issued.
        UserMessage(content=[ToolResultBlock("t2", "listing")]),
        UserMessage(content=[ToolResultBlock("t1", "results")]),
        AssistantMessage(content=[TextBlock("done")]),
        ResultMessage(result="done"),
    ]:
        projector(message)
    projected = projector.finalize()

    roles = [m["role"] for m in projected]
    assert roles == ["assistant", "tool", "tool", "assistant"]
    assert [m["tool_call_id"] for m in projected if m["role"] == "tool"] == ["t1", "t2"]
    call_ids = [tc["id"] for tc in projected[0]["tool_calls"]]
    assert call_ids == ["t1", "t2"]
    assert [tc["function"]["name"] for tc in projected[0]["tool_calls"]] == [
        "web_search",
        "terminal",
    ]
    # Never two assistant messages in a row.
    assert not any(
        roles[i] == roles[i + 1] == "assistant" for i in range(len(roles) - 1)
    )


def test_consecutive_assistant_text_is_merged_rather_than_duplicated():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(AssistantMessage(content=[TextBlock("part one")]))
    projector(AssistantMessage(content=[TextBlock("part two")]))
    projected = projector.finalize()

    assert [m["role"] for m in projected] == ["assistant"]
    assert "part one" in projected[0]["content"]
    assert "part two" in projected[0]["content"]


def test_an_unanswered_tool_call_still_gets_a_tool_message():
    """An unanswered tool_call_id makes the next provider request fail."""
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(
        AssistantMessage(
            content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
        )
    )
    projector(ResultMessage(result="gave up"))
    projected = projector.finalize()

    assert [m["role"] for m in projected] == ["assistant", "tool"]
    assert projected[1]["tool_call_id"] == "t1"
    assert projected[1]["content"]


def test_the_sdk_session_id_is_captured_for_pr5():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(SystemMessage("init", {"session_id": "sdk-session-xyz"}))
    projector(ResultMessage(session_id="sdk-session-xyz"))

    assert projector.session_id == "sdk-session-xyz"
    assert agent._claude_sdk_session_id == "sdk-session-xyz"


def test_a_compaction_boundary_is_observed():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(SystemMessage("compact_boundary", {"session_id": "s"}))
    assert projector.compacted is True


def test_terminal_reason_and_error_are_surfaced():
    agent = _make_agent("web_search")
    projector = ClaudeEventProjector(agent)

    projector(
        ResultMessage(
            subtype="error_during_execution",
            is_error=True,
            errors=["upstream refused"],
            terminal_reason="max_turns",
        )
    )

    assert projector.is_error is True
    assert "upstream refused" in projector.error
    assert projector.terminal_reason == "max_turns"


def test_image_tool_results_survive_projection():
    agent = _make_agent("web_search")
    agent._model_supports_vision = lambda *a, **k: True
    agent._provider_supports_vision_tool_messages = lambda *a, **k: True
    projector = ClaudeEventProjector(agent)

    projector(
        AssistantMessage(
            content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
        )
    )
    projector(
        UserMessage(
            content=[
                ToolResultBlock(
                    "t1",
                    [
                        {"type": "text", "text": "a screenshot"},
                        {"type": "image", "data": "QUJD", "mimeType": "image/png"},
                    ],
                )
            ]
        )
    )
    projected = projector.finalize()

    content = projected[1]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)


def test_a_buggy_display_callback_cannot_break_the_turn():
    agent = _make_agent("web_search")

    def _boom(*args, **kwargs):
        raise RuntimeError("display exploded")

    agent._fire_stream_delta = _boom
    agent.tool_progress_callback = _boom
    projector = ClaudeEventProjector(agent)

    projector(AssistantMessage(content=[TextBlock("hello")]))
    projector(ResultMessage(result="hello"))

    assert projector.final_text == "hello"


# ---------------------------------------------------------------------------
# Whole turn
# ---------------------------------------------------------------------------


class _StubSession:
    """Replays a scripted message list through the projector."""

    def __init__(self, script, *, raises=None):
        self.script = script
        self.raises = raises
        self.closed = False
        self.prompts = []
        self.session_ids = []

    def run_turn(self, prompt, *, on_message, timeout=None):
        self.prompts.append(prompt)
        if self.raises is not None:
            raise self.raises
        for message in self.script:
            on_message(message)
        return len(self.script)

    def note_session_id(self, session_id):
        self.session_ids.append(session_id)

    def close(self):
        self.closed = True


def _run_turn(agent, session, messages=None):
    messages = messages if messages is not None else []
    with (
        patch.object(claude_runtime, "claude_runtime_preflight", return_value=None),
        patch.object(claude_runtime, "_ensure_session", return_value=session),
    ):
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )
    return result, messages


def test_a_completed_turn_returns_the_run_conversation_shape():
    agent = _make_agent("web_search")
    session = _StubSession(
        [
            AssistantMessage(content=[TextBlock("hello")]),
            ResultMessage(
                result="hello",
                usage={"input_tokens": 12, "output_tokens": 3},
                total_cost_usd=0.001,
            ),
        ]
    )
    result, messages = _run_turn(agent, session)

    assert result["final_response"] == "hello"
    assert result["messages"] is messages
    assert result["completed"] is True
    assert result["partial"] is False
    assert result["interrupted"] is False
    assert result["error"] is None
    assert result["api_calls"] == 1
    assert result["agent_persisted"] is True
    assert result["claude_session_id"] == "sdk-session-1"
    assert result["prompt_tokens"] >= 12


def test_projected_messages_are_spliced_exactly_once_including_trailing_events():
    """Events after ResultMessage must be projected, and only once."""
    agent = _make_agent("web_search")
    session = _StubSession(
        [
            AssistantMessage(
                content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
            ),
            UserMessage(content=[ToolResultBlock("t1", "results")]),
            ResultMessage(result="done"),
            # Trailing frame the CLI flushed after the result.
            AssistantMessage(content=[TextBlock("done")]),
        ]
    )
    result, messages = _run_turn(agent, session)

    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert sum(1 for m in messages if m["role"] == "tool") == 1
    assert messages[-1]["content"] == "done"
    # A second finalize must not be able to duplicate anything.
    assert result["messages"] == messages


def test_a_wedged_turn_retires_the_session_so_the_next_one_respawns():
    agent = _make_agent("web_search")
    agent._claude_session = session = _StubSession([], raises=TimeoutError("stalled"))
    result, _messages = _run_turn(agent, session)

    assert session.closed is True
    assert getattr(agent, "_claude_session", None) is None
    assert result["completed"] is False
    assert "stalled" in result["error"]


def test_tool_iterations_feed_the_skill_nudge_counter():
    agent = _make_agent("web_search")
    agent._iters_since_skill = 0
    session = _StubSession(
        [
            AssistantMessage(
                content=[ToolUseBlock("t1", "mcp__hermes__web_search", {"query": "a"})]
            ),
            UserMessage(content=[ToolResultBlock("t1", "results")]),
            ResultMessage(result="done"),
        ]
    )
    _run_turn(agent, session)

    assert agent._iters_since_skill == 1


def test_a_turn_without_usage_still_counts_as_one_api_call():
    agent = _make_agent("web_search")
    before = agent.session_api_calls
    session = _StubSession([ResultMessage(result="hi")])
    result, _messages = _run_turn(agent, session)

    assert agent.session_api_calls == before + 1
    assert result["api_calls"] == 1


# ---------------------------------------------------------------------------
# Every other api_mode is untouched
# ---------------------------------------------------------------------------


def _exhaust_budget(agent):
    agent.max_iterations = 0
    agent.iteration_budget._used = agent.iteration_budget.max_total


@pytest.mark.parametrize(
    "api_mode",
    [
        "chat_completions",
        "anthropic_messages",
        "codex_responses",
        "codex_app_server",
        "bedrock_converse",
    ],
)
def test_the_early_branch_does_not_fire_for_other_api_modes(api_mode):
    agent = _make_agent("web_search", api_mode=api_mode)
    _exhaust_budget(agent)
    fired = []
    agent._run_claude_agent_sdk_turn = lambda **kw: fired.append(kw) or {}
    agent._run_codex_app_server_turn = lambda **kw: {
        "final_response": "codex",
        "messages": [],
        "api_calls": 0,
    }

    agent.run_conversation("hello")

    assert fired == []


def test_the_early_branch_fires_for_claude_agent_sdk():
    agent = _make_agent("web_search", api_mode="claude_agent_sdk")
    _exhaust_budget(agent)
    fired = []

    def _forward(**kwargs):
        fired.append(kwargs)
        return {"final_response": "claude", "messages": [], "api_calls": 1}

    agent._run_claude_agent_sdk_turn = _forward
    result = agent.run_conversation("hello")

    assert len(fired) == 1
    assert result["final_response"] == "claude"
    # The default provider loop was bypassed entirely.
    assert result["api_calls"] == 1
