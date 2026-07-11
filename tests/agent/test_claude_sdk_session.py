import asyncio
from pathlib import Path

import pytest

from agent.claude_sdk_session import (
    ClaudeAgentSdkSession,
    build_claude_agent_options,
)


class FakeOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeSdk:
    ClaudeAgentOptions = FakeOptions

    class HookMatcher:
        def __init__(self, *, matcher, hooks, timeout=None):
            self.matcher = matcher
            self.hooks = hooks
            self.timeout = timeout

    @staticmethod
    def tool(name, description, input_schema):
        def decorate(handler):
            handler.sdk_name = name
            return handler

        return decorate

    @staticmethod
    def create_sdk_mcp_server(*, name, version, tools):
        return {"name": name, "version": version, "tools": tools}


def _kanban_tool(name="kanban_complete"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_options_are_subscription_only_strict_and_workspace_sandboxed(tmp_path):
    host_home = tmp_path / "host"
    workspace = tmp_path / "workspace"
    host_home.mkdir()
    workspace.mkdir()
    inherited = {
        "PATH": "/usr/bin:/bin",
        "USER": "person",
        "LOGNAME": "person",
        "ANTHROPIC_API_KEY": "paid-key",
        "OPENAI_API_KEY": "other-key",
    }

    options = build_claude_agent_options(
        sdk=FakeSdk,
        model="claude-sonnet-4-6",
        system_prompt="stable prompt",
        workspace=workspace,
        host_home=host_home,
        profile_home=tmp_path / "profile",
        inherited_env=inherited,
        tool_definitions=[
            _kanban_tool(),
            _kanban_tool("terminal"),
            _kanban_tool("process"),
            _kanban_tool("read_file"),
            _kanban_tool("write_file"),
        ],
        dispatch=lambda *args, **kwargs: "ok",
        effective_task_id="task-run-id",
        kanban_task_id="BUILD-392",
        max_turns=40,
        resume="sdk-session-1",
        cli_path=tmp_path / "claude-wrapper",
    )

    assert options.system_prompt == "stable prompt"
    assert options.setting_sources == []
    assert options.strict_mcp_config is True
    assert options.skills == []
    assert options.plugins == []
    assert options.fallback_model is None
    assert options.max_budget_usd is None
    assert options.resume == "sdk-session-1"
    assert options.permission_mode == "acceptEdits"
    assert options.tools == [
        "mcp__hermes__kanban_complete",
        "mcp__hermes__process",
        "mcp__hermes__read_file",
        "mcp__hermes__terminal",
        "mcp__hermes__write_file",
    ]
    assert options.allowed_tools == options.tools
    assert options.mcp_servers["hermes"]["name"] == "hermes"
    assert options.env == {}
    assert options.cli_path == str(tmp_path / "claude-wrapper")
    assert "ANTHROPIC_API_KEY" not in options.env
    assert "OPENAI_API_KEY" not in options.env
    assert options.sandbox is None
    assert options.include_partial_messages is True
    assert len(options.hooks["PreToolUse"]) == 1
    read_tool = next(
        tool
        for tool in options.mcp_servers["hermes"]["tools"]
        if tool.sdk_name == "read_file"
    )
    outside = asyncio.run(read_tool({"path": "../host/sentinel"}))
    assert outside["is_error"] is True
    assert "relative path" in outside["content"][0]["text"]


def test_orchestrator_mode_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="Kanban workers only"):
        build_claude_agent_options(
            sdk=FakeSdk,
            model="claude-sonnet-4-6",
            system_prompt="prompt",
            workspace=tmp_path,
            host_home=tmp_path / "host",
            profile_home=tmp_path / "profile",
            inherited_env={"PATH": "/usr/bin"},
            tool_definitions=[_kanban_tool()],
            dispatch=lambda *args, **kwargs: "ok",
            effective_task_id="task-run-id",
            kanban_task_id=None,
        )


def test_auxiliary_mode_exposes_only_explicit_memory_skill_capabilities(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def dispatch(name, arguments, *, task_id):
        calls.append((name, arguments, task_id))
        return {"success": True, "tool": name}

    definitions = [
        _kanban_tool(name)
        for name in (
            "memory",
            "skill_manage",
            "terminal",
            "read_file",
            "kanban_complete",
        )
    ]
    options = build_claude_agent_options(
        sdk=FakeSdk,
        model="claude-sonnet-4-6",
        system_prompt="review",
        workspace=workspace,
        host_home=tmp_path / "host",
        profile_home=tmp_path / "profile",
        inherited_env={"PATH": "/usr/bin:/bin"},
        tool_definitions=definitions,
        dispatch=dispatch,
        effective_task_id="worker-task",
        kanban_task_id="BUILD-392",
        capability_mode="auxiliary",
        auxiliary_tool_names=("memory", "skill_manage"),
    )

    assert options.tools == ["mcp__hermes__memory", "mcp__hermes__skill_manage"]
    assert not any(
        name in " ".join(options.tools)
        for name in ("terminal", "read_file", "kanban_complete")
    )
    tools = {
        tool.sdk_name: tool for tool in options.mcp_servers["hermes"]["tools"]
    }
    memory_result = asyncio.run(tools["memory"]({"action": "add"}))
    skill_result = asyncio.run(tools["skill_manage"]({"action": "patch"}))
    hook = options.hooks["PreToolUse"][0].hooks[0]
    memory_decision = asyncio.run(
        hook(
            {"tool_name": "mcp__hermes__memory", "tool_input": {"action": "add"}},
            "memory-call",
            {"signal": None},
        )
    )
    terminal_decision = asyncio.run(
        hook(
            {"tool_name": "mcp__hermes__terminal", "tool_input": {"command": "pwd"}},
            "terminal-call",
            {"signal": None},
        )
    )

    assert calls == [
        ("memory", {"action": "add"}, "worker-task"),
        ("skill_manage", {"action": "patch"}, "worker-task"),
    ]
    assert "success" in memory_result["content"][0]["text"]
    assert "success" in skill_result["content"][0]["text"]
    assert memory_decision["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert terminal_decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_auxiliary_mode_fails_closed_when_required_definition_is_missing(tmp_path):
    with pytest.raises(RuntimeError, match="missing required Hermes tools: memory"):
        build_claude_agent_options(
            sdk=FakeSdk,
            model="claude-sonnet-4-6",
            system_prompt="review",
            workspace=tmp_path,
            host_home=tmp_path / "host",
            profile_home=tmp_path / "profile",
            inherited_env={"PATH": "/usr/bin"},
            tool_definitions=[_kanban_tool("skill_manage")],
            dispatch=lambda *args, **kwargs: "ok",
            effective_task_id="worker-task",
            kanban_task_id="BUILD-392",
            capability_mode="auxiliary",
            auxiliary_tool_names=("memory",),
        )


def test_session_projects_typed_messages_and_resumes(monkeypatch, tmp_path):
    from tests.agent.test_claude_agent_runtime import ResultMessage, TextBlock, AssistantMessage

    calls = []

    class FakeClient:
        def __init__(self, options):
            calls.append(options)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def query(self, prompt):
            calls.append(prompt)

        async def receive_response(self):
            yield AssistantMessage([TextBlock("done")])
            yield ResultMessage("new-sdk-session", result="done", usage={"input_tokens": 3})

    FakeSdk.ClaudeSDKClient = FakeClient
    session = ClaudeAgentSdkSession(
        sdk=FakeSdk,
        options_factory=lambda resume: FakeOptions(resume=resume),
    )

    first = session.run_turn("do it")
    second = session.run_turn("do another")

    assert first.final_text == "done"
    assert first.session_id == "new-sdk-session"
    assert first.usage == {"input_tokens": 3}
    assert calls[0].resume is None
    assert calls[2].resume == "new-sdk-session"

    restored = ClaudeAgentSdkSession(
        sdk=FakeSdk,
        options_factory=lambda resume: FakeOptions(resume=resume),
        initial_session_id="persisted-sdk-session",
    )
    restored.run_turn("after worker restart")
    assert calls[4].resume == "persisted-sdk-session"


def test_session_streams_partial_text_and_tool_progress_without_duplication():
    from dataclasses import dataclass
    from tests.agent.test_claude_agent_runtime import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
    )

    @dataclass
    class StreamEvent:
        event: dict

    class FakeClient:
        def __init__(self, options):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def query(self, prompt):
            pass

        async def receive_response(self):
            yield StreamEvent(
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hel"}}
            )
            tool = ToolUseBlock("call-1", "mcp__hermes__terminal", {"command": "pwd"})
            yield AssistantMessage([tool])
            yield AssistantMessage([tool])
            yield StreamEvent(
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}
            )
            yield AssistantMessage([TextBlock("hello")])
            yield ResultMessage("session", result="hello")

    FakeSdk.ClaudeSDKClient = FakeClient
    deltas = []
    progress = []
    session = ClaudeAgentSdkSession(
        sdk=FakeSdk,
        options_factory=lambda resume: FakeOptions(resume=resume),
        stream_delta_callback=deltas.append,
        tool_progress_callback=lambda *args: progress.append(args),
    )

    projection = session.run_turn("go")

    assert deltas == ["hel", "lo"]
    assert len(progress) == 1
    assert progress[0][0:2] == ("tool.started", "terminal")
    assert projection.final_text == "hello"


def test_session_interrupts_immediately_when_overage_event_arrives():
    from claude_agent_sdk import RateLimitEvent, RateLimitInfo

    consumed_after_overage = []

    class FakeClient:
        interrupted = False

        def __init__(self, options):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def query(self, prompt):
            pass

        async def interrupt(self):
            type(self).interrupted = True

        async def receive_response(self):
            yield RateLimitEvent(
                rate_limit_info=RateLimitInfo(
                    status="allowed",
                    rate_limit_type="five_hour",
                    overage_status="allowed_warning",
                ),
                uuid="overage",
                session_id="session",
            )
            consumed_after_overage.append(True)

    FakeSdk.ClaudeSDKClient = FakeClient
    session = ClaudeAgentSdkSession(
        sdk=FakeSdk, options_factory=lambda resume: FakeOptions(resume=resume)
    )

    projection = session.run_turn("stop before paid overage")

    assert projection.failure is not None
    assert projection.failure.reason.value == "billing"
    assert FakeClient.interrupted is True
    assert consumed_after_overage == []
