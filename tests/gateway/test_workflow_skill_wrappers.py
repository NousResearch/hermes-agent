"""Gateway dispatch tests for workflow skill slash-command wrappers."""

from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_reload_skills_command import _make_event, _make_runner


@pytest.mark.asyncio
async def test_gateway_workflow_wrapper_rewrites_to_skill_message(monkeypatch):
    import agent.skill_commands as skill_commands_mod
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._draining = False
    runner._running_agents_ts = {}
    runner._begin_session_run_generation = lambda _key: 1
    runner._handle_message_with_agent = AsyncMock(return_value="agent result")
    runner._telegram_topic_root_lobby_message = lambda: "lobby"
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._should_send_telegram_lobby_reminder = lambda _source: False

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        skill_commands_mod,
        "get_skill_commands",
        lambda: {"/verify": {"name": "verify"}},
    )
    monkeypatch.setattr(
        skill_commands_mod,
        "resolve_skill_command_key",
        lambda name: "/verify" if name == "verify" else None,
    )

    calls = {}

    def fake_build(cmd_key, user_instruction, task_id=None):
        calls["cmd_key"] = cmd_key
        calls["user_instruction"] = user_instruction
        calls["task_id"] = task_id
        return "LOADED VERIFY"

    monkeypatch.setattr(skill_commands_mod, "build_skill_invocation_message", fake_build)

    event = _make_event("/verify acceptance criteria")
    result = await runner._handle_message(event)

    assert result == "agent result"
    assert event.text == "LOADED VERIFY"
    assert calls["cmd_key"] == "/verify"
    assert calls["user_instruction"] == "acceptance criteria"
    assert calls["task_id"] == "agent:main:telegram:dm:c1"
    runner._handle_message_with_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_gateway_devflow_wrapper_rewrites_to_skill_message(monkeypatch):
    import agent.skill_commands as skill_commands_mod
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._draining = False
    runner._running_agents_ts = {}
    runner._begin_session_run_generation = lambda _key: 1
    runner._handle_message_with_agent = AsyncMock(return_value="agent result")
    runner._telegram_topic_root_lobby_message = lambda: "lobby"
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._should_send_telegram_lobby_reminder = lambda _source: False

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        skill_commands_mod,
        "get_skill_commands",
        lambda: {"/devflow": {"name": "devflow"}},
    )
    monkeypatch.setattr(
        skill_commands_mod,
        "resolve_skill_command_key",
        lambda name: "/devflow" if name == "devflow" else None,
    )

    calls = {}

    def fake_build(cmd_key, user_instruction, task_id=None):
        calls["cmd_key"] = cmd_key
        calls["user_instruction"] = user_instruction
        calls["task_id"] = task_id
        return "LOADED DEVFLOW"

    monkeypatch.setattr(skill_commands_mod, "build_skill_invocation_message", fake_build)

    event = _make_event("/devflow implement workflow gates")
    result = await runner._handle_message(event)

    assert result == "agent result"
    assert event.text == "LOADED DEVFLOW"
    assert calls["cmd_key"] == "/devflow"
    assert calls["user_instruction"] == "implement workflow gates"
    assert calls["task_id"] == "agent:main:telegram:dm:c1"
    runner._handle_message_with_agent.assert_awaited_once()
