"""Gateway generic /skill command tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.commands import telegram_bot_commands


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="user-1",
            chat_id="chat-1",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(send=AsyncMock(), _pending_messages={})}
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner.hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[]))
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._update_prompt_pending = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: "telegram:chat-1:user-1"
    runner._is_telegram_topic_root_lobby = lambda source: False
    runner._begin_session_run_generation = lambda key: 1
    return runner


def test_generic_skill_command_is_registered_in_telegram_menu():
    """The stable /skill entry must stay visible even when individual skills overflow Telegram's menu."""
    assert any(name == "skill" for name, _description in telegram_bot_commands())


@pytest.mark.asyncio
async def test_generic_skill_command_loads_skill_and_forwards_instruction(monkeypatch):
    """/skill <name> should load any skill even when it is absent from Telegram's 100-command menu."""
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/hermes-agent": {"name": "hermes-agent", "description": "Configure Hermes"}},
    )
    monkeypatch.setattr(
        "agent.skill_commands.resolve_skill_command_key",
        lambda command: "/hermes-agent" if command in {"hermes-agent", "hermes_agent"} else None,
    )

    seen = {}

    def _build(cmd_key, user_instruction="", task_id=None):
        seen["cmd_key"] = cmd_key
        seen["instruction"] = user_instruction
        seen["task_id"] = task_id
        return f"SKILL::{cmd_key}::{user_instruction}"

    monkeypatch.setattr("agent.skill_commands.build_skill_invocation_message", _build)

    runner = _make_runner()

    async def _capture(event, source, quick_key, run_generation):
        seen["event_text"] = event.text
        seen["quick_key"] = quick_key
        seen["run_generation"] = run_generation
        return "agent-response"

    runner._handle_message_with_agent = _capture

    result = await runner._handle_message(_make_event("/skill hermes_agent explique la config"))

    assert result == "agent-response"
    assert seen["cmd_key"] == "/hermes-agent"
    assert seen["instruction"] == "explique la config"
    assert seen["event_text"] == "SKILL::/hermes-agent::explique la config"


@pytest.mark.asyncio
async def test_generic_skill_command_requires_skill_name():
    runner = _make_runner()

    result = await runner._handle_message(_make_event("/skill"))

    assert "Usage: /skill <name>" in result
    assert "hermes-agent" in result
