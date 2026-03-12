"""Tests for in-flight interrupt/queue routing in GatewayRunner._handle_message."""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_type="group",
        user_id="user-1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner_with_active_agent(session_key: str = "agent:main:discord:group:chan-1"):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {session_key: MagicMock()}
    runner._pending_messages = {}
    runner._pending_approvals = {}

    runner.session_store = MagicMock()
    runner.session_store._generate_session_key.return_value = session_key

    runner._is_user_authorized = lambda source: True
    return runner, session_key


class TestRuntimeMessageControlParsing:
    def test_parse_queue_prefixes(self):
        from gateway.run import GatewayRunner

        assert GatewayRunner._parse_runtime_message_control("/queue do x") == ("queue", "do x")
        assert GatewayRunner._parse_runtime_message_control("queue: do y") == ("queue", "do y")
        assert GatewayRunner._parse_runtime_message_control("Q: do z") == ("queue", "do z")

    def test_parse_steer_prefixes(self):
        from gateway.run import GatewayRunner

        assert GatewayRunner._parse_runtime_message_control("/steer switch plan") == ("steer", "switch plan")
        assert GatewayRunner._parse_runtime_message_control("steer: switch plan") == ("steer", "switch plan")
        assert GatewayRunner._parse_runtime_message_control("plain message") == ("steer", "plain message")


class TestActiveSessionRouting:
    @pytest.mark.asyncio
    async def test_default_message_interrupts_and_queues_payload(self):
        runner, session_key = _make_runner_with_active_agent()
        event = _make_event("please stop and do this instead")

        result = await runner._handle_message(event)

        assert result is None
        runner._running_agents[session_key].interrupt.assert_called_once_with("please stop and do this instead")
        assert runner._pending_messages[session_key] == "please stop and do this instead"

    @pytest.mark.asyncio
    async def test_queue_message_does_not_interrupt_but_appends_payload(self):
        runner, session_key = _make_runner_with_active_agent()
        event = _make_event("/queue then summarize errors")

        result = await runner._handle_message(event)

        assert result is None
        runner._running_agents[session_key].interrupt.assert_not_called()
        assert runner._pending_messages[session_key] == "then summarize errors"

    @pytest.mark.asyncio
    async def test_queue_messages_append_deterministically(self):
        runner, session_key = _make_runner_with_active_agent()

        await runner._handle_message(_make_event("q: step one"))
        await runner._handle_message(_make_event("queue: step two"))

        runner._running_agents[session_key].interrupt.assert_not_called()
        assert runner._pending_messages[session_key] == "step one\nstep two"
