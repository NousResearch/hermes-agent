import os
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


class _FakeSessionEntry:
    session_id = "sid-gateway-goal-config"


class _FakeSessionStore:
    def __init__(self):
        self.entry = _FakeSessionEntry()

    def get_or_create_session(self, source):
        return self.entry

    def _generate_session_key(self, source):
        return "agent:main:discord:channel:goal-config"


@pytest.mark.asyncio
async def test_gateway_goal_uses_goals_max_turns_from_full_config(tmp_path, monkeypatch):
    """Gateway /goal should honor top-level goals.max_turns from config.yaml."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}

    event = MessageEvent(
        text="/goal ship the benchmark",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-goal-config",
    )

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "⊙ Goal set (7-turn budget): ship the benchmark" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.max_turns == 7
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_confirm_promotes_receipt_with_explicit_user_action(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    event = MessageEvent(
        text="/goal confirm 73",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-goal-config",
    )

    with patch(
        "agent.verification_evidence.confirm_outcome_receipt",
        return_value={"id": 73, "reusable": True},
    ) as confirm_receipt:
        response = await GatewayRunner._handle_goal_command(runner, event)

    assert "73" in response
    assert "reusable" in response
    confirm_receipt.assert_called_once_with(
        73,
        expected_session_id="sid-gateway-goal-config",
        cwd=os.environ.get("TERMINAL_CWD") or os.getcwd(),
        actor="user",
    )
    goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_wait_supports_session_and_time_barriers(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-goal-config",
        chat_type="channel",
        user_id="user-goal-config",
    )

    await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal ship the benchmark",
            message_type=MessageType.TEXT,
            source=source,
            message_id="goal-set",
        ),
    )
    session_wait = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal wait session ci-watch CI is running",
            message_type=MessageType.TEXT,
            source=source,
            message_id="goal-wait-session",
        ),
    )
    state = goals.GoalManager("sid-gateway-goal-config").state
    assert "session ci-watch" in session_wait
    assert state.waiting_on_session == "ci-watch"

    time_wait = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal wait for 45 retry backoff",
            message_type=MessageType.TEXT,
            source=source,
            message_id="goal-wait-time",
        ),
    )
    state = goals.GoalManager("sid-gateway-goal-config").state
    assert "45s" in time_wait
    assert state.waiting_on_session is None
    assert state.waiting_until > 0
    goals._DB_CACHE.clear()
