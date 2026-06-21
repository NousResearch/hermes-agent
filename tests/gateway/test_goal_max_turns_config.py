import uuid

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import goals


class _FakeSessionEntry:
    def __init__(self, session_id: str):
        self.session_id = session_id


class _FakeSessionStore:
    def __init__(self, session_id: str):
        self.entry = _FakeSessionEntry(session_id)

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
    override_token = set_hermes_home_override(home)
    goals._DB_CACHE.clear()
    session_id = f"sid-gateway-goal-config-{uuid.uuid4().hex}"

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore(session_id)
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

    try:
        response = await GatewayRunner._handle_goal_command(runner, event)

        assert "Goal Packet" in response
        assert "ship the benchmark" in response
        state = goals.GoalManager(session_id).state
        assert state is None

        approve_event = MessageEvent(
            text="진행",
            message_type=MessageType.TEXT,
            source=event.source,
            message_id="msg-goal-config-approve",
        )
        approved = await GatewayRunner._handle_pending_goal_reply(runner, approve_event)
        assert approved is not None
        assert "⊙ Goal set (7-turn budget):" in approved
        assert "ship the benchmark" in approved

        state = goals.GoalManager(session_id).state
        assert state is not None
        assert state.max_turns == 7
    finally:
        try:
            goals.clear_goal(session_id)
            goals.clear_goal_draft(session_id)
        except Exception:
            pass
        reset_hermes_home_override(override_token)
        goals._DB_CACHE.clear()
