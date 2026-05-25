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


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}


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
    runner.adapters = {Platform.DISCORD: _FakeAdapter()}
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
async def test_gateway_goal_leading_number_overrides_config_budget(tmp_path, monkeypatch):
    """Gateway /goal N <text> should use N instead of goals.max_turns."""
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
    runner.adapters = {Platform.DISCORD: _FakeAdapter()}
    runner._queued_events = {}

    event = MessageEvent(
        text="/goal 50 ship the benchmark",
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
        assert "⊙ Goal set (50-turn budget): ship the benchmark" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.max_turns == 50
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_resume_enqueues_immediate_continuation(tmp_path, monkeypatch):
    """/goal resume should immediately re-kick the continuation loop."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {Platform.DISCORD: _FakeAdapter()}
    runner._queued_events = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-goal-config",
        chat_type="channel",
        user_id="user-goal-config",
    )

    set_event = MessageEvent(
        text="/goal ship the benchmark",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-goal-set",
    )
    await GatewayRunner._handle_goal_command(runner, set_event)

    pause_event = MessageEvent(
        text="/goal pause",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-goal-pause",
    )
    await GatewayRunner._handle_goal_command(runner, pause_event)

    resume_event = MessageEvent(
        text="/goal resume",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-goal-resume",
    )
    response = await GatewayRunner._handle_goal_command(runner, resume_event)

    try:
        assert "Goal resumed" in response
        pending = runner.adapters[Platform.DISCORD]._pending_messages
        queued = pending["agent:main:discord:channel:goal-config"]
        assert "Continuing toward your standing goal" in queued.text
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.status == "active"
    finally:
        goals._DB_CACHE.clear()
