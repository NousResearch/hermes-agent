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


class _IdleAdapter:
    """Command-path adapter probe: no active adapter frame to drain FIFO."""

    def __init__(self):
        self._pending_messages = {}
        self._active_sessions = {}
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


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
async def test_gateway_goal_starts_kickoff_when_command_has_no_adapter_frame(tmp_path, monkeypatch):
    """A gateway command path without BasePlatformAdapter's in-band drain
    must actively wake the queued /goal kickoff instead of leaving turns_used
    at zero until the user sends another message."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-goal-kickoff",
        chat_type="channel",
        user_id="user-goal-kickoff",
    )
    adapter = _IdleAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._queued_events = {}

    event = MessageEvent(
        text="/goal ship the benchmark",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-goal-kickoff",
    )

    try:
        response = await GatewayRunner._handle_goal_command(runner, event)
        assert "Goal set" in response
        assert [item.text for item in adapter.handled] == ["ship the benchmark"]
        assert adapter.handled[0].message_id == "msg-goal-kickoff"
    finally:
        goals._DB_CACHE.clear()
