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


class _PendingAdapter:
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
async def test_gateway_goal_resume_enqueues_canonical_continuation(
    tmp_path, monkeypatch
):
    """Messaging /goal resume restarts work without a second user message."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.__dict__["session_store"] = _FakeSessionStore()
    adapter = _PendingAdapter()
    runner.__dict__["adapters"] = {Platform.DISCORD: adapter}
    runner._queued_events = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-goal-config",
        chat_type="channel",
        user_id="user-goal-config",
    )
    event = MessageEvent(
        text="/goal resume",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-goal-resume",
    )

    mgr = goals.GoalManager("sid-gateway-goal-config", default_max_turns=1)
    mgr.set("ship the benchmark")
    assert mgr.state is not None
    mgr.state.turns_used = 1
    mgr.pause(reason="turn budget exhausted (1/1)")

    try:
        response = await GatewayRunner._handle_goal_command(runner, event)

        assert "Goal resumed" in response
        session_key = runner._session_key_for_source(source)
        assert (
            adapter._pending_messages[session_key].text
            == goals.GoalManager("sid-gateway-goal-config").next_continuation_prompt()
        )
        resumed = goals.GoalManager("sid-gateway-goal-config").state
        assert resumed is not None
        assert resumed.status == "active"
        assert resumed.turns_used == 0
    finally:
        goals._DB_CACHE.clear()
