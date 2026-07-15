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


def test_gateway_preserves_explicit_zero_goal_turn_cap(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"goals": {"max_turns": 0}},
    )

    assert runner._goal_max_turns_from_config() == 0


def test_gateway_does_not_treat_boolean_false_as_unbounded(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"goals": {"max_turns": False}},
    )

    assert runner._goal_max_turns_from_config() == 20


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
    adapter = object()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._queued_events = {}
    queued = []
    runner._session_key_for_source = lambda source: "goal-config-key"
    runner._enqueue_fifo = lambda key, queued_event, selected_adapter: queued.append(
        (key, queued_event, selected_adapter)
    )

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
        assert len(queued) == 1
        assert "Begin working toward your standing goal" in queued[0][1].text
        assert "goal_outcome" in queued[0][1].text
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_draft_queues_primary_model_workspace_turn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner._queued_events = {}
    adapter = object()
    runner.adapters = {Platform.DISCORD: adapter}
    queued = []
    runner._session_key_for_source = lambda source: "goal-draft-key"
    runner._enqueue_fifo = lambda key, queued_event, selected_adapter: queued.append(
        (key, queued_event, selected_adapter)
    )

    event = MessageEvent(
        text="/goal draft migrate auth to JWT",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-draft",
            chat_type="channel",
            user_id="user-goal-draft",
        ),
        message_id="msg-goal-draft",
    )

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.goal == "migrate auth to JWT"
        assert not state.has_contract()
        assert "queued primary-model turn" in response
        assert len(queued) == 1
        assert queued[0][0] == "goal-draft-key"
        assert queued[0][2] is adapter
        assert "Author your standing-goal workspace" in queued[0][1].text
        assert "goal_contract through the todo tool" in queued[0][1].text
    finally:
        goals._DB_CACHE.clear()
