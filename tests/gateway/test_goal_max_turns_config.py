import asyncio
import time

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

    def get_or_create_session(self, source, **_kwargs):
        return self.entry

    def _generate_session_key(self, source):
        return "agent:main:discord:channel:goal-config"


def _make_runner() -> GatewayRunner:
    """GatewayRunner skeleton for the /goal command path (no adapters)."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    return runner


def _make_goal_event() -> MessageEvent:
    return MessageEvent(
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


@pytest.mark.asyncio
async def test_gateway_goal_uses_goals_max_turns_from_full_config(tmp_path, monkeypatch):
    """Gateway /goal should honor top-level goals.max_turns from config.yaml."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    # Pre-warm from sync context: the /goal handler runs on the event loop,
    # where a cold cache only waits the bounded bootstrap window — under CI
    # load the goal write can be dropped and the state assertion flakes.
    goals._get_session_db()

    runner = _make_runner()

    event = _make_goal_event()

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "⊙ Goal set (7-turn budget): ship the benchmark" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.max_turns == 7
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_goal_command_slow_db_init_still_persists(tmp_path, monkeypatch):
    """A slow state.db init (cold cache, first /goal of the process) must
    not silently drop the goal write: the gateway warms the cache off-loop,
    so the reply is honest at any init duration.

    The init is deliberately slower than the (shrunk) bootstrap init
    window, so the window-only path would drop the write — this test
    discriminates the off-loop warm-up from mere window-widening without
    multi-second sleeps. Loop-freeze bounds are covered separately in
    tests/hermes_cli/test_goals_db_bootstrap_off_loop.py; no wall-clock
    gap assertions here (those are their own flake class).
    """
    import hermes_state

    monkeypatch.setattr(goals, "_DB_BOOTSTRAP_INIT_WAIT_S", 0.2)
    INIT_S = 0.8  # past the shrunk init window

    real_session_db = hermes_state.SessionDB

    class _SlowSessionDB(real_session_db):
        def __init__(self, *a, **k):
            time.sleep(INIT_S)
            super().__init__(*a, **k)

    monkeypatch.setattr(hermes_state, "SessionDB", _SlowSessionDB)

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = _make_runner()
    event = _make_goal_event()

    try:
        response = await GatewayRunner._handle_goal_command(runner, event)

        assert "⊙ Goal set (7-turn budget): ship the benchmark" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None, "goal write must persist even with a slow init"
        assert state.max_turns == 7
    finally:
        goals._DB_CACHE.clear()
