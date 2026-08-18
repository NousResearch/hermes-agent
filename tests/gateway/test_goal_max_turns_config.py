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
async def test_goal_command_slow_db_init_keeps_loop_free_and_persists(tmp_path, monkeypatch):
    """A slow state.db init (cold cache, first /goal of the process)
    must not freeze the event loop or silently drop the goal write.
    The cache is warmed off-loop, so the reply is honest at any init
    duration. Review follow-up (#88965): the bootstrap window alone
    froze the loop for the init duration and still dropped the write
    past ~1.5s."""
    import hermes_state

    # Past the init window: the window-only path expires its wait and
    # drops the write, so this test discriminates warm-up from windows.
    # The margin is 2.5s so the loop-gap ceiling below can be 2.0s (the
    # flake-policy floor) and an on-loop init still exceeds it.
    INIT_S = goals._DB_BOOTSTRAP_INIT_WAIT_S + 2.5

    class _SlowSessionDB(hermes_state.SessionDB):
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

    gaps = {"max": 0.0}
    stop = asyncio.Event()

    async def _ticker():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.05)
            now = time.monotonic()
            gaps["max"] = max(gaps["max"], now - last)
            last = now

    ticker = asyncio.create_task(_ticker())
    await asyncio.sleep(0.15)
    try:
        response = await GatewayRunner._handle_goal_command(runner, event)

        assert "⊙ Goal set (7-turn budget): ship the benchmark" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None, "goal write must persist even with a slow init"
        # 2.0s is the loose-bound floor from the flake policy. The init
        # (4.0s) runs off-loop, so an on-loop regression exceeds this
        # ceiling and a loaded runner does not.
        assert gaps["max"] < 2.0, (
            f"event loop frozen for {gaps['max']:.2f}s while the init ran off-loop"
        )
    finally:
        stop.set()
        await ticker
        goals._DB_CACHE.clear()
