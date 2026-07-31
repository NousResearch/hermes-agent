import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
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


class _DrainAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="goal-resume-notice")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "channel"}


class _MultiplexSessionStore:
    def __init__(self):
        self.entry = _FakeSessionEntry()

    def get_or_create_session(self, source):
        return self.entry

    def _generate_session_key(self, source):
        return build_session_key(
            source,
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
            profile=source.profile,
        )


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


@pytest.mark.asyncio
async def test_gateway_goal_resume_drains_on_multiplexed_profile(tmp_path, monkeypatch):
    """The transport adapter must consume a secondary-profile continuation."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    platform_config = PlatformConfig(enabled=True, token="token")
    adapter = _DrainAdapter(platform_config, Platform.DISCORD)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        multiplex_profiles=True,
        platforms={Platform.DISCORD: platform_config},
    )
    runner.__dict__["session_store"] = _MultiplexSessionStore()
    runner.__dict__["adapters"] = {}
    runner._profile_adapters = {"coder": {Platform.DISCORD: adapter}}
    runner._active_profile_name = lambda: "default"
    runner._queued_events = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-goal-multiplex",
        chat_type="channel",
        user_id="user-goal-multiplex",
        profile="coder",
    )
    event = MessageEvent(
        text="/goal resume",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-goal-multiplex-resume",
    )
    mgr = goals.GoalManager("sid-gateway-goal-config", default_max_turns=1)
    mgr.set("ship the multiplexed benchmark")
    mgr.pause(reason="turn budget exhausted (1/1)")

    async def _get_goal_manager(_event):
        return mgr, _FakeSessionEntry()

    runner.__dict__["_get_goal_manager_for_event"] = _get_goal_manager
    handled: list[str] = []
    continuation_handled = asyncio.Event()

    async def _handle(incoming):
        handled.append(incoming.text)
        if incoming.get_command() == "goal":
            return await GatewayRunner._handle_goal_command(runner, incoming)
        continuation_handled.set()
        return ""

    adapter.set_message_handler(_handle)

    try:
        await adapter.handle_message(event)
        await asyncio.wait_for(continuation_handled.wait(), timeout=1.0)
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))

        assert handled == [
            "/goal resume",
            mgr.next_continuation_prompt(),
        ]
        assert adapter._pending_messages == {}
    finally:
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
        goals._DB_CACHE.clear()


def test_gateway_goal_resume_fifo_separates_multiplexed_slot_and_state_keys():
    """A racing user turn must not strand or cross-route the continuation."""
    adapter = _DrainAdapter(
        PlatformConfig(enabled=True, token="token"), Platform.DISCORD
    )
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-goal-multiplex-race",
        chat_type="channel",
        user_id="user-goal-multiplex-race",
        profile="coder",
    )
    adapter_key = adapter.session_key_for_source(source)
    state_key = build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
        profile="coder",
    )
    default_state_key = build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    assert adapter_key == default_state_key
    assert adapter_key != state_key

    user_event = MessageEvent(
        text="racing user turn",
        message_type=MessageType.TEXT,
        source=source,
    )
    continuation_event = MessageEvent(
        text="[Continuing toward your standing goal]\nGoal: ship safely",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter._pending_messages[adapter_key] = user_event

    runner._enqueue_fifo(
        state_key,
        continuation_event,
        adapter,
        adapter_session_key=adapter_key,
    )

    assert adapter._pending_messages.pop(adapter_key) is user_event
    assert runner._promote_queued_event(state_key, adapter, None) is continuation_event
    assert runner._promote_queued_event(default_state_key, adapter, None) is None
    assert adapter._pending_messages == {}

    adapter._pending_messages[adapter_key] = continuation_event
    runner._session_state(state_key).conversation.queued_events.extend(
        [user_event, continuation_event]
    )
    removed = runner._clear_goal_pending_continuations(
        state_key,
        adapter,
        adapter_session_key=adapter_key,
    )
    assert removed == 2
    assert adapter._pending_messages == {}
    assert runner._session_state(state_key).conversation.queued_events == [user_event]
