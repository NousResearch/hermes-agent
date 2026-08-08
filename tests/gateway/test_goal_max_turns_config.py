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


@pytest.mark.asyncio
async def test_gateway_replacing_goal_removes_only_old_multiplexed_continuations(
    tmp_path, monkeypatch
):
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
        chat_id="chat-goal-replace",
        chat_type="channel",
        user_id="user-goal-replace",
        profile="coder",
    )
    event = MessageEvent(
        text="/goal replacement goal",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-goal-replace",
    )
    mgr = goals.GoalManager("sid-gateway-goal-replace")
    mgr.set("old goal")

    async def _get_goal_manager(_event):
        return mgr, _FakeSessionEntry()

    runner.__dict__["_get_goal_manager_for_event"] = _get_goal_manager
    adapter_key = adapter.session_key_for_source(source)
    state_key = runner._session_key_for_source(source)
    old_prompt = mgr.next_continuation_prompt()
    assert old_prompt is not None
    old_continuation = MessageEvent(
        text=old_prompt,
        message_type=MessageType.TEXT,
        source=source,
    )
    real_user = MessageEvent(
        text="older real user input",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter._pending_messages[adapter_key] = old_continuation
    runner._session_state(state_key).conversation.queued_events.extend(
        [real_user, old_continuation]
    )

    try:
        await GatewayRunner._handle_goal_command(runner, event)

        assert adapter._pending_messages == {}
        first = runner._dequeue_and_promote_queued_event(state_key, adapter, source)
        second = runner._dequeue_and_promote_queued_event(state_key, adapter, source)
        assert first is real_user
        assert second is not None
        assert second.text == "replacement goal"
        assert runner._dequeue_and_promote_queued_event(state_key, adapter, source) is None
        assert mgr.state is not None
        assert mgr.state.goal == "replacement goal"
        assert runner._goal_still_active_for_session(mgr.session_id, old_prompt) is False
        assert (
            runner._goal_still_active_for_session(
                mgr.session_id, mgr.next_continuation_prompt()
            )
            is True
        )
    finally:
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

    first = runner._dequeue_and_promote_queued_event(state_key, adapter, source)
    assert first is user_event
    assert adapter._pending_messages[adapter_key] is continuation_event

    second = runner._dequeue_and_promote_queued_event(state_key, adapter, source)
    assert second is continuation_event
    assert runner._dequeue_and_promote_queued_event(state_key, adapter, source) is None
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


@pytest.mark.asyncio
async def test_gateway_busy_secondary_adapter_stamps_real_user_before_fifo(monkeypatch):
    """Real adapter ingestion must retain the secondary profile before queueing."""
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
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._restart_requested = False
    runner._is_user_authorized = lambda source: True

    async def _no_compression(session_key: str) -> bool:
        return False

    runner._session_has_compression_in_flight = _no_compression
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    setattr(adapter, "gateway_runner", runner)
    runner._configure_profile_adapter(adapter, "coder", Platform.DISCORD)
    adapter._busy_text_mode = "interrupt"

    source = adapter.build_source(
        chat_id="chat-goal-busy-race",
        chat_type="channel",
        user_id="user-goal-busy-race",
    )
    assert source.profile == "coder"
    adapter_key = adapter.session_key_for_source(source)
    state_key = runner._session_key_for_source(source)
    default_source = SessionSource(
        platform=source.platform,
        chat_id=source.chat_id,
        chat_type=source.chat_type,
        user_id=source.user_id,
    )
    default_state_key = runner._session_key_for_source(default_source)
    assert state_key != default_state_key

    # Exercise the pre-wrapper shape identified by review: even a source that
    # reaches transport ingress without a profile must be stamped before the
    # active-session busy slot owns it.
    source.profile = None
    first_user = MessageEvent(
        text="older real user 1",
        message_type=MessageType.TEXT,
        source=source,
    )
    second_user = MessageEvent(
        text="older real user 2",
        message_type=MessageType.TEXT,
        source=source,
    )

    class _BusyAgent:
        _active_children = []
        _supports_active_turn_redirect = False

        def interrupt(self, _text=None):
            return None

    runner._session_state(state_key).turn.agent = _BusyAgent()
    adapter._active_sessions[adapter_key] = asyncio.Event()
    current_task = asyncio.current_task()
    assert current_task is not None
    adapter._session_tasks[adapter_key] = current_task
    await adapter.handle_message(first_user)
    await adapter.handle_message(second_user)

    continuation = MessageEvent(
        text="[Continuing toward your standing goal]\nGoal: ship safely",
        message_type=MessageType.TEXT,
        source=source,
    )
    runner._enqueue_fifo(
        state_key,
        continuation,
        adapter,
        adapter_session_key=adapter_key,
    )
    assert (
        runner._queue_depth(
            state_key,
            adapter=adapter,
            adapter_session_key=adapter_key,
        )
        == 3
    )

    first = runner._dequeue_and_promote_queued_event(state_key, adapter, source)
    second = runner._dequeue_and_promote_queued_event(state_key, adapter, source)
    third = runner._dequeue_and_promote_queued_event(state_key, adapter, source)
    assert first is not None
    assert second is not None
    assert third is not None
    assert first is first_user
    assert second is second_user
    assert third is continuation
    assert all(item.source.profile == "coder" for item in (first, second, third))
    assert all(runner._session_key_for_source(item.source) == state_key for item in (first, second, third))
    assert runner._promote_queued_event(default_state_key, adapter, None) is None
    assert runner._dequeue_and_promote_queued_event(state_key, adapter, source) is None
    default_state = runner._peek_session_state(default_state_key)
    assert default_state is None or default_state.conversation.queued_events == []
    assert runner._session_state(state_key).conversation.queued_events == []
    assert adapter._pending_messages == {}
