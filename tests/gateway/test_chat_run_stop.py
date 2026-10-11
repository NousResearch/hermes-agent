"""Ordinary chat Stop is fenced to the tool-originating run and profile."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL, _profile_runtime_scope
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_run_generation, set_session_vars
from gateway.turn_context import TurnContext
from hermes_cli.profiles import get_profile_dir
from tools.registry import _current_tool_invocation_context


def _running_turn(home: Path, *, generation: int = 3, agent=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1", profile="default")
    session_key = runner._session_key_for_source(source)
    state = runner._session_state(session_key)
    state.persistent.run_generation = generation
    state.turn.agent = _AGENT_PENDING_SENTINEL if agent is None else agent
    state.turn.event = MessageEvent(text="work", message_type=MessageType.TEXT, source=source)
    return runner, state, source, session_key


@pytest.mark.asyncio
async def test_chat_stop_requires_exact_run_and_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    runner, state, _, key = _running_turn(home)
    token = state.persistent.run_generation
    env_tokens = set_session_vars(session_key=key, profile="default")
    try:
        set_session_run_generation(token)
        invocation = _current_tool_invocation_context()
        assert (invocation.session_key, invocation.run_generation) == (key, token)
    finally:
        clear_session_vars(env_tokens)
    monkeypatch.setenv("HERMES_SESSION_RUN_GENERATION", "999")
    assert _current_tool_invocation_context().run_generation is None

    wrong = await runner.request_chat_run_stop(
        session_key="telegram:other", expected_run_generation=token, profile_home=home)
    assert wrong["status"] == "not_running"
    wrong_home = await runner.request_chat_run_stop(
        session_key=key, expected_run_generation=token, profile_home=tmp_path / "b")
    assert wrong_home["status"] == "stale"
    stale = await runner.request_chat_run_stop(
        session_key=key, expected_run_generation=token - 1, profile_home=home)
    assert stale["status"] == "stale"
    assert state.persistent.run_generation == token

    # The same runner serves a second real temporary profile. A token from A cannot
    # address B even when a webhook knows B's session key.
    other_home = get_profile_dir("secondary")
    other_home.mkdir(parents=True)
    (other_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    other_source = SessionSource(platform=Platform.TELEGRAM, chat_id="c2", chat_type="dm",
                                 user_id="u2", profile="secondary")
    other_key = runner._session_key_for_source(other_source)
    other = runner._session_state(other_key)
    other.persistent.run_generation = token
    other.turn.agent = _AGENT_PENDING_SENTINEL
    other.turn.event = MessageEvent(text="other", message_type=MessageType.TEXT, source=other_source)
    assert (await runner.request_chat_run_stop(
        session_key=other_key, expected_run_generation=token, profile_home=home))["status"] == "stale"

    human = MessageEvent(text="queued human", message_type=MessageType.TEXT, source=state.turn.event.source)
    wake = MessageEvent(text="internal wake", message_type=MessageType.TEXT,
                        source=state.turn.event.source, internal=True)
    adapter = SimpleNamespace(_pending_messages={key: human}, get_pending_message=lambda k: human)
    runner._delivery_adapter_for = lambda source: adapter if source.chat_id == "c1" else None
    state.conversation.queued_events.append(wake)

    accepted = await runner.request_chat_run_stop(
        session_key=key, expected_run_generation=token, profile_home=home)
    assert accepted["status"] == "accepted"
    assert accepted["worker_completion"] == "unknown"
    assert state.persistent.run_generation > token
    assert state.turn.agent is None
    assert adapter._pending_messages[key] is wake
    assert (await runner.request_chat_run_stop(
        session_key=other_key, expected_run_generation=token, profile_home=other_home))["status"] == "stale"
    with _profile_runtime_scope(other_home):
        assert (await runner.request_chat_run_stop(
            session_key=other_key, expected_run_generation=token, profile_home=other_home))["status"] == "accepted"
        assert (await runner.get_chat_run_stop_observation(
            session_key=key, run_generation=token, profile_home=home))["status"] == "stale"
    with _profile_runtime_scope(home):
        assert (await runner.get_chat_run_stop_observation(
            session_key=key, run_generation=token, profile_home=home))["stop_status"] == "accepted"
    assert (await runner.request_chat_run_stop(
        session_key=key, expected_run_generation=token, profile_home=home))["status"] == "stale"


@pytest.mark.asyncio
async def test_chat_stop_observes_worker_event_not_slot_release(tmp_path, monkeypatch):
    home = tmp_path / "a"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    class Agent:
        def __init__(self):
            self.calls = 0

        def interrupt(self, reason):
            self.calls += 1

    agent = Agent()
    runner, state, _, key = _running_turn(home, agent=agent)
    started, release = threading.Event(), threading.Event()

    def run_sync():
        started.set()
        assert release.wait(5)

    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    worker = runner._run_agent_start_turn_worker(
        SimpleNamespace(agent_holder=[agent], session_key=key, run_generation=3, session_id="session-1"),
        run_sync,
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        receipt = await runner.request_chat_run_stop(
            session_key=key, expected_run_generation=3, profile_home=home)
        assert receipt["status"] == "accepted"
        assert receipt["worker_completion"] == "pending"
        assert (await runner.get_chat_run_stop_observation(
            session_key=key, run_generation=3, profile_home=home))["worker_completion"] == "pending"
    finally:
        release.set()
        await worker.executor_task
    assert (await runner.get_chat_run_stop_observation(
        session_key=key, run_generation=3, profile_home=home))["worker_completion"] == "completed"

    successor = Agent()
    state.persistent.run_generation += 1
    state.turn.agent = successor
    state.turn.event = MessageEvent(text="successor", message_type=MessageType.TEXT,
                                    source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1",
                                                         chat_type="dm", user_id="u1", profile="default"))
    assert (await runner.request_chat_run_stop(
        session_key=key, expected_run_generation=3, profile_home=home))["status"] == "stale"
    assert successor.calls == 0

    # A successor can claim during the adapter await; the displaced Stop tail
    # must leave its slot and queued text untouched.
    generation = state.persistent.run_generation

    class RacingAdapter:
        async def interrupt_session_activity(self, *_args):
            state.persistent.run_generation += 1
            state.turn.agent = Agent()
            state.persistent.pending_command_text = "successor text"

    runner._delivery_adapter_for = lambda _source: RacingAdapter()
    runner._thread_metadata_for_source = lambda _source: {}
    raced = await runner.request_chat_run_stop(
        session_key=key, expected_run_generation=generation, profile_home=home)
    assert raced["status"] == "accepted"
    assert state.turn.agent is not successor
    assert state.persistent.pending_command_text == "successor text"

    class BrokenAgent:
        _gateway_turn_process_task_id = "task"
        _gateway_turn_process_baseline = frozenset()

        def interrupt(self, reason):
            raise RuntimeError("interrupt failed")

    state.turn.agent = BrokenAgent()
    runner._delivery_adapter_for = lambda _source: None
    with monkeypatch.context() as patch:
        patch.setattr("gateway.run_agent_cache.threading.Thread.start", lambda _self: (_ for _ in ()).throw(RuntimeError("reap failed")))
        failed = await runner.request_chat_run_stop(
            session_key=key, expected_run_generation=state.persistent.run_generation, profile_home=home)
    assert failed["status"] == "unsupported"
    assert failed["interrupt_delivery"] == "unsupported"
    assert failed["process_reap"] == "failed"
    assert state.turn.agent is None


@pytest.mark.asyncio
async def test_chat_stop_before_promotion_fences_model_launch_and_observes_late_worker(tmp_path, monkeypatch):
    home = tmp_path / "a"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    runner, _state, source, key = _running_turn(home)
    gate = threading.Event()
    ctx = TurnContext(source=source, session_key=key, run_generation=3,
                      internal_plugin_execution_id=None, execution_launch_gate=gate,
                      execution_launch_allowed=False, message="work")

    receipt = await runner.request_chat_run_stop(
        session_key=key, expected_run_generation=3, profile_home=home)
    assert receipt["status"] == "accepted"
    assert receipt["interrupt_delivery"] == "prelaunch"
    assert receipt["worker_completion"] == "unknown"

    ctx.agent_holder[0] = SimpleNamespace()
    await runner._run_agent_track_agent(ctx)
    assert gate.is_set() and ctx.execution_launch_allowed is False

    # The displaced preparation can still schedule its worker after Stop. Its
    # physical completion must attach to the original receipt, and the runner's
    # launch fence must deny model work for an ordinary chat turn.
    turn = TurnRunner(runner, ctx)
    monkeypatch.setattr(turn, "_combined_ephemeral_prompt", lambda: "")
    monkeypatch.setattr(turn, "_setup_stream_consumer", lambda _platform: (None, None, None, False))
    monkeypatch.setattr(turn, "_resolve_turn_agent", lambda *_args: (ctx.agent_holder[0], False))
    monkeypatch.setattr(turn, "_wire_turn_agent_callbacks", lambda *_args: None)
    monkeypatch.setattr(turn, "_load_turn_history", lambda *_args: pytest.fail("stopped worker loaded history"))
    runner._resolve_session_agent_runtime = lambda **_kw: ("model", {"provider": "test"})
    runner._provider_routing = {}
    runner._resolve_session_reasoning_config = lambda **_kw: {}
    runner._resolve_session_service_tier = lambda **_kw: None
    runner._resolve_turn_agent_config = lambda *_args: {}
    started, release = threading.Event(), threading.Event()

    def run_stopped_worker():
        started.set()
        assert release.wait(5)
        return turn.run_sync()

    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    worker = runner._run_agent_start_turn_worker(ctx, run_stopped_worker)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        assert (await runner.get_chat_run_stop_observation(
            session_key=key, run_generation=3, profile_home=home))["worker_completion"] == "pending"
    finally:
        release.set()
    assert (await worker.executor_task)["interrupted"] is True
    observed = await runner.get_chat_run_stop_observation(
        session_key=key, run_generation=3, profile_home=home)
    assert observed["worker_completion"] == "completed"
    assert observed["stop_status"] == "accepted"
    assert observed["effects"] == "unknown"
