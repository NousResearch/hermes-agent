"""Gateway regressions for terminal /steer FIFO delivery.

A terminal agent result can carry ``pending_steer`` when the turn exits before
its finalizer can inject an accepted steer.  The gateway must deliver older
transport work first, then the steer exactly once, without turning the queued
fallback into an interrupt.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    PlatformConfig,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _TerminalSteerAdapter(BasePlatformAdapter):
    def __init__(self) -> None:
        super().__init__(
            PlatformConfig(enabled=True, token="test"),
            Platform.TELEGRAM,
        )
        self.sent: list[str] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _RecordingInterruptEvent(asyncio.Event):
    def __init__(self) -> None:
        super().__init__()
        self.set_calls = 0

    def set(self) -> None:
        self.set_calls += 1
        super().set()


class _TerminalSteerAgent:
    started = threading.Event()
    release_terminal_result = threading.Event()
    messages: list[str] = []
    interrupts: list[str] = []
    terminal_steer = "terminal steer"

    def __init__(self, **kwargs) -> None:
        self.tools = []

    def interrupt(self, reason="") -> bool:
        type(self).interrupts.append(reason)
        return True

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).messages.append(message)
        if len(type(self).messages) == 1:
            type(self).started.set()
            if not type(self).release_terminal_result.wait(timeout=5):
                raise AssertionError("test did not release terminal result")
            return {
                "final_response": "Response truncated at the output cap",
                "messages": [],
                "api_calls": 1,
                "completed": False,
                "partial": True,
                "pending_steer": type(self).terminal_steer,
            }
        return {
            "final_response": f"processed: {message}",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _make_runner(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", _TerminalSteerAgent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    adapter = _TerminalSteerAdapter()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._queued_events = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        multiplex_profiles=False,
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )
    return gateway_run, runner, adapter


def _queued_event(source: SessionSource) -> MessageEvent:
    return MessageEvent(
        text="older queued work",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-before-terminal",
    )


@pytest.mark.parametrize("profile", [None, "coder"], ids=["default", "named-profile"])
def test_terminal_steer_enqueue_respects_durable_overflow_after_goal_clear(
    monkeypatch,
    tmp_path,
    profile,
):
    """An empty physical slot does not make a non-empty logical FIFO empty."""
    _gateway_run, runner, adapter = _make_runner(monkeypatch, tmp_path)
    runner.config.multiplex_profiles = profile is not None
    if profile is not None:
        runner._profile_adapters = {profile: {Platform.TELEGRAM: adapter}}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-goal-clear-terminal-steer",
        chat_type="dm",
        user_id="user-goal-clear",
        profile=profile,
    )
    state_key = runner._session_key_for_source(source)
    adapter_key = adapter.session_key_for_source(source)
    stale_goal = MessageEvent(
        text="[Continuing toward your standing goal]\nGoal: stale goal",
        message_type=MessageType.TEXT,
        source=source,
    )
    older_user = _queued_event(source)
    terminal_steer = MessageEvent(
        text="terminal steer",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
    )
    setattr(terminal_steer, "_gateway_terminal_steer", True)

    runner._enqueue_fifo(
        state_key,
        stale_goal,
        adapter,
        adapter_key=adapter_key,
    )
    runner._enqueue_fifo(
        state_key,
        older_user,
        adapter,
        adapter_key=adapter_key,
    )
    assert runner._clear_goal_pending_continuations(
        state_key,
        adapter,
        source=source,
    ) == 1
    assert adapter_key not in adapter._pending_messages
    assert runner._queued_events[state_key] == [older_user]

    # This is the terminal-result caller's enqueue at gateway/run.py. The old
    # implementation saw the empty physical slot and inserted the steer there,
    # jumping it ahead of the older durable overflow item.
    runner._enqueue_fifo(
        state_key,
        terminal_steer,
        adapter,
        adapter_key=adapter_key,
    )

    drained = []
    while True:
        pending = adapter.get_pending_message(adapter_key)
        pending = runner._promote_queued_event(
            state_key,
            adapter,
            pending,
            adapter_key=adapter_key,
        )
        if pending is None:
            break
        drained.append(pending)

    assert [event.text for event in drained] == [
        "older queued work",
        "terminal steer",
    ]
    assert drained[1] is terminal_steer
    assert getattr(drained[1], "_gateway_terminal_steer", False) is True
    assert adapter_key not in adapter._pending_messages
    assert runner._queued_events.get(state_key, []) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queue_before_run",
    [True, False],
    ids=["queue-before-agent-start", "queue-while-terminal-result-blocked"],
)
async def test_terminal_steer_runs_once_behind_older_transport_work_without_interrupt(
    monkeypatch,
    tmp_path,
    queue_before_run,
):
    """Control both queue/result orderings with events, not scheduler timing."""
    _TerminalSteerAgent.started = threading.Event()
    _TerminalSteerAgent.release_terminal_result = threading.Event()
    _TerminalSteerAgent.messages = []
    _TerminalSteerAgent.interrupts = []
    _TerminalSteerAgent.terminal_steer = "terminal steer"

    _gateway_run, runner, adapter = _make_runner(monkeypatch, tmp_path)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-terminal-steer",
        chat_type="dm",
        user_id="user-1",
    )
    session_key = build_session_key(source)
    queued = _queued_event(source)
    interrupt_event = _RecordingInterruptEvent()
    adapter._active_sessions[session_key] = interrupt_event

    if queue_before_run:
        runner._enqueue_fifo(session_key, queued, adapter)

    task = asyncio.create_task(
        runner._run_agent(
            message="original turn",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-terminal-steer",
            session_key=session_key,
        )
    )
    assert await asyncio.to_thread(_TerminalSteerAgent.started.wait, 2)

    if not queue_before_run:
        runner._enqueue_fifo(session_key, queued, adapter)

    active_event = adapter._active_sessions[session_key]
    assert not active_event.is_set(), "queue fallback must not request an interrupt"

    _TerminalSteerAgent.release_terminal_result.set()
    result = await asyncio.wait_for(task, timeout=5)

    assert _TerminalSteerAgent.messages == [
        "original turn",
        "older queued work",
        "terminal steer",
    ]
    assert _TerminalSteerAgent.messages.count("terminal steer") == 1
    assert _TerminalSteerAgent.interrupts == []
    assert interrupt_event.set_calls == 0
    assert session_key not in adapter._pending_messages
    assert runner._queued_events.get(session_key, []) == []
    assert result["final_response"] == "processed: terminal steer"


@pytest.mark.asyncio
@pytest.mark.parametrize("older_head", [False, True], ids=["direct", "behind-older-head"])
async def test_command_shaped_terminal_steer_keeps_accepted_provenance(
    monkeypatch,
    tmp_path,
    older_head,
):
    """Trusted terminal steer text is user input even when it resembles /stop."""
    _TerminalSteerAgent.started = threading.Event()
    _TerminalSteerAgent.release_terminal_result = threading.Event()
    _TerminalSteerAgent.messages = []
    _TerminalSteerAgent.interrupts = []
    _TerminalSteerAgent.terminal_steer = "/stop"

    _gateway_run, runner, adapter = _make_runner(monkeypatch, tmp_path)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-command-steer",
        chat_type="dm",
        user_id="user-command",
    )
    state_key = runner._session_key_for_source(source)
    adapter_key = adapter.session_key_for_source(source)
    adapter._active_sessions[adapter_key] = _RecordingInterruptEvent()
    if older_head:
        runner._enqueue_fifo(
            state_key,
            _queued_event(source),
            adapter,
            adapter_key=adapter_key,
        )

    task = asyncio.create_task(
        runner._run_agent(
            message="original turn",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-command-steer",
            session_key=state_key,
        )
    )
    assert await asyncio.to_thread(_TerminalSteerAgent.started.wait, 2)
    _TerminalSteerAgent.release_terminal_result.set()
    result = await asyncio.wait_for(task, timeout=5)

    expected = ["original turn"]
    if older_head:
        expected.append("older queued work")
    expected.append("/stop")
    assert _TerminalSteerAgent.messages == expected
    assert result["final_response"] == "processed: /stop"
    assert adapter_key not in adapter._pending_messages
    assert runner._queued_events.get(state_key, []) == []


@pytest.mark.asyncio
async def test_named_profile_keeps_adapter_slot_and_durable_fifo_keys_aligned(
    monkeypatch,
    tmp_path,
):
    _TerminalSteerAgent.started = threading.Event()
    _TerminalSteerAgent.release_terminal_result = threading.Event()
    _TerminalSteerAgent.messages = []
    _TerminalSteerAgent.interrupts = []
    _TerminalSteerAgent.terminal_steer = "terminal steer"

    _gateway_run, runner, adapter = _make_runner(monkeypatch, tmp_path)
    runner.config.multiplex_profiles = True
    runner._profile_adapters = {"coder": {Platform.TELEGRAM: adapter}}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-multiplex-steer",
        chat_type="dm",
        user_id="user-coder",
        profile="coder",
    )
    state_key = runner._session_key_for_source(source)
    adapter_key = adapter.session_key_for_source(source)
    assert state_key == adapter_key
    adapter._active_sessions[adapter_key] = _RecordingInterruptEvent()
    runner._enqueue_fifo(
        state_key,
        _queued_event(source),
        adapter,
        adapter_key=adapter_key,
    )

    task = asyncio.create_task(
        runner._run_agent(
            message="original turn",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-multiplex-steer",
            session_key=state_key,
        )
    )
    assert await asyncio.to_thread(_TerminalSteerAgent.started.wait, 2)
    _TerminalSteerAgent.release_terminal_result.set()
    result = await asyncio.wait_for(task, timeout=5)

    assert _TerminalSteerAgent.messages == [
        "original turn",
        "older queued work",
        "terminal steer",
    ]
    assert result["final_response"] == "processed: terminal steer"
    assert adapter_key not in adapter._pending_messages
    assert state_key not in adapter._pending_messages
    assert runner._queued_events.get(state_key, []) == []
    assert runner._queued_events.get(adapter_key, []) == []


@pytest.mark.asyncio
async def test_depth_cap_requeues_current_head_without_displacing_terminal_steer(
    monkeypatch,
    tmp_path,
):
    """MAX_DEPTH + 1 older events and terminal steer retain exact FIFO order."""
    _TerminalSteerAgent.started = threading.Event()
    _TerminalSteerAgent.release_terminal_result = threading.Event()
    _TerminalSteerAgent.messages = []
    _TerminalSteerAgent.interrupts = []
    _TerminalSteerAgent.terminal_steer = "terminal steer"

    _gateway_run, runner, adapter = _make_runner(monkeypatch, tmp_path)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-depth-steer",
        chat_type="dm",
        user_id="user-depth",
    )
    state_key = runner._session_key_for_source(source)
    adapter_key = adapter.session_key_for_source(source)
    adapter._active_sessions[adapter_key] = _RecordingInterruptEvent()
    older_texts = [f"older-{idx}" for idx in range(runner._MAX_INTERRUPT_DEPTH + 1)]
    for idx, text in enumerate(older_texts):
        runner._enqueue_fifo(
            state_key,
            MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=f"older-{idx}",
            ),
            adapter,
            adapter_key=adapter_key,
        )

    first_chain = asyncio.create_task(
        runner._run_agent(
            message="original turn",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-depth-steer",
            session_key=state_key,
        )
    )
    assert await asyncio.to_thread(_TerminalSteerAgent.started.wait, 2)
    _TerminalSteerAgent.release_terminal_result.set()
    await asyncio.wait_for(first_chain, timeout=5)

    assert _TerminalSteerAgent.messages == ["original turn", *older_texts[:-1]]
    assert adapter._pending_messages[adapter_key].text == older_texts[-1]
    assert [event.text for event in runner._queued_events[state_key]] == [
        "terminal steer"
    ]

    current = adapter.get_pending_message(adapter_key)
    assert current is not None
    await asyncio.wait_for(
        runner._run_agent(
            message=current.text,
            context_prompt="",
            history=[],
            source=current.source,
            session_id="sess-depth-steer",
            session_key=state_key,
        ),
        timeout=5,
    )

    assert _TerminalSteerAgent.messages == [
        "original turn",
        *older_texts,
        "terminal steer",
    ]
    assert adapter_key not in adapter._pending_messages
    assert runner._queued_events.get(state_key, []) == []
