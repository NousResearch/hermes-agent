"""Cancelling a session's processing task must stop the agent it was awaiting.

The agent runs on an executor thread; cancelling the asyncio task that awaits it
does not stop that thread. The turn's ``finally`` then releases the running-agent
slot, so a later ``_interrupt_and_clear_session`` finds no agent to interrupt.
Without an interrupt on cancellation the abandoned agent keeps calling the model
and running tools: a socket the eviction aborts is retried as a transient stream
drop, because nothing ever set the agent's interrupt flag.
"""

import asyncio
import importlib
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource

_SESSION_KEY = "agent:main:telegram:group:-1001:17585"


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="m-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _LongTurnAgent:
    """Keeps "working" until it is hard-interrupted (or the test releases it)."""

    instances: list = []

    def __init__(self, **kwargs):
        self.tools = []
        self._interrupt_requested = False
        self._interrupt_message = None
        self.hard_interrupts: list = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        _LongTurnAgent.instances.append(self)

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def hard_interrupt(self, message=None, *, tool_reason=None):
        self.hard_interrupts.append((message, tool_reason))
        self._interrupt_requested = True
        self._interrupt_message = message
        self.release.set()

    def interrupt(self, message=None):
        self.hard_interrupt(message)

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.started.set()
        self.release.wait(5.0)
        self.finished.set()
        return {"final_response": "", "messages": [], "api_calls": 1,
                "interrupted": self._interrupt_requested}


def _make_runner(adapter):
    GatewayRunner = importlib.import_module("gateway.run").GatewayRunner
    runner = object.__new__(GatewayRunner)
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
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False, group_sessions_per_user=False, stt_enabled=False)
    return runner


async def _start_turn(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _LongTurnAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    _LongTurnAgent.instances = []

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    runner = _make_runner(_Adapter())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", thread_id="17585")
    turn = asyncio.ensure_future(runner._run_agent(
        message="hi", context_prompt="", history=[], source=source,
        session_id="sess-cancelled", session_key=_SESSION_KEY))
    for _ in range(500):
        if _LongTurnAgent.instances and _LongTurnAgent.instances[0].started.is_set():
            break
        await asyncio.sleep(0.01)
    agent = _LongTurnAgent.instances[0]
    assert agent.started.is_set(), "harness: the agent never started"
    return turn, agent


async def _cancel(turn, release=None):
    # What adapter.cancel_session_processing() does to the session's processing task.
    turn.cancel()
    if release is not None:
        # Let a still-running agent finish once the cancel handler has run, so the turn's
        # cleanup does not sit out its 5s stream-flush wait.
        asyncio.get_running_loop().call_later(0.1, release.set)
    with pytest.raises(asyncio.CancelledError):
        await turn


@pytest.mark.asyncio
async def test_cancelling_the_turn_task_hard_interrupts_the_running_agent(monkeypatch, tmp_path):
    turn, agent = await _start_turn(monkeypatch, tmp_path)
    await _cancel(turn)

    # The abandoned agent must have been told to stop, and must actually stop.
    assert agent.hard_interrupts, "cancelled turn left its agent running with no interrupt"
    assert await asyncio.to_thread(agent.finished.wait, 2.0)
    assert agent._interrupt_requested


@pytest.mark.asyncio
async def test_cancel_keeps_an_earlier_interrupt_reason(monkeypatch, tmp_path):
    """A shutdown, /stop or /new that already interrupted the agent keeps its own reason."""
    turn, agent = await _start_turn(monkeypatch, tmp_path)
    agent._interrupt_requested = True
    agent._interrupt_message = "Gateway shutting down"
    try:
        await _cancel(turn, release=agent.release)
        assert agent.hard_interrupts == []
        assert agent._interrupt_message == "Gateway shutting down"
    finally:
        agent.release.set()


@pytest.mark.asyncio
async def test_cancel_after_the_result_is_published_does_not_interrupt(monkeypatch, tmp_path):
    """Once the turn published its result the finalizer has cleared the flag; re-arming it
    would end the cached agent's next turn at once."""
    GatewayRunner = importlib.import_module("gateway.run").GatewayRunner

    async def published_then_waiting(self, worker, turn_ctx, *_args):
        turn_ctx.result_holder[0] = {"final_response": "done", "messages": [], "api_calls": 1}
        await asyncio.Event().wait()

    monkeypatch.setattr(GatewayRunner, "_run_agent_poll_turn_worker", published_then_waiting)
    turn, agent = await _start_turn(monkeypatch, tmp_path)
    try:
        await _cancel(turn, release=agent.release)
        assert agent.hard_interrupts == []
        assert not agent._interrupt_requested
    finally:
        agent.release.set()
