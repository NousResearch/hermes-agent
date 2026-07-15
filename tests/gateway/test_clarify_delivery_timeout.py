"""Regression coverage for synchronous gateway clarify delivery timeouts."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import types
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class _DelayedClarifyAdapter(BasePlatformAdapter):
    def __init__(self) -> None:
        super().__init__(
            PlatformConfig(enabled=True, token="test-token"),
            Platform.TELEGRAM,
        )
        self.started = threading.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.published: list[str] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="message")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send_clarify(self, **kwargs) -> SendResult:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        self.published.append(kwargs["clarify_id"])
        return SendResult(success=True, message_id="late-message")


class _ClarifyAgent:
    def __init__(self, **kwargs) -> None:
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        response = self.clarify_callback("Choose", ["A", "B"])
        return {"final_response": response, "messages": [], "api_calls": 1}


class _ShortWaitFuture:
    """Use a real scheduled future while shortening the production wait."""

    def __init__(self, inner: Future, started: threading.Event) -> None:
        self._inner = inner
        self._started = started

    def result(self, timeout=None):
        assert self._started.wait(timeout=1)
        return self._inner.result(timeout=0.01)

    def cancel(self) -> bool:
        return self._inner.cancel()

    def cancelled(self) -> bool:
        return self._inner.cancelled()


def _make_runner(adapter: BasePlatformAdapter):
    gateway_run = importlib.import_module("gateway.run")
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
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


@pytest.mark.asyncio
async def test_clarify_delivery_timeout_cancels_send_before_cleanup(monkeypatch, tmp_path):
    """A timed-out adapter send cannot publish after its request is cleared."""
    from agent.async_utils import safe_schedule_threadsafe as real_schedule
    from tools import clarify_gateway

    adapter = _DelayedClarifyAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _ClarifyAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "test-key"},
    )

    scheduled: list[_ShortWaitFuture] = []

    def _schedule(coro, loop, **kwargs):
        inner = real_schedule(coro, loop, **kwargs)
        assert inner is not None
        wrapped = _ShortWaitFuture(inner, adapter.started)
        scheduled.append(wrapped)
        return wrapped

    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", _schedule)

    cleanup_saw_cancelled: list[bool] = []
    real_clear_session = clarify_gateway.clear_session

    def _record_cleanup(session_key: str) -> int:
        cleanup_saw_cancelled.append(scheduled[0].cancelled())
        return real_clear_session(session_key)

    monkeypatch.setattr(clarify_gateway, "clear_session", _record_cleanup)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat")
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="session",
        session_key="agent:main:telegram:dm:chat",
    )

    adapter.release.set()
    await asyncio.sleep(0.05)

    assert result["final_response"] == "[clarify prompt could not be delivered]"
    assert cleanup_saw_cancelled[0] is True
    assert adapter.cancelled.is_set()
    assert adapter.published == []
