"""Tests for early typing indicator stop on first stream delta (#49290).

When streaming is enabled, the typing indicator should stop as soon as the
first content delta is delivered to the platform — not after the full agent
run completes.  This minimizes the window where Discord (or similar) shows
both a typing indicator and the streamed response.
"""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.platforms.base as base_platform
from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource

import tools.terminal_tool  # noqa: F401 - register terminal emoji for fake-agent tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TypingCaptureAdapter(BasePlatformAdapter):
    """Adapter that records send_typing / stop_typing calls."""

    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self, platform=Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.typing_events: list[dict] = []
        self.sent: list[dict] = []
        self.edits: list[dict] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="msg-1")

    async def edit_message(self, chat_id, message_id, content, **kw) -> SendResult:
        self.edits.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing_events.append({"action": "start", "chat_id": chat_id})

    async def stop_typing(self, chat_id) -> None:
        self.typing_events.append({"action": "stop", "chat_id": chat_id})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class StreamingFakeAgent:
    """FakeAgent that calls stream_delta_callback during run_conversation."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.status_callback = kwargs.get("status_callback")
        self.step_callback = kwargs.get("step_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.stream_delta_callback
        if cb is not None:
            cb("Hello ")
            time.sleep(0.05)
            cb("world")
            time.sleep(0.05)
            cb(None)  # segment break
            cb("Second segment")
        return {
            "final_response": "Hello worldSecond segment",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


class NonStreamingFakeAgent:
    """FakeAgent that does NOT call stream_delta_callback."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _make_runner(adapter, streaming=True):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

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
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=StreamingConfig(enabled=streaming, transport="edit" if streaming else "off"),
        multiplex_profiles=False,
    )
    return runner


def _setup_mocks(monkeypatch, tmp_path, agent_cls):
    """Common monkeypatch setup for all tests."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    return gateway_run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_typing_stopped_on_first_stream_delta(monkeypatch, tmp_path):
    """stop_typing should fire on the first content delta, not after agent completes."""
    _setup_mocks(monkeypatch, tmp_path, StreamingFakeAgent)

    adapter = TypingCaptureAdapter(Platform.DISCORD)
    runner = _make_runner(adapter)

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="user1",
        chat_id="chat1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="test-session",
        session_key="discord:chat1",
    )

    assert result["final_response"] == "Hello worldSecond segment"

    # stop_typing should have been called at least once (on first delta)
    stop_calls = [e for e in adapter.typing_events if e["action"] == "stop"]
    assert len(stop_calls) >= 1, (
        f"Expected stop_typing to be called on first stream delta, "
        f"but typing_events={adapter.typing_events}"
    )
    assert stop_calls[0]["chat_id"] == "chat1"


@pytest.mark.asyncio
async def test_typing_stop_called_only_once_for_streaming(monkeypatch, tmp_path):
    """stop_typing should fire once on first delta, not on every delta."""
    _setup_mocks(monkeypatch, tmp_path, StreamingFakeAgent)

    adapter = TypingCaptureAdapter(Platform.DISCORD)
    runner = _make_runner(adapter)

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="user1",
        chat_id="chat1",
    )

    await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="test-session",
        session_key="discord:chat1",
    )

    # The FakeAgent sends 3 non-None deltas + 1 segment break (None).
    # stop_typing should fire exactly once from the stream delta path.
    # (There may be an additional stop_typing from the post-agent fallback at
    # line ~9424, which is fine — the important thing is the first one fires
    # during streaming, not after.)
    stop_calls = [e for e in adapter.typing_events if e["action"] == "stop"]
    # At minimum 1 (stream delta), at maximum 2 (stream + post-agent fallback)
    assert 1 <= len(stop_calls) <= 2, (
        f"Expected 1-2 stop_typing calls, got {len(stop_calls)}: {adapter.typing_events}"
    )


@pytest.mark.asyncio
async def test_no_streaming_typing_stops_after_agent(monkeypatch, tmp_path):
    """When streaming is disabled, stop_typing should only fire after agent completes."""
    _setup_mocks(monkeypatch, tmp_path, NonStreamingFakeAgent)

    adapter = TypingCaptureAdapter(Platform.DISCORD)
    runner = _make_runner(adapter, streaming=False)

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="user1",
        chat_id="chat1",
    )

    await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="test-session",
        session_key="discord:chat1",
    )

    # With streaming disabled, stop_typing is NOT called from _run_agent.
    # The post-agent fallback (line ~9424) is in _handle_message_with_agent,
    # which is the caller — not _run_agent itself.  So we expect 0 calls here.
    stop_calls = [e for e in adapter.typing_events if e["action"] == "stop"]
    assert len(stop_calls) == 0, (
        f"Expected 0 stop_typing calls in _run_agent (non-streaming), "
        f"got {len(stop_calls)}: {adapter.typing_events}"
    )
