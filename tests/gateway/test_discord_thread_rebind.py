"""Regression tests for Discord thread rebinding after tool-created threads."""

import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class DiscordCaptureAdapter(BasePlatformAdapter):
    """Adapter that records sends and thread marks for inspection."""

    _next_mid = 400

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
        self.sent = []
        self.thread_marks = []
        self._threads = SimpleNamespace(mark=self.thread_marks.append)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        DiscordCaptureAdapter._next_mid += 1
        return str(DiscordCaptureAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
                "message_id": mid,
            }
        )
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class FakeSessionStore:
    def __init__(self):
        self.entries = {}
        self.switched = []

    def _generate_session_key(self, source: SessionSource) -> str:
        chat_id = str(source.chat_id or "")
        thread_id = str(source.thread_id or "")
        if thread_id:
            return f"agent:main:discord:thread:{chat_id}:{thread_id}"
        return f"agent:main:discord:{source.chat_type or 'channel'}:{chat_id}"

    def get_or_create_session(self, source: SessionSource):
        session_key = self._generate_session_key(source)
        entry = self.entries.get(session_key)
        if entry is None:
            entry = SimpleNamespace(session_key=session_key, session_id=f"auto:{session_key}")
            self.entries[session_key] = entry
        return entry

    def switch_session(self, session_key: str, target_session_id: str):
        entry = SimpleNamespace(session_key=session_key, session_id=target_session_id)
        self.entries[session_key] = entry
        self.switched.append((session_key, target_session_id))
        return entry


class ThreadRebindAgent:
    """Fake agent that creates a thread, then emits a status message."""

    instances = []

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tool_complete_callback = kwargs.get("tool_complete_callback")
        self.status_callback = kwargs.get("status_callback")
        self.model = kwargs.get("model", "fake-model")
        self.provider = kwargs.get("provider", "fake-provider")
        self._chat_id = kwargs.get("chat_id")
        self._chat_name = kwargs.get("chat_name")
        self._chat_type = kwargs.get("chat_type")
        self._thread_id = kwargs.get("thread_id")
        self._gateway_session_key = kwargs.get("gateway_session_key")
        self.tools = []
        type(self).instances.append(self)

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.tool_complete_callback is not None:
            self.tool_complete_callback(
                "call-1",
                "discord",
                {"action": "create_thread", "name": "Planning"},
                json.dumps(
                    {
                        "success": True,
                        "thread_id": "555",
                        "thread_name": "Planning",
                    }
                ),
            )
        if self.status_callback is not None:
            self.status_callback("info", "post-thread status")
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner.session_store = FakeSessionStore()
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._draining = False
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _install_fakes(monkeypatch):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ThreadRebindAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    return gateway_run


@pytest.mark.asyncio
async def test_run_agent_rebinds_to_tool_created_discord_thread(monkeypatch, tmp_path):
    ThreadRebindAgent.instances.clear()
    adapter = DiscordCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_name="general",
        chat_type="channel",
        user_id="u1",
        user_name="alice",
    )
    session_key = "agent:main:discord:channel:123"

    result = await runner._run_agent(
        message="start a planning thread",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )
    await asyncio.sleep(0.05)

    assert result["final_response"] == "done"
    assert source.chat_id == "555"
    assert source.chat_name == "Planning"
    assert source.chat_type == "thread"
    assert source.thread_id == "555"
    assert source.parent_chat_id == "123"

    rebound_session_key = runner._session_key_for_source(source)
    assert runner.session_store.switched == [(rebound_session_key, "sess-1")]
    assert runner.session_store.entries[rebound_session_key].session_id == "sess-1"
    assert adapter.thread_marks == ["555"]
    assert any(
        msg["chat_id"] == "555"
        and msg["content"] == "post-thread status"
        and msg["metadata"] == {"thread_id": "555"}
        for msg in adapter.sent
    )

    agent = ThreadRebindAgent.instances[-1]
    assert agent._chat_id == "555"
    assert agent._chat_name == "Planning"
    assert agent._chat_type == "thread"
    assert agent._thread_id == "555"
    assert agent._gateway_session_key == rebound_session_key
