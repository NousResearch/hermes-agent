"""Regression tests for memory-tool privacy in public gateway progress."""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append(chat_id)

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class MemoryAddThenTerminalAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb(
                "tool.started",
                "memory",
                '+memory: "private memory payload"',
                {
                    "action": "add",
                    "target": "memory",
                    "content": "private memory payload",
                },
            )
            time.sleep(0.2)
            cb("tool.started", "terminal", "pwd", {"command": "pwd"})
            time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class MemoryVerboseThenTerminalAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb(
                "tool.started",
                "memory",
                '~memory: "private old text"',
                {
                    "action": "replace",
                    "target": "memory",
                    "old_text": "private old text",
                    "content": "new private replacement",
                },
            )
            time.sleep(0.2)
            cb("tool.started", "terminal", "pwd", {"command": "pwd"})
            time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class MemoryCompletionOnlyAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.completed", "memory", None, None, duration=31)
            time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _make_runner(adapter):
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


async def _run_once(monkeypatch, tmp_path, agent_cls, *, progress_mode="all", config=None):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", progress_mode)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal emoji

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config or {})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-memory-progress",
        session_key="agent:main:telegram:group:-1001:17585",
    )
    rendered = " ".join(c["content"] for c in adapter.sent) + " " + " ".join(
        c["content"] for c in adapter.edits
    )
    return adapter, result, rendered


@pytest.mark.asyncio
async def test_memory_progress_is_suppressed_in_telegram_all_mode(monkeypatch, tmp_path):
    adapter, result, rendered = await _run_once(
        monkeypatch,
        tmp_path,
        MemoryAddThenTerminalAgent,
        progress_mode="all",
    )

    assert result["final_response"] == "done"
    assert "terminal" in rendered
    assert "pwd" in rendered
    assert "private memory payload" not in rendered
    assert "+memory" not in rendered
    assert '"content"' not in rendered
    assert all("memory" not in call["content"] for call in adapter.sent + adapter.edits)


@pytest.mark.asyncio
async def test_memory_progress_is_suppressed_in_telegram_verbose_mode(monkeypatch, tmp_path):
    adapter, result, rendered = await _run_once(
        monkeypatch,
        tmp_path,
        MemoryVerboseThenTerminalAgent,
        progress_mode="verbose",
    )

    assert result["final_response"] == "done"
    assert "terminal" in rendered
    assert "pwd" in rendered
    assert "private old text" not in rendered
    assert "new private replacement" not in rendered
    assert "old_text" not in rendered
    assert "~memory" not in rendered
    assert all("memory" not in call["content"] for call in adapter.sent + adapter.edits)


@pytest.mark.asyncio
async def test_memory_completed_event_does_not_emit_long_tool_hint(monkeypatch, tmp_path):
    config = {"display": {"tool_progress_command": True}, "onboarding": {"seen": {}}}
    adapter, result, rendered = await _run_once(
        monkeypatch,
        tmp_path,
        MemoryCompletionOnlyAgent,
        progress_mode="all",
        config=config,
    )

    assert result["final_response"] == "done"
    assert adapter.sent == []
    assert adapter.edits == []
    assert "First-time tip" not in rendered
    assert "/verbose" not in rendered
