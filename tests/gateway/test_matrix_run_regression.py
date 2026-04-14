"""Gateway integration regressions for Matrix thinking/acting panes."""

import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class MatrixThinkingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.MATRIX)
        self._thinking_enabled = True
        self.calls = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.calls.append(("send", content))
        return SendResult(success=True, message_id="msg-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.calls.append(("edit", content))
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def start_thinking(self, room_id, task_id, initial_summary="Processing request...", model_label="", initial_content_md="", thread_id=None):
        self.calls.append(("start_thinking", initial_summary, model_label, initial_content_md, thread_id))
        return "$thinking"

    async def update_thinking(self, task_id, step_info, content_md="", model_label=None, append_line=True):
        self.calls.append(("update_thinking", step_info, content_md, model_label, append_line))

    async def finalize_thinking(self, task_id, final_summary="Task complete", collapse=True, model_label=None):
        self.calls.append(("finalize_thinking", final_summary, model_label, collapse))

    async def abort_thinking(self, task_id, reason="Aborted", model_label=None):
        self.calls.append(("abort_thinking", reason, model_label))

    async def start_tool_activity(self, room_id, task_id, initial_summary="Tool activity", model_label="", initial_content_md="", thread_id=None):
        self.calls.append(("start_tool_activity", initial_summary, model_label, initial_content_md, thread_id))
        return "$tools"

    async def update_tool_activity(self, task_id, step_info, content_md="", model_label=None, append_line=True):
        self.calls.append(("update_tool_activity", step_info, content_md, model_label, append_line))

    async def finalize_tool_activity(self, task_id, final_summary="Tool activity complete", collapse=True, model_label=None):
        self.calls.append(("finalize_tool_activity", final_summary, model_label, collapse))

    async def abort_tool_activity(self, task_id, reason="Aborted", model_label=None):
        self.calls.append(("abort_tool_activity", reason, model_label))

    def has_pending_interrupt(self, session_key: str) -> bool:
        return False

    def get_pending_message(self, session_key: str):
        return None


class FakeAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.thinking_callback = kwargs.get("thinking_callback")
        self.reasoning_callback = kwargs.get("reasoning_callback")
        self.status_callback = kwargs.get("status_callback")
        self.tools = []
        self.model = "gpt-5.4"
        self.provider = "openai-codex"
        self.session_id = kwargs.get("session_id")
        self.context_compressor = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.status_callback:
            self.status_callback("info", "switching to fallback: gpt-5.4 via openai-codex")
        if self.reasoning_callback:
            self.reasoning_callback("Reasoning delta")
        if self.thinking_callback:
            self.thinking_callback("Thinking status")
        if self.tool_progress_callback:
            self.tool_progress_callback("tool.started", "terminal", '"pwd"')
            self.tool_progress_callback("tool.started", "search_files", '"turn_route"')
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FakeAgentReasoningFirst(FakeAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.reasoning_callback:
            self.reasoning_callback("Reasoning arrived first")
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _make_runner(adapter):
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
    runner._running_agent_started_at = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._effective_model = None
    runner._effective_provider = None
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._get_or_create_gateway_honcho = lambda session_key: (None, {})
    runner._load_reasoning_config = lambda: None
    runner._evict_cached_agent = lambda session_key: runner._agent_cache.pop(session_key, None)
    return runner


@pytest.mark.asyncio
async def test_matrix_run_routes_thinking_reasoning_and_tools_to_panes(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": {"tool_progress": "all"}})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***", "provider": "anthropic", "base_url": "https://api.anthropic.com"})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *args, **kwargs: "claude-opus-4-6")

    adapter = MatrixThinkingAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm", thread_id=None)

    result = await runner._run_agent(message="hello", context_prompt="", history=[], source=source, session_id="sess-1", session_key="agent:main:matrix:dm:!room:example.org")

    assert result["final_response"] == "done"
    call_names = [name for name, *_ in adapter.calls]
    assert "start_thinking" in call_names
    assert "update_thinking" in call_names
    assert "start_tool_activity" in call_names
    assert "update_tool_activity" in call_names
    assert "finalize_thinking" in call_names
    assert "finalize_tool_activity" in call_names
    assert "send" not in call_names


@pytest.mark.asyncio
async def test_matrix_run_reasoning_first_still_starts_thinking_pane(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgentReasoningFirst
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": {"tool_progress": "all"}})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***", "provider": "anthropic", "base_url": "https://api.anthropic.com"})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *args, **kwargs: "claude-opus-4-6")

    adapter = MatrixThinkingAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm", thread_id=None)

    result = await runner._run_agent(message="hello", context_prompt="", history=[], source=source, session_id="sess-2", session_key="agent:main:matrix:dm:!room:example.org")

    assert result["final_response"] == "done"
    call_names = [name for name, *_ in adapter.calls]
    assert "start_thinking" in call_names
    assert "start_tool_activity" not in call_names


@pytest.mark.asyncio
async def test_matrix_run_respects_tool_progress_off(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": {"tool_progress": "off"}})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***", "provider": "anthropic", "base_url": "https://api.anthropic.com"})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *args, **kwargs: "claude-opus-4-6")

    adapter = MatrixThinkingAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm", thread_id=None)

    result = await runner._run_agent(message="hello", context_prompt="", history=[], source=source, session_id="sess-3", session_key="agent:main:matrix:dm:!room:example.org")

    assert result["final_response"] == "done"
    call_names = [name for name, *_ in adapter.calls]
    assert "start_thinking" in call_names
    assert "start_tool_activity" not in call_names
    assert "update_tool_activity" not in call_names
    assert "finalize_tool_activity" not in call_names
