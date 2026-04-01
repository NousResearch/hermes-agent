"""Regression tests for gateway fallback caching with the provider circuit breaker."""

import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class DummyAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.MATRIX):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="msg-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    def has_pending_interrupt(self, session_key: str) -> bool:
        return False

    def get_pending_message(self, session_key: str):
        return None


class FakeFallbackAgent:
    def __init__(self, **kwargs):
        self.model = "gpt-5.4"
        self.provider = "openai-codex"
        self.tools = []
        self.session_id = kwargs.get("session_id")
        self.context_compressor = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.tool_progress_callback = None
        self.step_callback = None
        self.stream_delta_callback = None
        self.status_callback = None
        self.reasoning_config = kwargs.get("reasoning_config")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        return {
            "final_response": "fallback ok",
            "messages": [],
            "api_calls": 1,
        }


class FakeCircuitBreaker:
    def __init__(self):
        self.calls = []

    def should_use_fallback(self, provider, session_key=None):
        self.calls.append((provider, session_key))
        return True


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
async def test_run_agent_keeps_cached_fallback_without_nameerror(monkeypatch, tmp_path):
    """Regression: post-run fallback bookkeeping must not reference inner-scope turn_route."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeFallbackAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_cb = FakeCircuitBreaker()
    fake_cb_module = types.ModuleType("agent.circuit_breaker")

    class ProviderCircuitBreaker:
        @staticmethod
        def get_instance():
            return fake_cb

    fake_cb_module.ProviderCircuitBreaker = ProviderCircuitBreaker
    monkeypatch.setitem(sys.modules, "agent.circuit_breaker", fake_cb_module)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "api_key": "***",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
    })
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *args, **kwargs: "claude-opus-4-6")

    adapter = DummyAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.com",
        chat_type="dm",
        thread_id=None,
    )
    session_key = "agent:main:matrix:dm:!room:example.com"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-fallback",
        session_key=session_key,
    )

    assert result["final_response"] == "fallback ok"
    assert runner._effective_model == "gpt-5.4"
    assert runner._effective_provider == "openai-codex"
    assert session_key in runner._agent_cache
    assert fake_cb.calls == [("anthropic", session_key)]
