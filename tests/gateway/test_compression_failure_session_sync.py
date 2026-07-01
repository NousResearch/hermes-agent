import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


SESSION_KEY = "agent:main:telegram:dm:12345"


class _SessionStore:
    def __init__(self):
        self.entry = SimpleNamespace(
            session_key=SESSION_KEY,
            session_id="session-before-compression",
        )
        self._entries = {SESSION_KEY: self.entry}
        self.save_calls = 0

    def _save(self):
        self.save_calls += 1


class _CompressionThenFailureAgent:
    def __init__(self, **kwargs):
        self.session_id = kwargs["session_id"]
        self.model = kwargs["model"]
        self.tools = []
        self.context_compressor = SimpleNamespace(
            last_prompt_tokens=4321,
            context_length=200000,
        )
        self.session_prompt_tokens = 4321
        self.session_completion_tokens = 0

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        self.session_id = "session-after-compression"
        return {
            "failed": True,
            "error": "APIConnectionError: Codex auxiliary Responses stream exceeded 120.0s total timeout",
            "messages": [
                {"role": "user", "content": "[compressed summary]"},
                {"role": "user", "content": user_message},
            ],
            "api_calls": 1,
        }

    def interrupt(self, *_args, **_kwargs):
        pass


class _CompressionThenInterruptedAgent(_CompressionThenFailureAgent):
    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        self.session_id = "session-after-compression"
        return {
            "final_response": "",
            "interrupted": True,
            "interrupt_message": "queued follow-up",
            "session_id": self.session_id,
            "messages": [
                {"role": "user", "content": "[compressed summary]"},
                {"role": "user", "content": user_message},
            ],
            "api_calls": 1,
        }


class _StreamConsumer:
    final_response_sent = False

    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self):
        return None

    def finish(self):
        pass


class _Adapter:
    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self):
        self._pending_messages = {}

    def get_pending_message(self, session_key):
        return self._pending_messages.pop(session_key, None)

    async def send(self, *_args, **_kwargs):
        return None

    async def send_typing(self, *_args, **_kwargs):
        return None

    async def stop_typing(self, *_args, **_kwargs):
        return None


def _runner(session_store):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner.config = SimpleNamespace(streaming=None, group_sessions_per_user=True, thread_sessions_per_user=False)
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.session_store = session_store
    runner._session_db = MagicMock()
    runner._session_db.get_telegram_topic_binding_by_session.return_value = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._pending_skills_reload_notes = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._draining = False
    runner._get_proxy_url = lambda: None
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "gpt-5.4",
        {"provider": "openai-codex", "api_mode": "codex_responses", "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "token"},
    )
    runner._resolve_session_reasoning_config = lambda **_kwargs: None
    runner._resolve_turn_agent_config = lambda message, model, runtime: {"model": model, "runtime": runtime}
    runner._load_service_tier = lambda: None
    runner._agent_config_signature = lambda *_args, **_kwargs: ("sig",)
    runner._extract_cache_busting_config = lambda _config: ()
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: None
    runner._sync_telegram_topic_binding = MagicMock()
    runner._release_running_agent_state = MagicMock()
    return runner


def test_failed_turn_still_syncs_compression_session_split(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CompressionThenFailureAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", _StreamConsumer)

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda *_args, **_kwargs: {"core"})

    session_store = _SessionStore()
    runner = _runner(session_store)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")

    result = asyncio.run(
        asyncio.wait_for(
            runner._run_agent(
                message="continue",
                context_prompt="",
                history=[{"role": "user", "content": "old question"}],
                source=source,
                session_id="session-before-compression",
                session_key=SESSION_KEY,
            ),
            timeout=2,
        )
    )

    assert result["failed"] is True
    assert result["session_id"] == "session-after-compression"
    assert result["history_offset"] == 0
    assert session_store.entry.session_id == "session-after-compression"
    assert session_store.save_calls == 1
    runner._sync_telegram_topic_binding.assert_called_once_with(
        source, session_store.entry, reason="agent-run-compression"
    )


def test_queued_followup_after_compression_uses_rotated_session(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CompressionThenInterruptedAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", _StreamConsumer)

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda *_args, **_kwargs: {"core"})

    session_store = _SessionStore()
    runner = _runner(session_store)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")
    queued = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )
    runner.adapters[Platform.TELEGRAM]._pending_messages[SESSION_KEY] = queued
    runner._prepare_inbound_message_text = AsyncMock(return_value="queued follow-up")

    original_run_agent = runner._run_agent
    followup_session_ids = []

    async def spy_run_agent(*args, **kwargs):
        if kwargs.get("_interrupt_depth", 0) > 0:
            followup_session_ids.append(kwargs.get("session_id"))
            return {
                "final_response": "follow-up done",
                "messages": [],
                "api_calls": 1,
                "history_offset": 0,
                "session_id": kwargs.get("session_id"),
            }
        return await original_run_agent(*args, **kwargs)

    runner._run_agent = spy_run_agent

    result = asyncio.run(
        asyncio.wait_for(
            runner._run_agent(
                message="continue",
                context_prompt="",
                history=[{"role": "user", "content": "old question"}],
                source=source,
                session_id="session-before-compression",
                session_key=SESSION_KEY,
            ),
            timeout=2,
        )
    )

    assert result["final_response"] == "follow-up done"
    assert followup_session_ids == ["session-after-compression"]
