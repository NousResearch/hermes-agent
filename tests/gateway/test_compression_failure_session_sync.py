import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
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
        self.peer_records = []

    def _ensure_loaded(self):
        return None

    def _save(self):
        self.save_calls += 1

    def _record_gateway_session_peer(self, session_id, session_key, source):
        self.peer_records.append((session_id, session_key, source))


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
    _pending_messages = {}

    def get_pending_message(self, _session_key):
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
    assert session_store.peer_records == [
        ("session-after-compression", SESSION_KEY, source)
    ]
    runner._sync_telegram_topic_binding.assert_called_once_with(
        source, session_store.entry, reason="agent-run-compression"
    )


def test_stale_generation_cannot_publish_compression_split():
    session_store = _SessionStore()
    runner = _runner(session_store)
    runner._session_run_generation = {SESSION_KEY: 2}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )

    result = runner._publish_compression_session_split(
        session_key=SESSION_KEY,
        source=source,
        previous_session_id="session-before-compression",
        new_session_id="late-compressed-child",
        reason="unit-test",
        run_generation=1,
    )

    assert result is None
    assert session_store.entry.session_id == "session-before-compression"
    assert session_store.save_calls == 0
    assert session_store.peer_records == []


def test_moved_active_binding_cannot_publish_late_compression_split():
    session_store = _SessionStore()
    session_store.entry.session_id = "fresh-session-after-new"
    runner = _runner(session_store)
    runner._session_run_generation = {SESSION_KEY: 1}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )

    result = runner._publish_compression_session_split(
        session_key=SESSION_KEY,
        source=source,
        previous_session_id="session-before-compression",
        new_session_id="late-compressed-child",
        reason="unit-test",
        run_generation=1,
    )

    assert result is None
    assert session_store.entry.session_id == "fresh-session-after-new"
    assert session_store.save_calls == 0
    assert session_store.peer_records == []
