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

    def _save(self):
        self.save_calls += 1

    def _record_gateway_session_peer(self, session_id, session_key, source):
        # #55300 records the child's gateway peer metadata after a compression
        # split; the fake tracks the call so tests can assert it fired.
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

    def run_conversation(
        self, user_message, conversation_history=None, task_id=None, **_kwargs
    ):
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


def _install_compression_failure_agent(monkeypatch, agent_cls=_CompressionThenFailureAgent):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", _StreamConsumer)

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda *_args, **_kwargs: {"core"})


def _run_compression_failure_turn(runner, source, *, run_generation=None):
    return asyncio.run(
        asyncio.wait_for(
            runner._run_agent(
                message="continue",
                context_prompt="",
                history=[{"role": "user", "content": "old question"}],
                source=source,
                session_id="session-before-compression",
                session_key=SESSION_KEY,
                run_generation=run_generation,
            ),
            timeout=2,
        )
    )


def test_failed_turn_still_syncs_compression_session_split(monkeypatch):
    _install_compression_failure_agent(monkeypatch)

    session_store = _SessionStore()
    runner = _runner(session_store)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")

    result = _run_compression_failure_turn(runner, source)

    assert result["failed"] is True
    assert result["session_id"] == "session-after-compression"
    assert result["history_offset"] == 0
    assert session_store.entry.session_id == "session-after-compression"
    assert session_store.save_calls == 1
    # #55300: the child's gateway peer metadata is recorded on the persist path.
    assert session_store.peer_records == [
        ("session-after-compression", SESSION_KEY, source)
    ]
    runner._sync_telegram_topic_binding.assert_called_once_with(
        source, session_store.entry, reason="agent-run-compression"
    )


def test_stale_run_does_not_overwrite_new_session_after_compression(monkeypatch):
    """A /stop + /new can invalidate a run while its compression is still unwinding.

    The stale run may still return with a rotated agent.session_id, but it must
    not publish that old compressed child back into the channel's active session
    binding. The outer gateway stale-result check will discard the response too;
    this regression covers the earlier side effect inside _run_agent().
    """
    _install_compression_failure_agent(monkeypatch)

    session_store = _SessionStore()
    runner = _runner(session_store)
    runner._session_run_generation[SESSION_KEY] = 2
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")

    result = _run_compression_failure_turn(runner, source, run_generation=1)

    assert result["failed"] is True
    assert result["session_id"] == "session-after-compression"
    assert result["history_offset"] == 0
    assert session_store.entry.session_id == "session-before-compression"
    assert session_store.save_calls == 0
    assert session_store.peer_records == []
    assert getattr(runner._sync_telegram_topic_binding, "call_count") == 0


def test_session_split_sync_skips_when_binding_already_moved(monkeypatch):
    """A live session binding is identity-guarded, not blindly overwritten.

    This catches the exact race where an old run starts with session A, /new
    moves the binding to fresh session B, and the old run finishes compression
    into child C. C must not replace B.
    """
    _install_compression_failure_agent(monkeypatch)

    session_store = _SessionStore()
    session_store.entry.session_id = "fresh-session-after-new"
    runner = _runner(session_store)
    runner._session_run_generation[SESSION_KEY] = 1
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")

    result = _run_compression_failure_turn(runner, source, run_generation=1)

    assert result["failed"] is True
    assert result["session_id"] == "session-after-compression"
    assert result["history_offset"] == 0
    assert session_store.entry.session_id == "fresh-session-after-new"
    assert session_store.save_calls == 0
    assert session_store.peer_records == []
    assert getattr(runner._sync_telegram_topic_binding, "call_count") == 0


class _RateLimitFailureAgent(_CompressionThenFailureAgent):
    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        return {
            "final_response": "API call failed after 3 retries: 429 Too Many Requests",
            "failed": True,
            "completed": False,
            "error": "429 Too Many Requests",
            "failure_reason": "rate_limit",
            "messages": [
                *(conversation_history or []),
                {"role": "user", "content": user_message},
            ],
            "api_calls": 3,
        }


def test_nonempty_rate_limit_response_preserves_failure_metadata(monkeypatch):
    _install_compression_failure_agent(monkeypatch, _RateLimitFailureAgent)

    session_store = _SessionStore()
    runner = _runner(session_store)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )

    result = _run_compression_failure_turn(runner, source)

    assert result["final_response"].startswith("API call failed after 3 retries")
    assert result["failed"] is True
    assert result["failure_reason"] == "rate_limit"
    assert result["completed"] is False

class _ProviderSwitchAgent(_CompressionThenFailureAgent):
    created_providers = []
    second_turn_history = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.api_mode = kwargs.get("api_mode")
        type(self).created_providers.append(self.provider)

    def run_conversation(
        self, user_message, conversation_history=None, task_id=None, **_kwargs
    ):
        history = list(conversation_history or [])

        if self.provider == "provider-a":
            return {
                "final_response": (
                    "API call failed after 3 retries: 429 Too Many Requests"
                ),
                "failed": True,
                "completed": False,
                "error": "429 Too Many Requests",
                "failure_reason": "rate_limit",
                "messages": [
                    *history,
                    {"role": "user", "content": user_message},
                ],
                "api_calls": 3,
            }

        type(self).second_turn_history = history
        response = "Provider B completed the next turn"
        return {
            "final_response": response,
            "failed": False,
            "completed": True,
            "messages": [
                *history,
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response},
            ],
            "api_calls": 1,
        }


def test_rate_limit_then_provider_switch_continues_without_replaying_error(
    monkeypatch
):
    _ProviderSwitchAgent.created_providers = []
    _ProviderSwitchAgent.second_turn_history = None
    _install_compression_failure_agent(monkeypatch, _ProviderSwitchAgent)

    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda _config=None: "model-a",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "provider-a",
            "model": "model-a",
            "api_key": "key-a",
            "base_url": "https://provider-a.example/v1",
            "api_mode": "chat_completions",
        },
    )

    session_store = _SessionStore()
    runner = _runner(session_store)
    runner._resolve_session_agent_runtime = (
        gateway_run.GatewayRunner._resolve_session_agent_runtime.__get__(
            runner, gateway_run.GatewayRunner
        )
    )
    runner._agent_config_signature = (
        gateway_run.GatewayRunner._agent_config_signature
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )
    initial_history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]

    first_result = asyncio.run(
        runner._run_agent(
            message="First request",
            context_prompt="",
            history=initial_history,
            source=source,
            session_id="session-before-compression",
            session_key=SESSION_KEY,
        )
    )

    assert first_result["failed"] is True
    assert first_result["failure_reason"] == "rate_limit"

    runner._session_model_overrides[SESSION_KEY] = {
        "model": "model-b",
        "provider": "provider-b",
        "api_key": "key-b",
        "base_url": "https://provider-b.example/v1",
        "api_mode": "chat_completions",
    }

    second_result = asyncio.run(
        runner._run_agent(
            message="Second request",
            context_prompt="",
            history=first_result["messages"],
            source=source,
            session_id="session-before-compression",
            session_key=SESSION_KEY,
        )
    )

    assert _ProviderSwitchAgent.created_providers == [
        "provider-a",
        "provider-b",
    ]
    assert second_result["failed"] is False
    assert second_result["completed"] is True
    assert second_result["final_response"] == (
        "Provider B completed the next turn"
    )
    assert not any(
        "429 Too Many Requests" in str(message.get("content", ""))
        for message in (_ProviderSwitchAgent.second_turn_history or [])
    )
