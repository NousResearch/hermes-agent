import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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
    monkeypatch.setattr(gateway_run, "_load_gateway_config", dict)
    monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", _StreamConsumer)

    from hermes_cli import tools_config

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


class _EmptyRateLimitFailureAgent(_CompressionThenFailureAgent):
    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        return {
            "final_response": "",
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


def test_empty_rate_limit_response_preserves_failure_metadata(monkeypatch):
    """Sibling of the non-empty path (#64686): the empty-response return
    branch in _run_agent must also forward failure_reason, or downstream
    consumers lose the structured reason exactly when the run produced no
    text at all."""
    _install_compression_failure_agent(monkeypatch, _EmptyRateLimitFailureAgent)

    session_store = _SessionStore()
    runner = _runner(session_store)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )

    result = _run_compression_failure_turn(runner, source)

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




class _SessionDB:
    """The two reads the split guard makes: which rows exist and the compression chain."""

    def __init__(self, rows, chain):
        self.rows, self.chain = set(rows), list(chain)

    def get_session(self, session_id):
        return {"id": session_id} if session_id in self.rows else None

    def get_compression_chain(self, session_id):
        return self.chain if session_id == self.chain[0] else [session_id]


class _ProfileSessionStore(_SessionStore):
    def __init__(self, db):
        super().__init__()
        self.db = db

    def _db_for_key(self, session_key):
        assert session_key == SESSION_KEY
        return self.db


def test_session_change_that_is_not_compression_keeps_the_route(monkeypatch):
    """#154994: the agent came back on another session the store already holds (an async
    subagent's), not a compression child. Rerouting would persist a route outside the
    conversation's lineage, refused ``admission_conflict`` on every later request."""
    _install_compression_failure_agent(monkeypatch)
    db = _SessionDB(rows={"session-before-compression", "session-after-compression"},
                    chain=["session-before-compression"])
    session_store = _ProfileSessionStore(db)
    runner = _runner(session_store)
    runner._evict_cached_agent = MagicMock()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")

    result = _run_compression_failure_turn(runner, source)

    assert session_store.entry.session_id == "session-before-compression"
    assert session_store.save_calls == 0
    assert session_store.peer_records == []
    runner._sync_telegram_topic_binding.assert_not_called()
    # The result reports the route's session, or the agent-result sync after the turn would move
    # the route anyway; the cached agent (now on the other session) is dropped for the next turn.
    assert result["session_id"] == "session-before-compression"
    assert "_compressed" not in result
    # None of the other session's messages are this route's new turn.
    assert result["history_offset"] == len(result["messages"])
    runner._evict_cached_agent.assert_called_once_with(SESSION_KEY, release_after_turn=True)


def test_real_compression_split_still_syncs_with_the_guard(monkeypatch):
    _install_compression_failure_agent(monkeypatch)
    db = _SessionDB(rows={"session-before-compression", "session-after-compression"},
                    chain=["session-before-compression", "session-after-compression"])
    session_store = _ProfileSessionStore(db)
    runner = _runner(session_store)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")

    result = _run_compression_failure_turn(runner, source)

    assert result["session_id"] == "session-after-compression"
    assert session_store.entry.session_id == "session-after-compression"
    assert session_store.save_calls == 1
    assert session_store.peer_records == [("session-after-compression", SESSION_KEY, source)]


class _RefusingPeerStore(_SessionStore):
    def __init__(self):
        super().__init__()
        self.saved_ids = []
        self._lock = threading.Lock()

    def _save(self):
        super()._save()
        self.saved_ids.append(self.entry.session_id)

    def _record_gateway_session_peer(self, session_id, session_key, source):
        from hermes_state_runtime import RuntimeStoreError
        raise RuntimeStoreError("admission_conflict")


def test_failed_split_publication_rolls_the_route_back(monkeypatch):
    """#154994: the route was saved before the peer record refused, and stayed on the new id."""
    _install_compression_failure_agent(monkeypatch)
    session_store = _RefusingPeerStore()
    runner = _runner(session_store)
    runner._evict_cached_agent = MagicMock()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")

    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match="admission_conflict"):  # still the turn's error
        _run_compression_failure_turn(runner, source)

    assert session_store.entry.session_id == "session-before-compression"
    assert session_store.saved_ids == ["session-after-compression", "session-before-compression"]
    runner._sync_telegram_topic_binding.assert_not_called()
    runner._evict_cached_agent.assert_called_once_with(SESSION_KEY, release_after_turn=True)
