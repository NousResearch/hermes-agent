import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


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
    runner._session_db._db.get_telegram_topic_binding_by_session.return_value = None
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


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("rotates", [False, True])
def test_queued_burst_follows_each_compression_child(monkeypatch, interrupted, rotates):
    session_store = _SessionStore()
    runner = _runner(session_store)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")
    adapter = runner.adapters[Platform.TELEGRAM]
    next_key = runner._session_key_for_source(source)
    run_ids, agent_ids = [], []

    class BurstAgent(_CompressionThenFailureAgent):
        def run_conversation(self, user_message, conversation_history=None, **kwargs):
            agent_ids.append(self.session_id)
            turn = len(agent_ids)
            messages = [{"role": "user", "content": user_message}]
            if turn < 3:
                if rotates:
                    self.session_id = f"compression-child-{turn}"
                    messages.insert(0, {"role": "user", "content": f"summary-{turn}"})
                key = SESSION_KEY if turn == 1 else next_key
                adapter._pending_messages[key] = MessageEvent(
                    text=f"queued-{turn}", source=source, message_type=MessageType.TEXT,
                    message_id=f"queued-id-{turn}",
                )
            return {
                "final_response": f"turn-{turn}-done", "session_id": self.session_id,
                "interrupted": interrupted and turn < 3, "messages": messages, "api_calls": 1,
            }

    _install_compression_failure_agent(monkeypatch, BurstAgent)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kw: kw["event"].text,
    )
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._deliver_queued_first_response = AsyncMock()
    original_run_agent = runner._run_agent

    async def record_turn(*args, **kwargs):
        run_ids.append(kwargs["session_id"])
        return await original_run_agent(*args, **kwargs)

    runner._run_agent = record_turn
    result = _run_compression_failure_turn(runner, source)
    expected = ["session-before-compression", "compression-child-1", "compression-child-2"] if rotates else ["session-before-compression"] * 3
    assert run_ids == expected
    assert agent_ids == expected
    assert result["session_id"] == expected[-1]
    assert result["final_response"] == "turn-3-done"
    assert not adapter._pending_messages
    assert runner._deliver_queued_first_response.await_count == (0 if interrupted else 2)
    assert [call.args for call in runner._refresh_agent_cache_message_count.await_args_list] == [
        (next_key, session_id) for session_id in expected[1:]
    ]
    assert [call.kwargs["session_key"] for call in runner._prepare_profile_scoped_inbound_message_text.await_args_list] == [next_key, next_key]
    if rotates:
        assert runner._prepare_profile_scoped_inbound_message_text.await_args_list[1].kwargs["history"][0]["content"] == "summary-2"


@pytest.mark.parametrize("active", [False, True])
def test_queued_goal_checks_compression_child_before_preprocessing(active):
    runner = _runner(_SessionStore())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")
    ctx = TurnContext(source=source, session_id="closed-parent", session_key=SESSION_KEY, history=[])
    pending = MessageEvent(text="continue goal", source=source)
    response = {"session_id": "compression-child", "messages": [], "interrupted": True}
    runner._is_goal_continuation_event = lambda event: True
    runner._goal_still_active_for_session = MagicMock(side_effect=lambda sid: active and sid == response["session_id"])
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=pending.text)
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._run_agent = AsyncMock(return_value={"final_response": "continued", "session_id": response["session_id"]})

    result = asyncio.run(runner._run_agent_queued_followup(
        ctx, runner.adapters[Platform.TELEGRAM], pending.text, pending, response, response, None,
    ))

    runner._goal_still_active_for_session.assert_called_once_with(response["session_id"])
    if active:
        assert result["final_response"] == "continued"
        assert runner._run_agent.await_args.kwargs["session_id"] == response["session_id"]
        runner._prepare_profile_scoped_inbound_message_text.assert_awaited_once()
    else:
        assert result is response
        runner._prepare_profile_scoped_inbound_message_text.assert_not_awaited()
        runner._refresh_agent_cache_message_count.assert_not_awaited()
        runner._run_agent.assert_not_awaited()
