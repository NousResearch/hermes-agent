"""Regression tests for gateway turns invalidated during slow agent startup."""

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None, multiplex_profiles=False)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._gateway_loop = None
    return runner


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _setup_runtime(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "agent:\n  model: test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"model": "test-model"}},
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "test-model",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(
        tools_config,
        "_get_platform_tools",
        lambda user_config, platform_key: {"core"},
    )


def test_invalidated_slow_init_cannot_overwrite_cache_or_run(
    monkeypatch,
    tmp_path,
):
    """A /stop during AIAgent.__init__ must fence the old worker before use."""

    _setup_runtime(monkeypatch, tmp_path)
    runner = _make_runner()
    session_key = "agent:main:telegram:dm:12345"
    generation = runner._begin_session_run_generation(session_key)
    newer_agent = object()
    constructed = []

    class InvalidatedDuringInitAgent:
        run_calls = 0

        def __init__(self, *args, **kwargs):
            self.tools = []
            self.model = kwargs.get("model", "test-model")
            self.provider = kwargs.get("provider", "openrouter")
            self.session_id = kwargs.get("session_id", "session-1")
            self.context_compressor = None
            self.released = False
            constructed.append(self)

            # Reproduce the production race: while this constructor is blocked,
            # /stop invalidates its generation and a newer turn publishes a
            # replacement cached agent for the same routing key.
            runner._invalidate_session_run_generation(
                session_key,
                reason="test_stop_during_init",
            )
            with runner._agent_cache_lock:
                runner._agent_cache[session_key] = (
                    newer_agent,
                    "newer-signature",
                    0,
                    "session-1",
                )

        def release_clients(self):
            self.released = True

        def run_conversation(self, *args, **kwargs):
            type(self).run_calls += 1
            raise AssertionError("stale agent must never enter run_conversation")

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = InvalidatedDuringInitAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    result = asyncio.run(
        runner._run_agent(
            message="old prompt",
            context_prompt="",
            history=[],
            source=_source(),
            session_id="session-1",
            session_key=session_key,
            run_generation=generation,
        )
    )

    assert result["stale_run"] is True
    assert result["api_calls"] == 0
    assert InvalidatedDuringInitAgent.run_calls == 0
    assert constructed and constructed[0].released is True
    assert runner._agent_cache[session_key][0] is newer_agent


def test_stale_cache_hit_does_not_wire_or_release_newer_agent(monkeypatch, tmp_path):
    """A generation invalidated after cache lookup must leave the cached agent untouched."""

    _setup_runtime(monkeypatch, tmp_path)
    runner = _make_runner()
    runner._agent_config_signature = lambda *args, **kwargs: "cached-signature"
    runner._refresh_fallback_model = lambda: None
    runner._apply_fallback_chain_to_agent = lambda *args: None
    session_key = "agent:main:telegram:dm:12345"
    generation = runner._begin_session_run_generation(session_key)
    callback_sentinel = object()

    class CachedAgent:
        run_calls = 0
        release_calls = 0

        def __init__(self):
            self.tools = []
            self.model = "test-model"
            self.provider = "openrouter"
            self.session_id = "session-1"
            self.max_iterations = 4
            self.request_overrides = {}
            for name in (
                "tool_progress_callback", "tool_start_callback", "tool_complete_callback",
                "step_callback", "stream_delta_callback", "interim_assistant_callback",
                "status_callback", "notice_callback", "notice_clear_callback", "event_callback",
                "reasoning_config", "service_tier",
            ):
                setattr(self, name, callback_sentinel)

        def release_clients(self):
            type(self).release_calls += 1

        def run_conversation(self, *args, **kwargs):
            type(self).run_calls += 1
            raise AssertionError("stale cache-hit agent must never enter run_conversation")

    cached_agent = CachedAgent()
    runner._agent_cache[session_key] = (cached_agent, "cached-signature", 0, "session-1")

    from gateway.run_turn_runner import TurnRunner

    original_resolve = TurnRunner._resolve_turn_agent

    def resolve_then_invalidate(turn_runner, *args, **kwargs):
        resolved = original_resolve(turn_runner, *args, **kwargs)
        runner._invalidate_session_run_generation(session_key, reason="test_stop_after_cache_hit")
        return resolved

    monkeypatch.setattr(TurnRunner, "_resolve_turn_agent", resolve_then_invalidate)
    result = asyncio.run(
        runner._run_agent(
            message="old prompt",
            context_prompt="",
            history=[],
            source=_source(),
            session_id="session-1",
            session_key=session_key,
            run_generation=generation,
        )
    )

    assert result["stale_run"] is True
    assert CachedAgent.run_calls == 0
    assert CachedAgent.release_calls == 0
    assert runner._agent_cache[session_key][0] is cached_agent
    assert all(
        getattr(cached_agent, name) is callback_sentinel
        for name in (
            "tool_progress_callback", "tool_start_callback", "tool_complete_callback",
            "step_callback", "stream_delta_callback", "interim_assistant_callback",
            "status_callback", "notice_callback", "notice_clear_callback", "event_callback",
            "reasoning_config", "service_tier",
        )
    )
