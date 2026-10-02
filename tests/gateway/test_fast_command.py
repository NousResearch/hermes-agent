"""Tests for gateway /fast support and Priority Processing routing."""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import hermes_yaml as yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None
    last_run = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(
        self,
        user_message,
        conversation_history=None,
        task_id=None,
        persist_user_message=None,
        persist_user_timestamp=None,
    ):
        type(self).last_run = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
            "persist_user_message": persist_user_message,
            "persist_user_timestamp": persist_user_timestamp,
        }
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


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
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _make_discord_auto_thread_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="999",
        chat_type="thread",
        user_id="user-1",
        thread_id="999",
        parent_chat_id="100",
        auto_thread_created=True,
        auto_thread_initial_name="raw user prompt",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def test_turn_route_injects_priority_processing_without_changing_runtime():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://api.openai.com/v1",
        "provider": "openai",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)

    assert route["runtime"]["provider"] == "openai"
    assert route["runtime"]["api_mode"] == "chat_completions"
    assert route["request_overrides"] == {"service_tier": "priority"}

    # Proxied routes never receive the param (OpenRouter strips it / others 400).
    runtime_kwargs.update(base_url="https://openrouter.ai/api/v1", provider="openrouter")
    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)
    assert route["request_overrides"] == {}


@pytest.mark.asyncio
async def test_handle_fast_command_global_flag_persists_config(monkeypatch, tmp_path):
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    # /fast now resolves eligibility through the session runtime resolver; with
    # no session override that path calls the real provider resolver, so stub it.
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    response = await runner._handle_fast_command(_make_event("/fast fast --global"))

    assert "FAST" in response
    assert runner._service_tier == "priority"

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["agent"]["service_tier"] == "fast"
    # Global write supersedes the session override.
    assert not runner._session_service_tier_overrides


@pytest.mark.asyncio
async def test_session_fast_override_beats_config_default(monkeypatch, tmp_path):
    """A session /fast normal wins over agent.service_tier: fast in config."""
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"agent": {"service_tier": "fast"}},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    # Eligibility routes through the session runtime resolver; stub the real
    # provider resolution the no-override path would otherwise trigger.
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    event = _make_event("/fast normal")
    session_key = runner._session_key_for_source(event.source)

    response = await runner._handle_fast_command(event)

    assert "NORMAL" in response
    # Override stores explicit None (normal) and wins over config "fast".
    assert session_key in runner._session_service_tier_overrides
    assert runner._resolve_session_service_tier(session_key=session_key) is None
    # A different session still gets the config default.
    assert runner._resolve_session_service_tier(session_key="other-session") == "priority"


_ASTRA_ON_CODEX = {"model": "gpt-6-astra", "provider": "openai-codex",
                   "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "***"}
_GPT_ON_OPENROUTER = {"model": "openai/gpt-5.4", "provider": "openrouter",
                      "base_url": "https://openrouter.ai/api/v1", "api_key": "***"}


@pytest.mark.asyncio
@pytest.mark.parametrize("default_model, override, command, accepted", [
    # #118761: Astra picked with session /model over a default /fast can't serve.
    ("claude-sonnet-4-6", _ASTRA_ON_CODEX, "/fast ultrafast", True),
    # Converse: a fast-capable default must not admit a session route whose turns never carry the tier.
    ("claude-opus-5-5", _GPT_ON_OPENROUTER, "/fast fast", False),
])
async def test_fast_gate_follows_the_session_route(monkeypatch, tmp_path, default_model, override, command, accepted):
    """Real fast-mode tables: /fast accepts exactly the tiers the session's next turn would send."""
    runner = _make_runner()
    event = _make_event(command)
    session_key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[session_key] = dict(override)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: default_model)

    response = await runner._handle_fast_command(event)

    tier = runner._resolve_session_service_tier(session_key=session_key)
    model, runtime = runner._resolve_session_agent_runtime(source=event.source)
    route = runner._resolve_turn_agent_config("hi", model, runtime)
    if accepted:
        assert tier == "ultrafast" and "only available" not in response
        assert route["request_overrides"] == {"service_tier": "ultrafast"}
    else:
        assert "only available" in response
        assert session_key not in runner._session_service_tier_overrides


@pytest.mark.asyncio
async def test_fast_override_lands_under_the_recovered_telegram_topic_key(monkeypatch, tmp_path):
    """/fast keys its eligibility check AND its tier override by the topic-recovered source the next
    turn uses (#30479), not the raw lobby-shaped event source."""
    import dataclasses

    runner = _make_runner()
    source = _make_source()
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda src: "77")
    raw_key = runner._session_key_for_source(source)
    turn_key = runner._session_key_for_source(dataclasses.replace(source, thread_id="77"))
    assert turn_key != raw_key
    runner._session_model_overrides[turn_key] = dict(_ASTRA_ON_CODEX)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "claude-sonnet-4-6")

    response = await runner._handle_fast_command(MessageEvent(text="/fast fast", source=source, message_id="m1"))

    assert "FAST" in response
    assert runner._resolve_session_service_tier(session_key=turn_key) == "priority"
    assert raw_key not in runner._session_service_tier_overrides
