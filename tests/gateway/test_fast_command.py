"""Tests for gateway /fast support and Priority Processing routing."""

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import SendResult
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


class _PickerAdapter:
    def __init__(self):
        self.calls = []

    async def send_choice_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SendResult(success=True, message_id="m1")


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

    response = await runner._handle_fast_command(_make_event("/fast fast --global"))

    assert "FAST" in response
    assert runner._service_tier == "priority"

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["agent"]["service_tier"] == "fast"
    # Global write supersedes the session override.
    assert not runner._session_service_tier_overrides


@pytest.mark.asyncio
async def test_typed_fast_global_persists_to_originating_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / "default"
    named_home = tmp_path / "profiles" / "work"
    default_home.mkdir(parents=True)
    named_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        "agent:\n  service_tier: normal\n", encoding="utf-8"
    )
    (named_home / "config.yaml").write_text(
        "agent:\n  service_tier: normal\n", encoding="utf-8"
    )
    runner = _make_runner()
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda _source: named_home

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4"
    )
    event = _make_event("/fast fast --global")
    event.source.profile = "work"

    with gateway_run._profile_runtime_scope(named_home):
        response = await runner._handle_fast_command(event)

    default_config = yaml.safe_load(
        (default_home / "config.yaml").read_text(encoding="utf-8")
    )
    named_config = yaml.safe_load(
        (named_home / "config.yaml").read_text(encoding="utf-8")
    )
    assert default_config["agent"]["service_tier"] == "normal"
    assert named_config["agent"]["service_tier"] == "fast"
    assert "FAST" in response


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["fast", "normal", "auto", "cold"])
@pytest.mark.parametrize("persist,save_succeeds", [(True, True), (True, False), (False, True)])
async def test_fast_global_picker_persists_to_originating_profile(
    monkeypatch, tmp_path, choice, persist, save_succeeds
):
    """A delayed picker tap must not fall back to the default profile."""
    from agent.secret_scope import set_multiplex_active

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / "default"
    named_home = tmp_path / "profiles" / "work"
    default_home.mkdir(parents=True)
    named_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        "agent:\n  service_tier: normal\n", encoding="utf-8"
    )
    (named_home / "config.yaml").write_text(
        "agent:\n  service_tier: normal\n", encoding="utf-8"
    )

    adapter = _PickerAdapter()
    runner = _make_runner()
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda _source: named_home
    runner._adapter_for_source = lambda _source: adapter
    runner._thread_metadata_for_source = lambda _source, anchor=None: {}
    runner._reply_anchor_for_event = lambda _event: None
    event = _make_event("/fast --global" if persist else "/fast")
    event.source.profile = "work"
    session_key = runner._session_key_for_source(event.source)

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4"
    )
    if not save_succeeds:
        def fail_write(*args, **kwargs):
            raise OSError("config is read-only")

        monkeypatch.setattr("gateway.slash_commands_model.atomic_config_write", fail_write)

    set_multiplex_active(True)
    try:
        with gateway_run._profile_runtime_scope(named_home):
            runner._set_session_service_tier_override(session_key, None)
            await runner._handle_fast_command(event)
        on_choice = adapter.calls[0]["on_choice_selected"]
        reply = await on_choice(event.source.chat_id, choice)
        assert gateway_run._gateway_config_home() == default_home
        with gateway_run._profile_runtime_scope(named_home):
            if persist and save_succeeds:
                assert session_key not in runner._session_service_tier_overrides
            else:
                expected_tier = {"fast": "priority", "normal": None}.get(choice, choice)
                assert runner._session_service_tier_overrides[session_key] == expected_tier
    finally:
        set_multiplex_active(False)

    default_config = yaml.safe_load(
        (default_home / "config.yaml").read_text(encoding="utf-8")
    )
    named_config = yaml.safe_load(
        (named_home / "config.yaml").read_text(encoding="utf-8")
    )
    assert default_config["agent"]["service_tier"] == "normal"
    assert named_config["agent"]["service_tier"] == (
        choice if persist and save_succeeds else "normal"
    )
    assert ("saved" in reply.lower()) == (persist and save_succeeds)


@pytest.mark.asyncio
async def test_session_fast_override_beats_config_default(monkeypatch, tmp_path):
    """A session /fast normal wins over agent.service_tier: fast in config."""
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"service_tier": "fast"}},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    event = _make_event("/fast normal")
    session_key = runner._session_key_for_source(event.source)

    response = await runner._handle_fast_command(event)

    assert "NORMAL" in response
    # Override stores explicit None (normal) and wins over config "fast".
    assert session_key in runner._session_service_tier_overrides
    assert runner._resolve_session_service_tier(session_key=session_key) is None
    # A different session still gets the config default.
    assert runner._resolve_session_service_tier(session_key="other-session") == "priority"
