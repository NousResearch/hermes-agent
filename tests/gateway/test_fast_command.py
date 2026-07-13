"""Tests for gateway /fast support and Priority Processing routing."""

import sys
import json
import threading
import types
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import AsyncSessionStore, SessionSource, SessionStore


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


def _authoritative_route_store(*, ensure_loaded, entry_for):
    """Build a minimal real SessionStore capability without disk setup."""
    store = object.__new__(SessionStore)
    store._ensure_loaded = ensure_loaded
    store.entry_for = entry_for
    return store


def _route_entry(identity):
    return SimpleNamespace(
        model_override_identity=identity,
        model_override=None,
        _model_override_identity_invalid=False,
    )


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
        "provider": "openai-api",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)

    assert route["runtime"]["provider"] == "openai-api"
    assert route["runtime"]["api_mode"] == "codex_responses"
    assert route["request_overrides"] == {"service_tier": "priority"}


def test_turn_route_fails_closed_for_proxy_gpt_model():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner, "hi", "gpt-5.6-sol", runtime_kwargs
    )

    assert route["request_overrides"] == {}


@pytest.mark.parametrize(
    ("provider", "api_mode", "model", "expected"),
    [
        (
            "openai-codex",
            "codex_responses",
            "gpt-5.5",
            {"service_tier": "fast"},
        ),
        (
            "anthropic",
            "anthropic_messages",
            "claude-opus-4-8",
            {"speed": "fast"},
        ),
    ],
)
def test_turn_route_serializes_provider_specific_fast_override(
    provider, api_mode, model, expected
):
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "runtime-secret",
        "base_url": "https://example.invalid",
        "provider": provider,
        "api_mode": api_mode,
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner, "hi", model, runtime_kwargs
    )

    assert route["request_overrides"] == expected


def test_turn_route_skips_priority_processing_for_unsupported_models():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.3-codex", runtime_kwargs)

    assert route["request_overrides"] == {}


def test_gateway_configured_identity_uses_shared_api_mode_inference(monkeypatch):
    infer = MagicMock(return_value="shared-mode")
    monkeypatch.setattr(
        "hermes_cli.providers.infer_api_mode_from_provider", infer
    )

    identity = gateway_run.GatewayRunner._configured_route_identity(
        {"model": {"default": "m", "provider": "openai-codex"}}
    )

    assert identity == ("m", "openai-codex", "shared-mode")
    infer.assert_called_once_with("openai-codex")


@pytest.mark.asyncio
async def test_handle_fast_command_persists_config(monkeypatch, tmp_path):
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    runner._resolve_configured_session_route_identity = MagicMock(
        return_value=("gpt-5.4", "openai-api", "codex_responses")
    )

    response = await runner._handle_fast_command(_make_event("/fast fast"))

    assert "FAST" in response
    assert runner._service_tier == "priority"

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["agent"]["service_tier"] == "fast"


@pytest.mark.asyncio
async def test_fast_status_and_toggle_replies_use_i18n(monkeypatch, tmp_path):
    runner = _make_runner()
    translator = MagicMock(side_effect=lambda key, **_kwargs: key)
    monkeypatch.setattr("gateway.slash_commands.t", translator)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    runner._resolve_configured_session_route_identity = MagicMock(
        return_value=("gpt-5.5", "openai-codex", "codex_responses")
    )

    assert await runner._handle_fast_command(
        _make_event("/fast status")
    ) == "gateway.fast.route_status"
    assert await runner._handle_fast_command(
        _make_event("/fast fast")
    ) == "gateway.fast.route_saved"

    runner._resolve_configured_session_route_identity.return_value = (
        "gpt-5.6-sol",
        "openai-codex",
        "codex_responses",
    )
    assert await runner._handle_fast_command(
        _make_event("/fast status")
    ) == "gateway.fast.route_off"
    assert await runner._handle_fast_command(
        _make_event("/fast fast")
    ) == "gateway.fast.route_unavailable"

    def _fail_write(*_args, **_kwargs):
        raise OSError("read-only test config")

    monkeypatch.setattr("gateway.slash_commands.atomic_config_write", _fail_write)
    runner._resolve_configured_session_route_identity.return_value = (
        "gpt-5.5",
        "openai-codex",
        "codex_responses",
    )
    assert await runner._handle_fast_command(
        _make_event("/fast normal")
    ) == "gateway.fast.route_session_only"


@pytest.mark.asyncio
async def test_fast_persistence_failure_is_honestly_process_wide(
    monkeypatch, tmp_path
):
    runner = _make_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    runner._resolve_configured_session_route_identity = MagicMock(
        return_value=("gpt-5.5", "openai-codex", "codex_responses")
    )
    monkeypatch.setattr(
        "gateway.slash_commands.atomic_config_write",
        MagicMock(side_effect=OSError("read only")),
    )

    response = await runner._handle_fast_command(_make_event("/fast fast"))

    assert "gateway-process-wide" in response
    assert "this session only" not in response
    assert runner._service_tier == "priority"
    runtime = {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "api_key": "token",
    }
    # _service_tier belongs to the runner, so two distinct response scopes see
    # the same fallback state until restart.
    first = runner._resolve_turn_agent_config("s1", "gpt-5.5", runtime)
    second = runner._resolve_turn_agent_config("s2", "gpt-5.5", runtime)
    assert first["request_overrides"] == second["request_overrides"] == {
        "service_tier": "fast"
    }


@pytest.mark.asyncio
async def test_fast_status_uses_persisted_session_route_without_credentials(
    monkeypatch, tmp_path
):
    runner = _make_runner()
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    persisted = {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None,
        entry_for=lambda key: _route_entry(persisted)
        if key == session_key
        else None,
    )
    runner._session_model_overrides[session_key] = {
        **persisted,
        "api_key": "must-not-be-read",
    }

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "model": {
                "default": "claude-opus-4-8",
                "provider": "anthropic",
                "api_mode": "anthropic_messages",
            }
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        MagicMock(side_effect=AssertionError("credential checkout during /fast status")),
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        MagicMock(side_effect=AssertionError("provider checkout during /fast status")),
    )

    response = await runner._handle_fast_command(_make_event("/fast status"))

    assert "openai-codex/gpt-5.6-sol" in response
    assert "Codex Fast" in response
    assert "GPT-5.5" in response
    assert "claude" not in response.lower()


@pytest.mark.asyncio
async def test_fast_status_reports_persisted_preference_as_unavailable_without_checkout(
    monkeypatch, tmp_path
):
    runner = _make_runner()
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    persisted = {
        "model": "gpt-5.5",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None,
        entry_for=lambda _key: _route_entry(persisted),
    )
    runner._session_model_override_unavailable = {session_key}
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    checkout = MagicMock(side_effect=AssertionError("status checked credentials"))
    monkeypatch.setattr(runner, "_reresolve_model_override_credentials", checkout)

    response = await runner._handle_fast_command(_make_event("/fast status"))
    toggle_response = await runner._handle_fast_command(_make_event("/fast fast"))

    assert "Configured session preference" in response
    assert "currently unavailable" in response
    assert "openai-codex/gpt-5.5" in response
    assert toggle_response == response
    checkout.assert_not_called()


@pytest.mark.asyncio
async def test_fast_status_opus_48_proxy_points_to_separate_fast_model(
    monkeypatch, tmp_path
):
    runner = _make_runner()
    runner._resolve_configured_session_route_identity = MagicMock(
        return_value=(
            "claude-opus-4-8",
            "claude-apr",
            "anthropic_messages",
        )
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

    response = await runner._handle_fast_command(_make_event("/fast status"))

    assert "claude-apr/claude-opus-4-8" in response
    assert "speed=fast" in response
    assert "claude-opus-4-8-fast" in response


def test_pure_route_identity_matches_runtime_precedence(monkeypatch):
    source = _make_source()
    session_key = "agent:main:telegram:dm:12345"
    config = {
        "model": {
            "default": "gpt-5.4",
            "provider": "openai-api",
            "api_mode": "codex_responses",
        }
    }

    runner = _make_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                channel_overrides={
                    "12345": ChannelOverride(
                        model="gpt-5.5",
                        provider="openai-codex",
                    )
                }
            )
        }
    )
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None, entry_for=lambda _key: None
    )

    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-api",
            "api_mode": "codex_responses",
            "api_key": "global-key",
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {
            "provider": provider,
            "api_mode": "codex_responses",
            "api_key": "channel-key",
        },
    )

    pure = runner._resolve_configured_session_route_identity(
        source=source,
        session_key=session_key,
        user_config=config,
    )
    model, runtime = runner._resolve_session_agent_runtime(
        source=source,
        session_key=session_key,
        user_config=config,
    )

    assert pure == (model, runtime["provider"], runtime["api_mode"])
    assert pure == ("gpt-5.5", "openai-codex", "codex_responses")


def test_pure_route_identity_matches_runtime_for_session_override(monkeypatch):
    source = _make_source()
    runner = _make_runner()
    session_key = runner._session_key_for_source(source)
    identity = {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None,
        entry_for=lambda _key: _route_entry(identity),
    )
    runner._session_model_overrides[session_key] = {
        **identity,
        "api_key": "session-secret",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "credential_pool": object(),
    }
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        MagicMock(side_effect=AssertionError("session override must win")),
    )
    config = {
        "model": {
            "default": "claude-opus-4-8",
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
        }
    }

    pure = runner._resolve_configured_session_route_identity(
        source=source, session_key=session_key, user_config=config
    )
    model, runtime = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=config
    )

    assert pure == (model, runtime["provider"], runtime["api_mode"])
    assert pure == ("gpt-5.6-sol", "openai-codex", "codex_responses")


def test_pure_route_identity_matches_runtime_for_channel_and_session_override(
    monkeypatch,
):
    source = _make_source()
    runner = _make_runner()
    session_key = runner._session_key_for_source(source)
    identity = {
        "model": "claude-opus-4-6",
        "provider": "anthropic",
        "api_mode": "anthropic_messages",
    }
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                channel_overrides={
                    "12345": ChannelOverride(
                        model="gpt-5.5",
                        provider="openai-codex",
                    )
                }
            )
        }
    )
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None,
        entry_for=lambda _key: _route_entry(identity),
    )
    runner._session_model_overrides[session_key] = {
        **identity,
        "api_key": "session-secret",
    }
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        MagicMock(side_effect=AssertionError("session override must win")),
    )
    config = {
        "model": {
            "default": "gpt-5.4",
            "provider": "openai-api",
            "api_mode": "codex_responses",
        }
    }

    pure = runner._resolve_configured_session_route_identity(
        source=source, session_key=session_key, user_config=config
    )
    model, runtime = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=config
    )

    assert pure == (model, runtime["provider"], runtime["api_mode"])
    assert pure == ("claude-opus-4-6", "anthropic", "anthropic_messages")


def test_missing_persisted_api_mode_does_not_rehydrate_matching_cached_route(
    monkeypatch,
):
    source = _make_source()
    runner = _make_runner()
    session_key = runner._session_key_for_source(source)
    persisted = {"model": "gpt-5.5", "provider": "openai-codex"}
    cached = {
        **persisted,
        "api_mode": "codex_responses",
        "api_key": "cached-secret",
    }
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None,
        entry_for=lambda _key: _route_entry(persisted),
    )
    runner._session_model_overrides[session_key] = cached
    reresolve = MagicMock(
        side_effect=AssertionError("matching route must not rehydrate credentials")
    )
    monkeypatch.setattr(runner, "_reresolve_model_override_credentials", reresolve)

    model, runtime = runner._resolve_session_agent_runtime(
        source=source,
        session_key=session_key,
        user_config={"model": {"default": "global-model"}},
    )

    assert model == "gpt-5.5"
    assert runtime["api_mode"] == "codex_responses"
    assert runner._session_model_overrides[session_key] is cached
    reresolve.assert_not_called()


def test_pure_route_identity_matches_runtime_for_global_route(monkeypatch):
    runner = _make_runner()
    runner.config = GatewayConfig()
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None, entry_for=lambda _key: None
    )
    config = {
        "model": {
            "default": "claude-opus-4-6",
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
        }
    }
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "api_key": "global-secret",
        },
    )

    pure = runner._resolve_configured_session_route_identity(
        source=_make_source(), session_key="global-key", user_config=config
    )
    model, runtime = runner._resolve_session_agent_runtime(
        source=_make_source(), session_key="global-key", user_config=config
    )

    assert pure == (model, runtime["provider"], runtime["api_mode"])


@pytest.mark.parametrize(
    ("channel_model", "channel_provider", "session_override"),
    product((False, True), repeat=3),
)
def test_pure_route_identity_matches_runtime_exhaustive_precedence_matrix(
    monkeypatch, channel_model, channel_provider, session_override
):
    """Guard pure /fast identity selection against full runtime precedence."""
    source = _make_source()
    runner = _make_runner()
    session_key = runner._session_key_for_source(source)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                channel_overrides={
                    "12345": ChannelOverride(
                        model="claude-opus-4-6" if channel_model else None,
                        provider="anthropic" if channel_provider else None,
                    )
                }
            )
        }
    )
    identity = (
        {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
        }
        if session_override
        else None
    )
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None,
        entry_for=lambda _key: _route_entry(identity) if identity else None,
    )
    if identity:
        runner._session_model_overrides[session_key] = {
            **identity,
            "api_key": "session-secret",
        }

    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-api",
            "api_mode": "codex_responses",
            "api_key": "global-secret",
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {
            "provider": provider,
            "api_mode": "anthropic_messages",
            "api_key": "channel-secret",
        },
    )
    config = {
        "model": {
            "default": "gpt-5.4",
            "provider": "openai-api",
            "api_mode": "codex_responses",
        }
    }

    pure = runner._resolve_configured_session_route_identity(
        source=source, session_key=session_key, user_config=config
    )
    model, runtime = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=config
    )

    assert pure == (model, runtime["provider"], runtime["api_mode"])


def test_model_switch_note_reports_enabled_fast_becoming_unavailable():
    runner = _make_runner()
    runner._service_tier = "priority"
    result = SimpleNamespace(
        new_model="gpt-5.6-sol",
        target_provider="openai-codex",
        api_mode="codex_responses",
    )

    note = runner._fast_unavailable_model_switch_row(result)

    assert "Fast: unavailable" in note
    assert "openai-codex/gpt-5.6-sol" in note
    assert "normal speed" in note


def test_model_switch_note_derives_api_mode_when_result_omits_it():
    runner = _make_runner()
    runner._service_tier = "priority"
    result = SimpleNamespace(
        new_model="gpt-5.5",
        target_provider="openai-codex",
    )

    assert runner._fast_unavailable_model_switch_row(result) is None


def test_model_switch_unavailable_note_uses_i18n(monkeypatch):
    runner = _make_runner()
    runner._service_tier = "priority"
    translator = MagicMock(return_value="localized model-switch note")
    monkeypatch.setattr("gateway.slash_commands.t", translator)

    note = runner._fast_unavailable_model_switch_row(
        SimpleNamespace(
            new_model="gpt-5.6-sol",
            target_provider="openai-codex",
            api_mode="codex_responses",
        )
    )

    assert note == "localized model-switch note"
    translator.assert_called_once_with(
        "gateway.fast.model_switch_unavailable",
        route="openai-codex/gpt-5.6-sol",
    )


def test_new_fast_gateway_messages_have_english_i18n_fallbacks():
    from agent.i18n import t

    assert t(
        "gateway.fast.route_status",
        lang="es",
        state="on",
        family="Codex Fast",
        route="openai-codex/gpt-5.5",
    ) == "⚡ Fast: on — Codex Fast on `openai-codex/gpt-5.5`"
    assert t(
        "gateway.fast.model_switch_unavailable",
        lang="es",
        route="openai-codex/gpt-5.6-sol",
    ).endswith("requests will use normal speed.")


@pytest.mark.asyncio
async def test_run_agent_passes_priority_processing_to_gateway_agent(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    (tmp_path / "config.yaml").write_text("agent:\n  service_tier: fast\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    # ``_load_service_tier`` was refactored to call ``_load_gateway_runtime_config``
    # (which wraps ``_load_gateway_config`` plus env-expansion).  Since the test
    # stubs ``_load_gateway_config`` to ``{}``, also stub the runtime wrapper
    # directly so the priority routing assertions still exercise the live tier.
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"service_tier": "fast"}},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-api",
            "api_mode": "codex_responses",
            "base_url": "https://api.openai.com/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init["service_tier"] == "priority"
    assert _CapturingAgent.last_init["request_overrides"] == {"service_tier": "priority"}


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution_failure", [None, RuntimeError("keychain unavailable")])
async def test_run_agent_fails_closed_when_persisted_route_credentials_unavailable(
    monkeypatch, tmp_path, resolution_failure
):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    session_key = "agent:main:telegram:dm:12345"
    identity = {
        "model": "gpt-5.5",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    runner.session_store = _authoritative_route_store(
        ensure_loaded=lambda: None,
        entry_for=lambda key: (
            SimpleNamespace(
                model_override_identity=identity,
                model_override=None,
                _model_override_identity_invalid=False,
            )
            if key == session_key
            else None
        ),
    )
    resolver = MagicMock()
    if resolution_failure is None:
        resolver.return_value = None
    else:
        resolver.side_effect = resolution_failure
    monkeypatch.setattr(runner, "_reresolve_model_override_credentials", resolver)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda config=None: "global-fallback"
    )
    configured_provider = MagicMock(
        side_effect=AssertionError("configured provider must not be resolved")
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", configured_provider
    )

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="do not reroute this prompt",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key=session_key,
    )

    assert "currently unavailable" in result["final_response"]
    assert "preference was preserved" in result["final_response"]
    assert result["api_calls"] == 0
    configured_provider.assert_not_called()
    assert _CapturingAgent.last_init is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "ensure_loaded",
        "entry_for",
        "identity_read",
        "malformed_string",
        "malformed_missing_provider",
    ],
)
async def test_persisted_route_lookup_failures_abort_before_enrichment_or_provider(
    monkeypatch, failure
):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    source = _make_source()
    session_key = "agent:main:telegram:dm:12345"
    entry = SimpleNamespace(session_key=session_key, session_id="session-1")

    def ensure_loaded():
        if failure == "ensure_loaded":
            raise OSError("routing store unavailable")

    def entry_for(_key):
        if failure == "entry_for":
            raise OSError("routing entry unavailable")
        if failure == "identity_read":
            class UnreadableEntry:
                _model_override_identity_invalid = False

                @property
                def model_override_identity(self):
                    raise OSError("identity read unavailable")

            return UnreadableEntry()
        identity = (
            "not-a-mapping"
            if failure == "malformed_string"
            else {"model": "gpt-5.5"}
        )
        return SimpleNamespace(
            model_override_identity=identity,
            model_override=None,
            _model_override_identity_invalid=False,
        )

    store = _authoritative_route_store(
        ensure_loaded=ensure_loaded,
        entry_for=entry_for,
    )
    store.get_or_create_session = lambda _source: entry
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._recover_telegram_topic_thread_id = MagicMock(return_value=None)
    runner._prepare_inbound_message_text = AsyncMock(
        side_effect=AssertionError("enrichment must not run")
    )
    global_provider = MagicMock(
        side_effect=AssertionError("global provider must not run")
    )
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", global_provider)

    _CapturingAgent.last_init = None
    response = await runner._handle_message_with_agent(
        _make_event("route safely"), source, "quick", 1
    )

    assert "could not be read or validated" in response
    runner._prepare_inbound_message_text.assert_not_awaited()
    global_provider.assert_not_called()
    assert _CapturingAgent.last_init is None


@pytest.mark.parametrize("store", [MagicMock(), SimpleNamespace(_entries={})])
def test_non_persistent_fixture_stores_have_no_persisted_route_authority(store):
    runner = _make_runner()
    runner.session_store = store

    lookup = runner._persisted_session_route_identity("fixture-session")

    assert lookup.state == "absent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_identity",
    [
        "not-a-mapping",
        {"model": "gpt-5.5"},
        {"provider": "openai-codex"},
    ],
)
async def test_malformed_persisted_identity_roundtrip_fails_closed_before_enrichment(
    monkeypatch, tmp_path, malformed_identity
):
    """Exercise SessionEntry.from_dict via the real sessions.json load path."""
    import hermes_state

    monkeypatch.setattr(
        hermes_state,
        "SessionDB",
        MagicMock(side_effect=RuntimeError("JSON-only persistence fixture")),
    )
    source = _make_source()
    sessions_dir = tmp_path / "sessions"
    seed = SessionStore(sessions_dir, GatewayConfig())
    entry = seed.get_or_create_session(source)
    payload = json.loads((sessions_dir / "sessions.json").read_text())
    payload[entry.session_key]["model_override_identity"] = malformed_identity
    (sessions_dir / "sessions.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = SessionStore(sessions_dir, GatewayConfig())
    loaded._ensure_loaded()
    loaded_entry = loaded.entry_for(entry.session_key)
    assert loaded_entry.model_override_identity is None
    assert loaded_entry._model_override_identity_invalid is True

    runner = _make_runner()
    runner.session_store = loaded
    runner._async_session_store = AsyncSessionStore(loaded)
    runner._recover_telegram_topic_thread_id = MagicMock(return_value=None)
    runner._prepare_inbound_message_text = AsyncMock(
        side_effect=AssertionError("enrichment must not run")
    )
    global_provider = MagicMock(
        side_effect=AssertionError("global provider must not run")
    )
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", global_provider)
    _install_fake_agent(monkeypatch)

    _CapturingAgent.last_init = None
    response = await runner._handle_message_with_agent(
        _make_event("route safely"), source, "quick", 1
    )

    assert "could not be read or validated" in response
    runner._prepare_inbound_message_text.assert_not_awaited()
    global_provider.assert_not_called()
    assert _CapturingAgent.last_init is None


@pytest.mark.asyncio
async def test_run_agent_passes_discord_auto_thread_title_callback(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    runner._session_db = SimpleNamespace(_db=MagicMock())  # type: ignore[assignment]

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
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
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    with patch("agent.title_generator.maybe_auto_title") as mock_title:
        await runner._run_agent(
            message="raw user prompt",
            context_prompt="",
            history=[],
            source=_make_discord_auto_thread_source(),
            session_id="session-1",
            session_key="agent:main:discord:thread:999",
        )

    mock_title.assert_called_once()
    callback = mock_title.call_args.kwargs["title_callback"]
    with patch.object(runner, "_schedule_discord_semantic_thread_rename") as mock_schedule:
        callback("Semantic Session Title")
    mock_schedule.assert_called_once()
    assert mock_schedule.call_args.args[1] == "session-1"
    assert mock_schedule.call_args.args[2] == "Semantic Session Title"


def test_session_source_preserves_discord_auto_thread_metadata():
    source = _make_discord_auto_thread_source()

    restored = SessionSource.from_dict(source.to_dict())

    assert restored.auto_thread_created is True
    assert restored.auto_thread_initial_name == "raw user prompt"
