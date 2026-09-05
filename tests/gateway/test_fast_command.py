"""Tests for gateway /fast support and Priority Processing routing."""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
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
    runner._service_tier_escalation = None
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

    # OpenRouter forwards the wire tier (flex/priority) rather than stripping it.
    runtime_kwargs.update(base_url="https://openrouter.ai/api/v1", provider="openrouter")
    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)
    assert route["request_overrides"] == {"service_tier": "priority"}


def test_turn_route_injects_flex_for_openrouter_variant():
    runner = _make_runner()
    runner._service_tier = "flex"
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
        runner,
        "hi",
        "deepseek/deepseek-v4-flash-0731:nitro",
        runtime_kwargs,
    )

    assert route["request_overrides"] == {"service_tier": "flex"}


def test_load_service_tier_accepts_flex(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"service_tier": "FLEX"}},
    )

    assert gateway_run.GatewayRunner._load_service_tier() == "flex"


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


def test_turn_route_uses_passed_tier_not_runner_shared_field():
    """Resolved turn tier must not be reread from the shared runner snapshot."""
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

    route_flex = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner, "hi", "gpt-5.4", runtime_kwargs, service_tier="flex",
    )
    runner._service_tier = "priority"
    route_normal = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner, "hi", "gpt-5.4", runtime_kwargs, service_tier=None,
    )
    runner._service_tier = "flex"
    route_priority = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner, "hi", "gpt-5.4", runtime_kwargs, service_tier="priority",
    )

    assert route_flex["request_overrides"] == {"service_tier": "flex"}
    assert route_normal["request_overrides"] == {}
    assert route_priority["request_overrides"] == {"service_tier": "priority"}


@pytest.mark.asyncio
async def test_handle_fast_status_shows_flex(monkeypatch, tmp_path):
    runner = _make_runner()
    runner._try_send_choice_picker = AsyncMock(return_value=False)
    runner._peek_session_state = lambda _key: None
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "google/gemini-3.7-flash",
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "agent": {
                "service_tier": "",
                "service_tier_overrides": {"google/gemini-3.7-flash": "flex"},
            }
        },
    )
    from hermes_cli.models import model_supports_fast_mode

    # * Real OR model: capability is false; status must still show flex.
    assert model_supports_fast_mode("google/gemini-3.7-flash") is False

    response = await runner._handle_fast_command(_make_event("/fast status"))

    assert response is not None
    assert "flex" in response.lower()


@pytest.mark.asyncio
async def test_handle_fast_switch_gated_normal_always_available(monkeypatch, tmp_path):
    runner = _make_runner()
    runner._try_send_choice_picker = AsyncMock(return_value=False)
    runner._peek_session_state = lambda _key: None
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "google/gemini-3.7-flash",
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "agent": {
                "service_tier": "flex",
                "service_tier_overrides": {},
            }
        },
    )
    from hermes_cli.models import model_supports_fast_mode

    assert model_supports_fast_mode("google/gemini-3.7-flash") is False

    blocked = await runner._handle_fast_command(_make_event("/fast fast"))
    assert blocked is not None
    assert "only available" in blocked.lower()
    assert runner._service_tier == "flex"

    cleared = await runner._handle_fast_command(_make_event("/fast normal"))
    assert cleared is not None
    assert "NORMAL" in cleared
    assert runner._service_tier is None


def _store_factory(tmp_path, monkeypatch):
    """Real SessionStore over a shared sessions dir, without SQLite."""
    import hermes_state
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    def _raise(*_a, **_k):
        raise RuntimeError("SQLite disabled in test")

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make():
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None
        return store

    return _make


def _make_runner_with_store(store, config=None):
    runner = _make_runner()
    runner.session_store = store
    if config is not None:
        runner.config = config
    runner._try_send_choice_picker = AsyncMock(return_value=False)
    return runner


@pytest.mark.asyncio
async def test_fast_status_uses_persisted_model_after_restart(monkeypatch, tmp_path):
    """After a gateway restart, first /fast status uses the stored /model."""
    make_store = _store_factory(tmp_path, monkeypatch)
    source = _make_source()
    store = make_store()
    entry = store.get_or_create_session(source)
    store.set_model_override(
        entry.session_key,
        {"model": "google/gemini-3.7-flash", "provider": "openrouter"},
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "openai/gpt-5",
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "agent": {
                "service_tier": "priority",
                "service_tier_overrides": {
                    "google/gemini-3.7-flash": "flex",
                },
            }
        },
    )

    restarted = _make_runner_with_store(make_store())
    event = _make_event("/fast status")
    assert event.source.chat_id == source.chat_id

    response = await restarted._handle_fast_command(event)

    assert response is not None
    assert "flex" in response.lower()
    assert "FAST" not in response

    blocked = await restarted._handle_fast_command(_make_event("/fast fast"))
    assert blocked is not None
    assert "only available" in blocked.lower()

    cleared = await restarted._handle_fast_command(_make_event("/fast normal"))
    assert cleared is not None
    assert "NORMAL" in cleared


@pytest.mark.asyncio
async def test_fast_status_uses_channel_override_model(monkeypatch, tmp_path):
    from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "openai/gpt-5",
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "agent": {
                "service_tier": "priority",
                "service_tier_overrides": {
                    "google/gemini-3.7-flash": "flex",
                },
            }
        },
    )

    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "12345": ChannelOverride(model="google/gemini-3.7-flash"),
                },
            ),
        },
    )
    runner = _make_runner_with_store(
        types.SimpleNamespace(
            get_or_create_session=lambda source: types.SimpleNamespace(
                session_id="session-1"
            ),
            load_transcript=lambda session_id: [],
        ),
        config=config,
    )

    response = await runner._handle_fast_command(_make_event("/fast status"))

    assert response is not None
    assert "flex" in response.lower()
    assert "FAST" not in response


def _patch_empty_default_runtime(monkeypatch, *, provider="openai-codex", model=""):
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": provider,
            "api_key": "k",
            "base_url": "https://example.test",
            "api_mode": "chat_completions",
            **({"model": model} if model else {}),
        },
    )


@pytest.mark.asyncio
async def test_fast_status_and_turn_agree_on_empty_default(monkeypatch, tmp_path):
    """Empty model.default: /fast status gates on the same catalog id as the turn."""
    from hermes_cli.models import get_default_model_for_provider

    expected = get_default_model_for_provider("openai-codex")
    assert expected

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"default": "", "provider": "openai-codex"}},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "")
    _patch_empty_default_runtime(monkeypatch, provider="openai-codex")
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "agent": {
                "service_tier": "priority",
                "service_tier_overrides": {expected: "flex"},
            }
        },
    )

    runner = _make_runner()
    runner._try_send_choice_picker = AsyncMock(return_value=False)
    source = _make_source()
    session_key = runner._session_key_for_source(source)

    response = await runner._handle_fast_command(_make_event("/fast status"))
    turn_model, _ = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config={"model": {"default": "", "provider": "openai-codex"}}
    )

    assert response is not None
    assert "flex" in response.lower()
    assert turn_model == expected


def test_fast_and_turn_agree_on_runtime_provider_model(monkeypatch, tmp_path):
    """Empty model.default: /fast and the next turn pick the provider catalog id."""
    from hermes_cli.models import get_default_model_for_provider

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _patch_empty_default_runtime(monkeypatch, provider="openai-codex")
    expected = get_default_model_for_provider("openai-codex")
    assert expected

    runner = _make_runner()
    user_config = {"model": {"default": "", "provider": "openai-codex"}}
    source = _make_source()
    session_key = runner._session_key_for_source(source)

    fast_model = runner._resolve_session_effective_model(
        source=source, session_key=session_key, user_config=user_config
    )
    turn_model, _ = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=user_config
    )

    assert fast_model == turn_model == expected


def test_fast_and_turn_agree_on_provider_only_channel(monkeypatch, tmp_path):
    """Channel override with only provider: /fast and turn share the catalog model."""
    from gateway.config import ChannelOverride, GatewayConfig, PlatformConfig
    from hermes_cli.models import get_default_model_for_provider

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "anthropic",
            "api_key": "k",
            "base_url": "https://api.anthropic.com",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {
            "provider": provider,
            "api_key": "k2",
            "base_url": "https://example.test",
            "api_mode": "chat_completions",
        },
    )
    expected = get_default_model_for_provider("openai-codex")
    assert expected

    runner = _make_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "12345": ChannelOverride(provider="openai-codex"),
                },
            ),
        },
    )
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    user_config = {"model": {"default": "", "provider": "anthropic"}}

    fast_model = runner._resolve_session_effective_model(
        source=source, session_key=session_key, user_config=user_config
    )
    turn_model, _ = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=user_config
    )

    assert fast_model == turn_model == expected


def test_fast_and_turn_agree_on_last_resolved_model(monkeypatch, tmp_path):
    """Empty config: /fast and turn recover the same last-known-good model."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _patch_empty_default_runtime(monkeypatch, provider="")
    runner = _make_runner()
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._session_state(session_key).conversation.last_resolved_model = (
        "cached/last-good"
    )
    user_config = {"model": {}}

    fast_model = runner._resolve_session_effective_model(
        source=source, session_key=session_key, user_config=user_config
    )
    turn_model, _ = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=user_config
    )

    assert fast_model == turn_model == "cached/last-good"


@pytest.mark.asyncio
async def test_fast_status_skips_credential_rehydrate(monkeypatch, tmp_path):
    """Persisted /model must not trigger credential resolution on /fast status."""
    make_store = _store_factory(tmp_path, monkeypatch)
    source = _make_source()
    store = make_store()
    entry = store.get_or_create_session(source)
    store.set_model_override(
        entry.session_key,
        {"model": "google/gemini-3.7-flash", "provider": "openrouter"},
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "openai/gpt-5",
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "agent": {
                "service_tier": "priority",
                "service_tier_overrides": {
                    "google/gemini-3.7-flash": "flex",
                },
            }
        },
    )
    cred_spy = MagicMock(side_effect=AssertionError("status path must stay model-only"))
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs_for_provider", cred_spy
    )

    restarted = _make_runner_with_store(make_store())
    response = await restarted._handle_fast_command(_make_event("/fast status"))

    assert response is not None
    assert "flex" in response.lower()
    cred_spy.assert_not_called()
