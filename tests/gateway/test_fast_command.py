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
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)

    assert route["runtime"]["provider"] == "openrouter"
    assert route["runtime"]["api_mode"] == "chat_completions"
    assert route["request_overrides"] == {"service_tier": "priority"}


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


@pytest.mark.asyncio
async def test_handle_fast_command_session_scoped_by_default(monkeypatch, tmp_path):
    """Bare /fast fast applies a session override — config.yaml untouched."""
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    # ``/fast`` now resolves the effective session model through
    # ``_resolve_session_agent_runtime``; with no session override that path
    # calls runtime provider resolution, so stub it (no real credentials in CI).
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    response = await runner._handle_fast_command(_make_event("/fast fast"))

    assert "FAST" in response
    assert runner._service_tier == "priority"
    # Session override recorded; config.yaml NOT written.
    assert runner._session_service_tier_overrides
    assert not (tmp_path / "config.yaml").exists()


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
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
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


@pytest.mark.asyncio
async def test_handle_fast_command_allows_when_session_override_supports_fast(
    monkeypatch, tmp_path
):
    """Global default is fast-ineligible, but a session /model override is.

    The eligibility check must resolve the model the session will actually use
    (via ``_resolve_session_agent_runtime``), not the global config default —
    otherwise ``/fast`` is wrongly rejected after switching to a fast-capable
    model in-session.
    """
    runner = _make_runner()
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._session_model_overrides[session_key] = {
        "model": "gpt-5.4",
        "api_key": "***",
        "credential_pool": "pool",
    }

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    # Global default is Anthropic Opus 4.8, which does NOT support the fast
    # *parameter* path (fast is a separate model id there).
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda config=None: "claude-opus-4-8"
    )

    response = await runner._handle_fast_command(
        MessageEvent(text="/fast fast", source=source, message_id="m1")
    )

    assert "FAST" in response
    assert runner._service_tier == "priority"
    # Bare /fast (no --global) is session-scoped: the override is recorded and
    # config.yaml stays untouched (global persistence requires --global).
    assert runner._session_service_tier_overrides
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.asyncio
async def test_handle_fast_command_rejects_when_session_override_unsupported(
    monkeypatch, tmp_path
):
    """Global default supports fast, but the session override does not.

    ``/fast`` must be refused because the effective session model is
    fast-ineligible — the reverse of the previous test, guarding both
    directions.
    """
    runner = _make_runner()
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._session_model_overrides[session_key] = {
        "model": "claude-opus-4-8",
        "api_key": "***",
        "credential_pool": "pool",
    }

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4"
    )

    response = await runner._handle_fast_command(
        MessageEvent(text="/fast fast", source=source, message_id="m1")
    )

    assert "only available for OpenAI models" in response
    assert runner._service_tier is None
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.asyncio
async def test_handle_fast_command_rehydrates_persisted_session_override(
    monkeypatch, tmp_path
):
    """A persisted /model override that a restart cleared from memory is
    rehydrated by ``_resolve_session_agent_runtime`` before eligibility is
    judged, so ``/fast`` respects it even with an empty in-memory dict.
    """
    runner = _make_runner()
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    # In-memory overrides are empty (simulating a fresh gateway restart);
    # only the session store still knows the user switched to gpt-5.4.
    runner._session_model_overrides = {}
    _persisted_key = session_key
    runner.session_store.get_model_override = lambda session_key: (
        {"model": "gpt-5.4"} if session_key == _persisted_key else None
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda config=None: "claude-opus-4-8"
    )
    # The persisted override has no provider, so the resolver falls through to
    # env-based runtime resolution then re-applies the override model on top.
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    response = await runner._handle_fast_command(
        MessageEvent(text="/fast fast", source=source, message_id="m1")
    )

    assert "FAST" in response
    assert runner._service_tier == "priority"
    # The persisted override was rehydrated into the in-memory dict.
    assert runner._session_model_overrides.get(session_key, {}).get("model") == "gpt-5.4"


@pytest.mark.asyncio
async def test_handle_fast_command_normalizes_source_before_override_lookup(
    monkeypatch, tmp_path
):
    """The command source is normalized (Telegram DM topic recovery, #30479)
    before deriving the override key, so ``/fast`` reads the override under the
    same key the next message turn will — a lobby-shaped reply pinned to the
    user's last-active topic.
    """
    runner = _make_runner()
    source = _make_source()  # telegram DM, thread_id=None (lobby)
    # The next turn recovers this DM to the user's last-active topic 77.
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda src: "77")
    import dataclasses

    recovered_source = dataclasses.replace(source, thread_id="77")
    recovered_key = runner._session_key_for_source(recovered_source)
    raw_key = runner._session_key_for_source(source)
    assert recovered_key != raw_key  # normalization actually changes the key
    runner._session_model_overrides[recovered_key] = {
        "model": "gpt-5.4",
        "api_key": "***",
        "credential_pool": "pool",
    }

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda config=None: "claude-opus-4-8"
    )

    response = await runner._handle_fast_command(
        MessageEvent(text="/fast fast", source=source, message_id="m1")
    )

    # If the handler had keyed off the raw (un-normalized) source it would miss
    # the override and reject against the unsupported global default.
    assert "FAST" in response
    assert runner._service_tier == "priority"


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
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
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
