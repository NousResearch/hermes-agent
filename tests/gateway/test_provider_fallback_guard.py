from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(platform: Platform = Platform.DISCORD) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, *, platform: Platform = Platform.DISCORD) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(platform), message_id="m1")


def _make_runner(platform: Platform = Platform.DISCORD):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source(platform)),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._session_sources = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session.return_value = None
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._begin_session_run_generation = lambda _key: 1
    runner._release_running_agent_state = lambda _key: runner._running_agents.pop(_key, None)
    return runner


def test_fallback_config_none_raises_named_error():
    from agent.provider_errors import HermesFallbackConfigError
    from gateway.provider_guard import validate_gateway_agent_turn_config

    with pytest.raises(HermesFallbackConfigError) as exc:
        validate_gateway_agent_turn_config(
            {"fallback_model": None},
            {
                "provider": "openai-codex",
                "model": "gpt-5.5",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "token-present",
            },
        )

    assert "NoneType" not in str(exc.value)
    assert exc.value.error_class == "provider_config_or_fallback_failure"


def test_missing_fallback_reports_no_valid_fallback_status():
    from gateway.provider_guard import validate_gateway_agent_turn_config

    status = validate_gateway_agent_turn_config(
        {},
        {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "token-present",
        },
    )

    assert status.fallback_status == "no_valid_fallback_configured"


def test_openai_responses_api_config_is_valid_when_configured():
    from gateway.provider_guard import validate_gateway_agent_turn_config

    status = validate_gateway_agent_turn_config(
        {},
        {
            "provider": "openai",
            "model": "operator-configured-model",
            "base_url": "https://api.openai.com/v1",
            "api_key": "token-present",
        },
    )

    assert status.provider == "openai"
    assert status.base_url_class == "api.openai.com"


def test_startup_validation_runtime_is_structural_for_codex(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        lambda provider: {"tokens": {"access_token": "present"}} if provider == "openai-codex" else None,
    )

    runtime = GatewayRunner._resolve_agent_turn_validation_runtime(
        {"model": {"provider": "openai-codex", "default": "gpt-5.5"}}
    )

    assert runtime["provider"] == "openai-codex"
    assert runtime["model"] == "gpt-5.5"
    assert runtime["api_key"] == "configured"


def test_primary_local_typeerror_is_sanitized_provider_failure():
    from agent.provider_errors import provider_failure_result

    result = provider_failure_result(
        TypeError("'NoneType' object is not iterable"),
        provider="openai-codex",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        fallback_status="no_valid_fallback_configured",
    )

    assert result["error_class"] == "provider_config_or_fallback_failure"
    assert result["fallback_status"] == "no_valid_fallback_configured"
    assert "NoneType" not in result["error"]
    assert "chatgpt.com/backend-api/codex" not in result["error"]


def test_discord_status_callback_suppresses_internal_provider_retry_warning():
    from gateway.run import _prepare_gateway_status_message

    unsafe = "⚠️ Non-retryable error (HTTP None) — trying fallback..."

    assert _prepare_gateway_status_message(Platform.DISCORD, "api_error", unsafe) is None


@pytest.mark.asyncio
async def test_gateway_returns_controlled_diagnostic_for_provider_failure():
    from agent.provider_errors import HermesProviderConfigError

    runner = _make_runner()
    runner._agent_turn_config_error = HermesProviderConfigError(
        "primary provider config failed"
    )
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("provider failure leaked to agent")
    )

    result = await runner._handle_message(_make_event("hello Hermes"))

    assert "agent turn path is unavailable" in result
    assert "provider_config_or_fallback_failure" in result
    assert "NoneType" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_provider_failure_discord_output_excludes_internal_preambles():
    from agent.provider_errors import HermesProviderConfigError

    runner = _make_runner()
    runner._agent_turn_config_error = HermesProviderConfigError(
        "primary provider config failed"
    )
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("provider failure leaked to agent")
    )

    result = await runner._handle_message(_make_event("hello Hermes"))

    assert "provider_config_or_fallback_failure" in result
    forbidden = (
        "Non-retryable error",
        "HTTP None",
        "trying fallback",
        "NoneType",
        "traceback",
        "request dump",
        "token",
        "api key",
        "authorization",
        "cookie",
        "bearer",
    )
    lowered = result.lower()
    for term in forbidden:
        assert term.lower() not in lowered
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_still_works_when_agent_provider_unavailable():
    from agent.provider_errors import HermesProviderConfigError

    runner = _make_runner()
    runner._agent_turn_config_error = HermesProviderConfigError(
        "primary provider config failed"
    )
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("status command leaked to agent")
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Session ID:** `sess-1`" in result
    assert "**Agent Running:** No" in result
    assert "provider_config_or_fallback_failure" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_natural_language_status_uses_local_status_when_agent_provider_unavailable():
    from agent.provider_errors import HermesProviderConfigError

    runner = _make_runner()
    runner._agent_turn_config_error = HermesProviderConfigError(
        "primary provider config failed"
    )
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("status text leaked to agent")
    )

    result = await runner._handle_message(_make_event("status"))

    assert "**Session ID:** `sess-1`" in result
    assert "**Agent Running:** No" in result
    assert "provider_config_or_fallback_failure" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_natural_language_help_is_local_when_agent_provider_unavailable():
    from agent.provider_errors import HermesProviderConfigError

    runner = _make_runner()
    runner._agent_turn_config_error = HermesProviderConfigError(
        "primary provider config failed"
    )
    runner._handle_help_command = AsyncMock(return_value="local safe help")
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("help text leaked to agent")
    )

    result = await runner._handle_message(_make_event("help"))

    assert result == "local safe help"
    runner._handle_help_command.assert_awaited_once()
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "hermes gateway restart",
        "hermes gateway start",
        "hermes gateway stop",
    ],
)
async def test_discord_gateway_process_text_is_advisory_not_agent_mutation(text):
    runner = _make_runner()
    runner._agent_turn_config_error = None
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("unsupported gateway mutation leaked to agent")
    )

    result = await runner._handle_message(_make_event(text))

    assert "Gateway process control is not available from natural-language chat" in result
    assert "status" in result.lower()
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_discord_gateway_restart_text_is_advisory_while_agent_running():
    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)
    running_agent = SimpleNamespace(interrupt=MagicMock())
    runner._running_agents[session_key] = running_agent
    runner._running_agents_ts[session_key] = 1.0
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("unsupported gateway mutation leaked to agent")
    )

    result = await runner._handle_message(
        MessageEvent(text="hermes gateway restart", source=source, message_id="m1")
    )

    assert "Gateway process control is not available from natural-language chat" in result
    running_agent.interrupt.assert_not_called()
    runner._run_agent.assert_not_called()
