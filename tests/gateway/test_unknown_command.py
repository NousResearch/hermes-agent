"""Tests for gateway warning when an unrecognized /command is dispatched.

Without this warning, unknown slash commands get forwarded to the LLM as plain
text, which often leads to silent failure (e.g. the model inventing a bogus
delegate_task call instead of telling the user the command doesn't exist).
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
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
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_unknown_slash_command_returns_guidance(monkeypatch):
    """A genuinely unknown /foobar should return user-facing guidance, not
    silently drop through to the LLM."""
    import gateway.run as gateway_run

    runner = _make_runner()
    # If the LLM were called, this would fail: the guard must short-circuit
    # before _run_agent is invoked.
    runner._run_agent = AsyncMock(
        side_effect=AssertionError(
            "unknown slash command leaked through to the agent"
        )
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/definitely-not-a-command"))

    assert result is not None
    assert "Unknown command" in result
    assert "/definitely-not-a-command" in result
    assert "/commands" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_slash_command_underscored_form_also_guarded(monkeypatch):
    """Telegram may send /foo_bar — same guard must trigger for underscored
    commands that normalize to unknown hyphenated names."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError(
            "unknown slash command leaked through to the agent"
        )
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/made_up_thing"))

    assert result is not None
    assert "Unknown command" in result
    assert "/made_up_thing" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_known_slash_command_not_flagged_as_unknown(monkeypatch):
    """A real built-in like /status must NOT hit the unknown-command guard."""
    runner = _make_runner()
    # Make _handle_status_command exist via the normal path by running a real
    # dispatch. If the guard fires, the return string will mention "Unknown".
    runner._running_agents[build_session_key(_make_source())] = MagicMock()

    result = await runner._handle_message(_make_event("/status"))

    assert result is not None
    assert "Unknown command" not in result


@pytest.mark.asyncio
async def test_underscored_alias_for_hyphenated_builtin_not_flagged(monkeypatch):
    """Telegram autocomplete sends /reload_mcp for the /reload-mcp built-in.
    That must NOT be flagged as unknown."""
    import gateway.run as gateway_run

    runner = _make_runner()
    # Prevent real MCP work; we only care that the unknown guard doesn't fire.
    async def _noop_reload(*_a, **_kw):
        return "mcp reloaded"

    runner._handle_reload_mcp_command = _noop_reload  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/reload_mcp"))

    # Whatever /reload_mcp returns, it must not be the unknown-command guard.
    if result is not None:
        assert "Unknown command" not in result
