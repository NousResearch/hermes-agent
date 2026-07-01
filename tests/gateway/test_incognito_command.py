"""Tests for the /incognito gateway command."""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


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
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    return runner


class TestHandleIncognitoCommand:
    @pytest.mark.asyncio
    async def test_status_defaults_off(self):
        runner = _make_runner()

        result = await runner._handle_incognito_command(_make_event("/incognito status"))

        assert result == "Incognito mode: 🔓 OFF"

    @pytest.mark.asyncio
    async def test_on_sets_session_flag_and_cached_agent(self):
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)
        cached_agent = MagicMock()
        cached_agent.persist_session = True
        runner._agent_cache[session_key] = (cached_agent, object())

        result = await runner._handle_incognito_command(_make_event("/incognito on"))

        assert session_key in runner._incognito_sessions
        assert cached_agent.persist_session is False
        assert "Incognito ON" in result
        assert "NOT be persisted" in result

    @pytest.mark.asyncio
    async def test_off_clears_session_flag_and_restores_cached_agent(self):
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)
        cached_agent = MagicMock()
        cached_agent.persist_session = False
        runner._incognito_sessions = {session_key}
        runner._agent_cache[session_key] = (cached_agent, object())

        result = await runner._handle_incognito_command(_make_event("/incognito off"))

        assert session_key not in runner._incognito_sessions
        assert cached_agent.persist_session is True
        assert result == "🔓 Incognito OFF — persistence resumed for this chat."

    @pytest.mark.asyncio
    async def test_invalid_argument_shows_usage(self):
        runner = _make_runner()

        result = await runner._handle_incognito_command(_make_event("/incognito maybe"))

        assert result == "Usage: /incognito [on|off|status]\n(no argument toggles current state)"
