"""Tests for gateway /reasoning command behavior."""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="123", chat_id="456"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    session_entry = MagicMock()
    session_entry.session_id = "session_123"
    session_entry.session_key = "agent:main:telegram:dm"
    session_entry.reasoning_mode = "off"

    store = MagicMock()
    store.get_or_create_session.return_value = session_entry
    runner.session_store = store
    return runner, store, session_entry


@pytest.mark.asyncio
async def test_reasoning_command_shows_current_mode():
    runner, _, _ = _make_runner()

    result = await runner._handle_reasoning_command(_make_event("/reasoning"))

    assert "Current reasoning mode" in result
    assert "disable reasoning" in result


@pytest.mark.asyncio
async def test_reasoning_command_persists_stream_mode():
    runner, store, session_entry = _make_runner()

    result = await runner._handle_reasoning_command(_make_event("/reasoning stream"))

    store.set_reasoning_mode.assert_called_once_with(session_entry.session_key, "stream")
    assert "enabled and will stream live" in result


@pytest.mark.asyncio
async def test_reasoning_command_persists_on_mode_as_hidden():
    runner, store, session_entry = _make_runner()

    result = await runner._handle_reasoning_command(_make_event("/reasoning on"))

    store.set_reasoning_mode.assert_called_once_with(session_entry.session_key, "on")
    assert result == "Reasoning enabled and hidden for this chat."


def test_resolve_session_reasoning_config_disables_when_off():
    runner, _, _ = _make_runner()
    runner._reasoning_config = {"enabled": True, "effort": "high"}

    resolved = runner._resolve_session_reasoning_config("off")

    assert resolved == {"enabled": False}


def test_resolve_session_reasoning_config_enables_when_globally_disabled():
    runner, _, _ = _make_runner()
    runner._reasoning_config = {"enabled": False}

    resolved = runner._resolve_session_reasoning_config("on")

    assert resolved == {"enabled": True, "effort": "medium"}
