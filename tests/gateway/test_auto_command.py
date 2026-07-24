"""Tests for gateway /auto session scoping (Auto Mode).

Mirrors test_yolo_command.py's structure. Also covers the on/off/status
subcommand parsing (matching /footer's convention) that /yolo's bare-toggle
doesn't need — see gateway/slash_commands.py's _handle_auto_command.
"""

import os

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from tools.approval import disable_session_auto, is_session_auto_enabled


@pytest.fixture(autouse=True)
def _clean_auto_state():
    disable_session_auto("agent:main:telegram:dm:chat-a")
    disable_session_auto("agent:main:telegram:dm:chat-b")
    yield
    disable_session_auto("agent:main:telegram:dm:chat-a")
    disable_session_auto("agent:main:telegram:dm:chat-b")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _make_event(chat_id: str, text: str = "/auto") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


@pytest.mark.asyncio
async def test_auto_command_bare_toggle_scoped_to_current_session(monkeypatch):
    runner = _make_runner()

    event_a = _make_event("chat-a")
    session_a = runner._session_key_for_source(event_a.source)
    session_b = runner._session_key_for_source(_make_event("chat-b").source)

    result_on = await runner._handle_auto_command(event_a)

    assert "ON" in result_on
    assert is_session_auto_enabled(session_a) is True
    assert is_session_auto_enabled(session_b) is False

    result_off = await runner._handle_auto_command(event_a)

    assert "OFF" in result_off
    assert is_session_auto_enabled(session_a) is False


@pytest.mark.asyncio
async def test_auto_on_subcommand_enables_idempotently():
    runner = _make_runner()
    event = _make_event("chat-a", text="/auto on")
    session = runner._session_key_for_source(event.source)

    result = await runner._handle_auto_command(event)
    assert "ON" in result
    assert is_session_auto_enabled(session) is True

    # Calling again with "on" must stay ON, not toggle back off.
    result2 = await runner._handle_auto_command(event)
    assert "ON" in result2
    assert is_session_auto_enabled(session) is True


@pytest.mark.asyncio
async def test_auto_off_subcommand_disables_idempotently():
    runner = _make_runner()
    on_event = _make_event("chat-a", text="/auto on")
    off_event = _make_event("chat-a", text="/auto off")
    session = runner._session_key_for_source(on_event.source)

    await runner._handle_auto_command(on_event)
    assert is_session_auto_enabled(session) is True

    result = await runner._handle_auto_command(off_event)
    assert "OFF" in result
    assert is_session_auto_enabled(session) is False

    result2 = await runner._handle_auto_command(off_event)
    assert "OFF" in result2
    assert is_session_auto_enabled(session) is False


@pytest.mark.asyncio
async def test_auto_status_reports_without_changing_state():
    runner = _make_runner()
    status_event = _make_event("chat-a", text="/auto status")
    session = runner._session_key_for_source(status_event.source)

    result = await runner._handle_auto_command(status_event)
    assert "OFF" in result
    assert is_session_auto_enabled(session) is False

    on_event = _make_event("chat-a", text="/auto on")
    await runner._handle_auto_command(on_event)

    result2 = await runner._handle_auto_command(status_event)
    assert "ON" in result2
    assert is_session_auto_enabled(session) is True  # unchanged by status


@pytest.mark.asyncio
async def test_auto_unknown_argument_returns_usage_without_changing_state():
    runner = _make_runner()
    event = _make_event("chat-a", text="/auto banana")
    session = runner._session_key_for_source(event.source)

    result = await runner._handle_auto_command(event)
    assert "Usage" in result
    assert is_session_auto_enabled(session) is False


@pytest.mark.asyncio
async def test_auto_command_does_not_touch_yolo_state():
    from tools.approval import is_session_yolo_enabled

    runner = _make_runner()
    event = _make_event("chat-a", text="/auto on")
    session = runner._session_key_for_source(event.source)

    await runner._handle_auto_command(event)

    assert is_session_auto_enabled(session) is True
    assert is_session_yolo_enabled(session) is False
