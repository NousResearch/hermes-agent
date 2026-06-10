"""Tests for confirmation-safe gateway /call command."""

from __future__ import annotations

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
    adapter.send_slash_confirm = AsyncMock(return_value=None)
    runner.adapters = {Platform.TELEGRAM: adapter}

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
    runner.session_store.append_to_transcript = MagicMock()

    import itertools as _it

    runner._slash_confirm_counter = _it.count(1)
    runner._session_key_for_source = lambda src: build_session_key(src)
    runner._thread_metadata_for_source = lambda *a, **kw: None
    runner._reply_anchor_for_event = lambda _e: None
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]))
    return runner


def test_parse_call_args_accepts_task_flags():
    from gateway.run import GatewayRunner

    parsed = GatewayRunner._parse_call_command_args(
        '+15551234567 --task "Ask whether they carry lemon ejuice" '
        '--first-sentence "Hi, this is Hermes calling for Jesse." --max-duration 2'
    )

    assert parsed["phone_number"] == "+15551234567"
    assert parsed["task"] == "Ask whether they carry lemon ejuice"
    assert parsed["first_sentence"] == "Hi, this is Hermes calling for Jesse."
    assert parsed["max_duration"] == 2


def test_parse_call_args_rejects_non_e164_phone():
    from gateway.run import GatewayRunner

    parsed = GatewayRunner._parse_call_command_args("5551234567 --task hi")

    assert "error" in parsed
    assert "E.164" in parsed["error"]


@pytest.mark.asyncio
async def test_call_command_registers_one_time_confirm_without_calling():
    from tools import slash_confirm as slash_confirm

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    slash_confirm.clear(session_key)
    runner._execute_confirmed_vapi_call = AsyncMock(return_value="should not run")

    result = await runner._handle_call_command(
        _make_event('/call +15551234567 --task "Ask if they carry item X"')
    )

    runner._execute_confirmed_vapi_call.assert_not_awaited()
    assert "Confirm outbound Vapi call" in result
    runner.adapters[Platform.TELEGRAM].send_slash_confirm.assert_awaited_once()
    assert runner.adapters[Platform.TELEGRAM].send_slash_confirm.await_args.kwargs["allow_always"] is False
    assert slash_confirm.get_pending(session_key)["command"] == "call"
    slash_confirm.clear(session_key)


@pytest.mark.asyncio
async def test_call_confirm_cancel_does_not_call_vapi():
    from tools import slash_confirm as slash_confirm

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    slash_confirm.clear(session_key)
    runner._execute_confirmed_vapi_call = AsyncMock(return_value="should not run")

    await runner._handle_call_command(
        _make_event('/call +15551234567 --task "Ask if they carry item X"')
    )
    pending = slash_confirm.get_pending(session_key)
    assert pending is not None

    resolved = await slash_confirm.resolve(session_key, pending["confirm_id"], "cancel")

    runner._execute_confirmed_vapi_call.assert_not_awaited()
    assert "cancelled" in resolved.lower()


@pytest.mark.asyncio
async def test_call_confirm_always_is_rejected():
    from tools import slash_confirm as slash_confirm

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    slash_confirm.clear(session_key)
    runner._execute_confirmed_vapi_call = AsyncMock(return_value="should not run")

    await runner._handle_call_command(
        _make_event('/call +15551234567 --task "Ask if they carry item X"')
    )
    pending = slash_confirm.get_pending(session_key)
    assert pending is not None

    resolved = await slash_confirm.resolve(session_key, pending["confirm_id"], "always")

    runner._execute_confirmed_vapi_call.assert_not_awaited()
    assert "one-time approval" in resolved


@pytest.mark.asyncio
async def test_call_confirm_once_executes_vapi():
    from tools import slash_confirm as slash_confirm

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    slash_confirm.clear(session_key)
    runner._execute_confirmed_vapi_call = AsyncMock(return_value="queued")

    await runner._handle_call_command(
        _make_event('/call +15551234567 --task "Ask if they carry item X"')
    )
    pending = slash_confirm.get_pending(session_key)
    assert pending is not None

    resolved = await slash_confirm.resolve(session_key, pending["confirm_id"], "once")

    runner._execute_confirmed_vapi_call.assert_awaited_once()
    assert resolved == "queued"
