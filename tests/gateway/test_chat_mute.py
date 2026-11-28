"""Tests for gateway.chat_mute — the /mute per-chat silence store and gate.

The mute is a harness-level state command: while a chat is muted the gateway drops
conversational messages deterministically (idle and mid-run), and slash commands
(notably the second /mute that unmutes) always pierce it.
"""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import chat_mute
from gateway.chat_mute import clear_chat_mute, is_chat_muted, parse_mute_duration, set_chat_mute
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_mute, "get_hermes_home", lambda: tmp_path)
    yield tmp_path


def test_mute_is_chat_scoped_expires_and_fails_open(tmp_path, monkeypatch):
    set_chat_mute("telegram", "c1", duration_seconds=600)
    assert is_chat_muted("telegram", "c1")
    assert not is_chat_muted("telegram", "c2")
    assert not is_chat_muted("discord", "c1")
    expires = json.loads((tmp_path / ".chat_mutes.json").read_text(encoding="utf-8"))["telegram:c1"]["expires_at"]
    real_time = chat_mute.time.time
    monkeypatch.setattr(chat_mute.time, "time", lambda: expires + 1)
    assert not is_chat_muted("telegram", "c1")
    monkeypatch.setattr(chat_mute.time, "time", real_time)
    # A zero duration is an instant expiry, never an indefinite mute.
    set_chat_mute("telegram", "c1", duration_seconds=0)
    assert not is_chat_muted("telegram", "c1")
    set_chat_mute("telegram", "c1")
    assert clear_chat_mute("telegram", "c1") is True
    assert not is_chat_muted("telegram", "c1")
    # A corrupt store reads as "nothing muted": a broken file never silences every chat.
    (tmp_path / ".chat_mutes.json").write_text("{not json", encoding="utf-8")
    assert not is_chat_muted("telegram", "c1")


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("", (True, None)),
        ("30", (True, 1800.0)),
        ("2h", (True, 7200.0)),
        ("3 days", (True, 259200.0)),
        ("0", (False, None)),
        ("garbage", (False, None)),
    ],
)
def test_parse_mute_duration(arg, expected):
    assert parse_mute_duration(arg) == expected


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
                         user_name="tester", chat_type="dm")


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="m1")


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=build_session_key(_source()), session_id="sess-1",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm", total_tokens=0,
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._is_user_authorized = lambda _source: True
    runner._handle_message_with_agent = AsyncMock(return_value="agent reply")
    return runner


@pytest.mark.asyncio
async def test_muted_chat_drops_text_while_commands_pierce():
    """/mute through the real handler silences plain text (no agent run); /status still
    dispatches, and a second /mute toggles the chat live again."""
    runner = _runner()
    runner._handle_status_command = AsyncMock(return_value="status ok")

    assert "/mute" in await runner._handle_message(_event("/mute"))
    assert await runner._handle_message(_event("hello there")) is None
    runner._handle_message_with_agent.assert_not_awaited()

    assert await runner._handle_message(_event("/status")) == "status ok"
    await runner._handle_message(_event("/mute"))
    assert not is_chat_muted("telegram", "c1")
    assert await runner._handle_message(_event("hello again")) == "agent reply"


@pytest.mark.asyncio
async def test_muted_chat_text_never_reaches_a_running_turn():
    """Mid-run, a muted chat's text neither steers, interrupts nor queues behind the turn."""
    runner = _runner()
    runner._is_user_authorized_for_source = lambda _s: True
    runner._admit_bot_message_for_source = lambda _s: True
    runner._admit_bot_chain = lambda _e: True
    runner._effective_busy_input_mode = MagicMock(side_effect=AssertionError("reached busy routing"))
    runner._queue_or_replace_pending_event = MagicMock()
    set_chat_mute("telegram", "c1")

    handled = await runner._handle_active_session_busy_message(_event("stop talking"), build_session_key(_source()))

    assert handled is True
    runner._queue_or_replace_pending_event.assert_not_called()
