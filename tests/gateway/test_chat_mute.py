"""Tests for gateway.chat_mute — the /mute per-chat silence store and gate.

The mute is a harness-level state command: while a chat is muted the gateway drops
conversational messages deterministically (idle and mid-run), and slash commands
(notably the second /mute that unmutes) always pierce it.
"""
from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import chat_mute
from gateway.chat_mute import clear_chat_mute, is_muted, parse_mute_duration, set_chat_mute
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_mute, "get_process_hermes_home", lambda: tmp_path)
    yield tmp_path


def test_mute_interval_judges_arrival_expires_and_fails_open(tmp_path):
    set_chat_mute("telegram:c1", duration_seconds=600, at=1000.0)
    assert is_muted("telegram:c1", at=1000.0) and is_muted("telegram:c1", at=1599.0)
    assert not is_muted("telegram:c1", at=999.0)  # sent before /mute arrived
    assert not is_muted("telegram:c1", at=1600.0)  # the timed mute ran out
    assert not is_muted("telegram:c2", at=1200.0) and not is_muted("work:telegram:c1", at=1200.0)
    # An unmute closes the interval at its arrival: a message sent while muted stays muted even
    # when a text batch hands it over later; a zero duration is an instant expiry, never indefinite.
    assert clear_chat_mute("telegram:c1", at=1100.0) is True
    assert is_muted("telegram:c1", at=1050.0) and not is_muted("telegram:c1", at=1100.0)
    assert clear_chat_mute("telegram:c1", at=1200.0) is False
    set_chat_mute("telegram:c1", duration_seconds=0, at=2000.0)
    assert not is_muted("telegram:c1", at=2000.0)
    # An unreadable store (bad JSON, bad UTF-8) reads as "nothing muted", never an exception.
    for junk in (b"{not json", b"\xff\xfe\x00garbage"):
        (tmp_path / ".chat_mutes.json").write_bytes(junk)
        assert not is_muted("telegram:c1")


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


def _source(bot=None) -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
                         user_name="tester", chat_type="dm", profile=bot)


def _event(text: str, bot=None, received_at=None) -> MessageEvent:
    event = MessageEvent(text=text, source=_source(bot), message_id="m1")
    if received_at is not None:
        event._received_at = received_at
    return event


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    # The receiving bot's profile (its own resolver is tested with voice mode): None = default bot.
    runner._adapter_profile_for_source = lambda source: source.profile
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
async def test_muted_chat_drops_text_by_arrival_while_commands_and_queued_turns_pierce():
    """/mute through the real handler silences a person's text by when it ARRIVED: text sent just
    before /mute still runs, text sent while muted stays dropped after the second /mute. /status
    still dispatches, a turn the gateway already accepted (a /queue) is never re-judged, and a
    second bot in the same chat keeps its own state."""
    runner = _runner()
    runner._handle_status_command = AsyncMock(return_value="status ok")
    early = _event("sent before the mute", received_at=time.time() - 5)  # still in a text batch

    assert "/mute" in await runner._handle_message(_event("/mute"))
    late = _event("hello there")
    assert await runner._handle_message(late) is None
    runner._handle_message_with_agent.assert_not_awaited()
    assert await runner._handle_message(_event("/status")) == "status ok"
    queued = runner._queued_turn_event(_event("/queue later"), "later", False)
    runner._enqueue_fifo(build_session_key(_source()), queued, runner.adapters[Platform.TELEGRAM])
    assert await runner._handle_message(queued) == "agent reply"
    assert await runner._handle_message(_event("hi other bot", bot="work")) == "agent reply"
    assert await runner._handle_message(early) == "agent reply"

    assert "listening" in await runner._handle_message(_event("/mute"))
    runner._handle_message_with_agent.reset_mock()
    assert await runner._handle_message(late) is None  # flushed after the unmute: still dropped
    assert await runner._handle_message(_event("hello again")) == "agent reply"


@pytest.mark.asyncio
async def test_muted_chat_text_never_reaches_a_running_turn_but_answers_its_approval():
    """Mid-run, a muted chat's text neither steers, interrupts nor queues behind the turn; a bare
    "yes" to the approval the running turn is blocked on still answers it."""
    runner = _runner()
    runner._is_user_authorized_for_source = lambda _s: True
    runner._admit_bot_message_for_source = lambda _s: True
    runner._admit_bot_chain = lambda _e: True
    runner._effective_busy_input_mode = MagicMock(side_effect=AssertionError("reached busy routing"))
    runner._queue_or_replace_pending_event = MagicMock()
    runner._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
    set_chat_mute("telegram:c1")
    key = build_session_key(_source())

    assert await runner._handle_active_session_busy_message(_event("stop talking"), key) is True
    runner._queue_or_replace_pending_event.assert_not_called()
    runner._route_plaintext_approval_while_busy.assert_awaited_once()
