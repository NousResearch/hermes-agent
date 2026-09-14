"""Tests for gateway.chat_mute — the /mute per-chat silence store and gate.

The mute is a harness-level state command (Poke/Devin-Slack-etiquette
inspired): while a chat is muted the gateway drops conversational messages
deterministically, and slash commands (notably /unmute) always pierce it.
"""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.chat_mute as chat_mute
from gateway.chat_mute import (
    clear_chat_mute,
    format_remaining,
    get_mute_entry,
    is_chat_muted,
    parse_mute_duration,
    set_chat_mute,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the mute store at a temp HERMES_HOME so tests never touch real state."""
    monkeypatch.setattr(chat_mute, "get_hermes_home", lambda: tmp_path)
    yield tmp_path


# ---------------------------------------------------------------------------
# Store semantics
# ---------------------------------------------------------------------------


def test_mute_and_unmute_roundtrip():
    assert not is_chat_muted("telegram", "c1")
    set_chat_mute("telegram", "c1")
    assert is_chat_muted("telegram", "c1")
    # Chat-scoped: a different chat on the same platform is unaffected.
    assert not is_chat_muted("telegram", "c2")
    # Platform-scoped: same chat id on another platform is unaffected.
    assert not is_chat_muted("discord", "c1")
    assert clear_chat_mute("telegram", "c1") is True
    assert not is_chat_muted("telegram", "c1")
    # Clearing an unmuted chat reports False.
    assert clear_chat_mute("telegram", "c1") is False


def test_timed_mute_expires(monkeypatch):
    set_chat_mute("telegram", "c1", duration_seconds=600)
    assert is_chat_muted("telegram", "c1")
    entry = get_mute_entry("telegram", "c1")
    assert entry is not None and entry["expires_at"] is not None
    # Jump past expiry.
    future = entry["expires_at"] + 1
    monkeypatch.setattr(chat_mute.time, "time", lambda: future)
    assert not is_chat_muted("telegram", "c1")
    assert get_mute_entry("telegram", "c1") is None


def test_mute_survives_restart_via_file(tmp_path):
    """The mute persists on disk, so a gateway restart keeps it."""
    set_chat_mute("telegram", "c1")
    store_file = tmp_path / ".chat_mutes.json"
    assert store_file.exists()
    data = json.loads(store_file.read_text())
    assert "telegram:c1" in data
    # Fresh read (no in-memory state to begin with) still sees the mute.
    assert is_chat_muted("telegram", "c1")


def test_corrupt_store_fails_open(tmp_path):
    (tmp_path / ".chat_mutes.json").write_text("{not json", encoding="utf-8")
    assert not is_chat_muted("telegram", "c1")
    # A write recovers the store.
    set_chat_mute("telegram", "c1")
    assert is_chat_muted("telegram", "c1")


def test_unparsable_expiry_degrades_to_indefinite(tmp_path):
    (tmp_path / ".chat_mutes.json").write_text(
        json.dumps({"telegram:c1": {"muted_at": 0, "expires_at": "garbage"}}),
        encoding="utf-8",
    )
    assert is_chat_muted("telegram", "c1")


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("", (True, None)),
        ("on", (True, None)),
        ("30", (True, 1800.0)),
        ("30m", (True, 1800.0)),
        ("45min", (True, 2700.0)),
        ("2h", (True, 7200.0)),
        ("2 hours", (True, 7200.0)),
        ("1d", (True, 86400.0)),
        ("3 days", (True, 259200.0)),
        ("0", (False, None)),
        ("garbage", (False, None)),
        ("-5m", (False, None)),
    ],
)
def test_parse_mute_duration(arg, expected):
    assert parse_mute_duration(arg) == expected


def test_format_remaining_buckets():
    import time as _time

    now = _time.time()
    assert format_remaining({"expires_at": None}) == ""
    assert format_remaining({"expires_at": now + 90}, now=now) == "1m"
    assert format_remaining({"expires_at": now + 3 * 3600 + 600}, now=now) == "3h 10m"
    assert format_remaining({"expires_at": now + 86400 + 3600}, now=now) == "1d 1h"


# ---------------------------------------------------------------------------
# Gateway gate: muted chats drop conversational messages; commands pierce.
# Mirrors the bare-runner construction pattern of test_footer_command_mid_run.
# ---------------------------------------------------------------------------

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


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
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
    return runner, adapter


@pytest.mark.asyncio
async def test_muted_chat_drops_conversational_message():
    """A muted chat's plain message is dropped: None response, no agent run."""
    runner, _adapter = _make_runner(_session_entry())
    set_chat_mute("telegram", "c1")

    agent_run = AsyncMock()
    runner._handle_message_with_agent = agent_run

    result = await runner._handle_message(_make_event("hello there"))

    assert result is None
    agent_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_unmute_command_pierces_the_mute():
    """/unmute reaches its handler even while the chat is muted."""
    runner, _adapter = _make_runner(_session_entry())
    set_chat_mute("telegram", "c1")

    handler = AsyncMock(return_value="🔊 Chat unmuted — I'm listening again.")
    runner._handle_mute_command = handler
    agent_run = AsyncMock()
    runner._handle_message_with_agent = agent_run

    result = await runner._handle_message(_make_event("/unmute"))

    handler.assert_awaited_once()
    assert result == "🔊 Chat unmuted — I'm listening again."
    agent_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_mute_dispatches_mid_run():
    """/mute reaches its handler while an agent is running (busy dispatch)."""
    runner, _adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())
    runner._running_agents = {sk: MagicMock()}

    handler = AsyncMock(return_value="🔇 Muted.")
    runner._handle_mute_command = handler

    result = await runner._handle_message(_make_event("/mute"))

    handler.assert_awaited_once()
    assert result == "🔇 Muted."


@pytest.mark.asyncio
async def test_unmuted_chat_message_flows_to_agent():
    """Sanity: without a mute, conversational messages reach the agent path."""
    runner, _adapter = _make_runner(_session_entry())

    agent_run = AsyncMock(return_value="hi!")
    runner._handle_message_with_agent = agent_run
    # The claim/lease plumbing below the gate needs these:
    runner._claim_active_session_slot = lambda *_a, **_k: (None, None)
    runner._session_state = MagicMock()
    runner._persist_active_agents = lambda: None
    runner._begin_session_run_generation = lambda _k: 1
    runner._run_post_turn_hooks = AsyncMock()
    runner._restore_moa_one_shot = lambda *_a, **_k: None
    runner._restore_pending_one_turn_model_override = lambda *_a, **_k: None
    runner._clear_durable_active_turn = AsyncMock()
    runner._release_running_agent_state = lambda *_a, **_k: None
    runner._release_turn_lease = lambda *_a, **_k: None
    runner._external_drain_active = False
    runner._is_telegram_topic_root_lobby = lambda _s: False

    result = await runner._handle_message(_make_event("hello there"))

    agent_run.assert_awaited_once()
    assert result == "hi!"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
