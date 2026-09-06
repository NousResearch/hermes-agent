"""Tests for the gateway-only per-session ``/timer`` command."""

from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.slash_commands import (
    GatewaySlashCommandsMixin,
    _format_session_timer,
    _parse_session_timer_minutes,
)


class _Event:
    def __init__(self, source: SessionSource, args: str) -> None:
        self.source = source
        self._args = args

    def get_command_args(self) -> str:
        return self._args


class _Runner(GatewaySlashCommandsMixin):
    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self.async_session_store = AsyncSessionStore(store)

    def _session_key_for_source(self, source: SessionSource) -> str:
        return self._store._generate_session_key(source)


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="chat", thread_id="topic", user_id="user")


def _store(path: Path) -> SessionStore:
    return SessionStore(path, GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none")))


def test_parse_and_format_timer_duration():
    assert _parse_session_timer_minutes("30m") == 30
    assert _parse_session_timer_minutes("2 hours") == 120
    assert _parse_session_timer_minutes("1d") == 1440
    assert _parse_session_timer_minutes("0m") is None
    assert _parse_session_timer_minutes("six hours") is None
    assert _parse_session_timer_minutes("2w") is None
    assert _parse_session_timer_minutes("9" * 5000 + "m") is None
    assert _format_session_timer(30) == "30m"
    assert _format_session_timer(90) == "1h 30m"
    assert _format_session_timer(1500) == "1d 1h"


@pytest.mark.asyncio
async def test_timer_set_status_off_and_invalid_argument(tmp_path):
    store = _store(tmp_path)
    runner = _Runner(store)
    source = _source()

    reply = await runner._handle_timer_command(_Event(source, "6h"))
    assert "6h" in reply
    session_key = store._generate_session_key(source)
    assert store.get_session_idle_reset(session_key) == 360

    reply = await runner._handle_timer_command(_Event(source, "status"))
    assert "6h" in reply

    reply = await runner._handle_timer_command(_Event(source, "nonsense"))
    assert reply.startswith("Usage:")

    reply = await runner._handle_timer_command(_Event(source, "off"))
    assert "cleared" in reply
    assert store.get_session_idle_reset(session_key) is None


@pytest.mark.asyncio
async def test_timer_persists_across_store_reload_and_does_not_survive_new(tmp_path):
    source = _source()
    first_store = _store(tmp_path)
    first_runner = _Runner(first_store)
    await first_runner._handle_timer_command(_Event(source, "90m"))
    session_key = first_store._generate_session_key(source)
    assert first_store.get_session_idle_reset(session_key) == 90

    reloaded_store = _store(tmp_path)
    reloaded_runner = _Runner(reloaded_store)
    reply = await reloaded_runner._handle_timer_command(_Event(source, "status"))
    assert "1h 30m" in reply

    reloaded_store.reset_session(session_key)
    reply = await reloaded_runner._handle_timer_command(_Event(source, "status"))
    assert reply.startswith("No per-session timer")
