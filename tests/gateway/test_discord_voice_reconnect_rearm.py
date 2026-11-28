"""Voice capture recovery after a reconnect (#142093).

discord.py stops a voice connection's ``SocketReader`` when the voice socket
reconnects and never starts another one: a ``Thread`` cannot be restarted, and a
fresh reader is only built inside ``connect()``.  ``VoiceClient.is_connected()``
keeps returning True and the receiver's buffers simply stay empty, so the bot goes
deaf with nothing in the logs.

These tests cover the watchdog that notices the stopped reader and rebuilds the
connection, and the guard rails that keep a broken voice server from turning the
repair into a connect/disconnect storm.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import pytest

pytest.importorskip("discord")

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

GUILD = 4242


def _adapter() -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._voice_text_channels = {GUILD: 700}
    adapter._voice_sources = {GUILD: {"channel_id": 700, "label": "linked"}}
    return adapter


def _voice_client(alive: bool):
    return SimpleNamespace(
        is_connected=lambda: True,
        channel="voice-channel",
        _connection=SimpleNamespace(_socket_reader=SimpleNamespace(is_alive=lambda: alive)),
    )


def _receiver():
    return SimpleNamespace(_running=True, check_silence=list, discard_pending=lambda: None)


def test_reader_probe_reads_healthy_when_internals_are_missing():
    """A discord.py refactor must degrade to the old behaviour, not to a reconnect loop."""
    assert DiscordAdapter._voice_capture_dead(SimpleNamespace()) is False
    assert DiscordAdapter._voice_capture_dead(SimpleNamespace(_connection=SimpleNamespace())) is False
    assert DiscordAdapter._voice_capture_dead(SimpleNamespace(_connection=SimpleNamespace(_socket_reader=None))) is False


def test_reader_probe_reports_a_stopped_thread():
    assert DiscordAdapter._voice_capture_dead(_voice_client(alive=False)) is True
    assert DiscordAdapter._voice_capture_dead(_voice_client(alive=True)) is False


@pytest.mark.asyncio
async def test_stopped_reader_makes_the_listen_loop_hand_over_to_recovery():
    adapter = _adapter()
    receiver = _receiver()
    adapter._voice_receivers[GUILD] = receiver
    adapter._voice_clients[GUILD] = _voice_client(alive=False)
    adapter._recover_voice_connection = AsyncMock(return_value=None)

    await asyncio.wait_for(adapter._voice_listen_loop(GUILD), timeout=3)

    adapter._recover_voice_connection.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_reader_leaves_the_listen_loop_running():
    adapter = _adapter()
    adapter._voice_receivers[GUILD] = _receiver()
    adapter._voice_clients[GUILD] = _voice_client(alive=True)
    adapter._recover_voice_connection = AsyncMock(return_value=None)

    loop_task = asyncio.ensure_future(adapter._voice_listen_loop(GUILD))
    try:
        await asyncio.sleep(1.0)
        assert loop_task.done() is False
        adapter._recover_voice_connection.assert_not_awaited()
    finally:
        loop_task.cancel()
        await loop_task  # the loop absorbs CancelledError itself


@pytest.mark.asyncio
async def test_recovery_rejoins_with_the_binding_it_was_capturing_for():
    adapter = _adapter()
    adapter.leave_voice_channel = AsyncMock()
    adapter.join_voice_channel = AsyncMock(return_value=True)
    adapter._voice_listen_tasks[GUILD] = SimpleNamespace()

    with patch("plugins.platforms.discord.adapter.asyncio.sleep", new=AsyncMock()):
        await adapter._recover_voice_connection(GUILD, _voice_client(alive=False))

    adapter.leave_voice_channel.assert_awaited_once_with(GUILD)
    adapter.join_voice_channel.assert_awaited_once_with(
        "voice-channel",
        text_channel_id=700,
        source={"channel_id": 700, "label": "linked"},
    )


@pytest.mark.asyncio
async def test_recovery_is_not_started_twice_for_the_same_guild():
    adapter = _adapter()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_leave(guild_id):
        entered.set()
        await release.wait()

    adapter.leave_voice_channel = slow_leave
    adapter.join_voice_channel = AsyncMock(return_value=True)

    first = asyncio.ensure_future(adapter._recover_voice_connection(GUILD, _voice_client(alive=False)))
    await asyncio.wait_for(entered.wait(), timeout=2)
    with patch("plugins.platforms.discord.adapter.asyncio.sleep", new=AsyncMock()):
        await adapter._recover_voice_connection(GUILD, _voice_client(alive=False))
    release.set()
    await asyncio.wait_for(first, timeout=2)

    assert adapter.join_voice_channel.await_count == 1


@pytest.mark.asyncio
async def test_cooldown_holds_off_repeated_repairs_while_the_reader_stays_dead():
    adapter = _adapter()
    receiver = _receiver()
    adapter._voice_receivers[GUILD] = receiver
    adapter._voice_clients[GUILD] = _voice_client(alive=False)
    adapter._recover_voice_connection = AsyncMock(return_value=None)

    loop_task = asyncio.ensure_future(adapter._voice_listen_loop(GUILD))
    await asyncio.sleep(1.0)
    adapter._recover_voice_connection.assert_awaited_once()
    # The receiver never changed, so the loop keeps running and waits out the cooldown
    # instead of rebuilding the connection on every tick.
    assert loop_task.done() is False
    loop_task.cancel()
    await loop_task  # the loop absorbs CancelledError itself