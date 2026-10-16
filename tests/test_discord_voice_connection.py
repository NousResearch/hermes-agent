"""Exercise discord.py's real connection lifecycle, without Discord traffic."""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

# Gateway unit tests install a Discord stub. Run this dependency suite on its
# own to exercise the real package rather than treating that stub as evidence.
if "discord" in sys.modules and not hasattr(sys.modules["discord"], "__file__"):
    pytest.skip("gateway discord stub is loaded; run this file in its own process",
                allow_module_level=True)
pytest.importorskip("discord.voice_state")
from discord import ConnectionClosed
from plugins.platforms.discord.voice_connection import DiscordVoiceClient, _VoiceConnectionState
from discord.voice_state import ConnectionFlowState as Flow


@pytest_asyncio.fixture
async def state():
    client = SimpleNamespace(_connection=SimpleNamespace(
        loop=asyncio.get_running_loop(), _remove_voice_client=Mock()))
    guild = SimpleNamespace(id=1, change_voice_state=AsyncMock())
    channel = SimpleNamespace(id=10, guild=guild, _get_voice_client_key=lambda: (1, "guild_id"))
    vc = DiscordVoiceClient(client, channel)
    connection = vc._connection
    assert isinstance(connection, _VoiceConnectionState)
    connection._runner = asyncio.current_task()  # no background network poller
    try:
        yield connection
    finally:
        connection._socket_reader.stop()
        connection._socket_reader.join(timeout=1)
        assert not connection._socket_reader.is_alive()


async def connect_without_network(state, monkeypatch):
    async def voice_connect(**kwargs):
        state.state = Flow.got_voice_server_update
        await state.voice_state_update({'channel_id': '10', 'session_id': 'test'})
    monkeypatch.setattr(state, '_voice_connect', voice_connect)
    state.ip = '127.0.0.1'
    ws = SimpleNamespace(secret_key=b'x' * 32, close=AsyncMock())
    monkeypatch.setattr(state, '_connect_websocket', AsyncMock(return_value=ws))
    await state._connect(True, 1, False, False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize('code', [4014, 4022])
async def test_move_after_expected_disconnect_and_successful_reconnect(state, monkeypatch, code):
    state._expecting_disconnect = True
    await state.voice_state_update({'channel_id': None})
    assert state._disconnected.is_set()
    await connect_without_network(state, monkeypatch)
    state.ws.poll_event = AsyncMock(side_effect=ConnectionClosed(SimpleNamespace(close_code=code), code=code, shard_id=None))
    recovery = AsyncMock(return_value=False)
    disconnect = AsyncMock()
    monkeypatch.setattr(state, '_potential_reconnect', recovery)
    monkeypatch.setattr(state, 'disconnect', disconnect)
    await state._poll_voice_ws(True)
    recovery.assert_awaited_once()


@pytest.mark.asyncio
async def test_first_connection_and_keyword_connect(state, monkeypatch):
    await connect_without_network(state, monkeypatch)
    assert state.is_connected()
    assert not state._disconnected.is_set()
    state._disconnected.set()
    await state._connect(reconnect=True, timeout=1, self_deaf=False,
                         self_mute=False, resume=True)
    assert state.is_connected()
    assert not state._disconnected.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [RuntimeError, asyncio.CancelledError])
async def test_failed_or_cancelled_connect_preserves_acknowledgment(state, monkeypatch, error):
    state._disconnected.set()
    monkeypatch.setattr(state, '_inner_connect', AsyncMock(side_effect=error()))
    with pytest.raises(error):
        await state._connect(True, 1, False, False, False)
    assert state._disconnected.is_set()
    assert not state.is_connected()


@pytest.mark.asyncio
async def test_disconnect_before_connect_returns_is_not_cleared(state, monkeypatch):
    async def inner(**kwargs):
        # Represents a newer disconnect between handshake and wait_for return.
        await state.voice_state_update({'channel_id': None})
        state._expecting_disconnect = True
        await state.voice_state_update({'channel_id': None})
    monkeypatch.setattr(state, '_inner_connect', inner)
    await state._connect(True, 1, False, False, False)
    assert state._disconnected.is_set()
    assert not state.is_connected()


@pytest.mark.asyncio
@pytest.mark.parametrize('null_first', [True, False])
async def test_real_kick_terminates_in_both_event_orders(state, monkeypatch, null_first):
    await connect_without_network(state, monkeypatch)
    ws = state.ws
    closed = ConnectionClosed(Mock(close_code=4014), code=4014, shard_id=None)
    kick_tasks = []
    if null_first:
        await state.voice_state_update({'channel_id': None})
        # Discord acknowledges the native disconnect request.
        await state.voice_state_update({'channel_id': None})
        ws.poll_event = AsyncMock(side_effect=closed)
    else:
        async def poll():
            asyncio.get_running_loop().call_soon(
                lambda: kick_tasks.append(asyncio.create_task(
                    state.voice_state_update({'channel_id': None}))))
            raise closed
        ws.poll_event = poll
    await asyncio.wait_for(state._poll_voice_ws(True), timeout=1)
    await asyncio.gather(*kick_tasks)
    assert not state.is_connected()


@pytest.mark.asyncio
async def test_manual_leave_waits_for_acknowledgment(state, monkeypatch):
    await connect_without_network(state, monkeypatch)
    state._runner = None
    leave = asyncio.create_task(state.disconnect(wait=True))
    await asyncio.sleep(0)
    assert not leave.done()
    await state.voice_state_update({'channel_id': None})
    await asyncio.wait_for(leave, timeout=1)
    assert not state.is_connected()
    assert state._disconnected.is_set()


@pytest.mark.asyncio
async def test_shared_join_selects_real_client_factory(monkeypatch):
    from plugins.platforms.discord import adapter
    monkeypatch.setattr(adapter, 'DISCORD_AVAILABLE', True)
    class Selected(Exception):
        pass
    async def connect(*, cls):
        assert cls is DiscordVoiceClient
        raise Selected
    channel = SimpleNamespace(guild=SimpleNamespace(id=1), connect=connect)
    owner = object.__new__(adapter.DiscordAdapter)
    owner._client = Mock()
    owner._voice_locks = {}
    owner._voice_clients = {}
    with pytest.raises(Selected):
        await owner.join_voice_channel(channel)
