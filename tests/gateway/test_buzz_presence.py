"""Buzz presence: the adapter heartbeats an ephemeral kind-20001 event on its WebSocket.

Buzz clients colour an identity's presence dot from kind-20001 events it signs itself (content
``online`` / ``away`` / ``offline``, no tags). The relay keeps a status for 180s and only takes the kind
over WebSocket, so the adapter announces ``online`` on every connection, repeats it well inside that
window, and says ``offline`` on a graceful disconnect.
"""

import asyncio
import contextlib
import hashlib
import json
import logging

import pytest

from tests.gateway.test_buzz_typing import (
    CHANNEL, TEST_PRIVATE_KEY, _bip340_verify, _LiveSocket, _make_adapter,
)
from tests.waiting import async_wait_until

PRESENCE_TTL = 180.0  # block/buzz crates/buzz-pubsub/src/presence.rs PRESENCE_TTL_SECS


class _Relay(_LiveSocket):
    """A relay connection that stays open until ``close`` is set; ``refuse`` / ``hang`` name the presence
    statuses whose send fails or never returns."""

    def __init__(self, *, refuse=(), hang=()):
        super().__init__(self._until_closed)
        self.close, self.closed = asyncio.Event(), False
        self._refuse, self._hang = refuse, hang
        self.refused, self.hung, self.sent_after_close = [], [], []

    async def _until_closed(self):
        await self.close.wait()
        raise StopAsyncIteration

    async def send(self, raw):
        frame = json.loads(raw)
        status = frame[1]["content"] if frame[0] == "EVENT" and frame[1]["kind"] == 20001 else None
        if status in self._refuse:
            self.refused.append(status)
            raise ConnectionError("socket closed")
        if status in self._hang:
            self.hung.append(status)
            await asyncio.Event().wait()
        (self.sent_after_close if self.closed else self.sent).append(frame)

    async def __aexit__(self, *exc_info):
        self.closed = True


def _presence(relay):
    return [frame[1] for frame in relay.sent if frame[0] == "EVENT" and frame[1]["kind"] == 20001]


async def _parked(*args):
    await asyncio.Event().wait()


@contextlib.asynccontextmanager
async def _connected(monkeypatch, *relays):
    """An adapter whose WebSocket loop connects to *relays* in turn, then to idle ones."""
    import websockets

    adapter = _make_adapter()
    adapter._channel_state = {CHANNEL: {"last_ts": 1, "seen": {}}}
    adapter._ws_discovery_loop = _parked  # the discovery sweep shells out to the buzz CLI
    remaining = list(relays)
    monkeypatch.setattr(websockets, "connect", lambda *a, **k: remaining.pop(0) if remaining else _Relay())
    task = adapter._ws_task = asyncio.create_task(adapter._websocket_loop())
    try:
        yield adapter
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _fast_forward(monkeypatch):
    """Skip every sleep of a second or more (heartbeat interval, reconnect backoff); returns the skipped delays."""
    real_sleep, skipped = asyncio.sleep, []

    async def sleep(delay, result=None):
        if delay < 1:
            return await real_sleep(delay, result)
        skipped.append(delay)
        return await real_sleep(0, result)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return skipped


@pytest.mark.asyncio
async def test_connecting_announces_online_with_a_signed_kind_20001_event(monkeypatch):
    relay = _Relay()
    async with _connected(monkeypatch, relay) as adapter:
        event = (await async_wait_until(lambda: _presence(relay), timeout=5, what="a presence event"))[0]

    assert event["content"] == "online"
    assert event["tags"] == []
    assert event["pubkey"] == adapter._self_pubkey
    serialized = json.dumps(
        [0, event["pubkey"], event["created_at"], 20001, [], "online"], separators=(",", ":"), ensure_ascii=False
    ).encode()
    assert event["id"] == hashlib.sha256(serialized).hexdigest()
    assert _bip340_verify(adapter._self_pubkey, bytes.fromhex(event["id"]), bytes.fromhex(event["sig"]))


@pytest.mark.asyncio
async def test_heartbeat_repeats_online_inside_the_relay_ttl(monkeypatch):
    skipped = _fast_forward(monkeypatch)
    relay = _Relay()
    async with _connected(monkeypatch, relay):
        await async_wait_until(lambda: len(_presence(relay)) >= 3, timeout=5, what="three heartbeats")
        waits = list(skipped)

    assert {event["content"] for event in _presence(relay)} == {"online"}
    # Half the TTL at most, so one lost beat never lets the status expire.
    assert waits and all(0 < wait <= PRESENCE_TTL / 2 for wait in waits), waits


@pytest.mark.asyncio
async def test_a_reconnect_announces_online_again(monkeypatch):
    _fast_forward(monkeypatch)
    first, second = _Relay(), _Relay()
    async with _connected(monkeypatch, first, second):
        await async_wait_until(lambda: _presence(first), timeout=5, what="online on the first connection")
        first.close.set()
        await async_wait_until(lambda: _presence(second), timeout=5, what="online on the reconnected socket")

    assert _presence(second)[0]["content"] == "online"


@pytest.mark.asyncio
async def test_a_failing_heartbeat_keeps_trying_and_keeps_the_connection(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _fast_forward(monkeypatch)
    relay, spare = _Relay(refuse=("online",)), _Relay()
    async with _connected(monkeypatch, relay, spare) as adapter:
        await async_wait_until(lambda: len(relay.refused) >= 2, timeout=5, what="a second heartbeat attempt")

        assert adapter._ws is relay and not relay.closed, "a failed heartbeat tore the connection down"
        assert spare.sent == [], "a failed heartbeat forced a reconnect"
    assert TEST_PRIVATE_KEY not in caplog.text


@pytest.mark.asyncio
async def test_disconnect_says_offline_before_the_socket_closes(monkeypatch):
    relay = _Relay()
    async with _connected(monkeypatch, relay) as adapter:
        await async_wait_until(lambda: _presence(relay), timeout=5, what="online")
        await adapter.disconnect()

    last = _presence(relay)[-1]
    assert last["content"] == "offline", "disconnect() left the agent showing online until the TTL expires"
    assert last["tags"] == []
    assert _bip340_verify(adapter._self_pubkey, bytes.fromhex(last["id"]), bytes.fromhex(last["sig"]))
    assert relay.closed and relay.sent_after_close == []


@pytest.mark.asyncio
async def test_disconnect_does_not_wait_on_a_relay_that_never_takes_the_goodbye(monkeypatch):
    relay = _Relay(hang=("offline",))
    async with _connected(monkeypatch, relay) as adapter:
        await async_wait_until(lambda: _presence(relay), timeout=5, what="online")
        await asyncio.wait_for(adapter.disconnect(), 10.0)

    assert relay.hung == ["offline"], "disconnect() never tried to say offline"
    assert relay.closed


@pytest.mark.asyncio
async def test_offline_failures_never_raise_or_log_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    await _make_adapter(socket=None).disconnect()  # poll transport: no socket to say it on

    dead = _Relay(refuse=("offline",))
    await _make_adapter(socket=dead).disconnect()
    assert dead.refused == ["offline"]

    unsignable = "zz" * 32
    relay = _Relay()
    adapter = _make_adapter(socket=relay)
    adapter._private_key = unsignable
    await adapter.disconnect()

    assert relay.sent == []
    assert TEST_PRIVATE_KEY not in caplog.text and unsignable not in caplog.text
