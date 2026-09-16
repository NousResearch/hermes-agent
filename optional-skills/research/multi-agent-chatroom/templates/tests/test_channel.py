# tests/test_channel.py
import pytest
from server.channel import ChannelManager


@pytest.mark.asyncio
async def test_create_channel():
    mgr = ChannelManager()
    await mgr.create_channel("test")
    assert "test" in mgr._subscribers


@pytest.mark.asyncio
async def test_subscribe_and_publish():
    mgr = ChannelManager()
    await mgr.create_channel("test")
    received = []

    async def handler(msg):
        received.append(msg)

    await mgr.subscribe("test", handler)
    await mgr.publish("test", {"type": "message", "content": "hello"})
    await mgr.drain()

    assert len(received) == 1
    assert received[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_multiple_subscribers():
    mgr = ChannelManager()
    await mgr.create_channel("room")
    results = []

    for i in range(3):
        async def handler(msg, i=i):
            results.append((i, msg))
        await mgr.subscribe("room", handler)

    await mgr.publish("room", {"content": "broadcast"})
    await mgr.drain()
    assert len(results) == 3


@pytest.mark.asyncio
async def test_unsubscribe():
    mgr = ChannelManager()
    await mgr.create_channel("room")
    received = []

    async def handler(msg):
        received.append(msg)

    await mgr.subscribe("room", handler)
    await mgr.unsubscribe("room", handler)
    await mgr.publish("room", {"content": "should not arrive"})
    await mgr.drain()
    assert len(received) == 0


@pytest.mark.asyncio
async def test_get_history():
    mgr = ChannelManager()
    await mgr.create_channel("room")
    await mgr.publish("room", {"content": "msg1"})
    await mgr.publish("room", {"content": "msg2"})
    await mgr.publish("room", {"content": "msg3"})

    history = await mgr.get_history("room")
    assert len(history) == 3

    limited = await mgr.get_history("room", limit=1)
    assert len(limited) == 1
    assert limited[0]["content"] == "msg3"
