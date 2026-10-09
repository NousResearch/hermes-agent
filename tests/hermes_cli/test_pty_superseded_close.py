"""Slow superseded PTY teardown cannot block unrelated clients or escape shutdown."""
import asyncio
import threading

import pytest

from hermes_cli.pty_session import WS_CLOSE_SUPERSEDED, PtySession, PtySessionRegistry
from tests.hermes_cli.test_pty_session import FakeBridge, FakeWS


class GatedCloseBridge(FakeBridge):
    def __init__(self):
        super().__init__([])
        self.entered = threading.Event()
        self.release = threading.Event()

    def close(self):
        self.entered.set()
        self.release.wait(30)
        super().close()


@pytest.mark.asyncio
@pytest.mark.parametrize("attached", [False, True])
async def test_superseded_close_allows_unrelated_attach_before_lease_release(attached):
    registry = PtySessionRegistry(ttl=60, max_sessions=4, buffer_cap=1024, read_timeout=0.01)
    old_bridge = GatedCloseBridge()
    old = PtySession("token\0alpha\0session-a", old_bridge, buffer_cap=1024, read_timeout=0.01)
    registry._sessions[old.key] = old
    old_ws = FakeWS()
    if attached:
        await old.attach(old_ws)
    closer = asyncio.create_task(registry.close_other_sessions("token", keep_key="token\0beta\0session-b"))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(old_bridge.entered.wait, 5), 6)
        assert not closer.done()
        other_bridge = FakeBridge([])
        other, created = await asyncio.wait_for(registry.attach_or_spawn(
            "other-token\0alpha\0other-session", spawn=lambda: other_bridge), 2)
        assert created and other.bridge is other_bridge
        assert not closer.done()
        assert old_ws.close_code == (WS_CLOSE_SUPERSEDED if attached else None)
    finally:
        old_bridge.release.set()
        await asyncio.wait_for(closer, 5)
        await registry.close_all()
    assert old_bridge.closed


@pytest.mark.asyncio
async def test_cancelled_takeover_keeps_pending_close_owned_until_shutdown(tmp_path):
    registry = PtySessionRegistry(ttl=60, max_sessions=4, buffer_cap=1024, read_timeout=0.01)
    marker = tmp_path / "active-session"
    marker.write_text("session-a", encoding="utf-8")
    bridge = GatedCloseBridge()
    old = PtySession("token\0alpha\0session-a", bridge, buffer_cap=1024,
                     read_timeout=0.01, active_session_file=marker)
    registry._sessions[old.key] = old
    closer = asyncio.create_task(registry.close_other_sessions("token", keep_key="token\0beta\0session-b"))
    shutdown = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(bridge.entered.wait, 5), 6)
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer
        shutdown = asyncio.create_task(registry.close_all())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not shutdown.done()
        assert marker.exists()
    finally:
        bridge.release.set()
        if not closer.done():
            await asyncio.wait_for(closer, 5)
        if shutdown is not None:
            await asyncio.wait_for(shutdown, 5)
        await registry.close_all()
    assert bridge.closed
    assert not marker.exists()
