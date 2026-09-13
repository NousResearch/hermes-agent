"""A dropped viewer link (1006: lid closed, Wi-Fi) must NOT hand the screen back to the agent — the
human may be mid-login on it. Only a clean close (1000/1001) releases."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from hermes_cli.web_routers import display
from tools.bot_desktop import lease


class _Ws:
    """Just enough of a Starlette WebSocket: one disconnect message with the given close code."""

    def __init__(self, close_code: int):
        self._code = close_code
        self.accepted = False
        self.closed = False
        self.closed_code = None

    async def accept(self):
        self.accepted = True

    async def receive(self):
        await asyncio.sleep(0.05)
        return {"type": "websocket.disconnect", "code": self._code}

    async def send_bytes(self, data):
        pass

    async def close(self, code=1000, reason=""):
        if not self.closed:
            self.closed_code = code
        self.closed = True


async def _bridge_once(close_code: int, home: str) -> lease.Lease:
    sock_dir = os.path.join(home, "bot-desktop")
    os.makedirs(sock_dir, exist_ok=True)
    sock = os.path.join(sock_dir, "rfb.sock")

    async def _xvnc(reader, writer):  # a silent framebuffer
        await asyncio.sleep(1)
        writer.close()

    server = await asyncio.start_unix_server(_xvnc, path=sock)
    try:
        info = {"hermes_home": home, "viewer_id": "desk-1"}
        await display._bridge(_Ws(close_code), info)
    finally:
        server.close()
    return lease.get(profile_key=home)


# 1005 (no status code) is what noVNC's code-less socket.close() AND some proxies produce on a drop, so
# the server keeps the lease; the Desktop sends an explicit 1000 when the pane is closed on purpose.
@pytest.mark.parametrize(("close_code", "human_keeps_control"), [(1006, True), (1005, True), (1000, False)])
def test_only_a_clean_viewer_close_hands_the_screen_back(monkeypatch, close_code, human_keeps_control):
    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        lease.acquire("desk-1", profile_key=home)
        after = asyncio.run(_bridge_once(close_code, home))
    lease._reset_for_tests()
    assert (after.holder == lease.HUMAN) is human_keeps_control, after


def test_every_nonholder_is_evicted_during_human_takeover():
    assert display._should_evict(lease.Lease(holder=lease.HUMAN, viewer_id="desk-1"), "desk-1") is False
    assert display._should_evict(lease.Lease(holder=lease.AGENT), "desk-1") is False
    assert display._should_evict(lease.Lease(holder=lease.HUMAN, viewer_id="desk-2"), "desk-1") is True


def test_single_viewer_slots_are_scoped_per_bot_profile():
    evicted = []
    a1 = display._claim_active_viewer("bot-a", lambda: evicted.append("a1"))
    b1 = display._claim_active_viewer("bot-b", lambda: evicted.append("b1"))
    a2 = display._claim_active_viewer("bot-a", lambda: evicted.append("a2"))
    try:
        assert evicted == ["a1"]
    finally:
        display._release_active_viewer("bot-a", a1)
        display._release_active_viewer("bot-a", a2)
        display._release_active_viewer("bot-b", b1)


def test_authenticated_reconnect_during_takeover_receives_no_framebuffer():
    """A reusable login may mint another one-shot ticket; its non-holder viewer is still refused."""

    async def _run(home: str) -> _Ws:
        ws = _Ws(1000)
        await display._bridge(ws, {"hermes_home": home, "viewer_id": "desk-1"})
        return ws

    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        lease.acquire("desk-2", profile_key=home)
        ws = asyncio.run(_run(home))
    lease._reset_for_tests()
    assert ws.closed_code == display._CLOSE_CONTROL_TAKEN


class _OpenWs(_Ws):
    """Stays open until ``finish`` is set, then reports a clean close."""

    def __init__(self):
        super().__init__(1000)
        self.finish = asyncio.Event()

    async def receive(self):
        await self.finish.wait()
        return {"type": "websocket.disconnect", "code": self._code}


def test_new_viewer_replaces_the_existing_stream_for_one_profile():
    async def _run(home: str) -> tuple[_OpenWs, _OpenWs]:
        sock_dir = os.path.join(home, "bot-desktop")
        os.makedirs(sock_dir, exist_ok=True)

        async def _xvnc(reader, writer):
            await asyncio.sleep(2)
            writer.close()

        server = await asyncio.start_unix_server(_xvnc, path=os.path.join(sock_dir, "rfb.sock"))
        first, second = _OpenWs(), _OpenWs()
        first_task = asyncio.create_task(display._bridge(first, {"hermes_home": home, "viewer_id": "desk-1"}))
        second_task = None
        try:
            while not first.accepted:
                await asyncio.sleep(0.01)
            second_task = asyncio.create_task(display._bridge(second, {"hermes_home": home, "viewer_id": "desk-2"}))
            await asyncio.wait_for(first_task, timeout=2.0)
            assert first.closed_code == display._CLOSE_CONTROL_TAKEN
            return first, second
        finally:
            second.finish.set()
            if second_task is not None:
                await second_task
            if not first_task.done():
                first.finish.set()
                await first_task
            server.close()

    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        first, second = asyncio.run(_run(home))
    lease._reset_for_tests()
    assert first.closed and second.accepted


def test_a_takeover_made_by_another_process_evicts_within_the_refresh_interval(
    monkeypatch,
):
    """The bridge caches the input decision instead of reading lease.json per message; a takeover
    written by ANOTHER process (no in-process listener fires) must still be seen quickly."""
    from tools.bot_desktop import rfb_filter

    captured = {}

    class _Filter(rfb_filter.RfbClientFilter):
        def __init__(self, allow_input):
            super().__init__(allow_input)
            captured["allow"] = allow_input

    monkeypatch.setattr(rfb_filter, "RfbClientFilter", _Filter)

    async def _run(home: str) -> float:
        sock_dir = os.path.join(home, "bot-desktop")
        os.makedirs(sock_dir, exist_ok=True)

        async def _xvnc(reader, writer):
            await asyncio.sleep(2)
            writer.close()

        server = await asyncio.start_unix_server(_xvnc, path=os.path.join(sock_dir, "rfb.sock"))
        ws = _OpenWs()
        task = asyncio.create_task(display._bridge(ws, {"hermes_home": home, "viewer_id": "desk-1"}))
        try:
            while "allow" not in captured:
                await asyncio.sleep(0.01)
            assert captured["allow"]() is True
            # Another process takes over: the file changes, no listener in this process is told.
            lease._write(
                lease._path(home),
                lease.Lease(holder=lease.HUMAN, viewer_id="desk-2", epoch=2),
            )
            t0 = asyncio.get_running_loop().time()
            while captured["allow"]() and asyncio.get_running_loop().time() - t0 < 2.0:
                await asyncio.sleep(0.02)
            elapsed = asyncio.get_running_loop().time() - t0
            await asyncio.wait_for(task, timeout=2.0)
            assert ws.closed_code == display._CLOSE_CONTROL_TAKEN
            return elapsed
        finally:
            if not task.done():
                ws.finish.set()
                await task
            server.close()

    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        lease.acquire("desk-1", profile_key=home)
        elapsed = asyncio.run(_run(home))
    lease._reset_for_tests()
    assert elapsed < 0.5, elapsed
