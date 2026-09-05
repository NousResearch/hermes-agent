"""Keep-alive PTY sessions for dashboard terminals.

A PTY process outlives the WebSocket that created it: a single drain task always reads the PTY into
a bounded RingBuffer and forwards to the attached socket when present. Reconnecting with the same
opaque token replays the buffer and resumes live.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable, Dict, Optional, Tuple

WS_CLOSE_PROCESS_EXITED = 4410
WS_CLOSE_SUPERSEDED = 4409
TUI_FORCE_REDRAW = b"\x0c"


class RingBuffer:
    """Keeps only the most recent ``capacity`` bytes appended to it."""

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._buf = bytearray()
        self.truncated = False

    def append(self, data: bytes) -> None:
        self._buf.extend(data)
        overflow = len(self._buf) - self._cap
        if overflow > 0:
            del self._buf[:overflow]
            self.truncated = True

    def snapshot(self) -> bytes:
        return bytes(self._buf)


async def _close_ws(ws, code: int) -> None:
    try:
        if ws is not None:
            await ws.close(code=code)
    except Exception:
        pass


class PtySession:
    def __init__(self, key: str, bridge, *, buffer_cap: int, read_timeout: float) -> None:
        self.key = key
        self.bridge = bridge
        self.buffer = RingBuffer(buffer_cap)
        self.alive = True
        self.attached = False
        self.last_detached_at: Optional[float] = None
        self._read_timeout = read_timeout
        self._ws = None
        self._attach_generation = 0
        self._drain_task: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self.bridge.read, self._read_timeout)
            if chunk is None:                       # EOF — the agent process exited
                self.alive = False
                await _close_ws(self._ws, WS_CLOSE_PROCESS_EXITED)
                return
            if not chunk:                            # idle tick
                await asyncio.sleep(0)
                continue
            self.buffer.append(chunk)
            try:
                if self._ws is not None:
                    await self._ws.send_bytes(chunk)
            except Exception:
                pass                                 # detached mid-send; keep buffering

    async def write(self, ws, data: bytes) -> bool:
        """Serialize input and discard bytes from a superseded socket."""
        async with self._write_lock:
            if self._ws is not ws:
                return True
            generation = self._attach_generation
            delivered = await self.bridge.write(data)
            # A replacement socket can attach while the bridge write is
            # suspended on backpressure. A late failure from the superseded
            # socket must not poison the replacement's shared PTY session.
            if (
                not delivered
                and self._ws is ws
                and self._attach_generation == generation
            ):
                self.alive = False
            return delivered

    async def attach(self, ws, *, force_redraw: bool = False) -> bool:
        """Attach a browser terminal and replay buffered PTY output.

        The TUI renders differentially on an alternate screen, so a bounded ANSI tail is not a
        self-contained frame; ``force_redraw`` asks the live TUI for one full redraw after replay.
        """
        if self._ws is not ws:
            await _close_ws(self._ws, WS_CLOSE_SUPERSEDED)
        self._ws = ws
        self._attach_generation += 1
        self.attached = True
        self.last_detached_at = None
        if snap := self.buffer.snapshot():
            await ws.send_bytes(snap)
        if force_redraw:
            return await self.write(ws, TUI_FORCE_REDRAW)
        return True

    def detach(self, ws) -> None:
        # Only the currently-attached socket may mark the session detached: a superseded socket's
        # handler also calls detach on its way out (after the new tab attached), and flipping
        # ``attached`` then would make a session with a live viewer look idle and reapable.
        if self._ws is not ws:
            return
        self._ws = None
        self.attached = False
        self.last_detached_at = time.monotonic()

    async def close(self) -> None:
        self.alive = False
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            # bridge.close() joins the child — blocking; keep it off the event loop.
            # See #53227.
            await asyncio.to_thread(self.bridge.close)
        except Exception:
            pass


class RegistryFull(Exception):
    pass


async def run_reaper(registry: "PtySessionRegistry", *, interval: float = 60.0) -> None:
    """Periodically reap idle/dead keep-alive sessions. Cancelled on shutdown."""
    while True:
        await asyncio.sleep(interval)
        try:
            await registry.reap_idle()
        except Exception:
            pass


class PtySessionRegistry:
    def __init__(self, *, ttl: float, max_sessions: int, buffer_cap: int, read_timeout: float) -> None:
        self._ttl = ttl
        self._max = max_sessions
        self._buffer_cap = buffer_cap
        self._read_timeout = read_timeout
        self._sessions: Dict[str, PtySession] = {}

    async def attach_or_spawn(self, key: str, *, spawn: Callable[[], object]) -> Tuple[PtySession, bool]:
        await self.reap_idle()
        existing = self._sessions.get(key)
        if existing is not None and existing.alive:
            return existing, False
        if existing is not None:                       # dead remnant
            await existing.close()
            self._sessions.pop(key, None)
        if len(self._sessions) >= self._max:
            self._reap_one_idle_or_raise()
        # PTY spawn does blocking fork/exec work — keep it off the event loop.
        # See #53227.
        bridge = await asyncio.to_thread(spawn)
        session = PtySession(key, bridge, buffer_cap=self._buffer_cap, read_timeout=self._read_timeout)
        await session.start()
        self._sessions[key] = session
        return session, True

    def detach(self, key: str, ws) -> None:
        s = self._sessions.get(key)
        if s is not None:
            s.detach(ws)

    async def reap_idle(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        doomed = [
            key for key, s in self._sessions.items()
            if not s.alive or (not s.attached and s.last_detached_at is not None and (now - s.last_detached_at) > self._ttl)
        ]
        for key in doomed:
            await self._sessions.pop(key).close()

    def _reap_one_idle_or_raise(self) -> None:
        idle = [s for s in self._sessions.values() if not s.attached and s.last_detached_at is not None]
        if not idle:
            raise RegistryFull()
        oldest = min(idle, key=lambda s: s.last_detached_at or 0.0)
        self._sessions.pop(oldest.key, None)
        asyncio.create_task(oldest.close())

    async def close_all(self) -> None:
        for key in list(self._sessions):
            await self._sessions.pop(key).close()
