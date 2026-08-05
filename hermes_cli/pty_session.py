"""Keep-alive PTY sessions for dashboard terminals.

A PTY process outlives the WebSocket that created it: a single drain task
always reads the PTY into a bounded RingBuffer and forwards to the attached
socket when present. Reconnecting with the same opaque token replays the
buffer and resumes live. See
docs/superpowers/specs/2026-06-20-pty-keepalive-reattach-design.md.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

WS_CLOSE_PROCESS_EXITED = 4410
WS_CLOSE_SUPERSEDED = 4409
TUI_FORCE_REDRAW = b"\x0c"


class RingBuffer:
    """Keeps only the most recent ``capacity`` bytes appended to it.

    Tracks ``total_appended`` — the total number of bytes ever appended — so a
    reconnecting client can request an incremental replay starting from its
    last-known byte offset instead of re-receiving the entire buffer.
    """

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._buf = bytearray()
        self._truncated = False
        self._total_appended = 0

    def append(self, data: bytes) -> None:
        self._buf.extend(data)
        self._total_appended += len(data)
        overflow = len(self._buf) - self._cap
        if overflow > 0:
            del self._buf[:overflow]
            self._truncated = True

    def snapshot(self) -> bytes:
        return bytes(self._buf)

    def snapshot_from(self, offset: int) -> bytes | None:
        """Return bytes appended after ``offset``, or ``None`` if ``offset``
        has been evicted (rolled out of the ring).

        ``offset`` is a value of ``total_appended`` the client saved before
        disconnecting. If ``offset`` is still within the window we still hold
        (i.e. ``total_appended - len(buffer) <= offset``), we can send just
        the tail. Otherwise the client's offset is stale and must fall back
        to a full ``snapshot()``.
        """
        if offset < 0:
            return None
        earliest = self._total_appended - len(self._buf)
        if offset < earliest:
            return None  # rolled out — caller should full-replay
        skip = offset - earliest  # bytes in buffer that precede the offset
        return bytes(self._buf[skip:])

    @property
    def total_appended(self) -> int:
        return self._total_appended

    @property
    def truncated(self) -> bool:
        return self._truncated


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
        self._drain_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self.bridge.read, self._read_timeout)
            if chunk is None:                       # EOF — the agent process exited
                self.alive = False
                ws = self._ws
                if ws is not None:
                    try:
                        await ws.close(code=WS_CLOSE_PROCESS_EXITED)
                    except Exception:
                        pass
                return
            if not chunk:                            # idle tick
                await asyncio.sleep(0)
                continue
            self.buffer.append(chunk)
            ws = self._ws
            if ws is not None:
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    pass                             # detached mid-send; keep buffering

    async def attach(self, ws, client_offset: Optional[int] = None, *, force_redraw: bool = False) -> None:
        """Attach a browser terminal and replay buffered PTY output.

        The TUI uses an alternate screen and differential rendering, so a
        bounded ANSI tail is not guaranteed to be a self-contained frame.
        Reattaching a fresh xterm therefore asks the live TUI to emit one
        complete redraw after the replay.
        """
        old = self._ws
        if old is not None and old is not ws:
            try:
                await old.close(code=WS_CLOSE_SUPERSEDED)
            except Exception:
                pass
        self._ws = ws
        self.attached = True
        self.last_detached_at = None
        # Incremental replay: if the client sends its last-known byte offset
        # and that offset is still within our RingBuffer window, send only the
        # tail. This avoids re-replaying up to 1 MB of history the xterm.js
        # terminal already has painted (the dashboard keeps ChatPage mounted
        # persistently, so the terminal buffer survives tab switches — the
        # bytes are never lost on the client side).
        if client_offset is not None:
            incremental = self.buffer.snapshot_from(client_offset)
            if incremental is not None:
                # Offset still in window — send only the delta (if any).
                if incremental:
                    await self._send_chunked(ws, incremental)
                # Zero delta: the client's offset already sits at the tail.
                # Under the connection-lifecycle architecture (suspend on
                # hide / resume on show), a reconnect always carries a prior
                # offset and the client does NOT arm the hydration overlay
                # for incremental replays — so no sentinel frame is needed
                # here. Just return; live output resumes on the next frame.
                return
        # Full replay (first connect, or offset rolled out of the ring).
        # NOTE: for a reconnecting client whose offset rolled out of the
        # window, we deliberately do NOT fall back to a full 1MB replay: the
        # dashboard keeps ChatPage mounted, so xterm.js still holds the full
        # history the client painted before — a full replay would re-send
        # megabytes of bytes the client already has and stall rendering
        # ("tab-return black screen" on long-running sessions that exceed
        # the ring window while backgrounded). Emit the no-op SGR-reset
        # frame to clear the hydration gate and let live output resume.
        snap = self.buffer.snapshot()
        if client_offset is None and snap:
            await self._send_chunked(ws, snap)
            if force_redraw:
                self.bridge.write(TUI_FORCE_REDRAW)
        else:
            # First attach to an empty buffer, or reconnect with a
            # rolled-out offset: the client's resume-hydration gate is
            # edge-triggered on the first payload, so emit one no-op frame
            # or the "loading…" overlay stays up (black screen). SGR reset
            # is invisible on xterm.
            try:
                await ws.send_bytes(b"\x1b[0m")
            except Exception:
                pass

    async def _send_chunked(self, ws, data: bytes, chunk_size: int = 16384) -> None:
        """Send ``data`` in ``chunk_size`` frames, yielding between frames so a
        large replay (up to 1 MB) doesn't block the event loop."""
        for i in range(0, len(data), chunk_size):
            try:
                await ws.send_bytes(data[i:i + chunk_size])
            except Exception:
                return
            if i + chunk_size < len(data):
                await asyncio.sleep(0)

    def detach(self, ws) -> None:
        # Only the currently-attached socket may mark the session detached.
        # A superseded socket's handler also calls detach on its way out
        # (its ``finally`` runs after the new tab attached); flipping
        # ``attached`` then would make a session with a live viewer look
        # idle and reapable.
        if self._ws is not ws:
            return
        self._ws = None
        self.attached = False
        self.last_detached_at = time.monotonic()

    async def close(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            # bridge.close() joins the child — blocking; keep it off the
            # event loop (#53227).
            await asyncio.to_thread(self.bridge.close)
        except Exception:
            pass


from typing import Callable, Dict, Tuple


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
    def __init__(self, *, ttl: float, max_sessions: int,
                 buffer_cap: int, read_timeout: float) -> None:
        self._ttl = ttl
        self._max = max_sessions
        self._buffer_cap = buffer_cap
        self._read_timeout = read_timeout
        self._sessions: Dict[str, PtySession] = {}

    async def attach_or_spawn(self, key: str, *, spawn: Callable[[], object]
                              ) -> Tuple[PtySession, bool]:
        await self.reap_idle()
        existing = self._sessions.get(key)
        if existing is not None and existing.alive:
            return existing, False
        if existing is not None:                       # dead remnant
            await existing.close()
            self._sessions.pop(key, None)
        if len(self._sessions) >= self._max:
            self._reap_one_idle_or_raise()
        # PTY spawn does blocking fork/exec work — keep it off the event
        # loop (#53227).
        bridge = await asyncio.to_thread(spawn)
        session = PtySession(key, bridge, buffer_cap=self._buffer_cap,
                             read_timeout=self._read_timeout)
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
            if (not s.alive)
            or (not s.attached and s.last_detached_at is not None
                and (now - s.last_detached_at) > self._ttl)
        ]
        for key in doomed:
            await self._sessions.pop(key).close()

    def _reap_one_idle_or_raise(self) -> None:
        idle = [s for s in self._sessions.values()
                if not s.attached and s.last_detached_at is not None]
        if not idle:
            raise RegistryFull()
        oldest = min(idle, key=lambda s: s.last_detached_at or 0.0)
        self._sessions.pop(oldest.key, None)
        asyncio.create_task(oldest.close())

    async def close_all(self) -> None:
        for key in list(self._sessions):
            await self._sessions.pop(key).close()
