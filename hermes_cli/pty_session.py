"""Keep-alive PTY sessions for dashboard terminals.

A PTY process outlives the WebSocket that created it: a single drain task always reads the PTY into
a bounded RingBuffer and forwards to the attached socket when present. Reconnecting with the same
opaque token replays the buffer and resumes live.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from typing import Callable, Dict, Optional, Tuple

WS_CLOSE_PROCESS_EXITED = 4410
WS_CLOSE_SUPERSEDED = 4409
TUI_FORCE_REDRAW = b"\x0c"


class RingBuffer:
    """Keeps only the most recent ``capacity`` bytes appended to it.

    Tracks ``total_appended`` so a reconnecting client can request an incremental
    replay from its last-known byte offset instead of re-receiving the buffer.
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
        """Return bytes appended after ``offset``, or ``None`` if it is outside the window."""
        earliest = self.start_offset
        if offset < earliest or offset > self._total_appended:
            return None
        skip = offset - earliest
        return bytes(self._buf[skip:])

    @property
    def start_offset(self) -> int:
        """Absolute offset of the first byte still retained."""
        return self._total_appended - len(self._buf)

    @property
    def total_appended(self) -> int:
        return self._total_appended

    @property
    def truncated(self) -> bool:
        return self._truncated


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
        self._send_lock = asyncio.Lock()
        self.epoch = secrets.token_hex(16)

    async def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self.bridge.read, self._read_timeout)
            if chunk is None:
                async with self._send_lock:
                    self.alive = False
                    await _close_ws(self._ws, WS_CLOSE_PROCESS_EXITED)
                return
            if not chunk:
                await asyncio.sleep(0)
                continue
            async with self._send_lock:
                self.buffer.append(chunk)
                ws = self._ws
                if ws is not None:
                    try:
                        await ws.send_bytes(chunk)
                    except Exception:
                        pass

    async def write(self, ws, data: bytes) -> bool:
        """Serialize input and discard bytes from a superseded socket."""
        async with self._write_lock:
            if self._ws is not ws:
                return True
            generation = self._attach_generation
            delivered = await self.bridge.write(data)
            if (
                not delivered
                and self._ws is ws
                and self._attach_generation == generation
            ):
                self.alive = False
            return delivered

    async def attach(
        self,
        ws,
        client_offset: Optional[int] = None,
        client_epoch: Optional[str] = None,
        *,
        force_redraw: bool = False,
    ) -> bool:
        """Attach ``ws`` and replay from a byte cursor when it is safe.

        A text control frame always precedes binary PTY bytes. ``reset`` tells
        the browser to discard its old terminal state before applying the
        retained snapshot.
        """
        async with self._send_lock:
            old = self._ws
            self._ws = ws
            self._attach_generation += 1
            self.attached = True
            self.last_detached_at = None
            if old is not None and old is not ws:
                await _close_ws(old, WS_CLOSE_SUPERSEDED)

            reset = True
            reason = "initial"
            start_offset = self.buffer.start_offset
            replay = self.buffer.snapshot()

            if client_offset is not None or client_epoch is not None:
                if client_epoch != self.epoch:
                    reason = "epoch_mismatch"
                elif client_offset is None:
                    reason = "invalid_cursor"
                else:
                    incremental = self.buffer.snapshot_from(client_offset)
                    if incremental is not None:
                        reset = False
                        reason = "resume"
                        start_offset = client_offset
                        replay = incremental
                    elif client_offset > self.buffer.total_appended:
                        reason = "offset_ahead"
                    else:
                        reason = "offset_rolled_out"

            cutover_offset = start_offset + len(replay)
            control = json.dumps(
                {
                    "type": "pty.replay",
                    "epoch": self.epoch,
                    "start_offset": start_offset,
                    "replay_end_offset": cutover_offset,
                    "reset": reset,
                    "reason": reason,
                },
                separators=(",", ":"),
            )
            try:
                await ws.send_text(control)
                if replay:
                    await self._send_chunked(ws, replay)
            except BaseException:
                self.detach(ws)
                raise
            if force_redraw and reset:
                return await self.write(ws, TUI_FORCE_REDRAW)
            return True

    async def _send_chunked(self, ws, data: bytes, chunk_size: int = 16384) -> None:
        """Send ``data`` in ``chunk_size`` frames, yielding between frames."""
        for i in range(0, len(data), chunk_size):
            await ws.send_bytes(data[i:i + chunk_size])
            if i + chunk_size < len(data):
                await asyncio.sleep(0)

    def detach(self, ws) -> None:
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
        if existing is not None:
            await existing.close()
            self._sessions.pop(key, None)
        if len(self._sessions) >= self._max:
            self._reap_one_idle_or_raise()
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
