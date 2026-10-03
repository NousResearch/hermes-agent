"""Keep-alive PTY sessions for dashboard terminals.

A PTY process outlives the WebSocket that created it: a single drain task always reads the PTY into
a bounded RingBuffer and forwards to the attached socket when present. Reconnecting with the same
opaque token replays the buffer and resumes live.
"""
from __future__ import annotations

import asyncio
import threading
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
            await asyncio.wait_for(ws.close(code=code), timeout=5.0)
    except Exception:
        pass


class PtySession:
    def __init__(self, key: str, bridge, *, buffer_cap: int, read_timeout: float) -> None:
        self.key = key
        self.bridge = bridge
        self.buffer = RingBuffer(buffer_cap)
        self.alive = True
        self.attached = False
        # Admission can fail before the first attach (e.g. lost metadata).
        self.last_detached_at: Optional[float] = time.monotonic()
        self._read_timeout = read_timeout
        self._ws = None
        self._attach_generation = 0
        # Only final sink writes and viewer claims hold this lock. Profile
        # admission/backpressure waits happen outside it.
        self._viewer_lock = threading.Lock()
        self._drain_task: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()
        self._output_lock = asyncio.Lock()
        self._input_task: Optional[asyncio.Task] = None
        self._close_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while self.alive:
            try:
                chunk = await loop.run_in_executor(None, self.bridge.read, self._read_timeout)
            except OSError:
                chunk = None
            if chunk is None:                       # EOF — the agent process exited
                self.alive = False
                ws = self._ws
                async with self._output_lock:
                    if self._ws is ws:
                        await _close_ws(ws, WS_CLOSE_PROCESS_EXITED)
                return
            if not chunk:                            # idle tick
                await asyncio.sleep(0)
                continue
            self.buffer.append(chunk)
            ws = self._ws
            output_lock = self._output_lock
            try:
                if ws is not None:
                    async with output_lock:
                        if self._ws is ws:
                            await asyncio.wait_for(ws.send_bytes(chunk), timeout=5.0)
            except Exception:
                # The viewer is gone; nothing else observes this failure (the handler's finally
                # only runs once ws.receive() sees the disconnect). detach() is a no-op when a
                # replacement socket attached during the send, so the new viewer keeps its session.
                if self._ws is ws:
                    self.detach(ws)
                    await _close_ws(ws, 1011)

    async def write(self, ws, data: bytes, *, fence=None) -> bool:
        """Serialize input and discard bytes from a superseded socket.

        ``fence`` is handed to the bridge (see ``PtyBridge.write``); what it raises propagates.
        """
        async with self._write_lock:
            if self._ws is not ws or not self.alive:
                return True
            generation = self._attach_generation

            def viewer_fence(write):
                def admitted_write():
                    with self._viewer_lock:
                        if self._ws is not ws or self._attach_generation != generation or not self.alive:
                            return None
                        return write()

                return fence(admitted_write)

            write = self.bridge.write(data) if fence is None else self.bridge.write(data, fence=viewer_fence)
            task = self._input_task = asyncio.create_task(write)
            try:
                delivered = await task
            except asyncio.CancelledError:
                if self._attach_generation != generation:
                    return False
                raise
            finally:
                if self._input_task is task:
                    self._input_task = None
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

    def resize(self, ws, *, cols: int, rows: int) -> None:
        if self._ws is ws and self.alive:
            self.bridge.resize(cols=cols, rows=rows)

    async def attach(self, ws, *, force_redraw: bool = False, initial_text: Optional[str] = None) -> bool:
        """Attach a browser terminal and replay buffered PTY output.

        The TUI renders differentially on an alternate screen, so a bounded ANSI tail is not a
        self-contained frame; ``force_redraw`` asks the live TUI for one full redraw after replay.
        """
        self._attach_generation += 1
        generation = self._attach_generation
        if self._input_task is not None:
            self._input_task.cancel()
        # Revoke queued workers first, then serialize the claim after any
        # already-admitted sink write. ConPTY writes can block: keep the loop
        # free to run their timeout/cancellation cleanup while we wait.
        while not self._viewer_lock.acquire(blocking=False):
            await asyncio.sleep(0.001)
            if self._attach_generation != generation:
                return False
        try:
            if self._attach_generation != generation:
                return False
            old_ws = self._ws
            self._ws = ws
            self.attached = True
            self.last_detached_at = None
            # Per-viewer lock: a stalled old send must not block its replacement.
            output_lock = self._output_lock = asyncio.Lock()
        finally:
            self._viewer_lock.release()
        try:
            async with output_lock:
                snap = self.buffer.snapshot()
                if old_ws is not ws:
                    await _close_ws(old_ws, WS_CLOSE_SUPERSEDED)
                if self._ws is not ws:
                    return False
                if initial_text is not None:
                    await asyncio.wait_for(ws.send_text(initial_text), timeout=5.0)
                if self._ws is not ws:
                    return False
                # Metadata-bearing host clients treat exactly the next binary
                # frame as replay. Send an empty snapshot too, so the first live
                # frame never loses terminal-query replies on an empty history.
                if snap or initial_text is not None:
                    await asyncio.wait_for(ws.send_bytes(snap), timeout=5.0)
        except asyncio.CancelledError:
            self.detach(ws)
            raise
        except Exception:
            self.detach(ws)
            return False
        if self._ws is not ws:
            return False
        if not self.alive:
            self.detach(ws)
            await _close_ws(ws, WS_CLOSE_PROCESS_EXITED)
            return False
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
        # Disconnect cancellation must not strand a child after its registry
        # entry was removed. Cleanup has independent, idempotent ownership.
        if self._close_task is None:
            self.alive = False
            self._attach_generation += 1
            if self._input_task is not None:
                self._input_task.cancel()
            ws = self._ws
            self.detach(ws)
            self._close_task = asyncio.create_task(self._close(ws))
        await asyncio.shield(self._close_task)

    async def _close(self, ws) -> None:
        await _close_ws(ws, WS_CLOSE_PROCESS_EXITED)
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
    """Every keep-alive slot holds a PTY that some tab is still attached to."""

    def __init__(self, message: str = "Too many chat terminals are open in other tabs; close one and try again.") -> None:
        super().__init__(message)


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
        # One registry-wide reservation spans lookup, spawn, and registration:
        # racing connections with one attach token must share one tracked PTY.
        self._spawn_lock = asyncio.Lock()
        self._spawn_cleanup: set[asyncio.Task] = set()
        self._closed = False
        # Retain ownership of removed sessions until helper-process shutdown completes.
        self._background_closes: set[asyncio.Task] = set()

    async def attach_or_spawn(self, key: str, *, spawn: Callable[[], object]) -> Tuple[PtySession, bool]:
        # Reserve capacity and the key across blocking fork/exec. On cancellation
        # the registry, not the request's cancellation scope, owns that reservation
        # until the admission finishes and any unclaimed child has been closed.
        await self._spawn_lock.acquire()
        admission = asyncio.create_task(self._attach_or_spawn(key, spawn=spawn))
        try:
            # wait() leaves admission running on caller cancellation without
            # shield() reporting its late exception (Python 3.14). The registry's
            # discard task owns that result once the caller disconnects.
            await asyncio.wait({admission})
            result = admission.result()
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._discard_admission(admission))
            self._spawn_cleanup.add(cleanup)
            cleanup.add_done_callback(self._spawn_cleanup.discard)
            raise
        except BaseException:
            self._spawn_lock.release()
            raise
        self._spawn_lock.release()
        return result

    async def _discard_admission(self, admission: asyncio.Task) -> None:
        try:
            try:
                session, created = await admission
            except Exception:
                # The disconnected caller cannot observe a late fork/exec
                # failure, but its task exception must still be retrieved.
                return
            if created:
                if self._sessions.get(session.key) is session:
                    self._sessions.pop(session.key)
                await session.close()
        finally:
            self._spawn_lock.release()

    async def _attach_or_spawn(self, key: str, *, spawn: Callable[[], object]) -> Tuple[PtySession, bool]:
        if self._closed:
            raise RegistryFull("Terminal service is shutting down.")
        await self.reap_idle()
        existing = self._sessions.get(key)
        if existing is not None and existing.alive:
            return existing, False
        if existing is not None:                       # dead remnant
            self._sessions.pop(key, None)
            self._close_in_background(existing)
        if len(self._sessions) >= self._max:
            self._reap_one_idle_or_raise()
        # PTY spawn does blocking fork/exec work — keep it off the event loop.
        # See #53227.
        bridge = await asyncio.to_thread(spawn)
        session = PtySession(key, bridge, buffer_cap=self._buffer_cap, read_timeout=self._read_timeout)
        await session.start()
        self._sessions[key] = session
        return session, True

    async def close_other_sessions(self, prefix: str, *, keep_key: str) -> None:
        """Close sessions belonging to the same logical client except ``keep_key``.

        Dashboard profile changes keep the browser's attach token but change the
        canonical session key. The previous profile's detached PTY must not
        remain alive long enough to hold the TUI session lease and reject a
        later return to that chat.
        """
        async with self._spawn_lock:
            keys = [
                key for key in self._sessions
                if key != keep_key and (key == prefix or key.startswith(prefix + "\0"))
            ]
            for key in keys:
                session = self._sessions.pop(key, None)
                if session is not None:
                    # A sibling tab sharing the attach token may still be viewing this
                    # PTY: supersede it explicitly (4409) instead of leaving it silent
                    # until its next keystroke fails with 1013. Detach first so close()
                    # does not overwrite that code with WS_CLOSE_PROCESS_EXITED.
                    ws = session._ws
                    session.detach(ws)
                    await _close_ws(ws, WS_CLOSE_SUPERSEDED)
                    await session.close()

    def detach(self, key: str, ws) -> None:
        s = self._sessions.get(key)
        if s is not None:
            s.detach(ws)

    async def reap_idle(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        doomed = [
            (key, s) for key, s in self._sessions.items()
            if not s.alive or not s.bridge.is_alive()
            or (not s.attached and s.last_detached_at is not None and (now - s.last_detached_at) > self._ttl)
        ]
        for key, expected in doomed:
            # Reaps overlap (attach_or_spawn and the background reaper) and close()
            # awaits, so a concurrent reap can have popped this key already — skip
            # it instead of raising KeyError into the websocket handler.
            if self._sessions.get(key) is not expected:
                continue
            if expected.alive and expected.bridge.is_alive() and (expected.attached or expected.last_detached_at is None
                                   or now - expected.last_detached_at <= self._ttl):
                continue
            session = self._sessions.pop(key, None)
            if session is not None:
                await session.close()

    def _reap_one_idle_or_raise(self) -> None:
        idle = [s for s in self._sessions.values() if not s.attached and s.last_detached_at is not None]
        if not idle:
            raise RegistryFull()
        oldest = min(idle, key=lambda s: s.last_detached_at or 0.0)
        self._sessions.pop(oldest.key, None)
        self._close_in_background(oldest)

    def _close_in_background(self, session: "PtySession") -> None:
        task = asyncio.create_task(session.close())
        self._background_closes.add(task)
        task.add_done_callback(self._background_closes.discard)

    async def close_all(self) -> None:
        self._closed = True
        # Wait for any admitted spawn before snapshotting shutdown ownership.
        async with self._spawn_lock:
            pass
        if self._spawn_cleanup:
            await asyncio.shield(asyncio.gather(*self._spawn_cleanup))
        # Helper-process grace must run concurrently inside the backend shutdown budget.
        sessions = [self._sessions.pop(key) for key in list(self._sessions)]
        await asyncio.gather(*(s.close() for s in sessions), *self._background_closes, return_exceptions=True)
