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


def _process_ancestors(pid: int) -> set[int]:
    """Pids of ``pid``'s live ancestors; empty when the process is gone or unreadable."""
    import psutil  # type: ignore

    try:
        return {parent.pid for parent in psutil.Process(pid).parents()}
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ValueError):
        return set()


def _resume_target(key: str) -> str:
    """The resume target a keep-alive key was registered under.

    Registry keys read ``token\\0profile\\0resume``; a key without a resume
    target — a chat that was never resumed from — yields ``""``.
    """
    parts = key.split("\0")
    return parts[2] if len(parts) >= 3 else ""


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
            ws = self._ws
            try:
                if ws is not None:
                    await ws.send_bytes(chunk)
            except Exception:
                # The viewer is gone; nothing else observes this failure (the handler's finally
                # only runs once ws.receive() sees the disconnect). detach() is a no-op when a
                # replacement socket attached during the send, so the new viewer keeps its session.
                self.detach(ws)

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
            try:
                await ws.send_bytes(snap)
            except Exception:
                # Client dropped mid-replay; the caller never reaches its writer loop, so undo the
                # attach here or reap_idle() can never reclaim this PTY (#110849).
                self.detach(ws)
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

    def hosts_pid(self, pid: Optional[int]) -> bool:
        """Whether ``pid`` runs inside this PTY: the leader itself or one of its descendants.

        The dashboard's PTY leader is ``node entry.js``; the process that takes the
        session lease is its ``tui_gateway`` child, so the lease pid is never the
        bridge's own pid and only ancestry identifies the terminal. A bridge that
        cannot report a pid (already reaped, or a stub) is simply not a match.
        """
        if pid is None:
            return False
        try:
            leader = int(self.bridge.pid)
        except (AttributeError, TypeError, ValueError, OSError):
            return False
        return leader == pid or leader in _process_ancestors(pid)

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
        # The get-or-spawn decision spans awaits (reap_idle, the spawn thread,
        # session.start), so two connections racing one attach token both saw
        # "no session" and forked a PTY each: the token then mapped to whichever
        # registered last while the other tab's live session fell out of the
        # registry — never reaped, and a reattach landed on the wrong terminal
        # (#115304). Serialize the decision so a token maps to one PTY.
        # ponytail: one registry-wide lock, not per key — argv resolution is
        # already serialized globally for the same reason, and a spawn only
        # delays NEW chats. Per-key locks if spawn throughput ever matters.
        self._attach_lock = asyncio.Lock()
        # Sessions popped from the registry but still closing in the background; close_all()
        # awaits them too, and holding the tasks keeps them from being garbage-collected.
        self._background_closes: set[asyncio.Task] = set()

    async def attach_or_spawn(self, key: str, *, spawn: Callable[[], object]) -> Tuple[PtySession, bool]:
        await self.reap_idle()
        async with self._attach_lock:
            existing = self._sessions.get(key)
            if existing is not None and existing.alive:
                return existing, False
            if existing is not None:                       # dead remnant
                # Close in the background: ending a dead leader's helpers can take the helper
                # grace, and this lock serializes every new chat.
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
        async with self._attach_lock:
            keys = [
                key for key in self._sessions
                if key != keep_key and (key == prefix or key.startswith(prefix + "\0"))
            ]
            for key in keys:
                session = self._sessions.pop(key, None)
                if session is not None:
                    # A sibling tab sharing the attach token may still be viewing this
                    # PTY: supersede it explicitly (4409) instead of leaving it silent
                    # until its next keystroke fails with 1013.
                    await _close_ws(session._ws, WS_CLOSE_SUPERSEDED)
                    await session.close()

    async def close_orphaned_sessions(
        self, resume: Optional[str], *, keep_key: str, holder_pid: Optional[int] = None,
    ) -> None:
        """Close a keep-alive PTY stranded under a superseded attach token.

        Rotating the token — the dashboard's *New chat* — moves the tab to a
        fresh PTY while the previous one keeps the TUI's single-writer lease on
        its session. That terminal is out of the user's reach, so the next
        return to the same chat is refused as a session held elsewhere. A PTY
        with a live viewer is never a candidate: somebody is still using it.

        A chat that was never resumed from carries no resume target in its key,
        so ``holder_pid`` — the process holding the lease for the session being
        resumed — identifies that terminal instead.
        """
        if not resume:
            return
        async with self._attach_lock:
            doomed = [
                key for key, session in self._sessions.items()
                if key != keep_key and not session.attached
                and (_resume_target(key) == resume or session.hosts_pid(holder_pid))
            ]
            sessions = [self._sessions.pop(key) for key in doomed]
        # Close outside the registry lock — a close can wait out its helpers' SIGHUP grace and
        # this lock serializes every new chat — but still before the caller spawns: the child's
        # session lease is only released once its close finishes.
        for session in sessions:
            await session.close()

    def detach(self, key: str, ws) -> None:
        s = self._sessions.get(key)
        if s is not None:
            s.detach(ws)

    async def reap_idle(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        doomed = [
            key for key, s in self._sessions.items()
            if not s.alive
            or (not s.attached and s.last_detached_at is not None and (now - s.last_detached_at) > self._ttl)
            # EOF never arrives if a helper still holds the PTY slave after the child died (#76759);
            # ask the process itself (a WNOHANG waitpid).
            or not s.bridge.is_alive()
        ]
        for key in doomed:
            # Reaps overlap (attach_or_spawn and the background reaper) and close()
            # awaits, so a concurrent reap can have popped this key already — skip
            # it instead of raising KeyError into the websocket handler.
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
        # Close concurrently: each close() may wait out its helpers' SIGHUP grace, and shutdown runs
        # under the backend's SIGTERM -> SIGKILL budget (dashboard_procs._POSIX_TERM_GRACE_SECONDS).
        sessions = [self._sessions.pop(key) for key in list(self._sessions)]
        await asyncio.gather(*(s.close() for s in sessions), *self._background_closes, return_exceptions=True)
