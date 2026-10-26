"""Process-lifetime host shells, owned by Webapp identity and profile home.

This is a separate capability namespace from Dashboard's client-selected chat IDs.
The existing PTY registry owns processes/drains; this module owns host admission.
A shell dies with its profile: once a named home is missing or deleted (main's
tombstone rule, ``hermes_constants.assert_named_profile_home_live``) its input is
refused and the reaper closes it.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import json
from pathlib import Path
import re
import secrets
import time

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from hermes_cli.pty_session import (
    WS_CLOSE_VIEWER_STALLED, WS_CLOSE_VIEWER_STALLED_REASON, PtySession, PtySessionRegistry,
    RegistryFull, _close_ws, run_reaper,
)
from hermes_cli.web_host_terminal import META_PREFIX
from hermes_constants import assert_named_profile_home_live, get_hermes_home

# No survival across backend restart. Disconnects retain a bounded ANSI tail.
RETENTION_SECONDS = 15 * 60
_MAX_SHELLS = 16
# Inbound frames buffered while one batch is written (backpressure bound).
_INPUT_BATCH_FRAMES = 64
# How long a resize that undoes the previous one waits for the program to answer that one (``_resize``).
_RESIZE_ANSWER_SECONDS = 1.0
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")


class TerminalDenied(Exception):
    pass


class TerminalExpired(Exception):
    pass


class TerminalExited(TerminalExpired):
    """The shell exited by itself (``exit``, Ctrl-D); the server did not end it."""


def _home_live(home: Path) -> bool:
    """A named home must exist and carry no deletion tombstone; any other home only has to exist."""
    try:
        assert_named_profile_home_live(home)
    except FileNotFoundError:
        return False
    return home.is_dir()


@dataclass(frozen=True)
class HostOwner:
    identity: tuple[str, str]
    home: Path
    shell: str
    cwd: str

    def current(self) -> bool:
        """Blocking filesystem reads (home, tombstone): call off-loop."""
        return _home_live(self.home)

    def admit(self, write):
        """Run ``write`` while this owner's profile home is live, or raise ``TerminalExpired``.

        Input sent once the profile is deleted never reaches the shell. Blocking: bridges call
        it off-loop.
        """
        if not self.current():
            raise TerminalExpired()
        return write()


def _request_identity(ws: WebSocket) -> tuple[str, str]:
    """``(provider, user)`` of the terminal ticket that admitted *ws*; ``_ws_gate`` admits no other
    credential to the host terminal."""
    identity = getattr(ws, "_hermes_auth_identity", None) or {}
    user, provider = identity.get("user_id"), identity.get("provider")
    if not isinstance(user, str) or not user or not isinstance(provider, str) or not provider:
        raise TerminalDenied()
    return provider, user


def _request_home(profile: str | None) -> Path:
    from hermes_cli.web_server_profiles import _resolve_profile_dir
    requested = (profile or "").strip()
    return (
        _resolve_profile_dir(requested)
        if requested and requested.lower() != "current" else get_hermes_home()
    ).resolve()


class HostTerminalRegistry(PtySessionRegistry):
    # The most exited shells remembered at once: a tab that redials after the reaper took its
    # shell still learns that it exited (``resolve``).
    _EXITS_KEPT = 4 * _MAX_SHELLS

    def __init__(self):
        # Every live shell keeps a read in flight. On the loop's default executor, retained
        # shells would queue the keystroke fences, ownership checks, uploads and SessionDB
        # hops that share it; a pool sized to capacity gives each shell its own reader.
        super().__init__(
            ttl=RETENTION_SECONDS, max_sessions=_MAX_SHELLS, buffer_cap=1024 * 1024, read_timeout=0.2,
            drain_executor=ThreadPoolExecutor(_MAX_SHELLS, thread_name_prefix="host-pty-drain"))
        self.owners: dict[str, HostOwner] = {}
        # Token -> (owner identity, when it was reaped) of shells that exited by themselves, for
        # the retention TTL.
        self.exits: dict[str, tuple[tuple[str, str], float]] = {}

    def _reap_one_idle_or_raise(self) -> None:
        # Retention is a promise, not an LRU hint; never evict a live shell to
        # make room for a different pane, even if its viewer just disconnected.
        raise RegistryFull()

    async def create(self, ws: WebSocket, identity: tuple[str, str]):
        from hermes_cli.web_host_terminal import query_dimension, resolve_argv
        from hermes_cli.web_server_chat import PtyBridge
        token = secrets.token_urlsafe(32)
        owner = None

        def spawn():
            nonlocal owner
            home = _request_home(ws.query_params.get("profile"))
            if not _home_live(home):
                raise TerminalExpired()
            argv, cwd, env, shell = resolve_argv(home=home, requested_cwd=ws.query_params.get("cwd"))
            bridge = PtyBridge.spawn(
                argv, cwd=cwd, env=env,
                cols=query_dimension(ws.query_params.get("cols"), 80, 2000),
                rows=query_dimension(ws.query_params.get("rows"), 24, 1000))
            owner = HostOwner(identity, home, shell, cwd)
            return bridge

        session, _ = await self.attach_or_spawn(token, spawn=spawn)
        assert owner is not None  # spawn completed
        self.owners[token] = owner
        if not await asyncio.to_thread(owner.current):
            await self.remove(token)
            raise TerminalExpired()
        return session, owner

    async def resolve(self, token: str, ws: WebSocket, identity: tuple[str, str], *, closing=False):
        session = self._sessions.get(token)
        owner = self.owners.get(token)
        if session is None or owner is None:
            gone = self.exits.get(token)
            if gone is not None and gone[0] == identity:
                raise TerminalExited()
            raise TerminalExpired()
        if owner.identity != identity:
            raise TerminalDenied()

        def validate():
            if _request_home(ws.query_params.get("profile")) != owner.home:
                raise TerminalDenied()
            owner.admit(lambda: None)

        await asyncio.to_thread(validate)
        # No await between the final check and the caller's attach claim. The
        # home was just checked off-loop; every input write checks it again.
        if self._sessions.get(token) is not session:
            raise TerminalExpired()
        if not closing and _exited(session):
            raise TerminalExited()
        if not closing and self.expired(session):
            raise TerminalExpired()
        return session, owner

    async def remove(self, token: str):
        session = self._sessions.pop(token, None)
        self.owners.pop(token, None)
        if session is not None:
            await asyncio.shield(self._close_in_background(session))

    async def reap_idle(self, now=None):
        # Durable tombstones also detect deletion by a different process; attached
        # viewers do not exempt a deleted profile.
        # One thread hop per tick keeps every shell's marker reads off the loop.
        owners = list(self.owners.items())
        retired = await asyncio.to_thread(lambda: {token for token, owner in owners if not owner.current()})
        now = time.monotonic() if now is None else now
        for token, owner in owners:
            session = self._sessions.get(token)
            if session is not None and (token in retired or self._reapable(session, now)):
                if token not in retired and _exited(session):
                    self._remember_exit(token, owner, now)
                await self.remove(token)
        for token in list(self.owners):
            if token not in self._sessions:
                self.owners.pop(token, None)
        for token in [token for token, (_, at) in self.exits.items() if now - at > self._ttl]:
            del self.exits[token]

    def _remember_exit(self, token: str, owner: HostOwner, now: float) -> None:
        self.exits[token] = (owner.identity, now)
        while len(self.exits) > self._EXITS_KEPT:
            del self.exits[next(iter(self.exits))]

    async def close_all(self):
        await super().close_all()
        self.owners.clear()
        self.exits.clear()
        # Every drain is cancelled by now; a read still in flight returns within its
        # timeout, so let it finish behind us. A closed registry never spawns again.
        self._drain_executor.shutdown(wait=False)


def _exited(session: PtySession) -> bool:
    """Whether a session still in the registry lost its shell: the server pops a session before
    it closes one, so only the shell's own exit leaves a dead one behind."""
    return not session.alive or not session.bridge.is_alive()


def get_host_terminals(app) -> HostTerminalRegistry:
    registry = getattr(app.state, "host_terminals", None)
    if registry is None or registry._closed:
        raise RuntimeError("Host terminal service is not running; start the application lifespan first.")
    return registry


@asynccontextmanager
async def host_terminal_lifespan(app):
    # Overlapping lifespans must not share loop-bound sessions or teardown.
    registry = HostTerminalRegistry()
    registries = getattr(app.state, "_host_terminal_registries", None)
    if registries is None:
        registries = app.state._host_terminal_registries = []
    registries.append(registry)
    app.state.host_terminals = registry
    reaper = asyncio.create_task(run_reaper(registry, interval=1.0))
    try:
        yield
    finally:
        # Withdraw ownership before awaiting cleanup so another teardown cannot
        # restore this registry while its reaper/sessions are shutting down.
        registries.remove(registry)
        if getattr(app.state, "host_terminals", None) is registry:
            if registries:
                app.state.host_terminals = registries[-1]
            else:
                del app.state.host_terminals
        if not registries:
            del app.state._host_terminal_registries
        reaper.cancel()
        with suppress(asyncio.CancelledError):
            await reaper
        await registry.close_all()


def _metadata(token, owner, registry, *, reconnected, truncated):
    return META_PREFIX + json.dumps({
        "shell": owner.shell, "terminalId": token, "cwd": owner.cwd,
        "reconnected": reconnected, "retentionSeconds": registry._ttl,
        "truncated": truncated,
    }, separators=(",", ":"))


async def _send_closed(ws: WebSocket, token: str) -> None:
    await ws.send_text(META_PREFIX + json.dumps({"terminalId": token, "closed": True}))
    await ws.close(code=1000)


async def _admit(ws: WebSocket, registry: HostTerminalRegistry):
    """Authorize the request and return ``(session, owner)`` to attach.

    Returns None once a ``close`` request has been served. Raises the ``Terminal*``
    errors ``host_terminal`` maps to close codes.
    """
    token = ws.query_params.get("attach")
    action = ws.query_params.get("action")
    identity = _request_identity(ws)
    if action not in (None, "close") or (action == "close" and token is None):
        raise TerminalDenied()
    if token is not None and not TOKEN_RE.fullmatch(token):
        raise TerminalExpired()
    if action == "close" and token not in registry._sessions:
        # Missing is idempotent, but never bypass auth/host policy.
        await _send_closed(ws, token)
        return None
    if token is None:
        return await registry.create(ws, identity)
    session, owner = await registry.resolve(token, ws, identity, closing=action == "close")
    if action == "close":
        await registry.remove(token)
        await _send_closed(ws, token)
        return None
    return session, owner


async def _read_frames(ws: WebSocket, inbox: asyncio.Queue, ended: asyncio.Future) -> None:
    """Queue inbound frames. The last item is None, or the error that ended reading
    (re-raised by ``_pump_input`` so the route's handlers still see it). ``ended`` gets the same
    item as soon as reading ends, before it is queued: a write waiting on the shell, or a full
    inbox behind it, must not keep the pump from seeing that the socket is gone."""
    end = None
    try:
        while (msg := await ws.receive()).get("type") != "websocket.disconnect":
            await inbox.put(msg)
    except RuntimeError:  # a superseding/drain task already closed us
        pass
    # health: allow BLE001 -- forwarded, not swallowed: _pump_input re-raises it, and a reader that died unqueued would hang the pump
    except Exception as exc:
        end = exc
    ended.set_result(end)
    await inbox.put(end)


async def _write_while_open(session: PtySession, ws: WebSocket, data: bytes, fence,
                            ended: asyncio.Future) -> bool | None:
    """``session.write`` of *data* with no deadline while the socket is open, or None once the
    socket has ended and the write has to wait for the program to read.

    The program may leave its input unread for as long as it likes, but not past the viewer. Input
    the tab sent before it closed still reaches the program as far as the program takes it; the
    first time the write then has to wait for room, it is cancelled (the rest is dropped, as on a
    takeover) and the session stays alive for the caller to detach.
    """
    loop = asyncio.get_running_loop()
    stalled = asyncio.Event()

    def watched(write):
        accepted = fence(write)
        # The POSIX bridge reports 0 bytes taken when the program's input queue is full; it then
        # waits for room. Runs in the bridge's worker thread.
        full = type(accepted) is int and accepted == 0
        loop.call_soon_threadsafe(stalled.set if full else stalled.clear)
        return accepted

    write = asyncio.ensure_future(session.write(ws, data, fence=watched, timeout=None))
    try:
        await asyncio.wait((write, ended), return_when=asyncio.FIRST_COMPLETED)
        if not write.done():
            stall = asyncio.ensure_future(stalled.wait())
            try:
                await asyncio.wait((write, stall), return_when=asyncio.FIRST_COMPLETED)
            finally:
                stall.cancel()
    finally:
        if not write.done():
            write.cancel()
            await asyncio.wait((write,))
    return None if write.cancelled() else write.result()


def _input_runs(frames: list[dict], resize_re):
    """``frames`` in order: consecutive keystrokes joined into one run, resizes as ``(cols, rows)``."""
    run = b""
    for msg in frames:
        raw = msg.get("bytes")
        if raw is None:
            raw = (msg.get("text") or "").encode("utf-8")
        match = resize_re.fullmatch(raw) if raw else None
        if match is None:
            run += raw
            continue
        if run:
            yield run
            run = b""
        yield int(match[1]), int(match[2])
    if run:
        yield run


async def _resize(session: PtySession, ws: WebSocket, size: tuple[int, int], undo):
    """Apply ``size``; return what the next resize needs to tell whether it undoes this one.

    A program can read the size well after its SIGWINCH (Node, behind its event loop, and reports a
    'resize', which Hermes' TUI redraws on, only when the size it reads has changed). A step undone
    before it looks is never seen: both signals find the size it already had. So a resize back to
    the size before the previous one first waits until the program has written output since that
    one, as a program that redraws on a resize does, or ``_RESIZE_ANSWER_SECONDS`` for one that
    does not. ``undo`` is ``(size before the previous resize, outputs when it was applied)``, set
    only when that resize changed the size.
    """
    if undo is not None and undo[0] == size:
        await session.wait_for_output(undo[1], _RESIZE_ANSWER_SECONDS)
    before = session.size()
    session.resize(ws, cols=size[0], rows=size[1])
    return (before, session.outputs) if before is not None and before != size else None


async def _pump_input(ws: WebSocket, registry: HostTerminalRegistry, token: str,
                      session: PtySession, owner: HostOwner) -> None:
    """Forward keystrokes and resizes while ``ws`` is the attached viewer.

    Keystrokes reach the shell only inside ``owner.admit``, so input sent after the
    profile is deleted never does, even when it queued behind input that was admitted
    first. The fence reads files, so it runs off the loop, once per run of keystrokes in
    a batch (the frames that queued while the previous batch was written): a paste of
    many frames costs a few fences, not one each.

    Writes have no deadline. While the foreground program is not reading (``sleep``,
    ``npm install``) the PTY takes about 1 KB of input on macOS, so a paste waits for
    that program, in order, with any Ctrl-C behind it, as in native Desktop's terminal.
    A re-attach or the socket closing cancels the wait and keeps the shell (``host_terminal``
    then detaches it, so the retention reaper owns it again); only a closed bridge (the shell
    exited or the terminal was closed) or a deleted profile drops the session. ConPTY
    buffers input itself, so Windows writes land at once and keep the bridge's deadline.
    """
    from hermes_cli.web_server_chat import _RESIZE_RE
    inbox: asyncio.Queue = asyncio.Queue(maxsize=_INPUT_BATCH_FRAMES)
    ended = asyncio.get_running_loop().create_future()
    reader = asyncio.create_task(_read_frames(ws, inbox, ended))
    undo = None
    try:
        # No Ctrl-L/redraw input: arbitrary foreground shell programs own stdin.
        while session._ws is ws and session.alive:
            batch = [await inbox.get()]
            while not inbox.empty():
                batch.append(inbox.get_nowait())
            frames = [msg for msg in batch if isinstance(msg, dict)]
            for item in _input_runs(frames, _RESIZE_RE):
                if isinstance(item, tuple):
                    undo = await _resize(session, ws, item, undo)
                    continue
                try:
                    delivered = await _write_while_open(session, ws, item, owner.admit, ended)
                except (TerminalExpired, FileNotFoundError):
                    await registry.remove(token)
                    return
                if delivered is None:
                    # The socket ended and this input was left waiting for the program: stop here,
                    # so the caller detaches a shell that is still running.
                    if isinstance(ended.result(), Exception):
                        raise ended.result()
                    return
                if not delivered:
                    # A closed bridge marked the session dead. A takeover only cancelled this
                    # write: drop the rest of the batch and leave this socket for the new viewer
                    # to supersede, so this tab does not reconnect and take the shell back.
                    if not session.alive:
                        await registry.remove(token)
                        return
                    break
            if len(frames) < len(batch):
                if isinstance(batch[-1], Exception):
                    raise batch[-1]
                return
    finally:
        reader.cancel()
        with suppress(asyncio.CancelledError):
            await reader


async def host_terminal(ws: WebSocket) -> None:
    """Called only after both the existing WS gate and host policy have passed."""
    from hermes_cli.pty_session import WS_CLOSE_PROCESS_EXITED, WS_CLOSE_PROCESS_EXITED_REASON
    from hermes_cli.web_server_chat import _pty_fail
    session = None
    try:
        registry = get_host_terminals(ws.app)
        admitted = await _admit(ws, registry)
        if admitted is None:
            return
        session, owner = admitted
        token = session.key
        reconnected = ws.query_params.get("attach") is not None

        def initial_text(truncated: bool) -> str:
            # Says whether the replay that follows leaves out older output.
            return _metadata(token, owner, registry, reconnected=reconnected, truncated=truncated)

        if not await session.attach(ws, initial_text=initial_text):
            if session._ws is None:
                await _close_ws(ws, WS_CLOSE_VIEWER_STALLED, WS_CLOSE_VIEWER_STALLED_REASON)
            return
        await _pump_input(ws, registry, token, session, owner)
    except TerminalDenied:
        await ws.close(code=4403, reason="Terminal belongs to another identity or profile")
    except TerminalExited:
        # The same close an attached tab gets when its shell exits: the tab closes too.
        await ws.close(code=WS_CLOSE_PROCESS_EXITED, reason=WS_CLOSE_PROCESS_EXITED_REASON)
    except (TerminalExpired, FileNotFoundError, HTTPException):
        await ws.close(code=4410, reason="Terminal or profile expired; open a new terminal")
    except RegistryFull:
        await ws.close(code=1013, reason="Terminal capacity reached; close a terminal and retry")
    except WebSocketDisconnect:
        pass
    except (OSError, RuntimeError) as exc:
        await _pty_fail(ws, exc, surface="Terminal")
    finally:
        if session is not None:
            session.detach(ws)
