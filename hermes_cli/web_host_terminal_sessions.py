"""Process-lifetime host shells, owned by Webapp identity and profile generation.

This is a separate capability namespace from Dashboard's client-selected chat IDs.
The existing PTY registry owns processes/drains; this module owns host admission.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import json
from pathlib import Path
import re
import secrets
import time

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from hermes_cli.profile_incarnation import (
    ensure_profile_incarnation, profile_incarnation_lease, profile_incarnation_matches,
)
from hermes_cli.pty_session import PtySession, PtySessionRegistry, RegistryFull, run_reaper
from hermes_cli.web_host_terminal import META_PREFIX
from hermes_constants import get_hermes_home, named_profile_home_is_unavailable

# No survival across backend restart. Disconnects retain a bounded ANSI tail.
RETENTION_SECONDS = 15 * 60
# Inbound frames buffered while one batch is checked and written (backpressure bound).
_INPUT_BATCH_FRAMES = 64
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")


class TerminalDenied(Exception):
    pass


class TerminalExpired(Exception):
    pass


@dataclass(frozen=True)
class HostOwner:
    identity: tuple[str, str]
    home: Path
    incarnation: str | None
    shell: str
    cwd: str

    def current(self) -> bool:
        """Blocking filesystem reads (home, tombstone, incarnation marker): call off-loop."""
        return (
            self.home.is_dir()
            and not named_profile_home_is_unavailable(self.home)
            and (self.incarnation is None or profile_incarnation_matches(self.home, self.incarnation))
        )


def _request_identity(ws: WebSocket) -> tuple[str, str]:
    identity = getattr(ws, "_hermes_auth_identity", None)
    if identity:
        user, provider = identity.get("user_id"), identity.get("provider")
        if not isinstance(user, str) or not user or not isinstance(provider, str) or not provider:
            raise TerminalDenied()
        return provider, user
    from hermes_cli.web_server import app
    if getattr(app.state, "auth_required", False):
        raise TerminalDenied()
    # _ws_gate already validated the process's loopback credential.
    return "loopback", "session-token"


def _request_home(profile: str | None) -> Path:
    from hermes_cli.web_server_profiles import _resolve_profile_dir
    requested = (profile or "").strip()
    return (
        _resolve_profile_dir(requested)
        if requested and requested.lower() != "current" else get_hermes_home()
    ).resolve()


class HostTerminalRegistry(PtySessionRegistry):
    def __init__(self, *, ttl: float = RETENTION_SECONDS, max_sessions: int = 16,
                 buffer_cap: int = 1024 * 1024, read_timeout: float = 0.2):
        super().__init__(ttl=ttl, max_sessions=max_sessions, buffer_cap=buffer_cap, read_timeout=read_timeout)
        self.owners: dict[str, HostOwner] = {}
        self._closing: set[asyncio.Task] = set()

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
            # Capture the generation and spawn under this name's lease so
            # deletion/recreation cannot retarget an already-admitted request.
            with profile_incarnation_lease(home):
                if not home.is_dir():
                    raise TerminalExpired()
                incarnation = ensure_profile_incarnation(home)
                argv, cwd, env, shell = resolve_argv(home=home, requested_cwd=ws.query_params.get("cwd"))
                bridge = PtyBridge.spawn(
                    argv, cwd=cwd, env=env,
                    cols=query_dimension(ws.query_params.get("cols"), 80, 2000),
                    rows=query_dimension(ws.query_params.get("rows"), 24, 1000))
                owner = HostOwner(identity, home, incarnation, shell, cwd)
                return bridge

        session, _ = await self.attach_or_spawn(token, spawn=spawn)
        assert owner is not None  # spawn completed under the incarnation lease
        self.owners[token] = owner
        if not await asyncio.to_thread(owner.current):
            await self.remove(token)
            raise TerminalExpired()
        return session, owner

    async def resolve(self, token: str, ws: WebSocket, identity: tuple[str, str], *, closing=False):
        session = self._sessions.get(token)
        owner = self.owners.get(token)
        if session is None or owner is None:
            raise TerminalExpired()
        if owner.identity != identity:
            raise TerminalDenied()

        def validate():
            home = _request_home(ws.query_params.get("profile"))
            if home != owner.home:
                raise TerminalDenied()
            with profile_incarnation_lease(home, owner.incarnation):
                if not owner.current():
                    raise TerminalExpired()

        await asyncio.to_thread(validate)
        # No await between the final check and the caller's attach claim. The
        # generation was just checked off-loop; input re-checks it before any write.
        if (self._sessions.get(token) is not session
                or (not closing and (not session.alive or self.expired(session)))):
            raise TerminalExpired()
        return session, owner

    def expired(self, session, now=None):
        return (not session.attached and session.last_detached_at is not None
                and (time.monotonic() if now is None else now) - session.last_detached_at > self._ttl)

    async def remove(self, token: str):
        session = self._sessions.pop(token, None)
        self.owners.pop(token, None)
        if session is not None:
            await self._close_session(session)

    async def _close_session(self, session: PtySession):
        task = asyncio.create_task(session.close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)
        await asyncio.shield(task)

    async def reap_idle(self, now=None):
        # Durable tombstones/incarnations also detect retirement by a different
        # process; attached viewers do not exempt retired profile generations.
        # One thread hop per tick keeps every shell's marker reads off the loop.
        owners = list(self.owners.items())
        retired = await asyncio.to_thread(lambda: {token for token, owner in owners if not owner.current()})
        for token, owner in owners:
            session = self._sessions.get(token)
            if session is not None and (token in retired or not session.alive or self.expired(session, now)):
                await self.remove(token)
        for token in list(self.owners):
            if token not in self._sessions:
                self.owners.pop(token, None)

    async def close_all(self):
        await super().close_all()
        if self._closing:
            await asyncio.gather(*self._closing)
        self.owners.clear()


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


def _metadata(token, owner, session, registry, *, reconnected):
    return META_PREFIX + json.dumps({
        "shell": owner.shell, "terminalId": token, "cwd": owner.cwd,
        "reconnected": reconnected, "retentionSeconds": registry._ttl,
        "truncated": session.buffer.truncated,
    }, separators=(",", ":"))


async def _send_closed(ws: WebSocket, token: str) -> None:
    await ws.send_text(META_PREFIX + json.dumps({"terminalId": token, "closed": True}))
    await ws.close(code=1000)


async def _admit(ws: WebSocket, registry: HostTerminalRegistry):
    """Authorize the request and return ``(token, session, owner)`` to attach.

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
        session, owner = await registry.create(ws, identity)
        return session.key, session, owner
    session, owner = await registry.resolve(token, ws, identity, closing=action == "close")
    if action == "close":
        await registry.remove(token)
        await _send_closed(ws, token)
        return None
    return token, session, owner


async def _read_frames(ws: WebSocket, inbox: asyncio.Queue) -> None:
    """Queue inbound frames. The last item is None, or the error that ended reading
    (re-raised by ``_pump_input`` so the route's handlers still see it)."""
    end = None
    try:
        while (msg := await ws.receive()).get("type") != "websocket.disconnect":
            await inbox.put(msg)
    except RuntimeError:  # a superseding/drain task already closed us
        pass
    except Exception as exc:
        end = exc
    await inbox.put(end)


async def _pump_input(ws: WebSocket, registry: HostTerminalRegistry, token: str,
                      session: PtySession, owner: HostOwner) -> None:
    """Forward keystrokes and resizes while ``ws`` is the attached viewer.

    A frame is written only after an ownership check that began once it had
    arrived, so input sent after a profile generation retires never reaches the
    shell. The check reads files, so it runs off the loop, once per batch: the
    frames that queued while the previous batch was checked or written. A paste
    of many frames costs a few checks, not one each.
    """
    from hermes_cli.web_server_chat import _RESIZE_RE
    inbox: asyncio.Queue = asyncio.Queue(maxsize=_INPUT_BATCH_FRAMES)
    reader = asyncio.create_task(_read_frames(ws, inbox))
    try:
        # No Ctrl-L/redraw input: arbitrary foreground shell programs own stdin.
        while session._ws is ws and session.alive:
            batch = [await inbox.get()]
            while not inbox.empty():
                batch.append(inbox.get_nowait())
            frames = [msg for msg in batch if isinstance(msg, dict)]
            if frames and not await asyncio.to_thread(owner.current):
                await registry.remove(token)
                return
            for msg in frames:
                raw = msg.get("bytes")
                if raw is None:
                    raw = (msg.get("text") or "").encode("utf-8")
                if not raw:
                    continue
                match = _RESIZE_RE.fullmatch(raw)
                if match:
                    session.resize(ws, cols=int(match[1]), rows=int(match[2]))
                elif not await session.write(ws, raw):
                    if session._ws is ws:
                        await registry.remove(token)
                    return
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
    from hermes_cli.web_server_chat import _pty_fail
    session = None
    try:
        registry = get_host_terminals(ws.app)
        admitted = await _admit(ws, registry)
        if admitted is None:
            return
        token, session, owner = admitted
        initial_text = _metadata(
            token, owner, session, registry, reconnected=ws.query_params.get("attach") is not None)
        if not await session.attach(ws, initial_text=initial_text):
            if session._ws is None:
                with suppress(Exception):
                    await ws.close(code=1011, reason="Terminal replay interrupted; reconnect")
            return
        await _pump_input(ws, registry, token, session, owner)
    except TerminalDenied:
        await ws.close(code=4403, reason="Terminal belongs to another identity or profile")
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
