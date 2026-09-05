"""Chat-tab WebSocket routes: /api/console, /api/pty, the /api/ws gateway
sidecar and /api/pub + /api/events broadcast.

Helpers/state that tests monkeypatch on ``web_server`` stay there and are
reached through the late-binding seam (cycle-safe).
"""

import asyncio
import functools
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from hermes_cli.pty_session import RegistryFull
from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_chat import (
    _build_sidecar_url, _close_stalled_pty_input, _get_console_executor, _legacy_pump, _ws_auth_ok,
    _ws_request_is_allowed,
)

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_active_session_file_for_channel = late("_active_session_file_for_channel", "hermes_cli.web_server_chat")
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
_resolve_chat_argv_async = late("_resolve_chat_argv_async", "hermes_cli.web_server_chat")
_resolve_profile_dir = late("_resolve_profile_dir", "hermes_cli.web_server_profiles")
_ws_auth_reason = late("_ws_auth_reason", "hermes_cli.web_server_chat")
_ws_client_reason = late("_ws_client_reason", "hermes_cli.web_server_chat")
_ws_host_origin_reason = late("_ws_host_origin_reason", "hermes_cli.web_server_chat")
_DASHBOARD_EMBEDDED_CHAT_ENABLED = LateState("_DASHBOARD_EMBEDDED_CHAT_ENABLED")


def _get_event_state(app: "FastAPI"):
    """(event_channels, event_lock) from app.state, lazily initialised when the
    lifespan hasn't run (TestClient without a ``with`` block). The lifespan path
    is preferred because it creates the Lock on the correct event loop."""
    try:
        return app.state.event_channels, app.state.event_lock
    except AttributeError:
        app.state.event_channels = {}
        app.state.event_lock = asyncio.Lock()
        return app.state.event_channels, app.state.event_lock


_VALID_CHANNEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_PTY_EPOCH_RE = re.compile(r"[0-9a-f]{32}")
_PTY_MAX_CLIENT_OFFSET = (1 << 53) - 1


def _parse_pty_replay_cursor(ws: WebSocket) -> tuple[Optional[str], Optional[int]]:
    """Parse the optional all-or-nothing PTY replay cursor (epoch + offset)."""
    epochs = ws.query_params.getlist("epoch")
    offsets = ws.query_params.getlist("offset")
    if len(epochs) > 1 or len(offsets) > 1:
        raise ValueError("duplicate epoch or offset")

    raw_epoch = epochs[0] if epochs else None
    raw_offset = offsets[0] if offsets else None
    if raw_epoch is None and raw_offset is None:
        return None, None
    if raw_epoch is None or raw_offset is None:
        raise ValueError("epoch and offset must be provided together")
    if not (ws.query_params.get("attach") or "").strip():
        raise ValueError("epoch and offset require an attach token")
    if _PTY_EPOCH_RE.fullmatch(raw_epoch) is None:
        raise ValueError("epoch must be 32 lowercase hexadecimal characters")
    if re.fullmatch(r"0|[1-9][0-9]*", raw_offset) is None:
        raise ValueError("offset must be a canonical non-negative integer")

    offset = int(raw_offset)
    if offset > _PTY_MAX_CLIENT_OFFSET:
        raise ValueError("offset exceeds the JavaScript safe-integer range")
    return raw_epoch, offset


def _ws_auth_mode() -> str:
    """Short label for the active WS auth mode — logged on every connection."""
    from hermes_cli.web_server_chat import _LOOPBACK_HOSTS
    from hermes_cli.web_server import app
    if getattr(app.state, "auth_required", False):
        return "gated"
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return "insecure"
    return "loopback"


async def _broadcast_event(app: Any, channel: str, payload: str) -> None:
    """Fan out one publisher frame to every subscriber on `channel`."""
    event_channels, event_lock = _get_event_state(app)
    async with event_lock:
        subs = list(event_channels.get(channel, ()))
    for sub in subs:
        try:
            await sub.send_text(payload)
        except Exception:
            # Subscriber went away mid-send; /api/events' finally removes it.
            _log.warning("broadcast send failed for subscriber on %s", channel, exc_info=True)


def _channel_or_close_code(ws: WebSocket) -> Optional[str]:
    """Channel id from the query string, or None if invalid."""
    channel = ws.query_params.get("channel", "")
    return channel if _VALID_CHANNEL_RE.match(channel) else None


def _read_active_session_file(path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return str(data.get("session_id") or "").strip() or None


def _ws_close_reason(text: str) -> str:
    """Clamp to RFC 6455's 123-byte close-reason limit (uvicorn raises past it);
    reasons embed an attacker-controlled origin, so truncate rather than crash."""
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= 123:
        return text
    return encoded[:120].decode("utf-8", "ignore") + "..."


async def _ws_gate(ws: WebSocket, kind: str) -> Optional[tuple[str, str, str]]:
    """Run the pre-accept gates for /api/console and /api/pty.

    Each gate maps to a distinct close code so the log and the browser banner
    agree on the cause: 4404 chat disabled, 4401 bad credential, 4403
    host/origin mismatch, 4408 peer not allowed. Returns ``(peer, mode, cred)``
    once every gate passes, or None after closing the socket.
    """
    peer = ws.client.host if ws.client else "?"
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        _log.info("%s refused: embedded chat disabled peer=%s", kind, peer)
        await ws.close(code=4404, reason="embedded chat disabled")
        return None

    auth_reason, cred = _ws_auth_reason(ws)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning("%s auth rejected reason=%s mode=%s cred=%s peer=%s", kind, auth_reason, mode, cred, peer)
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return None

    host_origin_reason = _ws_host_origin_reason(ws)
    if host_origin_reason is not None:
        _log.warning("%s refused: %s peer=%s", kind, host_origin_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(host_origin_reason))
        return None

    client_reason = _ws_client_reason(ws)
    if client_reason is not None:
        _log.warning("%s refused: %s", kind, client_reason)
        await ws.close(code=4408, reason=_ws_close_reason(client_reason))
        return None
    return peer, mode, cred


async def _close_unless_sidecar_allowed(ws: WebSocket) -> bool:
    """Pre-accept gates for the /api/ws, /api/pub and /api/events sidecars:
    4403 when chat is disabled or the request isn't allowed, 4401 on bad auth."""
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return False
    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return False
    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return False
    return True


# --- /api/console: the curated console engine, in-process, exchanging JSON
# frames with the dashboard xterm overlay. Never spawns a PTY, shell or CLI.

_CONSOLE_PROMPT = "hermes> "
_CONSOLE_COMMAND_TIMEOUT_SECONDS = 60.0
_CONSOLE_OUTPUT_LIMIT = 50000


def _execute_console_line(engine: Any, line: str, *, confirmed: bool, profile: Optional[str]) -> Any:
    # _profile_scope swaps process-global skill module paths; keep it inside
    # the worker thread and never hold it across awaits.
    with _profile_scope(profile):
        return engine.execute(line, confirmed=confirmed)


class _ConsoleSender:
    """Serialises frames onto one console socket and owns the prompt suffix."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.lock = asyncio.Lock()

    async def send(self, payload: Dict[str, Any]) -> None:
        async with self.lock:
            await self.ws.send_json(payload)

    async def prompt(self, **payload: Any) -> None:
        await self.send({**payload, "prompt": _CONSOLE_PROMPT})

    async def error(self, message: str, *, id: Optional[int] = None, command: Optional[str] = None,
                    prompt: Optional[str] = None) -> None:
        # Key order matches the historical frames: type, id, message, command, prompt.
        frame: Dict[str, Any] = {"type": "error"}
        if id is not None:
            frame["id"] = id
        frame["message"] = message
        if command is not None:
            frame["command"] = command
        if prompt is not None:
            frame["prompt"] = prompt
        await self.send(frame)

    async def complete(self, status: str, command: str, command_id: int, *, prompt: str = _CONSOLE_PROMPT) -> None:
        await self.send({"type": "complete", "id": command_id, "status": status, "command": command, "prompt": prompt})

    async def error_then_complete(self, message: str, command: str, command_id: int, status: str) -> None:
        await self.error(message, id=command_id, command=command)
        await self.complete(status, command, command_id)

    async def send_result(self, result: Any, *, command_id: int) -> None:
        command = result.command or ""
        status = result.status
        if status == "ok":
            if result.output:
                await self.send({
                    "type": "output", "id": command_id, "stream": "stdout",
                    "data": result.output, "command": command,
                })
            await self.complete("ok", command, command_id)
        elif status == "error":
            await self.error_then_complete(result.output or "Command failed.", command, command_id, "error")
        elif status == "confirm_required":
            await self.prompt(
                type="confirm_required", id=command_id, command=command,
                message=result.confirmation_message or f"Run `{command}`?",
            )
            await self.complete("confirm_required", command, command_id)
        elif status == "clear":
            await self.send({"type": "clear", "id": command_id})
            await self.complete("clear", command, command_id)
        elif status == "exit":
            await self.complete("exit", command, command_id, prompt="")
        else:
            await self.error(f"Unknown console result status: {status}", id=command_id, command=command)


def _console_json_payload(msg: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    raw: str | bytes | None = msg.get("text")
    if raw is None:
        raw = msg.get("bytes")
    if raw is None:
        return None, None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "Console frames must be UTF-8 JSON."
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, "Console frames must be JSON objects."
    if not isinstance(payload, dict):
        return None, "Console frames must be JSON objects."
    return payload, None


@router.websocket("/api/console")
async def console_ws(ws: WebSocket) -> None:
    gate = await _ws_gate(ws, "console")
    if gate is None:
        return
    peer, mode, cred = gate
    await ws.accept()

    profile = (ws.query_params.get("profile") or "").strip() or None
    out = _ConsoleSender(ws)

    try:
        from hermes_cli.console_engine import HermesConsoleEngine

        engine = HermesConsoleEngine(output_limit=_CONSOLE_OUTPUT_LIMIT)
        if profile and profile.lower() != "current":
            _resolve_profile_dir(profile)
    except HTTPException as exc:
        await out.error(str(exc.detail), prompt="")
        await ws.close(code=4400, reason=_ws_close_reason(str(exc.detail)))
        return
    except Exception as exc:
        _log.exception("console failed to initialize")
        await out.error(f"Console unavailable: {exc}", prompt="")
        await ws.close(code=1011)
        return

    _log.info("console accepted peer=%s mode=%s cred=%s profile=%s", peer, mode, cred, profile or "current")
    await out.prompt(type="ready", profile=profile or "current")

    active_task: asyncio.Task | None = None
    pending_confirmation: Optional[str] = None
    command_generation = 0

    async def run_command(line: str, *, confirmed: bool, command_id: int) -> None:
        nonlocal active_task, pending_confirmation, command_generation
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _get_console_executor(),
                    functools.partial(_execute_console_line, engine, line, confirmed=confirmed, profile=profile),
                ),
                timeout=_CONSOLE_COMMAND_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            if command_id == command_generation:
                pending_confirmation = None
                await out.error_then_complete(
                    "Command timed out. Hermes Console returned to the prompt.", line, command_id, "timeout",
                )
        except Exception as exc:
            if command_id == command_generation:
                pending_confirmation = None
                _log.exception("console command failed")
                await out.error_then_complete(str(exc) or exc.__class__.__name__, line, command_id, "error")
        else:
            if command_id != command_generation:
                return
            pending_confirmation = result.command if result.status == "confirm_required" else None
            await out.send_result(result, command_id=command_id)
            if result.status == "exit":
                await ws.close(code=1000)
        finally:
            if command_id == command_generation:
                active_task = None

    def start_command(line: str, *, confirmed: bool = False) -> None:
        nonlocal active_task, command_generation
        command_generation += 1
        active_task = asyncio.create_task(run_command(line, confirmed=confirmed, command_id=command_generation))

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                break
            if msg.get("type") == "websocket.disconnect":
                break

            payload, error = _console_json_payload(msg)
            if error:
                await out.prompt(type="error", message=error)
                continue
            if payload is None:
                continue

            frame_type = str(payload.get("type") or "").strip().lower()
            if frame_type == "ping":
                await out.prompt(type="pong")
                continue

            if frame_type == "cancel":
                if active_task and not active_task.done():
                    command_generation += 1
                    active_task.cancel()
                    active_task = None
                    pending_confirmation = None
                    await out.prompt(type="complete", status="cancelled")
                elif pending_confirmation:
                    pending_confirmation = None
                    await out.prompt(type="complete", status="cancelled")
                else:
                    await out.prompt(type="complete", status="idle")
                continue

            if active_task and not active_task.done():
                await out.prompt(type="error", message="A console command is already running.")
                continue

            if frame_type == "confirm":
                command = str(payload.get("command") or pending_confirmation or "").strip()
                if not pending_confirmation:
                    await out.prompt(type="error", message="No command is waiting for confirmation.")
                    continue
                if command != pending_confirmation:
                    await out.prompt(type="error", message="Confirmation does not match the pending command.")
                    continue
                pending_confirmation = None
                start_command(command, confirmed=True)
                continue

            if frame_type in {"input", "command"}:
                line = str(payload.get("line") or payload.get("command") or "").strip()
                if not line:
                    await out.prompt(type="complete", status="ok")
                    continue
                if pending_confirmation:
                    await out.prompt(
                        type="error",
                        message="Confirm or cancel the pending command before running another one.",
                    )
                    continue
                start_command(line)
                continue

            await out.prompt(type="error", message=f"Unsupported console frame: {frame_type or '?'}")
    except WebSocketDisconnect:
        pass
    finally:
        if active_task and not active_task.done():
            active_task.cancel()
            try:
                await active_task
            except (asyncio.CancelledError, Exception):
                pass


async def _pty_fail(ws: WebSocket, text: str) -> None:
    await ws.send_text(f"\r\n\x1b[31m{text}\x1b[0m\r\n")
    await ws.close(code=1011)


@router.websocket("/api/pty")
async def pty_ws(ws: WebSocket) -> None:
    from hermes_cli.web_server_chat import PTY_REGISTRY, PtyBridge, PtyUnavailableError, _PTY_BRIDGE_AVAILABLE, _RESIZE_RE
    gate = await _ws_gate(ws, "pty")
    if gate is None:
        return
    peer, mode, cred = gate
    try:
        client_epoch, client_offset = _parse_pty_replay_cursor(ws)
    except ValueError as exc:
        _log.warning("pty refused: invalid replay cursor peer=%s error=%s", peer, exc)
        await ws.close(code=4400, reason=_ws_close_reason(f"invalid replay cursor: {exc}"))
        return
    await ws.accept()
    _log.info("pty accepted peer=%s mode=%s cred=%s", peer, mode, cred)

    # Native Windows can't import the POSIX PTY bridge: say so and close cleanly.
    if not _PTY_BRIDGE_AVAILABLE:
        await ws.send_text(
            "\r\n\x1b[31mChat unavailable: the embedded terminal requires a "
            "POSIX PTY, which native Windows Python doesn't provide.\x1b[0m\r\n"
            "\x1b[33mInstall Hermes inside WSL2 to use the dashboard's /chat "
            "tab — the rest of the dashboard works here.\x1b[0m\r\n"
        )
        await ws.close(code=1011)
        return

    raw_resume = ws.query_params.get("resume") or None
    resume = raw_resume
    profile = ws.query_params.get("profile") or None
    channel = _channel_or_close_code(ws)
    sidecar_url = _build_sidecar_url(channel) if channel else None
    force_fresh = (ws.query_params.get("fresh") or "").strip().lower() in {"1", "true", "yes", "on"}
    active_session_file: Optional[Path] = None

    if channel:
        active_session_file = _active_session_file_for_channel(ws.app, channel)
        if force_fresh:
            resume = None
            try:
                active_session_file.unlink(missing_ok=True)
            except OSError:
                pass
        elif not resume:
            resume = _read_active_session_file(active_session_file)
            if resume:
                # The client only pins the viewport to the bottom when it asked
                # for `?resume=`; announce the implicit active-session replay so
                # it gets the same follow-scroll treatment.
                # See #93518.
                await ws.send_json({"type": "resume", "id": resume})

    resolve_kwargs = {"resume": resume, "sidecar_url": sidecar_url, "profile": profile}
    if active_session_file is not None:
        resolve_kwargs["active_session_file"] = str(active_session_file)

    try:
        argv, cwd, env = await _resolve_chat_argv_async(**resolve_kwargs)
    except HTTPException as exc:  # unknown/invalid profile
        await _pty_fail(ws, f"Chat unavailable: {exc.detail}")
        return
    except SystemExit as exc:  # _make_tui_argv sys.exit(1)s when node/npm is missing
        await _pty_fail(ws, f"Chat unavailable: {exc}")
        return

    attach_token = ws.query_params.get("attach") or None
    registry_resume = raw_resume
    if raw_resume and env:
        registry_resume = env.get("HERMES_TUI_RESUME") or raw_resume
    if attach_token is not None and (registry_resume or profile):
        # Key explicit resumes on their canonical target, never the active-session fallback.
        attach_token = f"{attach_token}\0{profile or ''}\0{registry_resume or ''}"

    def _spawn():
        return PtyBridge.spawn(argv, cwd=cwd, env=env)

    if attach_token is None:
        # Legacy path: 1:1 socket<->PTY, killed on disconnect.
        try:
            bridge = _spawn()
        except PtyUnavailableError as exc:
            await _pty_fail(ws, f"Chat unavailable: {exc}")
            return
        except (FileNotFoundError, OSError) as exc:
            await _pty_fail(ws, f"Chat failed to start: {exc}")
            return
        await _legacy_pump(ws, bridge)
        return

    # Keep-alive path: the PTY outlives this socket; reattach by token.
    try:
        session, _created = await PTY_REGISTRY.attach_or_spawn(attach_token, spawn=_spawn)
    except (PtyUnavailableError, FileNotFoundError, OSError, RegistryFull) as exc:
        await _pty_fail(ws, f"Chat unavailable: {exc}")
        return

    # A fresh xterm can't rebuild the TUI from an arbitrary tail of alternate-
    # screen differential output; reused PTYs emit a full frame after replay.
    if not await session.attach(
        ws,
        client_offset=client_offset,
        client_epoch=client_epoch,
        force_redraw=not _created,
    ):
        await _close_stalled_pty_input(ws, path="keepalive-redraw")
        PTY_REGISTRY.detach(attach_token, ws)
        return

    # Writer loop only: the session's drain task (one per PTY, inside the
    # registry) forwards output to whichever socket is attached and ring-buffers
    # it while detached. On child EOF it closes the attached socket with 4410,
    # which unparks ws.receive() — same half-open protection as the legacy pump.
    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:  # receive() after the drain task already closed us
                break
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is None:
                text = msg.get("text")
                raw = text.encode("utf-8") if isinstance(text, str) else b""
            if not raw:
                continue
            # Resize escape is consumed locally, never written to the PTY.
            match = _RESIZE_RE.match(raw)
            if match and match.end() == len(raw):
                session.bridge.resize(cols=int(match.group(1)), rows=int(match.group(2)))
                continue
            if not await session.write(ws, raw):
                await _close_stalled_pty_input(ws, path="keepalive")
                break
    except WebSocketDisconnect:
        pass
    finally:
        # Detach only — the PTY keeps running for a reattach; the registry
        # reaper closes it after the TTL (or immediately on process exit).
        PTY_REGISTRY.detach(attach_token, ws)


# --- /api/ws: JSON-RPC sidecar for the Chat tab. Drives the same
# tui_gateway.dispatch surface Ink uses over stdio so the dashboard can render
# structured metadata next to the xterm; both transports bind to the same
# session id, so agent emits fan out to both sinks.


@router.websocket("/api/ws")
async def gateway_ws(ws: WebSocket) -> None:
    if not await _close_unless_sidecar_allowed(ws):
        return
    from tui_gateway.ws import handle_ws

    # The authenticated identity (ticket / internal credential) stamped by
    # _ws_auth_reason becomes the identity authority for privileged RPCs
    # (browser.controller.register). None on the legacy token path.
    await handle_ws(
        ws,
        auth_identity=getattr(ws, "_hermes_auth_identity", None),
        subprotocol=getattr(ws, "_hermes_ws_subprotocol", None),
    )


# --- /api/pub + /api/events: the PTY-side tui_gateway.entry opens /api/pub
# (HERMES_TUI_SIDECAR_URL from /api/pty's env) and writes every dispatcher emit
# through it; the dashboard fans frames out to /api/events subscribers on the
# same channel — the React sidebar's tool-call feed without touching the PTY
# child's stdio handshake with Ink.


async def _accept_channel_ws(ws: WebSocket) -> Optional[str]:
    if not await _close_unless_sidecar_allowed(ws):
        return None
    channel = _channel_or_close_code(ws)
    if not channel:
        await ws.close(code=4400)
        return None
    await ws.accept()
    return channel


@router.websocket("/api/pub")
async def pub_ws(ws: WebSocket) -> None:
    channel = await _accept_channel_ws(ws)
    if channel is None:
        return
    try:
        while True:
            await _broadcast_event(ws.app, channel, await ws.receive_text())
    except WebSocketDisconnect:
        pass


@router.websocket("/api/events")
async def events_ws(ws: WebSocket) -> None:
    channel = await _accept_channel_ws(ws)
    if channel is None:
        return
    event_channels, event_lock = _get_event_state(ws.app)
    async with event_lock:
        event_channels.setdefault(channel, set()).add(ws)
    try:
        while True:
            # Subscribers don't speak — receive() just blocks until disconnect.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with event_lock:
            subs = event_channels.get(channel)
            if subs is not None:
                subs.discard(ws)
                if not subs:
                    event_channels.pop(channel, None)
