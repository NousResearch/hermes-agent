"""Interactive terminal for the shared Desktop browser renderer.

Reuse the server's WebSocket authorization and existing POSIX PTY lifecycle.
A socket owns its shell; disconnect closes it. Never fall back from a sandboxed
terminal backend to a host shell, and never expose this on a locked hosted home.
"""
from __future__ import annotations

import asyncio
import contextlib
import codecs
import functools
import json
import logging
import os

import anyio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()
_MAX_TERMINALS = 8
_active_terminals = 0


def _terminal_launch(profile: str | None, cwd: str | None):
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.web_server_files import _fs_path, _managed_files_policy
    from hermes_cli.web_server_profiles import _config_profile_scope
    from hermes_platform.resolver import locate_command
    from tools.environments.local import build_subprocess_env

    with _config_profile_scope(profile):
        config = load_user_config_effective()
        terminal = config.get("terminal", {})
        if terminal.get("backend", "local") != "local":
            raise ValueError("Interactive browser shells require the local terminal backend.")
        if _managed_files_policy(None, create_root=False).locked_root is not None:
            raise ValueError("Interactive shells are unavailable on this hosted backend.")
        directory = _fs_path(cwd or terminal.get("cwd") or os.getcwd())
        if not directory.is_dir():
            raise ValueError("Terminal working directory is not a directory.")
        shell = locate_command(os.environ.get("SHELL", "sh")).command
        if not shell:
            raise ValueError("No interactive shell is available on this backend.")
        return shell, str(directory), build_subprocess_env(scrub_secrets=True)


async def _terminal_output(ws, bridge):
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    while True:
        data = await asyncio.to_thread(bridge.read)
        if data is None:
            tail = decoder.decode(b'', final=True)
            if tail:
                await ws.send_json({"type": "data", "data": tail})
            return
        if data:
            await ws.send_json({"type": "data", "data": decoder.decode(data)})


async def _terminal_input(ws, bridge):
    while True:
        raw = await ws.receive_text()
        if len(raw) > 65536:
            await ws.close(code=1009)
            return
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise ValueError("Invalid terminal message.")
        if message.get("type") == "resize":
            bridge.resize(cols=int(message["cols"]), rows=int(message["rows"]))
        elif message.get("type") == "write" and isinstance(message.get("data"), str):
            if not await bridge.write(message["data"].encode("utf-8")):
                return
        else:
            raise ValueError("Invalid terminal message.")


async def _terminal_cleanup(ws, tasks, spawn_task):
    global _active_terminals
    try:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if spawn_task is not None:
            # Startup runs in a thread and cannot be stopped by cancelling its
            # waiter. Retain ownership until it returns, including during shutdown.
            try:
                bridge = await spawn_task
            except Exception:
                logger.warning("Browser terminal startup failed before cleanup acquired a PTY", exc_info=True)
            else:
                await asyncio.to_thread(bridge.close)
        with contextlib.suppress(RuntimeError):
            await ws.close()
    finally:
        _active_terminals -= 1


async def _finish_terminal_cleanup(cleanup_task):
    cancelled = False
    # AnyIO shielding handles cancellation scopes; asyncio.shield plus retry
    # handles direct Task.cancel(), including repeated shutdown cancellation.
    with anyio.CancelScope(shield=True):
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cancelled = True
        cleanup_task.result()
    if cancelled:
        raise asyncio.CancelledError


@router.websocket("/api/browser/terminal")
async def browser_terminal(ws: WebSocket):
    from hermes_cli.web_server_chat import _ws_auth_ok, _ws_request_is_allowed
    from hermes_cli.web_routers.status import _frontend_identity

    global _active_terminals
    # This is a browser-only surface: require an actual matching web Origin in
    # addition to the shared Host/client/auth checks (Electron exemptions do not apply).
    from urllib.parse import urlparse
    origin = urlparse(ws.headers.get("origin", ""))
    if (origin.scheme not in ("http", "https") or origin.netloc != ws.headers.get("host")
            or not _ws_request_is_allowed(ws) or not _ws_auth_ok(ws)):
        await ws.close(code=1008)
        return
    if _frontend_identity() != "desktop" or _active_terminals >= _MAX_TERMINALS:
        await ws.close(code=1013)
        return
    # Reserve before the first await: concurrent handshakes must share the cap.
    _active_terminals += 1
    spawn_task = None
    tasks = []
    try:
        await ws.accept()
        # Import lazily: importing the PTY bridge on native Windows is unsupported.
        from hermes_platform.host.facts import os_family
        if os_family() == "win32":
            raise ValueError("Interactive browser terminals require a POSIX backend (Linux, macOS or WSL).")
        from hermes_cli.pty_bridge import PtyBridge
        from hermes_cli.web_server_profiles import _config_profile_scope
        profile = ws.query_params.get("profile")
        shell, cwd, env = await asyncio.to_thread(_terminal_launch, profile, ws.query_params.get("cwd"))
        with _config_profile_scope(profile):
            spawn_task = asyncio.create_task(asyncio.to_thread(functools.partial(
                PtyBridge.spawn, [*shell, '-i'], cwd=cwd, env=env,
                cols=int(ws.query_params.get("cols", "80")),
                rows=int(ws.query_params.get("rows", "24")))))
            bridge = await asyncio.shield(spawn_task)
        await ws.send_json({"type": "ready", "cwd": cwd, "shell": shell[0]})
        tasks = [asyncio.create_task(_terminal_output(ws, bridge)), asyncio.create_task(_terminal_input(ws, bridge))]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (ValueError, KeyError, OSError, ImportError, HTTPException) as exc:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.send_json({"type": "error", "error": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        cleanup_task = asyncio.create_task(_terminal_cleanup(ws, tasks, spawn_task))
        await _finish_terminal_cleanup(cleanup_task)
