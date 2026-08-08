"""Hermes Console command WebSocket (``/api/console``).

Extracted verbatim from ``hermes_cli/web_server.py`` (god-file decomposition
Wave 1, shard s5, cluster ``console_ws``): the safe in-process console
endpoint — structured JSON frames, bounded worker pool, command timeouts —
plus its pure helpers (frame parsing, result framing, profile extraction).

Behavior-neutral lift: bodies are unchanged.  ``_log`` is bound by explicit
name (``hermes_cli.web_server``) so log records emitted from these functions
keep the logger name they had before, and ``_DASHBOARD_EMBEDDED_CHAT_ENABLED``
/ ``_resolve_profile_dir`` / ``_profile_scope`` are imported lazily from
``hermes_cli.web_server`` at call time (same discipline as the accepted
precedent ``hermes_cli/cli_commands_mixin.py``) so tests that monkeypatch
``web_server._DASHBOARD_EMBEDDED_CHAT_ENABLED`` keep working.  The
``@app.websocket("/api/console")`` decorator is applied in ``web_server.py``
(which owns ``app``); the route stays registered ahead of the SPA catch-all.
"""

import asyncio
import atexit
import concurrent.futures
import functools
import json
import logging
import threading
from typing import Any, Dict, Optional

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from hermes_cli.ws_auth_gate_mixin import (
    _ws_auth_mode,
    _ws_auth_reason,
    _ws_client_reason,
    _ws_close_reason,
    _ws_host_origin_reason,
)

# Bind the web_server logger by name so records emitted here keep the exact
# logger name they had when these functions lived in web_server.py.
_log = logging.getLogger("hermes_cli.web_server")

# /api/console — safe Hermes Console command WebSocket.
#
# Unlike /api/pty, this endpoint never spawns a PTY, shell, or full Hermes CLI
# subprocess. It runs the curated console engine in-process and exchanges
# structured JSON frames with the dashboard xterm overlay.
# ---------------------------------------------------------------------------

_CONSOLE_PROMPT = "hermes> "
_CONSOLE_COMMAND_TIMEOUT_SECONDS = 60.0
_CONSOLE_OUTPUT_LIMIT = 50000

# Console commands run in a worker thread. On a timeout, asyncio.wait_for cancels
# the *awaitable*, but Python threads aren't preemptible, so a genuinely stuck
# worker keeps running to completion. To keep that from exhausting the shared
# default thread pool (asyncio.to_thread), we run console commands on a small
# dedicated, bounded pool: a leaked worker is capped, and concurrent console
# execution is bounded to a fixed number of threads regardless of reconnects.
_CONSOLE_EXECUTOR_MAX_WORKERS = 4
_console_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_console_executor_lock = threading.Lock()




def _get_console_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily create the bounded console worker pool (once per process)."""
    global _console_executor
    if _console_executor is None:
        with _console_executor_lock:
            if _console_executor is None:
                _console_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_CONSOLE_EXECUTOR_MAX_WORKERS,
                    thread_name_prefix="hermes-console",
                )
                # Ensure the pool is torn down on interpreter exit. Don't wait on
                # in-flight workers: a stuck 60s console command must not block
                # shutdown (cancel_futures drops anything not yet started).
                atexit.register(
                    lambda: _console_executor
                    and _console_executor.shutdown(wait=False, cancel_futures=True)
                )
    return _console_executor


def _console_profile_from_ws(ws: WebSocket) -> Optional[str]:
    profile = (ws.query_params.get("profile") or "").strip()
    return profile or None


def _execute_console_line(
    engine: Any,
    line: str,
    *,
    confirmed: bool,
    profile: Optional[str],
) -> Any:
    from hermes_cli.web_server import _profile_scope
    # _profile_scope swaps process-global skill module paths; keep it inside
    # the worker thread and never hold it across awaits.
    with _profile_scope(profile):
        return engine.execute(line, confirmed=confirmed)


async def _console_send(
    ws: WebSocket,
    send_lock: asyncio.Lock,
    payload: Dict[str, Any],
) -> None:
    async with send_lock:
        await ws.send_json(payload)


async def _console_send_result(
    ws: WebSocket,
    send_lock: asyncio.Lock,
    result: Any,
    *,
    command_id: int,
) -> None:
    command = result.command or ""
    status = result.status
    if status == "ok":
        if result.output:
            await _console_send(
                ws,
                send_lock,
                {
                    "type": "output",
                    "id": command_id,
                    "stream": "stdout",
                    "data": result.output,
                    "command": command,
                },
            )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "ok",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "error":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "id": command_id,
                "message": result.output or "Command failed.",
                "command": command,
            },
        )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "error",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "confirm_required":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "confirm_required",
                "id": command_id,
                "command": command,
                "message": result.confirmation_message or f"Run `{command}`?",
                "prompt": _CONSOLE_PROMPT,
            },
        )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "confirm_required",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "clear":
        await _console_send(ws, send_lock, {"type": "clear", "id": command_id})
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "clear",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "exit":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "exit",
                "command": command,
                "prompt": "",
            },
        )
        return

    await _console_send(
        ws,
        send_lock,
        {
            "type": "error",
            "id": command_id,
            "message": f"Unknown console result status: {status}",
            "command": command,
        },
    )


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


async def console_ws(ws: WebSocket) -> None:
    from hermes_cli.web_server import _DASHBOARD_EMBEDDED_CHAT_ENABLED, _resolve_profile_dir
    peer = ws.client.host if ws.client else "?"

    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        _log.info("console refused: embedded chat disabled peer=%s", peer)
        await ws.close(code=4404, reason="embedded chat disabled")
        return

    auth_reason, cred = _ws_auth_reason(ws)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning(
            "console auth rejected reason=%s mode=%s cred=%s peer=%s",
            auth_reason, mode, cred, peer,
        )
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return

    host_origin_reason = _ws_host_origin_reason(ws)
    if host_origin_reason is not None:
        _log.warning("console refused: %s peer=%s", host_origin_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(host_origin_reason))
        return

    client_reason = _ws_client_reason(ws)
    if client_reason is not None:
        _log.warning("console refused: %s", client_reason)
        await ws.close(code=4408, reason=_ws_close_reason(client_reason))
        return

    await ws.accept()

    profile = _console_profile_from_ws(ws)
    send_lock = asyncio.Lock()

    try:
        from hermes_cli.console_engine import HermesConsoleEngine

        engine = HermesConsoleEngine(output_limit=_CONSOLE_OUTPUT_LIMIT)
        if profile and profile.lower() != "current":
            _resolve_profile_dir(profile)
    except HTTPException as exc:
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "message": str(exc.detail),
                "prompt": "",
            },
        )
        await ws.close(code=4400, reason=_ws_close_reason(str(exc.detail)))
        return
    except Exception as exc:
        _log.exception("console failed to initialize")
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "message": f"Console unavailable: {exc}",
                "prompt": "",
            },
        )
        await ws.close(code=1011)
        return

    _log.info(
        "console accepted peer=%s mode=%s cred=%s profile=%s",
        peer,
        mode,
        cred,
        profile or "current",
    )
    await _console_send(
        ws,
        send_lock,
        {
            "type": "ready",
            "profile": profile or "current",
            "prompt": _CONSOLE_PROMPT,
        },
    )

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
                    functools.partial(
                        _execute_console_line,
                        engine,
                        line,
                        confirmed=confirmed,
                        profile=profile,
                    ),
                ),
                timeout=_CONSOLE_COMMAND_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            if command_id == command_generation:
                pending_confirmation = None
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "id": command_id,
                        "message": (
                            "Command timed out. Hermes Console returned to the prompt."
                        ),
                        "command": line,
                    },
                )
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "complete",
                        "id": command_id,
                        "status": "timeout",
                        "command": line,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
        except Exception as exc:
            if command_id == command_generation:
                pending_confirmation = None
                _log.exception("console command failed")
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "id": command_id,
                        "message": str(exc) or exc.__class__.__name__,
                        "command": line,
                    },
                )
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "complete",
                        "id": command_id,
                        "status": "error",
                        "command": line,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
        else:
            if command_id != command_generation:
                return
            pending_confirmation = (
                result.command if result.status == "confirm_required" else None
            )
            await _console_send_result(
                ws,
                send_lock,
                result,
                command_id=command_id,
            )
            if result.status == "exit":
                await ws.close(code=1000)
        finally:
            if command_id == command_generation:
                active_task = None

    async def start_command(line: str, *, confirmed: bool = False) -> None:
        nonlocal active_task, command_generation
        command_generation += 1
        command_id = command_generation
        active_task = asyncio.create_task(
            run_command(line, confirmed=confirmed, command_id=command_id)
        )

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                break
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break

            payload, error = _console_json_payload(msg)
            if error:
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "message": error,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue
            if payload is None:
                continue

            frame_type = str(payload.get("type") or "").strip().lower()
            if frame_type == "ping":
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "pong",
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue

            if frame_type == "cancel":
                if active_task and not active_task.done():
                    command_generation += 1
                    active_task.cancel()
                    active_task = None
                    pending_confirmation = None
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "cancelled",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                elif pending_confirmation:
                    pending_confirmation = None
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "cancelled",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                else:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "idle",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                continue

            if active_task and not active_task.done():
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "message": "A console command is already running.",
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue

            if frame_type == "confirm":
                command = str(payload.get("command") or pending_confirmation or "").strip()
                if not pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": "No command is waiting for confirmation.",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                if command != pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Confirmation does not match the pending command.",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                pending_confirmation = None
                await start_command(command, confirmed=True)
                continue

            if frame_type in {"input", "command"}:
                line = str(payload.get("line") or payload.get("command") or "").strip()
                if not line:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "ok",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                if pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": (
                                "Confirm or cancel the pending command before "
                                "running another one."
                            ),
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                await start_command(line)
                continue

            await _console_send(
                ws,
                send_lock,
                {
                    "type": "error",
                    "message": f"Unsupported console frame: {frame_type or '?'}",
                    "prompt": _CONSOLE_PROMPT,
                },
            )
    except WebSocketDisconnect:
        pass
    finally:
        if active_task and not active_task.done():
            active_task.cancel()
            try:
                await active_task
            except (asyncio.CancelledError, Exception):
                pass
