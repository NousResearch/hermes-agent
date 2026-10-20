"""Client-fs bridge: route file tool calls to the ACP client's filesystem.

When Hermes runs in a container and the IDE (JetBrains) runs on the host,
the agent's file tools cannot see the project: ``/home/tim/...`` paths do
not exist in-container. The ACP protocol solves this with reverse client
methods — ``fs/read_text_file`` and ``fs/write_text_file`` — which the
client (the IDE, which owns the host paths) services.

This module binds a :class:`ClientFSBridge` in a ContextVar for the
duration of one ACP agent run (mirroring the edit-approval requester
pattern in :mod:`acp_adapter.edit_approval`). CLI, gateway, and other
sessions leave it unset and are completely unaffected.

Routing rule (self-contained, no cwd assumption): a resolved target path
is *host-side* when neither the path nor its parent directory exists in
the container filesystem. New-file creation in a host project (parent
absent) routes to the IDE; writes into existing container dirs
(``/opt/obsidian``, ``/tmp``, ...) stay local.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CLIENT_FS_BRIDGE: ContextVar["ClientFSBridge | None"] = ContextVar(
    "ACP_CLIENT_FS_BRIDGE",
    default=None,
)


@dataclass(frozen=True)
class ClientFSBridge:
    """Everything the routing helpers need to reach the client's filesystem."""

    conn: Any  # acp.Client — exposes fs/* and terminal/* reverse methods
    session_id: str
    loop: asyncio.AbstractEventLoop
    fs_read: bool
    fs_write: bool
    project_cwd: str | None = None  # IDE project root (session cwd), for logging
    terminal: bool = False          # client advertises terminal/* methods

    def has(self, kind: str) -> bool:
        if self.conn is None:
            return False
        if kind == "read":
            return self.fs_read
        if kind == "write":
            return self.fs_write
        return False


def set_client_fs_bridge(bridge: Optional[ClientFSBridge]) -> Token:
    return _CLIENT_FS_BRIDGE.set(bridge)


def reset_client_fs_bridge(token: Token) -> None:
    _CLIENT_FS_BRIDGE.reset(token)


def get_client_fs_bridge() -> Optional[ClientFSBridge]:
    return _CLIENT_FS_BRIDGE.get()


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

# System prefixes that must never be written through the IDE, guarding the
# "parent absent in container" case for system directories a host user
# would never expect the IDE to write to. Project roots under
# /home/<user>/... are intentionally NOT blocked: that is exactly the
# absent-parent case the bridge exists for, and every routed write still
# passes the editor's diff-approval gate first.
_ROUTE_BLOCK_PREFIXES = (
    "/etc/", "/usr/", "/bin/", "/sbin/", "/boot/", "/proc/", "/sys/",
    "/dev/", "/var/", "/opt/", "/root/", "/run/", "/lib/", "/lib64/",
    "/srv/", "/mnt/", "/media/", "/snap/",
)


def is_host_side_path(resolved: str) -> bool:
    """Return True when *resolved* points outside the container filesystem.

    A path is host-side when it does not exist here AND its parent
    directory does not exist here either. A brand-new file in an existing
    container directory (e.g. ``/opt/obsidian/.../new.md``) is local.
    """
    if not resolved or not os.path.isabs(resolved):
        return False
    if os.path.exists(resolved):
        return False
    parent = os.path.dirname(resolved)
    if parent and os.path.isdir(parent):
        return False
    for prefix in _ROUTE_BLOCK_PREFIXES:
        if resolved.startswith(prefix):
            return False
    return True


# ---------------------------------------------------------------------------
# Sync bridges into the event loop (tools run in executor threads)
# ---------------------------------------------------------------------------

def bridge_read_text(bridge: ClientFSBridge, path: str) -> Optional[str]:
    """Fetch a file's text through the client's fs/read_text_file.

    Returns the content, or None when the client cannot serve it.
    """
    if not bridge.has("read"):
        return None
    read_fn = getattr(bridge.conn, "read_text_file", None)
    if read_fn is None:
        return None
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        read_fn(path=path, session_id=bridge.session_id),
        bridge.loop,
        logger=logger,
        log_message="client fs/read_text_file: failed to schedule on loop",
    )
    if future is None:
        return None
    try:
        resp = future.result(timeout=60.0)
    except Exception as exc:
        logger.warning("client fs/read_text_file failed for %s: %s", path, exc)
        return None
    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        content = resp.get("content")
    if isinstance(content, str) and content:
        logger.info(
            "ACP file read via client fs/read_text_file: %s (%d chars)",
            path, len(content),
        )
        return content
    logger.debug("client fs/read_text_file returned no content for %s", path)
    return None


def bridge_write_text(bridge: ClientFSBridge, path: str, content: str) -> None:
    """Write a file through the client's fs/write_text_file.

    Raises on failure so callers can report a tool error.
    """
    if not bridge.has("write"):
        raise RuntimeError("client does not advertise fs.write_text_file")
    write_fn = getattr(bridge.conn, "write_text_file", None)
    if write_fn is None:
        raise RuntimeError("client connection has no write_text_file method")
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        write_fn(content=content, path=path, session_id=bridge.session_id),
        bridge.loop,
        logger=logger,
        log_message="client fs/write_text_file: failed to schedule on loop",
    )
    if future is None:
        raise RuntimeError("could not schedule fs/write_text_file on the event loop")
    future.result(timeout=60.0)  # raises on client/protocol error
    logger.info(
        "ACP file written via client fs/write_text_file: %s (%d chars)",
        path, len(content),
    )


# ---------------------------------------------------------------------------
# Tool-call routing (called from model_tools.handle_function_call)
# ---------------------------------------------------------------------------

def _number_lines(text: str, start_line: int) -> str:
    lines = text.splitlines()
    return "\n".join(f"{start_line + i}|{line}" for i, line in enumerate(lines))


def _resolve_for_task(path: str, task_id: str) -> str:
    """Resolve a tool path the same way the file tools do (lazy import)."""
    from tools.file_tools import _resolve_path_for_task

    return str(_resolve_path_for_task(path, task_id or "default"))


def _resolve_args_path(function_name: str, function_args: dict) -> str | None:
    """Extract the target file path from a file-tool call, or None."""
    if function_name in ("write_file", "read_file"):
        return str(function_args.get("path") or "") or None
    if function_name == "patch":
        if function_args.get("mode", "replace") != "replace":
            return None  # V4A multi-file patches are not routed (yet)
        return str(function_args.get("path") or "") or None
    return None


def route_file_tool_to_client(
    function_name: str,
    function_args: dict,
    task_id: str,
) -> str | None:
    """Execute a file-tool call through the IDE client when the target is
    host-side. Returns a JSON tool-result string on success, a JSON
    error string when the client cannot serve it, or None to let the
    normal (container-local) dispatch proceed.

    Called from model_tools.handle_function_call after the edit-approval
    gate has already approved the mutation (write_file / patch).
    """
    bridge = get_client_fs_bridge()
    if bridge is None:
        return None
    if function_name not in ("read_file", "write_file", "patch"):
        return None

    raw_path = _resolve_args_path(function_name, function_args)
    if not raw_path:
        return None
    try:
        resolved = _resolve_for_task(raw_path, task_id)
    except Exception:
        resolved = raw_path

    if not is_host_side_path(resolved):
        return None  # container-local: normal dispatch

    # ---- read_file -----------------------------------------------------
    if function_name == "read_file":
        if not bridge.has("read"):
            return json.dumps({
                "error": (
                    f"Cannot read '{raw_path}': the file is not visible from "
                    "this agent (it lives on the editor's machine) and the "
                    "IDE client does not advertise fs.read_text_file."
                ),
            }, ensure_ascii=False)
        offset = int(function_args.get("offset", 1) or 1)
        limit = int(function_args.get("limit", 500) or 500)
        content = bridge_read_text(bridge, resolved)
        if content is None:
            return json.dumps({
                "error": f"Could not read IDE file: {resolved}",
            }, ensure_ascii=False)
        lines = content.splitlines()
        total_lines = len(lines)
        end_line = offset + limit - 1
        page = lines[offset - 1:end_line]
        result: dict[str, Any] = {
            "content": _number_lines("\n".join(page), offset) if page else "",
            "total_lines": total_lines,
            "truncated": total_lines > end_line,
            "ide_read": True,
        }
        if result["truncated"]:
            result["hint"] = (
                f"Use offset={end_line + 1} to continue reading "
                f"(showing {offset}-{min(end_line, total_lines)} of {total_lines} lines)"
            )
        return json.dumps(result, ensure_ascii=False)

    # ---- write_file / patch (replace) ----------------------------------
    if not bridge.has("write"):
        return json.dumps({
            "error": (
                f"Cannot write '{raw_path}': the file lives on the editor's "
                "machine and the IDE client does not advertise "
                "fs.write_text_file. The editor cannot apply this edit."
            ),
        }, ensure_ascii=False)

    if function_name == "write_file":
        new_content = str(function_args.get("content") or "")
    else:  # patch (replace): compute the full new text. The edit-approval
        # gate already ran this exact computation for its diff, so the
        # write matches what the editor approved.
        old_text = bridge_read_text(bridge, resolved)
        if old_text is None:
            return json.dumps({
                "error": (
                    f"Cannot patch '{raw_path}': could not fetch the current "
                    "content from the editor to compute the edit."
                ),
            }, ensure_ascii=False)
        from tools.fuzzy_match import fuzzy_find_and_replace

        new_content, match_count, _strategy, error = fuzzy_find_and_replace(
            old_text,
            str(function_args.get("old_string") or ""),
            str(function_args.get("new_string") or ""),
            bool(function_args.get("replace_all", False)),
        )
        if error or match_count == 0:
            return json.dumps({
                "error": error or f"Could not find match for old_string in {raw_path}",
            }, ensure_ascii=False)

    try:
        bridge_write_text(bridge, resolved, new_content)
    except Exception as exc:
        return json.dumps({
            "error": f"IDE write failed for {resolved}: {exc}",
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "files_modified": [resolved],
        "resolved_path": resolved,
        "ide_write": True,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Terminal execution through the IDE (terminal/create)
# ---------------------------------------------------------------------------

def run_command_via_client(
    bridge: "ClientFSBridge",
    command: str,
    workdir: str | None = None,
    timeout: float = 180.0,
) -> "str | None":
    """Run a shell command in the IDE's terminal via ``terminal/create``.

    Mirrors the ACP ``terminals`` flow (create -> wait_for_exit -> output ->
    release) and the reference implementation used by other ACP agents
    against JetBrains. Returns a JSON result string (success or error) when
    the client services the command, or ``None`` when the client cannot
    serve it at all (no connection / no ``create_terminal`` method) so the
    caller can fall back to container-local execution.

    The caller is expected to have already run the dangerous-command guard
    and the ACP permission request; this only performs the execution in the
    host terminal.
    """
    conn = bridge.conn
    create_fn = getattr(conn, "create_terminal", None)
    output_fn = getattr(conn, "terminal_output", None)
    wait_fn = getattr(conn, "wait_for_terminal_exit", None)
    release_fn = getattr(conn, "release_terminal", None)
    if conn is None or create_fn is None or output_fn is None:
        return None  # client can't serve -> caller falls back to local

    from agent.async_utils import safe_schedule_threadsafe

    loop = bridge.loop
    sid = bridge.session_id

    def _sched(coro):
        future = safe_schedule_threadsafe(
            coro, loop, logger=logger,
            log_message="client terminal: failed to schedule on loop",
        )
        if future is None:
            raise RuntimeError("could not schedule a terminal call on the event loop")
        return future

    terminal_id = None
    try:
        create_kw = {"command": command, "session_id": sid}
        if workdir:
            create_kw["cwd"] = workdir
        try:
            resp = _sched(create_fn(**create_kw)).result(timeout=60.0)
        except Exception as exc:
            logger.warning("terminal/create failed for %r: %s", command[:120], exc)
            return json.dumps(
                {"error": f"IDE terminal/create failed: {exc}", "status": "blocked"},
                ensure_ascii=False,
            )
        terminal_id = getattr(resp, "terminal_id", None)
        if not terminal_id:
            return json.dumps(
                {"error": "IDE terminal/create returned no terminalId", "status": "blocked"},
                ensure_ascii=False,
            )
        logger.info(
            "ACP command routed to IDE terminal %s (cwd=%s): %s",
            terminal_id, workdir or "<default>", command[:120],
        )

        # Wait for the command to exit, bounded by the caller's timeout.
        exit_resp = None
        timed_out = False
        if wait_fn is not None:
            wait_future = _sched(wait_fn(session_id=sid, terminal_id=terminal_id))
            try:
                exit_resp = wait_future.result(timeout=timeout)
            except Exception as exc:
                wait_future.cancel()
                timed_out = True
                logger.warning(
                    "terminal/wait_for_exit timed out/failed for %s: %s", terminal_id, exc
                )
        else:
            timed_out = True

        out_resp = _sched(output_fn(session_id=sid, terminal_id=terminal_id)).result(timeout=60.0)
        output = getattr(out_resp, "output", "") or ""
        truncated = bool(getattr(out_resp, "truncated", False))

        exit_code = getattr(exit_resp, "exit_code", None)
        signal = getattr(exit_resp, "signal", None)
        if exit_code is None:
            est = getattr(out_resp, "exit_status", None)
            if est is not None:
                exit_code = getattr(est, "exit_code", None)
                signal = getattr(est, "signal", None)

        if timed_out:
            return json.dumps({
                "output": output,
                "exit_code": -1,
                "error": f"Command timed out after {timeout:.0f}s in the IDE terminal",
                "status": "timeout",
                "truncated": truncated,
                "ide_terminal": True,
            }, ensure_ascii=False)

        ok = (exit_code or 0) == 0
        return json.dumps({
            "output": output,
            "exit_code": exit_code if exit_code is not None else 0,
            "error": "" if ok else (f"exited with code {exit_code}" if exit_code is not None else f"killed by signal {signal}"),
            "status": "success" if ok else "nonzero_exit",
            "truncated": truncated,
            "signal": signal,
            "ide_terminal": True,
        }, ensure_ascii=False)
    finally:
        if terminal_id and release_fn is not None:
            try:
                _sched(release_fn(session_id=sid, terminal_id=terminal_id)).result(timeout=30.0)
            except Exception:
                logger.debug("terminal/release failed for %s", terminal_id, exc_info=True)
