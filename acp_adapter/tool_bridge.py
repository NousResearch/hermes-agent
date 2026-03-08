"""ToolBridge — delegates file/terminal tool calls to the editor via ACP.

When Hermes runs inside an ACP-aware editor (Zed, JetBrains, etc.), file
and terminal operations should go through the editor so it can display diffs,
track changes, and manage terminals visually.  All other Hermes tools
(web_search, memory, delegate_task, …) continue to run locally.

Usage from the agent thread (synchronous):
    bridge = ToolBridge(conn, session_id, loop)
    result_json = bridge.dispatch("read_file", {"file_path": "/tmp/foo.py"})
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tools whose execution is delegated to the editor.
DELEGATED_TOOLS = {"read_file", "write_file", "patch", "terminal"}


class ToolBridge:
    """Routes delegated tool calls to the ACP client (editor).

    All public methods are *synchronous* — they use
    ``asyncio.run_coroutine_threadsafe`` to dispatch to the event loop
    that owns the ACP connection, then block until the result is ready.
    This lets them be called directly from the ``AIAgent`` thread.
    """

    DELEGATED_TOOLS = DELEGATED_TOOLS

    def __init__(self, conn: Any, session_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._conn = conn
        self._session_id = session_id
        self._loop = loop

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call to the editor and return the JSON result string."""
        handler = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "patch": self._patch_file,
            "terminal": self._terminal,
        }.get(tool_name)

        if handler is None:
            return json.dumps({"error": f"Unknown delegated tool: {tool_name}"})

        try:
            return handler(args)
        except Exception as exc:
            logger.exception("ToolBridge.dispatch(%s) failed", tool_name)
            return json.dumps({"error": f"ACP tool delegation failed: {exc}"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        """Schedule *coro* on the ACP event loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=120)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _read_file(self, args: dict[str, Any]) -> str:
        path = args.get("file_path") or args.get("path", "")
        line = args.get("line")
        limit = args.get("limit")

        resp = self._run(
            self._conn.read_text_file(
                path=path,
                session_id=self._session_id,
                line=line,
                limit=limit,
            )
        )
        content = resp.content if hasattr(resp, "content") else str(resp)
        return json.dumps({"success": True, "content": content})

    def _write_file(self, args: dict[str, Any]) -> str:
        path = args.get("file_path") or args.get("path", "")
        content = args.get("content", "")

        self._run(
            self._conn.write_text_file(
                path=path,
                content=content,
                session_id=self._session_id,
            )
        )
        return json.dumps({"success": True, "message": f"Wrote {len(content)} chars to {path}"})

    def _patch_file(self, args: dict[str, Any]) -> str:
        path = args.get("file_path") or args.get("path", "")
        old_text = args.get("old_string") or args.get("old_text", "")
        new_text = args.get("new_string") or args.get("new_text", "")

        # Read current content via ACP
        resp = self._run(
            self._conn.read_text_file(path=path, session_id=self._session_id)
        )
        current = resp.content if hasattr(resp, "content") else str(resp)

        if old_text not in current:
            return json.dumps({
                "success": False,
                "error": f"old_string not found in {path}",
            })

        updated = current.replace(old_text, new_text, 1)

        self._run(
            self._conn.write_text_file(
                path=path,
                content=updated,
                session_id=self._session_id,
            )
        )
        return json.dumps({"success": True, "message": f"Patched {path}"})

    def _terminal(self, args: dict[str, Any]) -> str:
        command = args.get("command", "")
        cwd = args.get("working_directory") or args.get("cwd")

        # Create terminal and run command
        create_resp = self._run(
            self._conn.create_terminal(
                command=command,
                session_id=self._session_id,
                cwd=cwd,
            )
        )

        terminal_id = create_resp.terminal_id if hasattr(create_resp, "terminal_id") else str(create_resp)

        # Wait for the command to finish
        exit_resp = self._run(
            self._conn.wait_for_terminal_exit(
                session_id=self._session_id,
                terminal_id=terminal_id,
            )
        )

        exit_code = exit_resp.exit_code if hasattr(exit_resp, "exit_code") else None

        # Retrieve output
        output_resp = self._run(
            self._conn.terminal_output(
                session_id=self._session_id,
                terminal_id=terminal_id,
            )
        )

        output = output_resp.output if hasattr(output_resp, "output") else str(output_resp)

        # Release the terminal
        try:
            self._run(
                self._conn.release_terminal(
                    session_id=self._session_id,
                    terminal_id=terminal_id,
                )
            )
        except Exception:
            logger.debug("Failed to release terminal", exc_info=True)

        return json.dumps({
            "success": True,
            "exit_code": exit_code,
            "output": output,
            "command": command,
        })
