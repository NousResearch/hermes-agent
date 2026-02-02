"""
tmux tool schema (sandbox).

This is a sandbox tool that provides basic tmux session control suitable for
TUI-style terminal interactions:
- send keys (arrow keys, enter, etc.)
- capture the current screen buffer

Execution is routed by ToolExecutor to the sandbox server's `tmux` backend.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolSchema


class TmuxTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="tmux",
            description=(
                "Control a per-trajectory tmux session inside the sandbox (stateful terminal). "
                "Use this for TUI-style interactions: send keys and capture the current screen."
            ),
            parameters={
                "action": {
                    "type": "string",
                    "description": "Action to perform: start | send_keys | stream | stop.",
                    "enum": ["start", "send_keys", "stream", "stop", "capture"],
                },
                "keys": {
                    "description": "Keys to send (string or list of strings) when action=send_keys.",
                },
                "block": {
                    "type": "boolean",
                    "description": "If true, wait for shell command completion (only valid at a shell prompt).",
                    "default": False,
                },
                "min_wait_s": {
                    "type": "number",
                    "description": "For non-blocking send_keys, sleep this long after sending keys (seconds).",
                    "default": 0.0,
                },
                "max_wait_s": {
                    "type": "number",
                    "description": "For blocking send_keys, max time to wait for completion (seconds).",
                },
                "capture_entire": {
                    "type": "boolean",
                    "description": "Deprecated. Streaming is preferred.",
                    "default": False,
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Max bytes to return per stream call.",
                },
                "reset": {
                    "type": "boolean",
                    "description": "If true, reset stream offset to the beginning of the asciinema recording.",
                    "default": False,
                },
                "pane_width": {
                    "type": "integer",
                    "description": "Pane width for action=start (columns).",
                    "minimum": 20,
                },
                "pane_height": {
                    "type": "integer",
                    "description": "Pane height for action=start (rows).",
                    "minimum": 10,
                },
            },
            required=["action"],
        )

    def is_available(self) -> tuple[bool, str | None]:
        return True, None

    async def execute(self, **kwargs: Dict[str, Any]) -> ToolResult:
        # This tool is intended to be executed via ToolExecutor -> sandbox server.
        # We keep a safe fallback for non-sandbox contexts.
        action = str(kwargs.get("action") or "").strip()
        return ToolResult(
            success=False,
            error=f"tmux tool must be executed in the sandbox (got action={action!r})",
        )
