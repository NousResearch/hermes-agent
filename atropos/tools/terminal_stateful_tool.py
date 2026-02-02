"""
Stateful terminal tool schema.

This is a sandbox tool that routes to the sandbox server as `bash_stateful`
via ToolExecutor mapping. It exists to expose an explicit, opt-in terminal
primitive suitable for stateful workflows (e.g. tmux sessions / TUIs).
"""

from __future__ import annotations

from typing import Optional

from .base import Tool, ToolResult, ToolSchema
from .basic_tools import BashTool


class TerminalStatefulTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="terminal_stateful",
            description=(
                "Execute a command in the sandbox, allowing stateful/background processes to persist "
                "across tool calls within the same trajectory slot (e.g. tmux sessions). "
                "Use sparingly; output is still non-interactive."
            ),
            parameters={
                "command": {"type": "string", "description": "The command to execute"},
                "timeout": {
                    "type": "integer",
                    "description": "Command timeout in seconds (optional).",
                    "minimum": 1,
                },
            },
            required=["command"],
        )

    def is_available(self) -> tuple[bool, str | None]:
        return True, None

    async def execute(self, command: str, timeout: Optional[int] = None) -> ToolResult:
        # Fallback direct execution (not stateful) when used outside ToolExecutor.
        bash = BashTool(timeout=float(timeout) if timeout else 30.0)
        return await bash.execute(command=command)

