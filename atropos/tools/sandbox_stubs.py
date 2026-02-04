"""
Sandbox tool stubs for Atropos ToolExecutor.

These tools are executed inside the sandbox containers via:
ToolExecutor -> SlotPool -> sandbox_server.py

They intentionally do NOT execute anything on the host process. If they are
called directly (outside ToolExecutor), they return a clear error.
"""

from __future__ import annotations

from typing import Optional

from .base import Tool, ToolResult, ToolSchema


class TerminalTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="terminal",
            description=(
                "Execute a command inside the sandbox slot workspace and return stdout/stderr. "
                "Filesystem persists within a trajectory slot. Background processes are not supported "
                "in stateless mode. Commands run under POSIX /bin/sh and each tool call runs in a fresh "
                "shell (no persisted env vars). Avoid bash-only syntax like `source`; prefer `. .venv/bin/activate` "
                "or invoke `.venv/bin/python ...` directly."
            ),
            parameters={
                "command": {"type": "string", "description": "The command to execute"},
                "timeout": {
                    "type": "integer",
                    "description": "Command timeout in seconds (optional).",
                    "minimum": 1,
                },
                "background": {
                    "type": "boolean",
                    "description": "Not supported in sandbox terminal (always false).",
                    "default": False,
                },
            },
            required=["command"],
            external=False,
        )

    async def execute(self, **_kwargs) -> ToolResult:
        return ToolResult(
            success=False,
            error="terminal must be executed via ToolExecutor inside the sandbox",
        )


class BashTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="bash",
            description="Execute a bash command inside the sandbox slot workspace.",
            parameters={"command": {"type": "string", "description": "The bash command to execute"}},
            required=["command"],
            external=False,
        )

    async def execute(self, **_kwargs) -> ToolResult:
        return ToolResult(success=False, error="bash must be executed via ToolExecutor inside the sandbox")


class ReadFileTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read a file from the sandbox slot workspace.",
            parameters={"path": {"type": "string", "description": "Path to the file"}},
            required=["path"],
            external=False,
        )

    async def execute(self, **_kwargs) -> ToolResult:
        return ToolResult(success=False, error="read_file must be executed via ToolExecutor inside the sandbox")


class WriteFileTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write a file into the sandbox slot workspace.",
            parameters={
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "File content"},
            },
            required=["path", "content"],
            external=False,
        )

    async def execute(self, **_kwargs) -> ToolResult:
        return ToolResult(success=False, error="write_file must be executed via ToolExecutor inside the sandbox")
