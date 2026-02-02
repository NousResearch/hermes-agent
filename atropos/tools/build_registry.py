"""
Unified tool registry builder for Hermes-Agent Atropos integration.

This composes:
- sandbox tool stubs (terminal/bash/read_file/write_file + stateful terminal/tmux)
- Hermes external tools (web/vision/image/moa/skills/browser), executed via ToolServer

ToolExecutor only needs the schema + `external` routing bit; ToolServer executes
the external tools via Hermes' existing implementations.
"""

from __future__ import annotations

from typing import List, Optional

from .base import ToolRegistry
from .hermes_external_tools import build_external_tools
from .sandbox_stubs import BashTool, ReadFileTool, TerminalTool, WriteFileTool
from .terminal_stateful_tool import TerminalStatefulTool
from .tmux_tool import TmuxTool
from .toolset_resolver import resolve_multiple_toolsets


def build_tool_registry(
    *,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    tool_server_url: Optional[str] = None,
) -> ToolRegistry:
    """
    Build a ToolRegistry for AgentEnv / ToolExecutor / ToolServer.

    If `tool_server_url` is not provided, external tools will be omitted so we do
    not advertise tools that cannot execute.
    """
    enabled_toolsets = enabled_toolsets or ["default"]

    # Resolve tool names using Hermes toolsets plus Atropos additions.
    selected = set(resolve_multiple_toolsets(enabled_toolsets))
    if disabled_toolsets:
        selected -= set(resolve_multiple_toolsets(disabled_toolsets))

    reg = ToolRegistry()

    # Always register sandbox tools if selected.
    sandbox_by_name = {
        "terminal": TerminalTool(),
        "bash": BashTool(),
        "read_file": ReadFileTool(),
        "write_file": WriteFileTool(),
        "terminal_stateful": TerminalStatefulTool(),
        "tmux": TmuxTool(),
    }
    for name, tool in sandbox_by_name.items():
        if name in selected:
            reg.register(tool)

    # External tools: only include when ToolServer is configured.
    if tool_server_url:
        for tool in build_external_tools(selected_tool_names=selected):
            if tool.name in selected:
                reg.register(tool)

    return reg
