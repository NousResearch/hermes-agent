"""
Tool abstractions for atropos-agent.

Provides base Tool class, ToolCall/ToolResult types, and specialized tools.

Kept modules:
- base.py: ToolSchema, ToolCall, ToolResult, Tool ABC, ToolRegistry
- tool_executor.py: Batched execution queue with slot routing
- terminal_stateful_tool.py: Persistent terminal sessions
- tmux_tool.py: Tmux-based streaming terminal

Removed (replaced by hermes-agent equivalents):
- build_registry.py → model_tools.py + toolsets.py
- sandbox_stubs.py → atropos/backends/ execute() methods
- hermes_external_tools.py → environments/agent_loop.py handle_function_call()
- toolset_resolver.py → toolsets.py
"""

from .base import Tool, ToolCall, ToolRegistry, ToolResult, ToolSchema
from .terminal_stateful_tool import TerminalStatefulTool
from .tmux_tool import TmuxTool

__all__ = [
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "TerminalStatefulTool",
    "TmuxTool",
]
