"""Action Runtime — the execution layer the Orchestration Core delegates to.

Phase 2 of docs/architecture/central-brain-openclaw.md: a single contract
(:class:`ExecutionTask` / :class:`ExecutionResult`) plus thin per-handler
adapters, so the gateway's exec handlers funnel through one honest-status schema
while their JSON-RPC wire shapes stay byte-compatible.
"""

from __future__ import annotations

from action_runtime.adapters import (
    cli_to_result,
    cli_to_wire,
    plugin_to_result,
    plugin_to_wire,
    shell_to_result,
    shell_to_wire,
    slash_to_result,
    slash_to_wire,
)
from action_runtime.contract import (
    Constraints,
    ErrorType,
    ExecError,
    ExecutionResult,
    ExecutionTask,
    NeedsInput,
    SideEffect,
    Status,
)

__all__ = [
    "Constraints",
    "ErrorType",
    "ExecError",
    "ExecutionResult",
    "ExecutionTask",
    "NeedsInput",
    "SideEffect",
    "Status",
    "cli_to_result",
    "cli_to_wire",
    "plugin_to_result",
    "plugin_to_wire",
    "shell_to_result",
    "shell_to_wire",
    "slash_to_result",
    "slash_to_wire",
]
