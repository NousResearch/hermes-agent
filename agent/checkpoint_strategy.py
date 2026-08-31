"""Checkpoint strategy: decide *when* to trigger agent state persistence.

This is a pure decision layer.  It returns ``True``/``False`` and never
calls the checkpointer directly, so different loop implementations can
plug in their own persistence backend (e.g. the existing
``tools.checkpoint_manager.CheckpointManager``).

Four strategies:

| Strategy | Triggers checkpoint                                 |
|----------|-----------------------------------------------------|
| NEVER    | Never                                               |
| ALL      | After every tool call                               |
| RISKY    | After destructive/mutating operations only          |
| SMART    | After destructive ops OR error results (see below)  |

SMART rationale: checkpoint after any state-changing op (write, patch,
terminal, move/delete) and after any tool that returns an error dict —
the agent may be in an inconsistent state and worth snapshotting before
a retry loop starts.  Successful read-only API calls do *not* trigger a
checkpoint; the result is ephemeral data and nothing has changed.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointStrategy(Enum):
    """When to checkpoint agent state."""

    NEVER = "never"   # No checkpoints
    ALL = "all"       # After every tool call
    RISKY = "risky"   # Only after destructive/mutating operations
    SMART = "smart"   # After destructive ops or error results


# ---------------------------------------------------------------------------
# Tool category sets
# ---------------------------------------------------------------------------

_DESTRUCTIVE_TOOLS = {
    "write_file",
    "patch",
    "terminal",   # can modify files
    "delete_file",
    "move_file",
}

# MCP tools use the naming convention  mcp_<server_id>_<tool_name>
# Match by prefix rather than a literal "mcp_*" entry.
_MCP_PREFIX = "mcp_"

_UNCERTAIN_TOOLS = {
    "execute_code",
    "terminal",
}


def _is_mcp_tool(tool_name: str) -> bool:
    """Return True if *tool_name* looks like a MCP tool (mcp_<id>_<name>)."""
    return tool_name.startswith(_MCP_PREFIX)


def should_checkpoint(
    tool_name: str,
    result: Any,
    strategy: CheckpointStrategy = CheckpointStrategy.SMART,
) -> bool:
    """Decide whether to checkpoint after a tool call.

    Args:
        tool_name: Name of the tool that just executed.
        result:    The result returned by the tool.
        strategy:  Which checkpoint strategy to apply.

    Returns:
        True if the caller should persist agent state, False otherwise.
    """
    if strategy == CheckpointStrategy.NEVER:
        return False

    if strategy == CheckpointStrategy.ALL:
        return True

    if strategy == CheckpointStrategy.RISKY:
        if tool_name in _DESTRUCTIVE_TOOLS:
            logger.debug("Checkpointing after destructive tool: %s", tool_name)
            return True
        return False

    if strategy == CheckpointStrategy.SMART:
        # 1. Checkpoint after any destructive / mutating operation.
        if tool_name in _DESTRUCTIVE_TOOLS:
            logger.debug("Checkpointing after destructive tool: %s", tool_name)
            return True

        # 2. Checkpoint when a tool returns an error dict — agent state may be
        #    inconsistent and worth preserving before a retry loop begins.
        if isinstance(result, dict) and "error" in result:
            logger.debug("Checkpointing after error result from %s", tool_name)
            return True

        # 3. Everything else (successful reads, API calls, MCP queries) is
        #    ephemeral data; no state has changed, no checkpoint needed.
        return False

    return False


def get_checkpoint_label(tool_name: str) -> str:
    """Return a human-readable label for a checkpoint taken after *tool_name*."""
    if tool_name in _DESTRUCTIVE_TOOLS:
        return "after_%s_mutation" % tool_name
    if _is_mcp_tool(tool_name):
        # e.g. "mcp_github_list_prs" → "after_mcp_github_call"
        parts = tool_name.split("_", 2)
        server_id = parts[1] if len(parts) >= 2 else tool_name
        return "after_mcp_%s_call" % server_id
    return "after_%s" % tool_name


class CheckpointManager:
    """Track which checkpoints have been taken in a conversation.

    This is a *record-keeping* companion to the decision layer; the actual
    filesystem snapshot is delegated to ``tools.checkpoint_manager.CheckpointManager``
    (which is owned by the agent and called from tool_executor).  This class
    just keeps a lightweight history so callers can audit what happened.
    """

    def __init__(self, strategy: CheckpointStrategy = CheckpointStrategy.SMART):
        self.strategy = strategy
        self.checkpoints_taken: int = 0
        self.checkpoint_history: List[Dict[str, Any]] = []

    def should_checkpoint(self, tool_name: str, result: Any) -> bool:
        """Return True if agent state should be persisted after this tool."""
        return should_checkpoint(tool_name, result, self.strategy)

    def record_checkpoint(self, tool_name: str, checkpoint_id: str) -> None:
        """Record that a checkpoint was taken (called by the loop after persisting)."""
        self.checkpoints_taken += 1
        label = get_checkpoint_label(tool_name)
        self.checkpoint_history.append(
            {
                "checkpoint_id": checkpoint_id,
                "label": label,
                "tool": tool_name,
                "sequence": self.checkpoints_taken,
            }
        )
        logger.debug("Recorded checkpoint #%d: %s", self.checkpoints_taken, label)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all checkpoints taken so far."""
        return {
            "strategy": self.strategy.value,
            "checkpoints_taken": self.checkpoints_taken,
            "history": self.checkpoint_history,
        }
