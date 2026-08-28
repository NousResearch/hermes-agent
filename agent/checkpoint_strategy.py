"""Checkpoint strategy middleware.

Decides when to save agent state for replay/recovery.

Strategies:
- checkpoint_never: no checkpoints
- checkpoint_all: after every tool call
- checkpoint_risky: after destructive operations (write, delete, API calls)
- checkpoint_smart: after uncertain operations (API calls, LLM reasoning changes)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


class CheckpointStrategy(Enum):
    """When to checkpoint agent state."""

    NEVER = "never"  # No checkpoints
    ALL = "all"  # After every tool call
    RISKY = "risky"  # Only after destructive operations
    SMART = "smart"  # After API calls, file mutations, uncertain outcomes


# Tool categories for checkpoint decisions
_DESTRUCTIVE_TOOLS = {
    "write_file",
    "patch",
    "terminal",  # Can modify files
    "delete_file",
    "move_file",
}

_API_TOOLS = {
    "web_search",
    "web_extract",
    "mail",
    "mcp_*",  # MCP tools
}

_UNCERTAIN_TOOLS = {
    "execute_code",
    "terminal",
}


def should_checkpoint(
    tool_name: str,
    result: Any,
    strategy: CheckpointStrategy = CheckpointStrategy.SMART,
) -> bool:
    """
    Decide if agent should checkpoint after this tool call.

    Args:
        tool_name: Name of the tool that just executed
        result: The result from the tool
        strategy: Which checkpoint strategy to use

    Returns:
        True if should save checkpoint, False otherwise
    """
    if strategy == CheckpointStrategy.NEVER:
        return False

    if strategy == CheckpointStrategy.ALL:
        return True

    if strategy == CheckpointStrategy.RISKY:
        # Checkpoint after destructive operations
        destructive_prefixes = tuple(f"{t}:" for t in _DESTRUCTIVE_TOOLS)
        if tool_name in _DESTRUCTIVE_TOOLS or tool_name.startswith(destructive_prefixes):
            logger.debug(f"Checkpointing after destructive tool: {tool_name}")
            return True
        return False

    if strategy == CheckpointStrategy.SMART:
        # 1. Checkpoint after destructive / mutating operations (always)
        destructive_prefixes = tuple(f"{t}:" for t in _DESTRUCTIVE_TOOLS)
        if tool_name in _DESTRUCTIVE_TOOLS or tool_name.startswith(destructive_prefixes):
            logger.debug("Checkpointing after destructive tool: %s", tool_name)
            return True

        # 2. Checkpoint after any tool that returns an error dict — state may be
        #    inconsistent and worth preserving before a retry loop starts.
        if isinstance(result, dict) and "error" in result:
            logger.debug("Checkpointing after error result from %s", tool_name)
            return True

        # 3. Successful API calls do NOT trigger a checkpoint — the result is
        #    ephemeral read-only data; nothing has changed that would need replay.
        return False

    return False


def get_checkpoint_label(tool_name: str) -> str:
    """Generate a descriptive label for a checkpoint."""
    if tool_name in _DESTRUCTIVE_TOOLS:
        return f"after_{tool_name}_mutation"
    if any(tool_name.startswith(f"{api}:") for api in _API_TOOLS):
        return f"after_{tool_name.split(':')[0]}_call"
    return f"after_{tool_name}"


class CheckpointManager:
    """Manage checkpoints throughout a conversation."""

    def __init__(self, strategy: CheckpointStrategy = CheckpointStrategy.SMART):
        self.strategy = strategy
        self.checkpoints_taken: int = 0
        self.checkpoint_history: List[Dict[str, Any]] = []

    def should_checkpoint(self, tool_name: str, result: Any) -> bool:
        """Check if should checkpoint after this tool."""
        return should_checkpoint(tool_name, result, self.strategy)

    def record_checkpoint(self, tool_name: str, checkpoint_id: str) -> None:
        """Record that a checkpoint was taken."""
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
        logger.debug(f"Recorded checkpoint #{self.checkpoints_taken}: {label}")

    def get_summary(self) -> Dict[str, Any]:
        """Get checkpoint summary."""
        return {
            "strategy": self.strategy.value,
            "checkpoints_taken": self.checkpoints_taken,
            "history": self.checkpoint_history,
        }
