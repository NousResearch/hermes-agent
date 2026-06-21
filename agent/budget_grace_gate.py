"""Budget-grace tool gate — deny-by-default side-effect lockout for the grace turn.

Background
----------
The tool-calling loop in ``agent/conversation_loop.py`` runs while the agent has
iteration budget remaining, plus **one optional grace turn** after the budget is
exhausted (the ``agent._budget_grace_call`` flag). The grace turn exists so the
model can write a clean final summary after it runs out of budget.

Today the grace turn would execute *whatever* tools the model returns — including
side-effecting ones (``terminal``, ``execute_code``, ``write_file``,
``delegate_task``, ``send_message``, ...). For a runaway worker that is one more
chance to spawn a subprocess or write to disk *after* its budget is already gone.

This module is the gate: during the grace turn, allow only an explicit read-only
allowlist (the reads a model needs to compose a final answer) and **refuse every
other tool — deny-by-default**, including any unrecognized / future / third-party
tool name. It is the true root-cause complement to the runaway-worker guards
(B/E/F) which cap blast radius *around* the loop; this stops the loop itself from
taking a side-effecting action past its budget.

Design notes
------------
- **Deny-by-default, NOT a denylist.** A new or unknown tool name is refused. The
  allowlist is a small, explicit, audited set of read-only finalization tools.
- **Pure / side-effect free.** Mirrors ``agent/tool_guardrails.py``: this module
  only decides; the runtime owns turning a decision into a synthetic tool result.
- **Mutating-tool overlap guard.** As defense-in-depth, any name that the existing
  guardrails consider mutating is refused even if (by mistake) it were ever added
  to the allowlist — the mutating set always wins.
"""

from __future__ import annotations

import json

from agent.tool_guardrails import MUTATING_TOOL_NAMES


# Read-only tools the model may still call during the grace turn so it can
# gather context and write a clean final summary. Deny-by-default: anything not
# in this set (including unknown/future tools) is refused.
#
# Kept deliberately small and audited. Every entry must be genuinely read-only
# (no disk writes, no subprocess, no network mutation, no message send, no
# subagent spawn). ``memory``/``todo`` are intentionally EXCLUDED: they mutate
# state and are not needed to compose a final answer.
GRACE_READONLY_TOOLS = frozenset(
    {
        "read_file",
        "search_files",
        "TaskSearch",
        "session_search",
        "mem0_search",
        "mem0_profile",
        "skill_view",
        "skills_list",
        # read-only MCP filesystem tools (mirror tool_guardrails idempotent set)
        "mcp_filesystem_read_file",
        "mcp_filesystem_read_text_file",
        "mcp_filesystem_read_multiple_files",
        "mcp_filesystem_list_directory",
        "mcp_filesystem_list_directory_with_sizes",
        "mcp_filesystem_directory_tree",
        "mcp_filesystem_get_file_info",
        "mcp_filesystem_search_files",
    }
)


def is_readonly_grace_tool(tool_name: str) -> bool:
    """Return True iff *tool_name* may execute during the budget grace turn.

    Deny-by-default: only the explicit read-only allowlist returns True. The
    mutating set always wins — a name that is both allowlisted (by mistake) and
    mutating is still refused — so the allowlist can never re-open a side effect.
    """
    if not isinstance(tool_name, str) or not tool_name:
        return False
    if tool_name in MUTATING_TOOL_NAMES:
        return False
    return tool_name in GRACE_READONLY_TOOLS


def grace_block_message(tool_name: str) -> str:
    """Human-readable reason a tool was refused during the grace turn."""
    return (
        f"Refused '{tool_name}': the iteration budget is exhausted and this is the "
        "final grace turn. Only read-only tools may run now. Write your final "
        "summary/answer directly instead of calling side-effecting tools."
    )


def grace_block_result(tool_name: str) -> str:
    """Build the synthetic role=tool content string for a grace-refused call.

    Shape mirrors the plugin/guardrail block path in the dispatchers
    (``{"error": ...}``) so the model sees a consistent blocked-tool result.
    """
    return json.dumps(
        {
            "error": grace_block_message(tool_name),
            "budget_grace_block": {"tool_name": tool_name, "reason": "budget_exhausted_grace_turn"},
        },
        ensure_ascii=False,
    )


__all__ = [
    "GRACE_READONLY_TOOLS",
    "is_readonly_grace_tool",
    "grace_block_message",
    "grace_block_result",
]
