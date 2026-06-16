"""``hermes action`` subcommand — review/approve deferred tool-gate actions.

The generic tool-approval gate (``tools/tool_gate.py``) stages designated tool
calls to ``pending/actions/`` and opens a Kanban approval card when no live
interactive channel is available. This command is the terminal-side review
affordance referenced by those cards: list staged actions, inspect one, and
approve (spawn a one-shot execution worker) or reject (discard).
"""

from __future__ import annotations

from typing import Callable


def build_action_parser(subparsers, *, cmd_action: Callable) -> None:
    """Attach the ``action`` subcommand to ``subparsers``."""
    action_parser = subparsers.add_parser(
        "action",
        help="Review and approve deferred (staged) tool-gate actions",
        description=(
            "Human-in-the-loop review for tool calls held by the approval "
            "gate (approvals.tool_gate). Approving spawns a one-shot worker "
            "that replays the staged call; rejecting discards it."
        ),
    )
    action_sub = action_parser.add_subparsers(dest="action_command")

    action_sub.add_parser("list", help="List pending staged actions")

    _show = action_sub.add_parser("show", help="Show one staged action")
    _show.add_argument("pending_id", help="The pending action id")

    _approve = action_sub.add_parser(
        "approve", help="Approve a staged action (queues it for execution)")
    _approve.add_argument("pending_id", help="The pending action id")

    _reject = action_sub.add_parser(
        "reject", help="Reject and discard a staged action")
    _reject.add_argument("pending_id", help="The pending action id")

    action_parser.set_defaults(func=cmd_action)
