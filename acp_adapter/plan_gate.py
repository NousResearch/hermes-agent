"""Plan-mode gate: while a session is in ``plan`` mode, the agent may run
read-only tools (to form a plan) but mutating tools (file edits, command
execution) are withheld until the user approves the plan by switching out of
plan mode. This is the only new agent-side capability for VS Code plan parity
(SPEC-1 FR-19, Component K).

The gate is enforced via the existing thread-local tool whitelist
(``hermes_cli.plugins.set_thread_tool_whitelist``): the ACP adapter installs the
plan-mode allow-list on the agent's executor thread, and any tool *not* in the
allow-list is blocked before execution with :data:`PLAN_BLOCK_FMT`.

Mutating-ness is derived from the single source of truth in
``acp_adapter.tools.get_tool_kind`` so the gate stays in sync with how tool
kinds are defined elsewhere.
"""
from __future__ import annotations

from typing import Iterable, Set

from acp_adapter.tools import get_tool_kind

MODE_PLAN = "plan"
MODE_OBSERVE = "observe"

# Tool kinds that change the workspace or run commands.
_MUTATING_KINDS = frozenset({"edit", "execute"})

# Observe mode withholds ONLY command execution — edits are allowed (the client
# captures them into a reviewable changeset) so a proactive background turn can
# propose fixes but can never run a command unprompted.
_EXECUTE_KINDS = frozenset({"execute"})

# Kinds that only read or observe — always safe while planning.
_READ_ONLY_KINDS = frozenset({"read", "search", "fetch", "think"})

# Side-effect-free ``other``-kind tools the agent needs to *form* a plan:
# ``todo`` emits the AgentPlanUpdate the editor renders; the rest are read/ask.
_PLANNING_SAFE_OTHER = frozenset({"todo", "memory", "session_search", "clarify"})

PLAN_BLOCK_FMT = (
    "Blocked in Plan mode: '{tool_name}' would change the workspace or run a "
    "command. Do not retry. Present your complete plan to the user as a normal "
    "message, then stop and wait — the user will review it and switch out of "
    "Plan mode to let you proceed."
)

OBSERVE_BLOCK_FMT = (
    "Blocked in Observe mode: '{tool_name}' would run a command. Observe mode "
    "may read the workspace and propose file edits (which the user reviews) but "
    "must NOT execute commands. Do not retry; finish using read and edit tools "
    "only."
)


def is_mutating_tool(tool_name: str) -> bool:
    """True if the tool edits files or executes commands."""
    return get_tool_kind(tool_name) in _MUTATING_KINDS


def plan_mode_allowed_tools(available: Iterable[str]) -> Set[str]:
    """Return the subset of ``available`` tools permitted while in Plan mode:
    read-only kinds plus a small set of planning-safe ``other`` tools.

    Mutating tools (``edit``/``execute``) are excluded so the resulting set can
    be handed to ``set_thread_tool_whitelist`` — anything outside it is blocked.
    Building an allow-list (rather than a deny-list) is fail-safe: an unforeseen
    tool defaults to blocked during planning rather than silently mutating.
    """
    allowed: Set[str] = set()
    for name in available:
        if get_tool_kind(name) in _READ_ONLY_KINDS or name in _PLANNING_SAFE_OTHER:
            allowed.add(name)
    return allowed


def observe_mode_allowed_tools(available: Iterable[str]) -> Set[str]:
    """Return the subset of ``available`` tools permitted in Observe mode:
    everything EXCEPT command execution.

    Unlike Plan mode (which withholds edits too), Observe mode lets the agent
    read AND propose edits — the client auto-captures those into a reviewable
    changeset — while ``execute``-kind tools are withheld. The result is handed
    to ``set_thread_tool_whitelist`` so any ``execute`` tool is blocked before
    it runs, giving a server-enforced "never run a command" guarantee for
    proactive background turns (defense-in-depth atop the client's auto-deny).
    """
    return {name for name in available if get_tool_kind(name) not in _EXECUTE_KINDS}
