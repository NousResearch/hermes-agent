"""What a finite one-shot session (``hermes chat -q`` / ``--oneshot``, ``hermes -z``) does NOT do.

A one-shot run has no later session in its HERMES_HOME to learn for: the process answers one query and
exits. The interactive self-improvement loop is pure overhead there, and a measured one — across 21
one-shot benchmark trajectories the agent authored 7 new skills and patched a bundled one mid-task, 37 of
~215 tool calls were ``skill_view``/``skill_manage``, and skill text was 34% of every tool-result byte fed
back into context. Two rules follow, both keyed on the same session marker the approval gate and the
delegation dispatcher already read (``HERMES_SINGLE_QUERY_SESSION``), so interactive sessions are untouched
(``skill_manage`` itself stays offered: a one-shot may still author skills):

* the ## Skills prompt drops the "record it / patch it / offer to save" coaching and the "load process skills
  even for tasks you already know" push, keeping only "load a skill when it adds knowledge you lack";
* delegation is capped per session (``delegation.oneshot_max_children``): subagents each re-pay a cold
  system prompt and re-explore the repo, and the observed spawns were mostly "independent review of my own
  work" rather than parallel work.

Nobody answers or steers a one-shot either, so it also drops what only an interactive user can use:
``clarify`` (its toolset is disabled for the session), the mid-turn steering briefing in the system prompt,
and ``delegate_task``'s live-control surface (children are joined inside the call, so ``action``
list/steer/stop never has a running child to reach).
"""
from __future__ import annotations

from typing import Any

# Toolsets a one-shot session never gets (``toolsets.session_disabled_toolsets(oneshot=True)``): the toolset is
# the surface gate, so the tool list, the tool_search bridge and delegate children all lose them together.
ONESHOT_DISABLED_TOOLSETS = ("clarify",)


def is_single_query_session() -> bool:
    """The finite ``-q`` marker, read through the session env so gateway-bound sessions never see it.

    A gateway-owned ``chat -q`` / ``-z`` session never has that env (only the client process set it): its
    one-shot fact is the frozen launch policy's ``oneshot`` source, a property of the SESSION, not of one
    admission, so the cached agent's tools[] and system prompt stay byte-stable across its turns."""
    try:
        from gateway.session_policy import active_policy
    except ImportError:  # install without the gateway package: no owner-bound session
        pass
    else:
        policy = active_policy()
        if policy is not None:
            return policy.source == "oneshot"
    try:
        from gateway.session_context import get_session_env
    except Exception:
        import os
        get_session_env = os.environ.get
    return str(get_session_env("HERMES_SINGLE_QUERY_SESSION", "") or "") == "1"


ONESHOT_SKILLS_LOAD_GUIDANCE = (
    "## Skills\n"
    "Scan the skills below and load one with skill_view(name) only when it carries domain knowledge you lack "
    "for THIS task (an API, a tool's commands, a project's conventions). Do not load general process skills "
    "(testing, debugging, review methodology) for work you already know how to do.\n"
)


def agent_is_oneshot(agent: Any) -> bool:
    """:func:`is_single_query_session`, frozen on *agent* at its first tool build so a later MCP refresh or
    prompt build answers the same even outside the turn's policy scope (tools[] and the prompt stay byte-stable)."""
    cached = getattr(agent, "_oneshot_session", None)
    if not isinstance(cached, bool):
        cached = is_single_query_session()
        agent._oneshot_session = cached
    return cached


def oneshot_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*tools* with ``delegate_task`` swapped for its one-shot variant (no live-control parameters)."""
    from tools.delegate_tool import oneshot_delegate_function
    return [
        {**tool, "function": oneshot_delegate_function(tool["function"])}
        if (tool.get("function") or {}).get("name") == "delegate_task" else tool
        for tool in tools
    ]
