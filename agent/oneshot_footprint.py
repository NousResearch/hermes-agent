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
"""
from __future__ import annotations


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
