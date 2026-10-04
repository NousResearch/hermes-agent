"""What a finite one-shot session (``hermes chat -q`` / ``--oneshot``, ``hermes -z``) does NOT do.

A one-shot run has no later session in its HERMES_HOME to learn for: the process answers one query and
exits. The interactive self-improvement loop is pure overhead there, and a measured one — across 21
one-shot benchmark trajectories the agent authored 7 new skills and patched a bundled one mid-task, 37 of
~215 tool calls were ``skill_view``/``skill_manage``, and skill text was 34% of every tool-result byte fed
back into context. Three rules follow, all keyed on the same session marker the approval gate and the
delegation dispatcher already read (``HERMES_SINGLE_QUERY_SESSION``), so interactive sessions are untouched:

* ``skill_manage`` is not offered (``skills_list``/``skill_view`` stay: reading a domain skill can still win);
* the ## Skills prompt drops the "record it / patch it / offer to save" coaching and the "load process skills
  even for tasks you already know" push, keeping only "load a skill when it adds knowledge you lack";
* delegation is capped per session (``delegation.oneshot_max_children``): subagents each re-pay a cold
  system prompt and re-explore the repo, and the observed spawns were mostly "independent review of my own
  work" rather than parallel work.

An install whose one-shot runs ARE a longer-lived identity (an orchestrator that resumes the same session
once per wake) opts back into learning with ``auxiliary.background_review.oneshot_learning``: skill_manage
stays, the post-turn review may fire, and the CLI waits a bounded time for it at exit.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

ONESHOT_HIDDEN_TOOLS = frozenset({"skill_manage"})


def is_single_query_session() -> bool:
    """The finite ``-q`` marker, read through the session env so gateway-bound sessions never see it."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        import os
        get_session_env = os.environ.get
    return str(get_session_env("HERMES_SINGLE_QUERY_SESSION", "") or "") == "1"


def oneshot_learning() -> bool:
    """A one-shot session whose install opted into learning (``auxiliary.background_review.oneshot_learning``):
    the run is one turn of a longer-lived identity (an orchestrator resuming the same session per wake),
    so ``skill_manage`` stays available for the post-turn review. Never raises."""
    if not is_single_query_session():
        return False
    try:
        from agent.background_review import oneshot_learning_enabled
        return oneshot_learning_enabled()
    except Exception:
        return False


def prune_oneshot_tools(tools: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """*tools* minus ``ONESHOT_HIDDEN_TOOLS``; identity when the session is not one-shot or opted into learning."""
    tools = list(tools)
    if not is_single_query_session() or oneshot_learning():
        return tools
    return [t for t in tools if (t.get("function") or {}).get("name") not in ONESHOT_HIDDEN_TOOLS]


ONESHOT_SKILLS_LOAD_GUIDANCE = (
    "## Skills\n"
    "Scan the skills below and load one with skill_view(name) only when it carries domain knowledge you lack "
    "for THIS task (an API, a tool's commands, a project's conventions). Do not load general process skills "
    "(testing, debugging, review methodology) for work you already know how to do, and do not create or edit "
    "skills: this is a one-shot run with no later session to reuse them.\n"
)

# Same economy, but the session continues on a later run: a skill that was wrong may be fixed.
ONESHOT_LEARNING_SKILLS_LOAD_GUIDANCE = (
    "## Skills\n"
    "Scan the skills below and load one with skill_view(name) only when it carries domain knowledge you lack "
    "for THIS task (an API, a tool's commands, a project's conventions). Do not load general process skills "
    "(testing, debugging, review methodology) for work you already know how to do. If a skill you loaded had "
    "wrong commands or missing steps, fix it with skill_manage(action='patch'); a review after this run "
    "handles anything else worth keeping.\n"
)


def oneshot_skills_guidance() -> str:
    """The one-shot ## Skills header for this session, or ``""`` when the session is not one-shot."""
    if not is_single_query_session():
        return ""
    return ONESHOT_LEARNING_SKILLS_LOAD_GUIDANCE if oneshot_learning() else ONESHOT_SKILLS_LOAD_GUIDANCE
