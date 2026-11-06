"""Session source vocabulary: the ``source`` label a session row carries, and which sources have a
person reading the turn as it runs. A leaf (its one import is the stdlib-only session context), so
the ``run_agent`` facade and its siblings import it directly."""

from __future__ import annotations

from typing import Any, Optional

from gateway.session_context import get_session_env

# Sources that label the human conversation an interactive UI transport hosts. A finite ``hermes chat -q`` /
# one-shot child spawned from such a session inherits HERMES_SESSION_SOURCE (the terminal tool bridges the
# session env into child processes) but is NOT that conversation: labelling it ``tui``/``desktop`` lists it
# in the TUI/WebUI pickers as a resumable chat and lets ``hermes -c`` in the TUI continue it (#112550).
# Automation sources (tool, a2a, ...) are inherited on purpose; a Kanban worker's descendants drop its
# gateway-only ``kanban`` label (delegation_context.scrub_kanban_env).
UI_TRANSPORT_SOURCES = frozenset({"tui", "desktop"})

# Finite non-interactive CLI runs (``hermes chat -q``/``--oneshot``, ``hermes -z``) get their own source so human
# pickers hide them without title/cwd heuristics; ``hermes -c`` still treats them as CLI history.
ONESHOT_SOURCE = "oneshot"
CLI_FAMILY_SOURCES = frozenset({"cli", ONESHOT_SOURCE})

# The owner's own local surfaces: the only sources that carry OpenAI Daybreak (agent.daybreak). It is the
# approved account's own access (no third-party or downstream use); a plugin session tagged ``telegram`` is
# someone else's chat even when a local backend builds its agent.
DAYBREAK_SOURCES = UI_TRANSPORT_SOURCES | {"cli"}

# A person reads the turn as it runs: not cron, batch, a messaging chat, or a one-shot run.
ATTENDED_SOURCES = UI_TRANSPORT_SOURCES | {"cli", "acp"}


def create_surface(source: str) -> str:
    """The ``session.create`` label for a session source: ``desktop`` is the Desktop's platform name for
    the ``gui`` surface (gateway.session_policy.SURFACES), which the create contract does not list."""
    return "gui" if source == "desktop" else source


def session_source_for(platform: Optional[str]) -> str:
    source = str(get_session_env("HERMES_SESSION_SOURCE", "") or "").strip()
    single_query = get_session_env("HERMES_SINGLE_QUERY_SESSION", "") == "1"
    explicit = get_session_env("HERMES_SESSION_SOURCE_EXPLICIT", "") == "1"
    if single_query and not explicit and source in UI_TRANSPORT_SOURCES:
        source = ""
    if single_query and not source and (platform or "cli") == "cli":
        return ONESHOT_SOURCE
    return source or platform or "cli"


def turn_daybreak_surface() -> Optional[bool]:
    """The gateway's verdict on whether the agent running this turn serves the owner's own chat, for code with no
    agent in hand (a delegation backend's ``route_task``): ``True`` or ``False`` when the gateway judged it (a turn
    it runs, a ``/bg`` task), ``None`` when nothing did (classic CLI, the TUI backend, cron). A gateway-hosted chat's
    platform is the ``local`` transport, and a ``/bg`` task binds no session (it reads as ``cli``), so ``False``
    must refuse whatever the session source says."""
    from agent.subagent_lifecycle import get_active_subagent_parent
    verdict = getattr(get_active_subagent_parent(), "daybreak_surface", None)
    return verdict if isinstance(verdict, bool) else None


def is_attended(agent: Any) -> bool:
    """A delegated child answers its parent, not a person, though its copied context still carries
    the parent's source; a library caller (no platform) has no one reading either."""
    platform = getattr(agent, "platform", None)
    if not platform or platform == "subagent":
        return False
    # A gateway-admitted ``hermes chat -q`` / ``-z`` turn carries its one-shot fact on the admission,
    # not in HERMES_SINGLE_QUERY_SESSION (the daemon never had that env).
    from gateway.session_finite import finite_turn_required
    if finite_turn_required() is True:
        return False
    return session_source_for(platform) in ATTENDED_SOURCES
