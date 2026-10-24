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
# Automation sources (kanban, tool, cron, a2a, ...) are inherited on purpose.
UI_TRANSPORT_SOURCES = frozenset({"tui", "desktop"})

# Finite non-interactive CLI runs (``hermes chat -q``/``--oneshot``, ``hermes -z``) get their own source so human
# pickers hide them without title/cwd heuristics; ``hermes -c`` still treats them as CLI history.
ONESHOT_SOURCE = "oneshot"
CLI_FAMILY_SOURCES = frozenset({"cli", ONESHOT_SOURCE})

# A person reads the turn as it runs: not cron, batch, a messaging chat, or a one-shot run.
ATTENDED_SOURCES = UI_TRANSPORT_SOURCES | {"cli", "acp"}


def session_source_for(platform: Optional[str]) -> str:
    source = str(get_session_env("HERMES_SESSION_SOURCE", "") or "").strip()
    single_query = get_session_env("HERMES_SINGLE_QUERY_SESSION", "") == "1"
    explicit = get_session_env("HERMES_SESSION_SOURCE_EXPLICIT", "") == "1"
    if single_query and not explicit and source in UI_TRANSPORT_SOURCES:
        source = ""
    if single_query and not source and (platform or "cli") == "cli":
        return ONESHOT_SOURCE
    return source or platform or "cli"


# The ``session.create`` wire vocabulary: the labels the create RPC can store — the gateway's
# ``session_policy.SURFACES`` keys, mirrored by the canonical create contract's ``source`` Literal.
# Automation labels (``kanban``, ``cron``, ``bot_room``, ...) name worker executions and
# direct-store rows, but cannot travel through the create wire.
CREATE_SOURCES = frozenset({"cli", "tui", "gui", "acp", "tool", ONESHOT_SOURCE})


def create_wire_source_for(platform: Optional[str]) -> str:
    """``session_source_for`` bounded to ``CREATE_SOURCES``: the row label ``session.create``
    can actually store. An inherited automation label the wire cannot express (a finite
    ``hermes chat -q`` child of a kanban worker session inherits ``kanban``, ...) falls back to
    the finite run's designed ``oneshot`` label — or ``platform``/``cli`` when attended — instead
    of failing the whole launch with ``invalid_params``. An explicit ``--source`` is kept
    verbatim: the gateway refuses what it cannot store, and that refusal is the user's to see."""
    source = session_source_for(platform)
    if source in CREATE_SOURCES or get_session_env("HERMES_SESSION_SOURCE_EXPLICIT", "") == "1":
        return source
    if get_session_env("HERMES_SINGLE_QUERY_SESSION", "") == "1":
        return ONESHOT_SOURCE
    return platform or "cli"


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
