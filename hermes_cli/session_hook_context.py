"""Dispatch-owned identity snapshots for native plugin hooks.

Capture profile identity while constructing the owning session. Never recover an
identifier from environment variables or a process-wide "current session".
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def hook_profile_scope(hermes_home):
    """Discover and invoke plugins in the payload owner's profile, including worker threads."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(hermes_home)) if hermes_home is not None else None
    try:
        yield
    finally:
        if token is not None:
            reset_hermes_home_override(token)


def capture_session_identity(*, session_id=None, stored_session_id=None,
                             runtime_session_id=None, source=None, surface=None,
                             hermes_home=None):
    """Capture only at the session's construction/registration boundary, under its home scope."""
    from hermes_constants import get_hermes_home
    from hermes_cli.profiles import get_active_profile_name

    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    with hook_profile_scope(home):
        profile = get_active_profile_name()
    return dict(runtime_session_id=runtime_session_id, stored_session_id=stored_session_id,
                session_id=session_id, task_id=None, profile=profile, hermes_home=home,
                source=source, surface=surface)


def agent_session_identity(agent) -> dict[str, Any]:
    """Read the dispatching agent, including a conversation ID rotated during this turn.

    Legacy/library agents without a captured owner expose unknown fields as None,
    never as the identity of whichever session most recently touched the process.
    """
    captured = getattr(agent, "_plugin_session_identity", None)
    identity: dict[str, Any] = dict(captured) if isinstance(captured, dict) else dict.fromkeys((
        "runtime_session_id", "stored_session_id", "session_id", "task_id",
        "profile", "hermes_home", "source", "surface"))
    identity["session_id"] = getattr(agent, "session_id", None)
    identity["task_id"] = getattr(agent, "_current_task_id", None)
    return identity
