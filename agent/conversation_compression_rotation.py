"""Adopting a rotated session mid-turn.

Compression publishes a continuation child (or a resumed owner adopts a live one) and the agent's
``session_id``, the worker thread's session context and the active turn's review ownership move
to it together, in that order.
"""

import contextlib
import os
from typing import Any


def rebind_session_context(session_id: str) -> None:
    """Point the worker thread's session ContextVar and log context at ``session_id``."""
    try:
        from gateway.session_context import set_current_session_id

        set_current_session_id(session_id)
    except Exception:
        os.environ["HERMES_SESSION_ID"] = session_id
    with contextlib.suppress(Exception):
        from hermes_logging import set_session_context

        set_session_context(session_id)


def adopt_rotated_session(agent: Any, session_id: str) -> None:
    """Make ``session_id`` the agent's live session: alias the active turn's review ownership to
    it first (a review registered against the old id would otherwise outlive the rotation), then
    the agent and the thread's session context."""
    from agent.background_review import rebind_foreground_review_ownership

    rebind_foreground_review_ownership(agent, session_id)
    agent.session_id = session_id
    rebind_session_context(session_id)
