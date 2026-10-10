"""Publish a constructed agent's session ID to task-local and legacy tool consumers."""

from __future__ import annotations

import os


def _publish_session_id(session_id: str) -> None:
    """Expose the session ID to tools via ContextVar (+ legacy os.environ fallback).

    If the ContextVar bridge fails to import, keep the root-agent env fallback but never let
    delegated construction publish a child ID process-wide.
    """
    try:
        from gateway.session_context import set_current_session_id
        set_current_session_id(session_id)
    except Exception:
        try:
            from agent.delegation_context import is_delegated_child_context
            delegated_child = is_delegated_child_context()
        except Exception:
            delegated_child = False
        if not delegated_child:
            os.environ["HERMES_SESSION_ID"] = session_id
