"""Session plumbing an agent sets up at construction: publishing its session id, and naming the storage
interface handle it writes its own store through."""

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


def writer_interface(persistence, store):
    """``persistence`` when it is a writer's handle beside ``store``, else None: no store, no handle, or a reader
    beside a store that is a client of the file's owner (a managed worker's, the classic CLI's), whose calls stay
    on that store."""
    return persistence if store is not None and persistence is not None and not persistence.read_only else None


def agent_interface(agent):
    """The agent's storage interface handle for its own store (:func:`writer_interface`), or None."""
    return writer_interface(getattr(agent, "persistence", None), getattr(agent, "_session_db", None))
