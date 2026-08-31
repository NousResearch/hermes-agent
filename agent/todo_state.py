"""Bind the in-memory todo store to the durable session sidecar."""

from __future__ import annotations

import json
import logging
import weakref

from tools.todo_tool import TodoStore

logger = logging.getLogger(__name__)


def _persistence_key(session_id: str, state) -> tuple[str, str]:
    return (
        session_id,
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def persist_todo_store(agent, state=None) -> bool:
    """Persist the current snapshot under the agent's live session identity."""
    if bool(getattr(agent, "_persist_disabled", False)):
        return False
    session_db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    store = getattr(agent, "_todo_store", None)
    if session_db is None or not session_id:
        return False
    if state is None:
        if store is None:
            return False
        payload = store.snapshot_state()
    else:
        payload = state
    persistence_key = _persistence_key(session_id, payload)
    if getattr(agent, "_todo_state_persist_key", None) == persistence_key:
        return True
    try:
        persisted = bool(session_db.update_session_todo_state(session_id, payload))
        if persisted:
            agent._todo_state_persist_key = persistence_key
        return persisted
    except Exception as exc:
        logger.debug("Could not persist todo state for %s: %s", session_id, exc)
        return False


def build_todo_store(agent, *, fallback_state=None) -> TodoStore:
    """Create a task store restored from, and writing to, the session DB.

    ``fallback_state`` carries a parent snapshot into a newly-created branch.
    A real sidecar for the target session always wins. Persistence is
    best-effort, and helper/fork agents that disable canonical writes remain
    memory-only.
    """
    store = TodoStore()
    session_db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    persistence_enabled = not bool(getattr(agent, "_persist_disabled", False))
    loaded_persisted = False
    if session_db is not None and session_id and persistence_enabled:
        try:
            persisted = session_db.get_session_todo_state(session_id)
            loaded_persisted = persisted is not None and store.load_state(persisted)
            if loaded_persisted:
                agent._todo_state_persist_key = _persistence_key(
                    session_id, store.snapshot_state()
                )
        except Exception as exc:
            logger.debug("Could not restore todo state for %s: %s", session_id, exc)

    loaded_fallback = False
    if not loaded_persisted and fallback_state is not None:
        loaded_fallback = store.load_state(fallback_state)

    if not persistence_enabled or session_db is None or not session_id:
        return store

    try:
        agent_ref = weakref.ref(agent)
    except TypeError:
        # Lightweight test/embedding hosts may not support weak references.
        agent_ref = lambda: agent

    def persist(state):
        current = agent_ref()
        # A late callback from a replaced store must never write its old
        # snapshot under a newly resumed/branched session identity.
        if current is not None and getattr(current, "_todo_store", None) is store:
            persist_todo_store(current, state)

    store.set_on_change(persist)
    if loaded_fallback:
        persist_todo_store(agent, store.snapshot_state())
    return store
