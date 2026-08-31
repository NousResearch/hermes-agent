"""Bind the in-memory todo store to the durable session sidecar."""

from __future__ import annotations

import json
import logging
import threading
import weakref

from tools.todo_tool import TodoStore

logger = logging.getLogger(__name__)
_LOCK_INIT = threading.Lock()


def _persistence_key(session_id: str, state) -> tuple[str, str]:
    return (
        session_id,
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _state_generation(state) -> int:
    if not isinstance(state, dict):
        return 0
    raw = state.get("generation", state.get("revision", 0))
    return max(0, raw) if isinstance(raw, int) and not isinstance(raw, bool) else 0


def _persistence_lock(agent):
    lock = getattr(agent, "_todo_state_persist_lock", None)
    if lock is not None:
        return lock
    # Lightweight embedding/test hosts do not pass through init_agent. Make
    # lazy creation safe when their first two callbacks arrive concurrently.
    with _LOCK_INIT:
        lock = getattr(agent, "_todo_state_persist_lock", None)
        if lock is None:
            lock = threading.RLock()
            agent._todo_state_persist_lock = lock
    return lock


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
        if not isinstance(state, dict):
            return False
        payload = state
    persistence_key = _persistence_key(session_id, payload)
    generation_key = (session_id, _state_generation(payload))
    with _persistence_lock(agent):
        if getattr(agent, "_todo_state_persist_key", None) == persistence_key:
            return True
        previous_generation = getattr(
            agent, "_todo_state_persist_generation", None
        )
        if (
            isinstance(previous_generation, tuple)
            and len(previous_generation) == 2
            and previous_generation[0] == session_id
            and generation_key[1] <= previous_generation[1]
        ):
            # Every durable TodoStore change advances generation. A same/older
            # snapshot arriving after a completed write is therefore stale.
            return True
        try:
            persisted = bool(session_db.update_session_todo_state(session_id, payload))
            if persisted:
                agent._todo_state_persist_key = persistence_key
                agent._todo_state_persist_generation = generation_key
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
                snapshot = store.snapshot_state()
                agent._todo_state_persist_key = _persistence_key(session_id, snapshot)
                agent._todo_state_persist_generation = (
                    session_id,
                    _state_generation(snapshot),
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
