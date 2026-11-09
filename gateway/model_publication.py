"""Admission gates for model publication, separate from command issue ordering."""
import asyncio
from weakref import WeakValueDictionary


def publication_lock(runner, session_key=None):
    """A session pick fences its route; a profile-default pick fences every route.

    Command writers still hold the runner's model-switch lock. Admission never
    takes that lock: provider I/O for another session must not stop this turn.
    Weak values retain gates while held or awaited without retaining idle routes.
    """
    locks = runner.__dict__.get("_model_publication_locks")
    if locks is None:
        locks = runner.__dict__["_model_publication_locks"] = WeakValueDictionary()
    lock = locks.get(session_key)
    if lock is None:
        lock = locks[session_key] = asyncio.Lock()
    return lock
