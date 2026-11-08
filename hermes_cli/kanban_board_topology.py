"""Cross-process topology fence for Kanban board discovery and lifecycle."""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

from hermes_cli.active_sessions import _FileLock

_PROCESS_LOCK = threading.RLock()
_LOCAL = threading.local()


@contextlib.contextmanager
def board_topology_lock(kanban_home: Path, *, purpose: str = "open"):
    """Serialize board publication/removal/open with destructive snapshots.

    The process RLock makes the file lock reentrant for create_board -> init_db ->
    connect while still blocking sibling threads. A snapshot rejects same-thread
    mutation/open re-entry, so callbacks cannot publish an unfenced board after
    enumeration. The persistent flock file provides the same ordering across
    Hermes processes and is never unlinked.
    """
    if purpose not in {"open", "mutation", "snapshot"}:
        raise ValueError(f"invalid board topology lock purpose: {purpose}")
    lock_path = (
        Path(kanban_home).expanduser().resolve(strict=False)
        / "kanban"
        / ".boards-topology.lock"
    )
    key = str(lock_path)
    _PROCESS_LOCK.acquire()
    states = getattr(_LOCAL, "states", None)
    if states is None:
        states = {}
        _LOCAL.states = states
    state = states.get(key)
    entered = False
    try:
        if state is None:
            file_lock = _FileLock(lock_path)
            file_lock.__enter__()
            states[key] = [1, file_lock, purpose]
            entered = True
        else:
            outer_purpose = state[2]
            if outer_purpose == "snapshot" and purpose != "snapshot":
                raise RuntimeError("board topology is frozen by a destructive snapshot")
            if outer_purpose == "open" and purpose == "mutation":
                raise RuntimeError("board mutation cannot re-enter an open topology lease")
            state[0] += 1
            entered = True
        yield
    finally:
        state = states.get(key)
        if entered and state is not None:
            state[0] -= 1
            if state[0] == 0:
                states.pop(key, None)
                state[1].__exit__(None, None, None)
        _PROCESS_LOCK.release()
