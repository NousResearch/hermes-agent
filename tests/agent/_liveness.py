"""Wait helpers for tests that coordinate with a peer thread.

A wall-clock budget in a test is a claim about the code under test. "A peer
thread got scheduled within N seconds", though, is a claim about the host: on
the loaded sweep machine (16 cores, load average 152, ten concurrent workers)
the same tree stretched ~24x, and a worker thread can wait seconds for a core.
A tight bound then fails for a reason that has nothing to do with the code
under test — it asserts the OS scheduler.

So coordinated waits in ``tests/agent`` use :data:`PEER_THREAD_LIVENESS_S`.
It is a **hang detector, not a budget**: it exists so a genuine hang (the peer
thread never reaches the awaited state) fails as an assertion with a readable
message instead of blocking the file until the runner's per-file timeout. When
the property under test can be stated without a clock at all — a
synchronization barrier, or an ordering between two events — prefer that over
any finite wait.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

# Not a performance budget (see module docstring) — the point is that a
# starved-but-live thread always wins this race and a dead one always loses it.
PEER_THREAD_LIVENESS_S = 60.0


def wait_event(event: threading.Event, what: str) -> None:
    """Wait for ``event``; fail with ``what`` if no peer thread ever sets it."""
    if not event.wait(timeout=PEER_THREAD_LIVENESS_S):
        raise AssertionError(f"{what} (waited {PEER_THREAD_LIVENESS_S:.0f}s)")


def join_thread(thread: threading.Thread, what: str) -> None:
    """Join ``thread``; fail with ``what`` if it never exits."""
    thread.join(timeout=PEER_THREAD_LIVENESS_S)
    if thread.is_alive():
        raise AssertionError(f"{what} (thread alive after {PEER_THREAD_LIVENESS_S:.0f}s)")


def await_state(predicate: Callable[[], bool], what: str, *, poll_seconds: float = 0.01) -> None:
    """Poll ``predicate``; fail with ``what`` if it never becomes true."""
    deadline = time.monotonic() + PEER_THREAD_LIVENESS_S
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll_seconds)
    raise AssertionError(f"{what} (state never became true within {PEER_THREAD_LIVENESS_S:.0f}s)")
