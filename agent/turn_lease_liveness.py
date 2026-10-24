"""Process-local registry of live durable turn-lease holders.

A durable turn lease (``session_turn_leases`` in state.db) is a CROSS-PROCESS lock: it
serialises [load history -> run -> flush] for one conversation between the gateway, the
Desktop app, a CLI resume and background delivery. The row is released in the turn's
``finally`` (see ``agent/turn_facade.py``) and renewed every 60s by ``DurableTurnLease``.

The failure this module exists to fix: **a turn that dies without running its ``finally``
leaks its lease forever.** The refresher timer is only cancelled from that ``finally``, so
a turn abandoned mid-flight (gateway SIGKILL, a wedged turn reaped by the inactivity
watchdog, a hard thread kill) keeps its lease row alive indefinitely. The next inbound on
that session then waits out the full lease budget (1800s) and produces nothing.

Observed 2026-10-23 on the homelab gateway: three telegram conversation leases held by
the LIVE gateway pid were still being refreshed 91 minutes after their turns died, so
those sessions could not be served again.

``_compression_lock_holder_process_is_dead`` cannot catch this: a lease held by OUR OWN
pid is deliberately treated as "alive" (PID reuse must never steal another thread's live
lease). But "the pid is alive" is not the same question as "the TURN is alive" — and only
the turn's owner can answer the second one.

This registry is that answer. A holder string registered here names a turn that is running
right now, in this process. A holder string that names our pid but is NOT registered here
names a turn that is gone: reclaim it.

Scope and safety:
  * The registry is in-memory and process-local. It is authoritative ONLY for holders
    whose pid equals THIS process's pid — a foreign pid's liveness is the kernel's
    business, and a foreign holder is never touched here.
  * Registration is bounded by the turn's own lifetime: ``register`` happens at turn
    admission, ``unregister`` in the turn's ``finally``. A crashed process loses the
    registry with itself, and its rows are reclaimed by the kernel-PID probe as before.
  * An un-registered own-pid holder is reclaimed on PROOF of absence from the registry.
    A holder whose registration raced the probe (acquired but not yet registered) is NOT
    absent-from-the-registry in a meaningful sense, so the acquisition path registers
    BEFORE the row is written. See ``agent/turn_facade_lease.py``.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, Optional

logger = __import__("logging").getLogger("run_agent")

# Holder string -> the conversation id it was admitted for. Keyed by holder so two
# concurrent turns in one process (different conversations) never collide.
_lock = threading.Lock()
_live_holders: Dict[str, str] = {}
# Holders that were registered and then explicitly released: the turn behind them is
# PROVABLY gone, which is the only evidence that may free their row. A holder that was
# never registered stays "unknown" — see ``is_holder_live``.
_released_holders: set = set()


def register_holder(holder: str, conversation_id: str = "") -> None:
    """Record that ``holder`` names a turn that is running in THIS process.

    Called at turn admission, BEFORE the lease row is written, so a probe can never see
    "own pid, released" for a turn that is merely being admitted.
    """
    if not holder:
        return
    with _lock:
        _released_holders.discard(holder)
        _live_holders[holder] = str(conversation_id or "")


def unregister_holder(holder: str) -> None:
    """Record that the turn behind ``holder`` is gone (called from the turn's ``finally``).

    The holder moves to the RELEASED set rather than simply disappearing: absence from
    the live set alone is ambiguous (it could mean "never registered"), while an explicit
    release is proof that the turn exited.
    """
    if not holder:
        return
    with _lock:
        _live_holders.pop(holder, None)
        _released_holders.add(holder)


def is_holder_live(holder: str) -> Optional[bool]:
    """True/False when the answer is knowable here; None when it is not this process's holder.

    ``None`` means "this registry cannot speak for that holder" — a foreign pid, an
    unstructured holder, or a holder from another PID namespace. Callers must treat None
    as "unknown, do not reclaim".

    The registry distinguishes three states, not two:

    * ``True``  — the holder is registered: a turn in THIS process is running behind it.
    * ``False`` — the holder was registered and then explicitly released: that turn is
      provably gone, so the row is an orphan and may be reclaimed.
    * ``None``  — the holder was never registered here (a foreign pid, an unstructured
      holder, or a lease written by an older build). "Unknown" is NOT "dead": the
      pre-existing contract is that a lease held by our own pid is never reclaimed on a
      guess, because another live thread's lease would be stolen. Only the explicit
      release proves death.

    This distinction is load-bearing: ``tests/hermes_state/test_session_turn_lease.py``
    pins "a lease held by our own pid is not reclaimable", and that property must survive
    the orphan fix — the fix narrows the orphan case to a holder we PROVED was released.
    """
    if not holder:
        return None
    import re

    match = re.search(r"(?:^|:)pid=(\d+)(?::|$)", holder)
    if not match:
        return None
    try:
        pid = int(match.group(1))
    except (TypeError, ValueError):
        return None
    if pid <= 0 or pid != os.getpid():
        # Not ours: the kernel-PID probe owns this question (PID reuse, namespaces).
        return None
    with _lock:
        if holder in _live_holders:
            return True
        return False if holder in _released_holders else None


def live_holder_count() -> int:
    """Number of turn leases this process currently believes are live (diagnostics/tests)."""
    with _lock:
        return len(_live_holders)


def _reset_for_tests() -> None:
    """Clear the registry (tests only)."""
    with _lock:
        _live_holders.clear()
        _released_holders.clear()
