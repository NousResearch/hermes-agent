"""Process-local liveness registry for durable session turn leases.

A durable turn lease row (``session_turn_leases``) records a holder string built by
``agent/turn_facade_lease.py``::

    pid=3860160:pidns=4026531836:turn=<session_id>:<task_id>:<uuid8>:platform=telegram

The reclaim predicate (``hermes_state._compression_lock_holder_process_is_dead``) can only
prove a holder stale via its ``pid=`` component: a lease whose *turn* died while the owning
*process* stayed alive is indistinguishable from a live turn, so the row survives until its
TTL expires.  That is not a theoretical gap — a turn wedged inside a blocking call never
reaches the façade's ``finally``, so ``DurableTurnLease.release()`` never runs, while the
periodic refresher (a separate daemon thread) keeps renewing the row.  The result is a lease
that blocks every later inbound on that session indefinitely, with a perfectly healthy PID.

This registry closes that gap without weakening the PID probe:

* a turn registers its unique turn token on admission and retires it when the turn ends;
* a holder is reclaimable when its turn token is *known* to have been retired in THIS
  process, even though its PID is still alive.

Deliberate properties:

* **Process-local.**  Only a holder minted by THIS process can be retired here.  A lease
  held by another process keeps deferring to its TTL, exactly as before — this never steals
  a live turn from a sibling process.
* **Fail-safe.**  An unknown token is NOT treated as dead.  A holder we cannot vouch for
  keeps deferring to TTL expiry, which self-heals.  A false reclaim would end a live turn,
  which is far worse than a late reclaim.
* **Turn-scoped, not generation-scoped.**  The registry is keyed by the unique per-turn token
  already present in the holder string, so a successor turn on the same session is never
  affected by its predecessor's retirement.

The registry is intentionally tiny and lock-guarded: one dict, one lock, O(1) lookups.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

logger = logging.getLogger("run_agent")

# ``:turn=<token>`` inside a structured holder string.  The token is built as
# ``f"{session_id}:{task_id}:{uuid4hex8}"`` and BOTH parts may themselves contain ':'
# (a session id is a fingerprint-derived string; a task id is a uuid).  The token
# therefore cannot be delimited by a fixed number of colons, so it is delimited by the
# next KNOWN structured component instead: ``:platform=`` for turn holders and
# ``:cron:`` for cron holders (``turn=<id>:cron:<job>:<hash>:<hash>``).
#
# The greedy ``[^:]+`` alternatives below mean "any characters that are not a colon",
# so the character class is written as ``[^:]*`` groups joined by literal colons.
_TURN_TOKEN_RE = re.compile(r"(?:^|:)turn=(.+?)(?=:platform=|:cron:|$)")

# Retired tokens older than this are forgotten.  A row whose TTL has long expired needs no
# registry entry to be reclaimable, so keeping them forever only leaks memory.
_RETIRED_TTL_S = 3600.0


class _TurnLivenessRegistry:
    """Tracks the turn tokens minted by this process and the ones that have retired."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # token -> monotonic timestamp of registration (for retired-entry expiry)
        self._active: dict[str, float] = {}
        # token -> monotonic timestamp of retirement
        self._retired: dict[str, float] = {}

    def register(self, token: str) -> None:
        """Record that ``token`` names a turn that is running in this process."""
        if not token:
            return
        with self._lock:
            self._active[token] = _now()
            # Re-registering a token clears any prior retirement: the same token is never
            # minted twice (uuid4 suffix), so this only guards against a caller re-adding.
            self._retired.pop(token, None)

    def retire(self, token: str) -> None:
        """Record that the turn named by ``token`` has ended (normally or abnormally)."""
        if not token:
            return
        with self._lock:
            self._active.pop(token, None)
            self._retired[token] = _now()
            self._prune_locked()

    def is_retired(self, token: str) -> bool:
        """True when ``token`` names a turn this process has already finished."""
        if not token:
            return False
        with self._lock:
            if token in self._retired:
                return True
            return False

    def is_active(self, token: str) -> bool:
        """True when ``token`` names a turn this process is currently running."""
        if not token:
            return False
        with self._lock:
            return token in self._active

    def _prune_locked(self) -> None:
        cutoff = _now() - _RETIRED_TTL_S
        stale = [tok for tok, ts in self._retired.items() if ts < cutoff]
        for tok in stale:
            self._retired.pop(tok, None)
        # An active token that is not retired is either running or leaked by a turn that
        # never retired it (a hard process death).  Both are covered by TTL expiry, so
        # active entries are pruned on the same slow horizon.
        stale_active = [tok for tok, ts in self._active.items() if ts < cutoff]
        for tok in stale_active:
            self._active.pop(tok, None)

    def reset(self) -> None:
        """Drop all state (tests)."""
        with self._lock:
            self._active.clear()
            self._retired.clear()


def _now() -> float:
    import time

    return time.monotonic()


_REGISTRY = _TurnLivenessRegistry()


def register_turn_token(token: str) -> None:
    """Declare ``token`` as a live turn of this process."""
    _REGISTRY.register(token)


def retire_turn_token(token: str) -> None:
    """Declare the turn named by ``token`` finished."""
    _REGISTRY.retire(token)


def turn_token_is_retired(token: str) -> bool:
    """True when this process has finished the turn named by ``token``."""
    return _REGISTRY.is_retired(token)


def turn_token_is_active(token: str) -> bool:
    """True when this process is running the turn named by ``token``."""
    return _REGISTRY.is_active(token)


def holder_turn_token(holder: str) -> Optional[str]:
    """Extract the per-turn token from a structured holder string, or None."""
    if not holder:
        return None
    match = _TURN_TOKEN_RE.search(holder)
    if not match:
        return None
    token = match.group(1)
    return token or None


def holder_turn_is_dead(holder: str) -> bool:
    """True when this process can prove the turn named by ``holder`` has ended.

    Only a token this process minted AND retired counts as proof.  An unknown token — a
    sibling process's turn, an unstamped legacy row, or a holder we cannot parse — returns
    False so the row keeps deferring to its TTL.  A false reclaim ends a live turn; a late
    reclaim merely delays the next inbound.
    """
    token = holder_turn_token(holder)
    if not token:
        return False
    return _REGISTRY.is_retired(token)
