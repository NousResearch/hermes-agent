"""Per-session turn lease — serializes the [load history -> run -> flush] region.

Busy guards are keyed by ROUTING KEY but the transcript is owned by SESSION_ID, and
``switch_session()`` makes key->id many-to-one (/resume from a second chat, CLI-continuity,
delegation pinning, topic tip-walks), so two keys could interleave flushes on one transcript
(``user;user`` wedge). The lease serializes per RESOLVED session_id: acquired right before the
transcript load, released in the dispatch layer's ``finally``; identity-checked release; a
timed-out waiter fails CLOSED (:class:`TurnLeaseTimeoutError`); only idle entries evict. Limits:
CLI-continuity processes are outside this lock; mid-turn rotation alias is closed by ``rebind``.
"""

import asyncio
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Cap on tracked leases. Idle entries evict oldest-first; live leases never do, so a burst of
# distinct sessions may transiently exceed the cap rather than break serialization.
DEFAULT_MAX_LEASES = 512
# Fallback wait (seconds) when the caller passes no positive timeout (bridged via
# HERMES_TURN_LEASE_TIMEOUT — lease contention is not agent inactivity). Fail-closed but short:
# never pin a sequential platform updater for minutes.
DEFAULT_LEASE_WAIT = 5.0


def _holder_desc(holder: Optional["TurnLeaseToken"]) -> tuple:
    return (holder.owner_key, holder.generation) if holder else ("?", "?")


class TurnLeaseTimeoutError(TimeoutError):
    """Lease held for the full wait budget; fail-closed: caller must not enter the turn region."""

    def __init__(self, session_id: str, *, owner_key: str, generation: int, wait_seconds: float):
        self.session_id, self.owner_key = session_id, owner_key
        self.generation, self.wait_seconds = generation, wait_seconds
        super().__init__(f"turn lease wait timed out after {wait_seconds:.0f}s on session "
                         f"{session_id} for routing key {owner_key} (gen {generation})")


class TurnLeaseToken:
    """Held-lease handle from :meth:`SessionTurnLeaseRegistry.acquire`; ``released`` makes
    release idempotent."""

    __slots__ = ("session_id", "owner_key", "generation", "released")

    def __init__(self, session_id: str, owner_key: str, generation: int) -> None:
        self.session_id, self.owner_key, self.generation = session_id, owner_key, generation
        self.released = False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"TurnLeaseToken(session_id={self.session_id!r}, owner_key={self.owner_key!r}, "
                f"generation={self.generation}, released={self.released})")


class _SessionLease:
    __slots__ = ("lock", "holder", "acquired_at", "last_used", "pending_acquires")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.holder: Optional[TurnLeaseToken] = None
        self.acquired_at, self.last_used, self.pending_acquires = 0.0, time.time(), 0

    @property
    def idle(self) -> bool:
        """True when evictable: nobody holds or awaits it."""
        return self.holder is None and not self.lock.locked() and self.pending_acquires == 0


class SessionTurnLeaseRegistry:
    """Asyncio lease per resolved session_id. Process-local, single-event-loop by design (same
    visibility scope as the routing-key guards it extends); call only from the gateway loop."""

    def __init__(self, max_entries: int = DEFAULT_MAX_LEASES) -> None:
        self._leases: Dict[str, _SessionLease] = {}
        self._max_entries = max(1, int(max_entries))

    def _get_or_create(self, session_id: str) -> _SessionLease:
        if (lease := self._leases.get(session_id)) is None:
            self._evict_idle()
            lease = self._leases[session_id] = _SessionLease()
        lease.last_used = time.time()
        return lease

    def _evict_idle(self) -> None:
        """Drop oldest idle entries to fit a new lease under the cap; never a held/contended one."""
        if (overflow := len(self._leases) - self._max_entries + 1) <= 0:
            return
        idle = sorted((sid for sid, l in self._leases.items() if l.idle),
                      key=lambda sid: self._leases[sid].last_used)
        for sid in idle[:overflow]:
            self._leases.pop(sid, None)

    async def acquire(
        self, session_id: str, *, owner_key: str, generation: int, timeout: Optional[float] = None
    ) -> Optional[TurnLeaseToken]:
        """Acquire the lease for ``session_id``, waiting if held. Raises
        :class:`TurnLeaseTimeoutError` when the wait budget expires; None for a falsy id."""
        if not session_id:
            return None
        wait = float(timeout) if timeout and timeout > 0 else DEFAULT_LEASE_WAIT
        token = TurnLeaseToken(session_id, owner_key, int(generation))
        lease = self._get_or_create(session_id)
        if lease.lock.locked():
            logger.warning(
                "turn lease contention on session %s: routing key %s (gen %s) waiting behind "
                "in-flight turn held by routing key %s (gen %s, held %.0fs) — two routing keys "
                "are mapped to one session_id (#64934); serializing this turn behind the previous "
                "turn's flush",
                session_id, owner_key, generation, *_holder_desc(lease.holder),
                time.time() - lease.acquired_at if lease.acquired_at else -1.0)
        # Lock.release() wakes a waiter while leaving the lock momentarily unlocked. Count every
        # in-progress acquire across that handoff (even apparently-uncontended ones — wait_for()
        # may schedule them before the lock coroutine runs) so eviction cannot orphan the old
        # lock and create a second lock for the same session.
        lease.pending_acquires += 1
        try:
            await asyncio.wait_for(lease.lock.acquire(), timeout=wait)
        except asyncio.TimeoutError:
            logger.error(
                "turn lease wait timed out after %.0fs on session %s (waiter: routing key %s gen "
                "%s; holder: routing key %s gen %s) — failing closed: refusing to run this turn "
                "UNSERIALIZED against the still-held lease",
                wait, session_id, owner_key, generation, *_holder_desc(lease.holder))
            raise TurnLeaseTimeoutError(
                session_id, owner_key=owner_key, generation=generation, wait_seconds=wait) from None
        finally:
            lease.pending_acquires -= 1
        # Lock held and no await before holder publication, so the lease cannot become
        # evictable after the pending count is cleared.
        lease.holder = token
        lease.acquired_at = lease.last_used = time.time()
        return token

    def rebind(self, token: Optional[TurnLeaseToken], new_session_id: str) -> bool:
        """Alias a HELD lease onto ``new_session_id`` after mid-turn rotation (compression) so the
        flush target stays serialized: the SAME ``_SessionLease`` is registered under the new id
        (old mapping idle-evicts later), only the holder may rebind, the token follows. A live
        lease on the new id: log loudly, keep the old id (fail-open)."""
        if (token is None or token.released or not new_session_id
                or new_session_id == token.session_id):
            return False
        if (lease := self._leases.get(token.session_id)) is None or lease.holder is not token:
            return False
        existing = self._leases.get(new_session_id)
        if existing is not None and existing is not lease and not existing.idle:
            logger.warning(
                "turn lease rebind blocked: session %s rotated to %s mid-turn (holder: routing key "
                "%s gen %s) but the target session's lease is already live (holder: routing key %s "
                "gen %s) — keeping the lease on the old id; transcript writes on %s may "
                "interleave (#64934 rotation-alias edge)",
                token.session_id, new_session_id, token.owner_key, token.generation,
                *_holder_desc(existing.holder), new_session_id)
            return False
        self._leases[new_session_id] = lease
        lease.last_used = time.time()
        token.session_id = new_session_id
        return True

    def release(self, token: Optional[TurnLeaseToken]) -> bool:
        """Release ``token``'s lease. Idempotent; True only when this exact token was the current
        holder (a re-release or a stale token whose slot went to a newer turn is a safe no-op)."""
        if token is None or token.released:
            return False
        token.released = True
        if (lease := self._leases.get(token.session_id)) is None:
            return False
        if lease.holder is not token:
            logger.debug("turn lease release skipped on session %s: token (key %s gen %s) is not "
                         "the current holder", token.session_id, token.owner_key, token.generation)
            return False
        lease.holder, lease.acquired_at, lease.last_used = None, 0.0, time.time()
        if lease.lock.locked():
            lease.lock.release()
        return True
