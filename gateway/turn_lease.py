"""Per-session turn lease — serializes the [load history → run → flush] region.

Why this exists (#64934): the gateway's busy guards are keyed by ROUTING KEY
(``_active_sessions`` in the adapter, ``_running_agents`` in the runner), but
the durable transcript is owned by SESSION_ID — and ``switch_session()`` makes
the key→id mapping many-to-one (``/resume`` of a named session from a second
chat/topic, CLI-continuity rebinding, async-delegation completion pinning,
Telegram topic-binding tip-walks). Two routing keys mapped to one session_id
run concurrent turns on two different agent objects, so no per-key guard ever
sees the collision. The two turns then interleave their flushes on one
transcript: rows persist in completion order instead of arrival order, the
identity-marker dedup over shared history dicts can swallow a row outright,
and the second turn runs on a history base that never saw the first turn's
exchange — leaving a permanent ``user;user`` alternation wedge that
``repair_message_sequence`` re-repairs on every request forever.

The lease closes that route by serializing per RESOLVED session_id: it is
acquired after session resolution is final (post ``switch_session``/tip-walk),
immediately before the transcript load, and released in the dispatch layer's
``finally`` on every exit path. Same-key messages never reach the acquisition
point while a turn runs (both routing-key guards hold them), so the lock is
uncontended everywhere except the alias-key route — where the second turn now
waits for the first turn's flush and logs one WARNING naming the session and
both routing keys (pairing with the cross-agent tripwire in
``agent/agent_runtime_helpers.note_turn_start``).

Safety properties:

- **Generation-scoped, identity-checked release.** A token records its owner
  (routing key, run generation) and release only frees the lease when that
  exact token is the current holder — a stale unwind can never release a
  newer turn's lease (the #28686 ownership lesson applied). Release is
  idempotent.
- **Fail-open on timeout.** A stuck holder degrades to today's unserialized
  behavior with a loud ERROR after the configured wait — never a wedged
  session. A degraded token holds nothing and releases nothing.
- **Bounded registry.** The per-session lease map is size-capped; eviction
  only ever removes idle (unheld, uncontended) entries, never a live lease.

Known limits (deliberate, flagged on #64934):

- A CLI process sharing the session via CLI-continuity is outside any
  in-process lock — that pair needs a DB-level lease (separate design).
- Mid-turn compression rotation leaves a small alias window: the tip-walk can
  resolve a fresh child id while the parent-holding turn is still in flight.
  The mid-turn binding-sync sites are the right place to alias the lease in a
  follow-up.
"""

import asyncio
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Upper bound on tracked per-session leases. Idle entries (no holder, no
# waiter) are evicted oldest-first once the cap is reached; live leases are
# never evicted, so a burst of distinct sessions can transiently exceed the
# cap rather than break serialization.
DEFAULT_MAX_LEASES = 512

# Fallback wait (seconds) when the caller passes no positive timeout. Matches
# the gateway's default agent inactivity timeout so a stuck holder fails open
# on the same clock the turn itself would be declared stuck on.
DEFAULT_LEASE_WAIT = 1800.0


class TurnLeaseToken:
    """Handle returned by :meth:`SessionTurnLeaseRegistry.acquire`.

    ``degraded`` means the acquire timed out and the turn is proceeding
    UNSERIALIZED (fail-open); such a token holds nothing and its release is a
    no-op. ``released`` makes release idempotent.
    """

    __slots__ = ("session_id", "owner_key", "generation", "degraded", "released")

    def __init__(
        self,
        session_id: str,
        owner_key: str,
        generation: int,
        degraded: bool = False,
    ) -> None:
        self.session_id = session_id
        self.owner_key = owner_key
        self.generation = generation
        self.degraded = degraded
        self.released = False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"TurnLeaseToken(session_id={self.session_id!r}, "
            f"owner_key={self.owner_key!r}, generation={self.generation}, "
            f"degraded={self.degraded}, released={self.released})"
        )


class _SessionLease:
    __slots__ = ("lock", "holder", "acquired_at", "last_used")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.holder: Optional[TurnLeaseToken] = None
        self.acquired_at = 0.0
        self.last_used = time.time()

    @property
    def idle(self) -> bool:
        """True when this lease can be evicted: nobody holds or awaits it."""
        return self.holder is None and not self.lock.locked()


class SessionTurnLeaseRegistry:
    """Asyncio lease per resolved session_id serializing transcript turns.

    Process-local and single-event-loop by design — the same visibility scope
    as the routing-key guards it extends. All methods must be called from the
    gateway's event loop.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_LEASES) -> None:
        self._leases: Dict[str, _SessionLease] = {}
        self._max_entries = max(1, int(max_entries))

    def __len__(self) -> int:
        return len(self._leases)

    def _get_or_create(self, session_id: str) -> _SessionLease:
        lease = self._leases.get(session_id)
        if lease is None:
            self._evict_idle()
            lease = _SessionLease()
            self._leases[session_id] = lease
        lease.last_used = time.time()
        return lease

    def _evict_idle(self) -> None:
        """Drop oldest idle entries so a new lease fits under the cap.

        Never evicts a held or contended lease — correctness beats the cap.
        """
        overflow = len(self._leases) - self._max_entries + 1
        if overflow <= 0:
            return
        idle_ids = sorted(
            (sid for sid, lease in self._leases.items() if lease.idle),
            key=lambda sid: self._leases[sid].last_used,
        )
        for sid in idle_ids[:overflow]:
            self._leases.pop(sid, None)

    async def acquire(
        self,
        session_id: str,
        *,
        owner_key: str,
        generation: int,
        timeout: Optional[float] = None,
    ) -> Optional[TurnLeaseToken]:
        """Acquire the turn lease for ``session_id``, waiting if held.

        Returns a :class:`TurnLeaseToken` — degraded when the wait timed out
        (fail-open: caller proceeds unserialized). Returns ``None`` for a
        falsy ``session_id``.
        """
        if not session_id:
            return None
        wait = float(timeout) if timeout and timeout > 0 else DEFAULT_LEASE_WAIT
        token = TurnLeaseToken(session_id, owner_key, int(generation))
        lease = self._get_or_create(session_id)

        if lease.lock.locked():
            holder = lease.holder
            logger.warning(
                "turn lease contention on session %s: routing key %s (gen %s) "
                "waiting behind in-flight turn held by routing key %s (gen %s, "
                "held %.0fs) — two routing keys are mapped to one session_id "
                "(#64934); serializing this turn behind the previous turn's "
                "flush",
                session_id,
                owner_key,
                generation,
                holder.owner_key if holder else "?",
                holder.generation if holder else "?",
                time.time() - lease.acquired_at if lease.acquired_at else -1.0,
            )

        try:
            await asyncio.wait_for(lease.lock.acquire(), timeout=wait)
        except asyncio.TimeoutError:
            holder = lease.holder
            logger.error(
                "turn lease wait timed out after %.0fs on session %s "
                "(waiter: routing key %s gen %s; holder: routing key %s "
                "gen %s) — failing open: this turn runs UNSERIALIZED against "
                "the stuck holder rather than wedging the session; transcript "
                "writes may interleave",
                wait,
                session_id,
                owner_key,
                generation,
                holder.owner_key if holder else "?",
                holder.generation if holder else "?",
            )
            token.degraded = True
            return token

        lease.holder = token
        lease.acquired_at = time.time()
        lease.last_used = lease.acquired_at
        return token

    def release(self, token: Optional[TurnLeaseToken]) -> bool:
        """Release ``token``'s lease. Idempotent; ownership-checked.

        Returns True only when this exact token was the current holder and
        the lock was freed. A degraded token, a re-release, or a stale token
        whose slot has since been granted to a newer turn are all safe
        no-ops — a stale unwind can never release a newer turn's lease.
        """
        if token is None or token.degraded or token.released:
            return False
        token.released = True
        lease = self._leases.get(token.session_id)
        if lease is None:
            return False
        if lease.holder is not token:
            logger.debug(
                "turn lease release skipped on session %s: token (key %s "
                "gen %s) is not the current holder",
                token.session_id,
                token.owner_key,
                token.generation,
            )
            return False
        lease.holder = None
        lease.acquired_at = 0.0
        lease.last_used = time.time()
        if lease.lock.locked():
            lease.lock.release()
        return True
