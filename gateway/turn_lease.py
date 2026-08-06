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

``SessionTurnLeaseRegistry`` remains the lightweight gateway-local guard.
``DurableTurnLease`` uses the same token shape with a SQLite row for entry
points that can share a session across processes (Desktop and API wake turns).
"""

import asyncio
import logging
import os
import threading
import time
import uuid
from threading import Event as _ThreadEvent
from threading import Thread as _Thread
from typing import Callable, Dict, Optional

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

# Durable rows are renewed well before expiry so a long tool-using turn cannot
# be mistaken for a stale owner. A crashed process is reclaimed immediately by
# SessionDB's structured-holder PID check; the TTL is the conservative fallback
# for an unprobeable/recycled PID.
DEFAULT_DURABLE_LEASE_TTL = 120.0
DEFAULT_DURABLE_POLL_INTERVAL = 0.1


class TurnLeaseTimeoutError(TimeoutError):
    """A durable turn lease stayed owned for the full bounded wait."""


class TurnLeaseCancelledError(RuntimeError):
    """The caller stopped waiting before the durable lease was acquired."""


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


class DurableTurnLease(TurnLeaseToken):
    """DB-backed turn lease shared by Desktop and API-server processes."""

    __slots__ = ("db", "holder", "session_ids", "ttl", "_stop", "_thread")

    @classmethod
    def acquire(
        cls,
        db,
        session_id: str,
        *,
        owner_key: str,
        generation: int,
        timeout: Optional[float] = None,
        ttl_seconds: float = DEFAULT_DURABLE_LEASE_TTL,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> Optional["DurableTurnLease"]:
        if db is None or not session_id:
            return None
        lease_methods = (
            "try_acquire_session_turn_lease",
            "get_session_turn_lease_holder",
            "refresh_session_turn_lease",
            "release_session_turn_lease",
        )
        if any(not callable(getattr(db, name, None)) for name in lease_methods):
            return None
        wait = float(timeout) if timeout and timeout > 0 else DEFAULT_LEASE_WAIT
        deadline = time.monotonic() + wait
        holder = (
            f"pid={os.getpid()}:tid={threading.get_ident()}"
            f":owner={str(owner_key or '?').replace(':', '_')[:80]}"
            f":gen={int(generation)}:nonce={uuid.uuid4().hex[:8]}"
        )
        logged = False
        while not db.try_acquire_session_turn_lease(
            session_id, holder, ttl_seconds=ttl_seconds
        ):
            if cancelled is not None and cancelled():
                raise TurnLeaseCancelledError(
                    f"cancelled while waiting for session turn lease on {session_id!r}"
                )
            if not logged:
                logged = True
                logger.warning(
                    "durable turn lease contention on session %s: %s waits behind %s",
                    session_id,
                    owner_key,
                    db.get_session_turn_lease_holder(session_id) or "?",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TurnLeaseTimeoutError(
                    f"timed out after {wait:.0f}s waiting for session turn lease "
                    f"on {session_id!r}"
                )
            time.sleep(min(DEFAULT_DURABLE_POLL_INTERVAL, remaining))

        if cancelled is not None and cancelled():
            db.release_session_turn_lease(session_id, holder)
            raise TurnLeaseCancelledError(
                f"cancelled while waiting for session turn lease on {session_id!r}"
            )

        token = cls(session_id, owner_key, generation)
        token.db = db
        token.holder = holder
        token.session_ids = [session_id]
        token.ttl = max(1.0, float(ttl_seconds))
        # Keep stable references to the stdlib primitives.  Some TUI tests
        # replace ``server.threading.Thread`` with an immediate runner; because
        # ``threading`` is a shared module object, looking the class up through
        # the module here would also make the lease refresher run synchronously
        # and block forever inside ``acquire``.
        token._stop = _ThreadEvent()
        token._thread = _Thread(
            target=token._refresh,
            name="session-turn-lease-refresh",
            daemon=True,
        )
        token._thread.start()
        return token

    def _refresh(self) -> None:
        interval = max(0.1, min(30.0, self.ttl / 2.0))
        while not self._stop.wait(interval):
            for session_id in tuple(self.session_ids):
                if not self.db.refresh_session_turn_lease(
                    session_id, self.holder, ttl_seconds=self.ttl
                ):
                    logger.error(
                        "lost durable turn lease for session %s (holder %s)",
                        session_id,
                        self.holder,
                    )

    def rebind(self, new_session_id: str) -> bool:
        if self.released or not new_session_id or new_session_id == self.session_id:
            return False
        # Keep the old id leased too: a caller may still resume the parent
        # while compression has already rotated this turn onto its child.
        if not self.db.try_acquire_session_turn_lease(
            self.session_id, self.holder, ttl_seconds=self.ttl
        ) or not self.db.try_acquire_session_turn_lease(
            new_session_id, self.holder, ttl_seconds=self.ttl
        ):
            return False
        self.session_ids.append(new_session_id)
        self.session_id = new_session_id
        return True

    def release(self) -> bool:
        if self.released:
            return False
        self.released = True
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
        released = False
        for session_id in reversed(self.session_ids):
            released = self.db.release_session_turn_lease(
                session_id, self.holder
            ) or released
        return released


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

    def rebind(self, token: Optional[TurnLeaseToken], new_session_id: str) -> bool:
        """Alias a HELD lease onto ``new_session_id`` after mid-turn rotation.

        Compression can rotate the durable session_id while a turn is in
        flight (session-hygiene pre-compression, in-agent compression). The
        turn's flush then targets the NEW id — so the serialization boundary
        must follow it, or an alias routing key resolving the new id (e.g. a
        topic tip-walk landing on the fresh child) could start a concurrent
        turn the lease never sees. This closes the rotation-alias window
        flagged on #64934.

        Mechanism: the SAME ``_SessionLease`` object is registered under the
        new id (the old mapping stays until it goes idle and is evicted), so
        acquirers on either id serialize against one lock — no lock state is
        moved, no asyncio internals are touched. Only the current holder can
        rebind (identity-checked like release), and the token follows to the
        new id so release frees the shared object.

        Edge: if the new id already has a live lease of its own (another
        turn is running on the target session), the two serialization
        domains cannot be merged mid-wait — log loudly and keep the token on
        the old id. Fail-open, never deadlock: a holder cannot wait mid-turn.
        """
        if (
            token is None
            or token.degraded
            or token.released
            or not new_session_id
            or new_session_id == token.session_id
        ):
            return False
        lease = self._leases.get(token.session_id)
        if lease is None or lease.holder is not token:
            return False

        existing = self._leases.get(new_session_id)
        if existing is not None and existing is not lease and not existing.idle:
            holder = existing.holder
            logger.warning(
                "turn lease rebind blocked: session %s rotated to %s mid-turn "
                "(holder: routing key %s gen %s) but the target session's "
                "lease is already live (holder: routing key %s gen %s) — "
                "keeping the lease on the old id; transcript writes on %s "
                "may interleave (#64934 rotation-alias edge)",
                token.session_id,
                new_session_id,
                token.owner_key,
                token.generation,
                holder.owner_key if holder else "?",
                holder.generation if holder else "?",
                new_session_id,
            )
            return False

        self._leases[new_session_id] = lease
        lease.last_used = time.time()
        token.session_id = new_session_id
        return True

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
