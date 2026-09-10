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
import sqlite3
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



# ---------------------------------------------------------------------------
# DB-level turn lease — async wrapper over the merged cross-process lease
# ---------------------------------------------------------------------------
#
# The storage substrate (``session_turn_leases`` table, lineage-root key,
# dead-PID reclaim, holder fencing via ``SessionTurnLeaseLostError``) landed
# in ``6e929a9694`` ("fix(sessions): serialize turns across processes") and
# follow-ups; the sync ``AIAgent`` consumer lives in ``run_agent.py``.  This
# section is the async adapter for event-loop contexts (gateway dispatch
# layer): the same acquire/refresh/release protocol exposed as a context
# manager that never blocks the loop.

# Default lease TTL (seconds).  Matches the merged storage default and the
# ``run_agent`` consumer's ``_lease_ttl`` (300.0), so an async wrapper turn
# and a sync AIAgent turn serialize on the same clock.
DEFAULT_DB_LEASE_TTL = 300.0

# Default total wait for the DB lease poll loop (seconds).  Mirrors the
# merged ``acquire_session_turn_lease`` wait budget.  A DB-held lease that
# outlives its TTL is reclaimed by the next acquirer (expired rows and rows
# whose local holder PID is known dead are deleted inside the same write
# transaction), so a crashed holder wedges the session for at most
# ``ttl_seconds`` — the wait budget only bounds how long this caller polls
# before failing open.
DEFAULT_DB_LEASE_WAIT = 1800.0

# Initial backoff interval between DB poll attempts (seconds).
DEFAULT_DB_LEASE_POLL = 0.25

# Maximum backoff interval (capped exponential).
DEFAULT_DB_LEASE_POLL_MAX = 2.0

# Backoff multiplier per retry.
DEFAULT_DB_LEASE_BACKOFF = 1.5


class TurnLease:
    """Async context manager over the merged cross-process session turn lease.

    ``async with TurnLease(state, session_id, holder) as acquired:``
    serializes the [load history → run → flush] region against the merged
    ``session_turn_leases`` table (``6e929a9694``, #67442), so a CLI process,
    a gateway process, or two ``hermes serve`` backends sharing one
    ``HERMES_HOME`` coordinate through the same DB row.  The in-process
    :class:`SessionTurnLeaseRegistry` above cannot see across process
    boundaries — this closes that gap for async callers.

    The storage layer is sync
    (``HermesState.try_acquire_session_turn_lease`` /
    ``refresh_session_turn_lease`` / ``release_session_turn_lease``); this
    wrapper adapts it to the event loop: each attempt is a short SQLite write
    and the wait between attempts is ``asyncio.sleep``, so the loop never
    blocks.

    Usage::

        async with TurnLease(
            state, session_id,
            holder=f"pid={os.getpid()}:turn={gen}:surface={surface}",
        ) as acquired:
            if not acquired:
                # Timed out — fail-open: proceed unserialized.
                ...
            # Critical section: load history → run → flush.
            ...

    .. rubric:: Protocol

    On enter, polls :meth:`HermesState.try_acquire_session_turn_lease` with
    capped exponential backoff until the lease is acquired or the overall
    ``wait_timeout`` expires.  Returns ``True`` when the lease was acquired,
    ``False`` on timeout (fail-open — the caller proceeds unserialized rather
    than wedging the session).  ``degraded`` is ``True`` exactly when
    exclusivity was NOT proven.

    On exit, calls :meth:`HermesState.release_session_turn_lease`.  Guaranteed
    on every exit path: normal return, exception, and cancellation.  If a
    background refresh task is running it is cancelled first.

    .. rubric:: Holder contract

    ``holder`` is the fence identity: the merged flush layer raises
    :class:`hermes_state.SessionTurnLeaseLostError` when a transcript write
    presents a holder that no longer owns the lease, and the dead-PID reclaim
    parses ``pid=`` out of the holder string to reap crashed holders without
    waiting for the TTL.  Pass a holder embedding the local PID (same
    convention as the merged ``run_agent`` consumer's
    ``f"pid={os.getpid()}:turn=<id>:platform=<...>"``).

    .. rubric:: Background refresh

    Set ``refresh_interval`` (seconds) to periodically bump the lease expiry
    during long turns via :meth:`HermesState.refresh_session_turn_lease`.  The
    refresh task is cancelled on exit.  Without it, a turn that runs longer
    than ``ttl_seconds`` will see its lease expire — the next acquirer
    reclaims the stale row and the two turns interleave.

    .. rubric:: Safety properties

    - **Fail-open on timeout.**  Returns ``False`` after ``wait_timeout``
      with a loud ERROR log, never a wedged session.  Posture per #67442:
      interactive user turns wait with a bounded budget; machine-initiated
      paths should choose the storage layer's fail-closed semantics instead.
    - **Guaranteed release.**  ``__aexit__`` is called on every path,
      including cancellation.  The DB row is deleted iff this holder still
      owns it (identity-checked by ``release_session_turn_lease``).
    - **Expired/dead-holder reclamation.**  The storage layer deletes stale
      rows transparently, so a crashed holder wedges the session for at most
      ``ttl_seconds``, not indefinitely.
    """

    def __init__(
        self,
        state,  # HermesState / SessionDB
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = DEFAULT_DB_LEASE_TTL,
        wait_timeout: float = DEFAULT_DB_LEASE_WAIT,
        poll_interval: float = DEFAULT_DB_LEASE_POLL,
        refresh_interval: Optional[float] = None,
    ) -> None:
        self._state = state
        self._session_id = session_id
        self._holder = holder
        self._ttl_seconds = float(ttl_seconds)
        self._wait_timeout = float(wait_timeout)
        self._poll_interval = float(poll_interval)
        self._refresh_interval = refresh_interval
        self._acquired = False
        self._refresh_task: Optional[asyncio.Task] = None

    @property
    def degraded(self) -> bool:
        """True when exclusivity was NOT proven (fail-open timeout).

        Posture contract from #67442: report whether the lease was proven,
        let each caller class decide how to react.
        """
        return not self._acquired

    async def __aenter__(self) -> bool:
        """Acquire the DB turn lease with non-blocking polling backoff.

        Returns ``True`` on success, ``False`` on timeout (fail-open).
        """
        if not self._session_id:
            self._acquired = False
            return False

        deadline = time.time() + self._wait_timeout
        backoff = self._poll_interval
        attempts = 0

        while True:
            attempts += 1
            try:
                if self._state.try_acquire_session_turn_lease(
                    self._session_id,
                    self._holder,
                    ttl_seconds=self._ttl_seconds,
                ):
                    self._acquired = True
                    logger.debug(
                        "DB turn lease acquired for session %s (holder=%s, "
                        "attempts=%s, ttl=%.0fs)",
                        self._session_id,
                        self._holder,
                        attempts,
                        self._ttl_seconds,
                    )
                    break
            except sqlite3.Error as exc:
                # A holder's long write transaction (compression publish,
                # large flush) can exhaust a single write-patience budget.
                # Keep polling until the wait budget expires — same posture
                # as the merged sync ``acquire_session_turn_lease`` loop.
                logger.debug(
                    "DB turn lease poll attempt %s on session %s hit sqlite "
                    "error %s — retrying",
                    attempts,
                    self._session_id,
                    exc,
                )

            remaining = deadline - time.time()
            if remaining <= 0:
                logger.error(
                    "DB turn lease wait timed out after %.1fs on session %s "
                    "(holder=%s, attempts=%s) — failing open: this turn runs "
                    "UNSERIALIZED; transcript writes may interleave with "
                    "another process (#67442)",
                    self._wait_timeout,
                    self._session_id,
                    self._holder,
                    attempts,
                )
                self._acquired = False
                break

            # Log contention at WARNING on the first attempt only (avoid log
            # spam during the poll loop).
            if attempts == 1:
                logger.warning(
                    "DB turn lease contention on session %s: holder=%s "
                    "waiting to acquire (ttl=%.0fs, wait=%.0fs) — another "
                    "process holds the lease (#67442)",
                    self._session_id,
                    self._holder,
                    self._ttl_seconds,
                    self._wait_timeout,
                )

            sleep = min(backoff, remaining)
            await asyncio.sleep(sleep)
            backoff = min(backoff * DEFAULT_DB_LEASE_BACKOFF,
                          DEFAULT_DB_LEASE_POLL_MAX)

        if self._acquired and self._refresh_interval is not None:
            self._start_refresh()

        return self._acquired

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the DB turn lease.  Guaranteed on every exit path."""
        try:
            if self._refresh_task is not None:
                self._refresh_task.cancel()
                try:
                    await self._refresh_task
                except asyncio.CancelledError:
                    pass
                self._refresh_task = None

            if self._acquired and self._session_id:
                self._state.release_session_turn_lease(
                    self._session_id,
                    self._holder,
                )
                logger.debug(
                    "DB turn lease released for session %s (holder=%s)",
                    self._session_id,
                    self._holder,
                )
        except Exception:
            logger.debug(
                "DB turn lease release error for session %s",
                self._session_id,
                exc_info=True,
            )
        finally:
            self._acquired = False

    def _start_refresh(self) -> None:
        """Launch a background task that refreshes the lease periodically."""
        if self._refresh_interval is None or self._refresh_interval <= 0:
            return

        async def _refresh_loop() -> None:
            interval = self._refresh_interval
            while True:
                await asyncio.sleep(interval)
                try:
                    ok = self._state.refresh_session_turn_lease(
                        self._session_id,
                        self._holder,
                        ttl_seconds=self._ttl_seconds,
                    )
                    if not ok:
                        logger.warning(
                            "DB turn lease refresh failed for session %s "
                            "(holder=%s) — lease may have been reclaimed by "
                            "another process (#67442)",
                            self._session_id,
                            self._holder,
                        )
                except Exception:
                    logger.debug(
                        "DB turn lease refresh error for session %s",
                        self._session_id,
                        exc_info=True,
                    )

        self._refresh_task = asyncio.create_task(_refresh_loop())

