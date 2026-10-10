"""The review fork's durable session-row lease (``agent/background_review.py`` sibling).

A review fork holds the same ``session_turn_leases`` row a foreground turn takes, so no two
processes decode one session at once. Unlike the foreground's :class:`DurableTurnLease` it never
outranks anyone: it YIELDS to a foreground waiter or a user transcript rewrite on its next
renewal tick, keeps renewing until the fork's own exit releases the row, and stops the fork when
the row is reclaimed under it or can no longer be renewed before it expires.
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from typing import Any, Optional

from agent.turn_facade_lease import (
    _REFRESH_EXPIRY_MARGIN_S,
    _REFRESH_WRITE_PATIENCE_S,
    DurableTurnLease,
)

# Same logger name as the facade so agent.log greps and caplog filters on review lines are unchanged.
logger = logging.getLogger("agent.background_review")

# A review fork renews its durable row often, and keeps renewing after it is asked to yield: the
# row is what a foreground waiter in another process (or a user transcript rewrite) stamps to
# make the review yield, and only the fork's own exit releases it. The TTL bounds a row whose
# process died without releasing it (a dead local PID is reclaimed at once). Foreground turns
# keep the longer LEASE_TTL_SECONDS / 60s cadence.
REVIEW_LEASE_TTL_SECONDS = 60.0
REVIEW_LEASE_REFRESH_SECONDS = 3.0


class _ReviewTurnLease(DurableTurnLease):
    """The review fork's durable session-row lease: it YIELDS to a foreground waiter.

    A foreground turn in another process (CLI resume, Desktop, a second gateway) stamps the row
    (``SessionDB.acquire_session_turn_lease``); so does a user transcript rewrite
    (``hermes_state_messages._check_transcript_write_guards``). Each stamper stores its own
    body-free cause, and the next renewal tick logs THAT slug (``review_preempted_cross_process``
    / ``review_preempted_by_transcript_edit``), fences the run and hard-interrupts the fork — and
    keeps renewing: the row is released only by the fork's own exit, so the waiter never decodes
    beside a still-running fork. The tick doubles as the escalation clock
    (``escalate_unacknowledged_cancel``), so a wedged fork is unwedged rather than left to the
    TTL. A holder-fenced renewal miss means the row was reclaimed under the fork (a dead-PID
    sweep, or a TTL that elapsed with no renewal) and another process may own the session:
    logged once as ``review_lease_lost``, and the fork is stopped the same way. Either stop is
    recorded on the run, so a deferred review is dropped instead of replaying a snapshot the
    transcript has moved past.

    A state.db write lock is neither: like the foreground lease, a locked renewal is a missed
    tick while the next attempt can still land before the row's authority (its last committed
    expiry) runs out. Only a lock that outlasts that authority stops the fork — a successor may
    reclaim the row on expiry — logged once as ``review_lease_renewal_locked`` and NOT recorded
    as a yield: the transcript did not move, so a deferred review is requeued rather than dropped.
    """

    def __init__(
        self,
        review_agent: Any,
        db: Any,
        session_id: str,
        holder: str,
        review_run: Any,
        owner: str,
        *,
        expires_at: Optional[float] = None,
    ) -> None:
        super().__init__(review_agent, db, session_id, holder, expires_at=expires_at)
        self.refresh_interval = REVIEW_LEASE_REFRESH_SECONDS
        self.ttl_seconds = REVIEW_LEASE_TTL_SECONDS
        self._review_run = review_run
        self._owner = owner
        self._yielded_at: Optional[float] = None
        self._renewal_stopped = False  # a lost row or a locked-out renewal: logged once
        self._now = time.monotonic  # test seam

    def refresh_tick(self):
        if self.stop.is_set():
            return False
        if self._yielded_at is None and (cause := self._yield_reason()):
            logger.info(
                "Background review preempted (owner=%s, reason=%s)", self._owner, cause
            )
            self._stop_fork(f"background review asked to yield ({cause})", cause)
        if self._yielded_at is not None:
            from agent.background_review import _CANCEL_ACK_ESCALATION_SECONDS

            elapsed = self._now() - self._yielded_at
            if (
                self._review_run is not None
                and elapsed >= _CANCEL_ACK_ESCALATION_SECONDS
            ):
                self._review_run.escalate_unacknowledged_cancel(
                    2 if elapsed >= 2 * _CANCEL_ACK_ESCALATION_SECONDS else 1
                )
        if self._renewal_stopped:
            return None
        return self._renew()

    def _yield_reason(self) -> Optional[str]:
        """The stamper's slug, or None while nobody asked the review to yield."""
        try:
            return self.db.session_turn_lease_yield_reason(
                self._current_session_id(), self.holder
            )
        except Exception:  # health: allow BLE001 -- a broken store reads as no yield; the renewal decides
            return None

    def _renew(self):
        """One renewal against the row's remaining authority (``DurableTurnLease.refresh_tick``'s
        rule): the write waits no longer than the authority allows, a SQLite lock is a missed
        tick while the next attempt still lands before expiry, and only a holder-fenced miss or
        a non-lock error is a lost row."""
        started = time.time()
        patience = self._authority_deadline - _REFRESH_EXPIRY_MARGIN_S - started
        if patience <= 0:
            return self._stop_locked_out()
        try:
            renewed = bool(
                self.db.refresh_session_turn_lease(
                    self._current_session_id(),
                    self.holder,
                    ttl_seconds=self.ttl_seconds,
                    patience_s=min(_REFRESH_WRITE_PATIENCE_S, patience),
                )
            )
        except Exception as exc:  # health: allow BLE001 -- classified below; anything else is a lost row
            if self.stop.is_set():
                return False  # the fork's finally released the row under the wait: not a loss
            from hermes_state_errors import is_sqlite_lock_error

            if is_sqlite_lock_error(exc):
                if (
                    time.time() + self.refresh_interval + _REFRESH_EXPIRY_MARGIN_S
                    >= self._authority_deadline
                ):
                    return self._stop_locked_out()
                logger.warning(
                    "Background review lease renewal hit a SQLite lock; will retry (owner=%s)",
                    self._owner,
                )
                return None
            logger.debug("Background review lease renewal failed", exc_info=True)
            renewed = False
        if renewed:
            self._authority_deadline = started + self.ttl_seconds
            return None
        if self.stop.is_set():
            return False  # the fork's finally released the row between the checks: not a loss
        self._renewal_stopped = True
        from agent.review_admission import REASON_LEASE_LOST

        logger.warning(
            "Background review lease lost (owner=%s, reason=%s)",
            self._owner,
            REASON_LEASE_LOST,
        )
        self._stop_fork(
            "session turn lease lost; another process may own the session",
            REASON_LEASE_LOST,
        )
        return None

    def _stop_locked_out(self):
        """state.db stayed locked until no renewal can land before the row expires: stop the fork
        now, before a successor reclaims the row beside it. No yield mark: the snapshot still
        describes the transcript, so a deferred review is requeued."""
        self._renewal_stopped = True
        from agent.review_admission import REASON_LEASE_RENEWAL_LOCKED

        logger.warning(
            "Background review lease renewal locked out (owner=%s, reason=%s)",
            self._owner,
            REASON_LEASE_RENEWAL_LOCKED,
        )
        self._stop_fork(
            "session turn lease could not be renewed before it expires",
            REASON_LEASE_RENEWAL_LOCKED,
            record_yield=False,
        )
        return None

    def _stop_fork(self, reason: str, slug: str, *, record_yield: bool = True) -> None:
        """Fence the run and hard-interrupt the fork, once; starts the escalation clock. With
        ``record_yield`` the ``slug`` is recorded on the run before the fence so the requeue
        policy sees it with the cancel (and drops the deferred review)."""
        if self._yielded_at is not None:
            return
        self._yielded_at = self._now()
        fork = self.agent
        if self._review_run is not None:
            if record_yield:
                self._review_run.lease_yield_reason = slug
            fork = self._review_run.cancel() or fork
        with suppress(Exception):
            from agent.interrupt_compat import request_hard_interrupt

            request_hard_interrupt(
                fork, reason, tool_reason="background review superseded"
            )
