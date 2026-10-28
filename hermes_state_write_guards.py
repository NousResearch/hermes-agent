"""Transcript write guards for SessionDB: the admission checks every transcript writer runs inside its
write transaction (compression lock, cross-process turn lease, closed compression parent), and what a
writer means by a live BACKGROUND-REVIEW turn lease (``REVIEW_LEASE_*``). Mixin bound via the MRO; the
policy constants are imported by every canonical mutation that names one."""

from __future__ import annotations

import logging
import time
from typing import Optional

from agent.review_admission import REASON_PREEMPTED_BY_TRANSCRIPT_EDIT
from hermes_state_common import (
    _COMPRESSION_LOCK_ROW_SQL,
    _ENDED_ROW_SQL,
    _ended_by_compression,
)
from hermes_state_compression import BACKGROUND_REVIEW_LEASE_HOLDER_MARK

logger = logging.getLogger("hermes_state")  # caplog tests pin the origin module's name

_TURN_LEASE_ROW_SQL = (
    "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = ?"
)
_STAMP_REVIEW_YIELD_SQL = (
    "UPDATE session_turn_leases SET yield_requested_at = ?, yield_reason = ? "
    "WHERE conversation_id = ? AND holder = ? AND yield_requested_at IS NULL"
)
# What a transcript writer does about a live BACKGROUND-REVIEW turn lease. The review fork never
# writes the transcript (immutable snapshot, persistence disabled), so the row cannot fence a write
# the way a foreground holder's does; what it means depends on the writer, never on the guard:
# REFUSE like any live lease (a maintenance probe spares the row, asks nothing), YIELD because a
# user rewrite or delete invalidated the fork's replay basis (stamp it to yield), IGNORE because an
# append cannot invalidate the snapshot (land beside the review).
REVIEW_LEASE_REFUSE = "refuse"
REVIEW_LEASE_YIELD = "yield"
REVIEW_LEASE_IGNORE = "ignore"
_DELETE_COMPRESSION_LOCK_SQL = (
    "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?"
)


def _stale_holder(row, now: float) -> bool:
    """A lock/lease row whose holder is expired or a provably dead local process."""
    from hermes_state import _compression_lock_holder_process_is_dead

    return float(row["expires_at"]) <= now or _compression_lock_holder_process_is_dead(
        row["holder"]
    )


def _log_review_yield_for_edit(session_id: str) -> None:
    """One body-free, owner-tagged line per stamp: a review preempted by a user edit is as greppable
    as one preempted by a foreground turn (``agent/review_admission.py``)."""
    from agent.review_admission import current_profile_key, owner_tag

    logger.info(
        "Background review asked to yield for a transcript edit (owner=%s, reason=%s)",
        owner_tag(current_profile_key(), session_id),
        REASON_PREEMPTED_BY_TRANSCRIPT_EDIT,
    )


class SessionWriteGuardsMixin:
    """``_check_transcript_write_guards`` (see module docstring)."""

    def _check_transcript_write_guards(
        self,
        conn,
        session_id: str,
        compression_lock_holder: Optional[str],
        turn_lease_holder: Optional[str] = None,
        turn_lease_ttl_seconds: float = 300.0,
        reject_active_turn_lease: bool = False,
        reject_active_compression_lock: bool = False,
        allow_closed_compression_parent: bool = False,
        review_lease_policy: str = REVIEW_LEASE_REFUSE,
    ) -> None:
        """Transcript-write admission checks, run INSIDE the write txn by every writer. Ordinary appends do
        NOT check compression_locks: the lock only stops two COMPRESSIONS colliding and archive_and_compact()
        commits against a watermark, so concurrent appends are safe (blocking them killed turns during slow
        summaries). Destructive user mutations opt in via ``reject_active_*`` so a compressor that captured
        its watermark cannot resurrect the removed turn.

        Shared by :meth:`append_message` and :meth:`append_messages_batch` so the two writers can never
        diverge on these correctness invariants (this guard has already needed targeted fixes — see the
        #74478 patience note below). User-initiated transcript mutations may opt in to rejecting an active
        unowned turn lease in that same transaction; ``review_lease_policy`` (``REVIEW_LEASE_*``) is what a
        live background-review holder means to THIS writer: a maintenance probe refuses like any live lease
        (nothing stamped or logged), a user rewrite asks it to yield, an append ignores it.
        """
        from hermes_state import SessionCompressionInProgressError
        from hermes_state_errors import (
            CompressionSessionClosedError,
            SessionTurnLeaseLostError,
        )

        # NOTE (#75316 redesign): appends do NOT check compression_locks. The lock's job is to stop two
        # COMPRESSIONS colliding, not to fence ordinary transcript writes. Concurrent appends during a
        # compression are safe by construction: archive_and_compact() commits against a watermark captured
        # at compression start and clones every row that arrived after it back into the live transcript, in
        # the same write transaction. Blocking appends here was the root cause of a whole symptom family —
        # turns dying as session_persistence_failed while a slow provider summary held the lease (#74568,
        # #77386), including stale locks from dead PIDs blocking writes for the full TTL. Keep that narrow
        # fence opt-in so ordinary appends retain the watermark behavior.
        if reject_active_compression_lock:
            active_lock = conn.execute(
                _COMPRESSION_LOCK_ROW_SQL, (session_id,)
            ).fetchone()
            if active_lock is not None:
                if _stale_holder(active_lock, time.time()):
                    conn.execute(
                        _DELETE_COMPRESSION_LOCK_SQL,
                        (session_id, active_lock["holder"]),
                    )
                elif active_lock["holder"] != compression_lock_holder:
                    raise SessionCompressionInProgressError(
                        f"Session {session_id!r} is being compressed by another writer"
                    )
        if turn_lease_holder or reject_active_turn_lease:
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            lease = conn.execute(_TURN_LEASE_ROW_SQL, (conversation_id,)).fetchone()
            now = time.time()
            if turn_lease_holder:
                if lease is None or lease["holder"] != turn_lease_holder:
                    raise SessionTurnLeaseLostError(
                        f"Session turn lease lost; refusing transcript write for {session_id!r}"
                    )
                if float(lease["expires_at"]) <= now:
                    # Expiry makes the row reclaimable, not taken over; BEGIN IMMEDIATE serializes this
                    # renewal with acquisition, so a still-matching owner recovers from a starved refresher.
                    conn.execute(
                        "UPDATE session_turn_leases SET expires_at = ? "
                        "WHERE conversation_id = ? AND holder = ?",
                        (
                            now + max(0.1, float(turn_lease_ttl_seconds)),
                            conversation_id,
                            turn_lease_holder,
                        ),
                    )
            elif lease is not None:
                if _stale_holder(lease, now):
                    # Same reclaim rule as acquisition; deleting also fences a stale late flush after the mutation.
                    conn.execute(
                        "DELETE FROM session_turn_leases WHERE conversation_id = ? AND holder = ?",
                        (conversation_id, lease["holder"]),
                    )
                elif (
                    BACKGROUND_REVIEW_LEASE_HOLDER_MARK not in str(lease["holder"])
                    or review_lease_policy == REVIEW_LEASE_REFUSE
                ):
                    raise SessionTurnLeaseLostError(
                        f"Session has an active turn lease; refusing transcript mutation for {session_id!r}"
                    )
                elif review_lease_policy == REVIEW_LEASE_YIELD:
                    # The rewrite invalidated the fork's replay basis, so it is asked to yield: the fork stops
                    # on its next renewal tick, logs this stored cause, and releases the row itself
                    # (agent/background_review.py::_ReviewTurnLease). Stamped, and logged, once per hold.
                    if (
                        conn.execute(
                            _STAMP_REVIEW_YIELD_SQL,
                            (
                                now,
                                REASON_PREEMPTED_BY_TRANSCRIPT_EDIT,
                                conversation_id,
                                lease["holder"],
                            ),
                        ).rowcount
                        > 0
                    ):
                        _log_review_yield_for_edit(session_id)
                # REVIEW_LEASE_IGNORE: the append lands beside the live review.
        if (
            _ended_by_compression(
                conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()
            )
            and not allow_closed_compression_parent
        ):
            raise CompressionSessionClosedError(session_id)
