"""A managed worker's waiting foreground acquire preempts a cross-process background review.

The worker reaches ``session_turn_leases`` only through ``turn.*`` receipts, so the owner's
claim transaction carries the same ``preempt_background_review`` stamp the SQLite mixin applies
for an in-process waiter: a review holder is asked to yield (never reclaimed), a plain try
stamps nothing, a foreground holder is never stamped, and a review row that outlived its
renewals is reclaimed and logged like any other stale holder. ``turn.yield_reason`` is the
read the review fork's own lease polls, so the store answers it like the owner's SessionDB.
"""

import os
import time

import pytest

from agent import review_admission
from hermes_state_compression import BACKGROUND_REVIEW_LEASE_HOLDER_MARK
from hermes_state_runtime import RuntimeStoreError, mutate_worker_execution
from tests.waiting import wait_until


def _review_holder(tag="abc"):
    return f"pid={os.getpid()}{BACKGROUND_REVIEW_LEASE_HOLDER_MARK}{tag}"


def test_waiting_worker_acquire_asks_a_review_holder_to_yield(worker):
    db, store = worker
    review = _review_holder()
    assert db.try_acquire_session_turn_lease("owned", review, ttl_seconds=60)
    assert store.try_acquire_session_turn_lease("owned", "worker-turn") is False
    assert (
        db.session_turn_lease_yield_requested("owned", review) is False
    )  # a plain try asks nothing
    assert (
        store.try_acquire_session_turn_lease(
            "owned", "worker-turn", preempt_background_review=True
        )
        is False
    )
    assert (
        db.session_turn_lease_yield_reason("owned", review)
        == review_admission.REASON_PREEMPTED_CROSS_PROCESS
    )
    # The store reads the stamp like the owner's SessionDB (the review fork's lease polls it).
    assert (
        store.session_turn_lease_yield_reason("owned", review)
        == review_admission.REASON_PREEMPTED_CROSS_PROCESS
    )
    assert store.session_turn_lease_yield_requested("owned", review) is True
    assert store.session_turn_lease_yield_reason("owned", "worker-turn") is None
    # Asked, never reclaimed: the fork's own exit releases the row.
    assert db.refresh_session_turn_lease("owned", review, ttl_seconds=60) is True


def test_waiting_worker_acquire_never_stamps_a_foreground_holder(worker):
    db, store = worker
    assert db.try_acquire_session_turn_lease(
        "owned", f"pid={os.getpid()}:turn=foreground", ttl_seconds=60
    )
    assert (
        store.try_acquire_session_turn_lease(
            "owned", "worker-turn", preempt_background_review=True
        )
        is False
    )
    assert db._read_one("SELECT yield_requested_at FROM session_turn_leases")[0] is None


def test_worker_turn_receipts_validate_the_preempt_flag(worker):
    db, store = worker

    def apply(operation, payload):
        return mutate_worker_execution(
            db,
            **store.scope,
            sequence=store.journal["next_sequence"],
            operation=operation,
            payload=payload,
        )

    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        apply(
            "turn.acquire",
            {"holder": "h", "ttl_seconds": 300, "preempt_background_review": "yes"},
        )
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        apply(
            "turn.renew",
            {"holder": "h", "ttl_seconds": 300, "preempt_background_review": True},
        )
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        apply("turn.yield_reason", {"holder": "h", "ttl_seconds": 300})


def test_waiting_worker_orchestrator_preempts_the_review_on_its_first_attempt(worker):
    """The store reuses the mixin's waiting orchestrator, which is the FOREGROUND's path: every
    failed attempt asks a review holder to yield, so self-improvement in another process never
    outranks a managed worker's turn."""
    db, store = worker
    review = _review_holder()
    assert db.try_acquire_session_turn_lease("owned", review, ttl_seconds=60)
    notices = []
    assert (
        store.acquire_session_turn_lease(
            "owned",
            "worker-turn",
            wait_seconds=0.3,
            poll_interval_seconds=0.05,
            on_wait=notices.append,
        )
        is False
    )
    assert notices
    assert (
        db.session_turn_lease_yield_reason("owned", review)
        == review_admission.REASON_PREEMPTED_CROSS_PROCESS
    )
    assert db.refresh_session_turn_lease("owned", review, ttl_seconds=60) is True


def test_worker_acquire_reclaims_an_unrenewed_review_row_and_logs_it(worker, caplog):
    db, store = worker
    review = _review_holder()
    assert db.try_acquire_session_turn_lease("owned", review, ttl_seconds=0.1)
    expiry = db.session_turn_lease_expires_at("owned", review)
    wait_until(lambda: time.time() > expiry, timeout=5)
    with caplog.at_level("INFO"):
        assert store.try_acquire_session_turn_lease("owned", "worker-turn") is True
    assert review_admission.REASON_LEASE_EXPIRED_RECLAIMED in caplog.text
    assert db._read_one("SELECT holder FROM session_turn_leases")[0] == "worker-turn"
