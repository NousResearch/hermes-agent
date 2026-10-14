"""Canonical session mutations against a live background-review lease.

A review fork holds the session's ``session_turn_leases`` row while it decodes an immutable
snapshot with persistence disabled, so its lease can never justify answering a user mutation
``session_busy`` for the review's lifetime. The canonical receipts carry the same policy the
storage facade applies (``hermes_state_messages.REVIEW_LEASE_*``): a rewind asks the holder to
yield (the rewrite invalidated its replay basis); a reset or branch publishes a new physical
target and leaves the review beside the closed transcript; a delete spares the row exactly as
the maintenance sweep does. A foreground holder still refuses every one of them.
"""

import os

import pytest

from agent import review_admission
from hermes_state import SessionDB
from hermes_state_compression import BACKGROUND_REVIEW_LEASE_HOLDER_MARK
from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS
import hermes_state_runtime as rt


def _review_holder(tag="abc"):
    return f"pid={os.getpid()}{BACKGROUND_REVIEW_LEASE_HOLDER_MARK}{tag}"


def _foreground_holder():
    return f"pid={os.getpid()}:turn=foreground"


def _mutate(db, epoch, session_id, operation, payload, request_id):
    row = db.get_session(session_id)
    return rt.mutate_runtime_session(
        db,
        epoch=epoch,
        principal_id="human",
        session_id=session_id,
        request_id=request_id,
        expected_revision=row["runtime_revision"],
        expected_generation=row["runtime_generation"],
        operation=operation,
        payload=payload,
    )


def _three_rows(db, session_id="s"):
    db.create_session(session_id, source="test")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "again"},
        ],
    )
    return db.get_messages(session_id)[-1]["id"]


def test_rewind_asks_a_background_review_holder_to_yield_instead_of_refusing(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        target = _three_rows(db)
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("s", holder, ttl_seconds=60)
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        receipt = _mutate(
            db, epoch, "s", "rewind", {"target_message_id": target}, "rewind"
        )
        assert receipt["rewound_count"] == 1
        assert [m["content"] for m in db.get_messages("s")] == ["first", "reply"]
        assert (
            db.session_turn_lease_yield_reason("s", holder)
            == review_admission.REASON_PREEMPTED_BY_TRANSCRIPT_EDIT
        )
        # Asked to yield, never reclaimed: the fork's own exit releases the row.
        assert db.refresh_session_turn_lease("s", holder, ttl_seconds=60) is True


def test_rewind_still_refuses_a_foreground_holder_without_stamping_it(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        target = _three_rows(db)
        holder = _foreground_holder()
        assert db.try_acquire_session_turn_lease("s", holder, ttl_seconds=60)
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        with pytest.raises(MUTATION_GUARD_REFUSALS):
            _mutate(db, epoch, "s", "rewind", {"target_message_id": target}, "rewind")
        assert [m["content"] for m in db.get_messages("s")] == [
            "first",
            "reply",
            "again",
        ]
        assert (
            db._read_one("SELECT yield_requested_at FROM session_turn_leases")[0]
            is None
        )


def test_delete_spares_a_background_review_holder_without_asking_it_to_yield(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        _three_rows(db)
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("s", holder, ttl_seconds=60)
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        with pytest.raises(MUTATION_GUARD_REFUSALS):
            _mutate(db, epoch, "s", "delete", {}, "delete")
        assert db.get_session("s") is not None
        assert db.session_turn_lease_yield_requested("s", holder) is False


def test_reset_publishes_a_new_target_beside_a_background_review(tmp_path):
    from tests.gateway.test_unknown_resolution_physical_worker import _local_reset

    with SessionDB(db_path=tmp_path / "state.db") as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        logical = _local_reset(db, epoch, "logical", "physical")
        db.append_message("physical", "user", "kept")
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("physical", holder, ttl_seconds=60)
        receipt = _mutate(db, epoch, logical, "reset", {}, "reset")
        assert receipt["previous_target_session_id"] == "physical"
        assert receipt["target_session_id"] not in {"physical", logical}
        # The closed transcript still describes what the fork is reviewing: nothing to yield for.
        assert db.session_turn_lease_yield_requested("physical", holder) is False
        assert db.refresh_session_turn_lease("physical", holder, ttl_seconds=60) is True


def test_reset_still_refuses_a_foreground_holder(tmp_path):
    from tests.gateway.test_unknown_resolution_physical_worker import _local_reset

    with SessionDB(db_path=tmp_path / "state.db") as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        logical = _local_reset(db, epoch, "logical", "physical")
        assert db.try_acquire_session_turn_lease(
            "physical", _foreground_holder(), ttl_seconds=60
        )
        with pytest.raises(MUTATION_GUARD_REFUSALS):
            _mutate(db, epoch, logical, "reset", {}, "reset")
        assert db.get_session("physical")["end_reason"] is None


def test_branch_copies_the_transcript_beside_a_background_review(tmp_path):
    from tests.gateway.test_unknown_resolution_physical_worker import _local_reset

    with SessionDB(db_path=tmp_path / "state.db") as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        logical = _local_reset(db, epoch, "logical", "physical")
        db.append_messages_batch(
            "physical",
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        )
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("physical", holder, ttl_seconds=60)
        receipt = _mutate(db, epoch, logical, "branch", {}, "branch")
        assert receipt["copied_messages"] == 2
        assert [
            m["content"] for m in db.get_messages(receipt["branched_session_id"])
        ] == ["q", "a"]
        assert db.session_turn_lease_yield_requested("physical", holder) is False
        assert db.refresh_session_turn_lease("physical", holder, ttl_seconds=60) is True
