"""Canonical session mutations against a live background-review lease.

A review fork holds the session's ``session_turn_leases`` row while it decodes an immutable
snapshot with persistence disabled, so its lease can never justify answering a user mutation
``session_busy`` for the review's lifetime. The canonical receipts carry the same policy the
storage facade applies (``hermes_state_write_guards.REVIEW_LEASE_*``): a rewind asks the holder to
yield (the rewrite invalidated its replay basis); a reset or branch publishes a new physical
target and leaves the review beside the closed transcript; a delete removes the fork's replay
basis outright and asks it to yield the same way, over the canonical receipt and the facade's
``delete_session`` alike (only a maintenance probe spares a review-held row). A foreground holder
still refuses every one of them. A detached result (an unviewed completion's note, a delegation's
delivery) is an APPEND after the review's snapshot: it lands beside the review and asks nothing.

The lease row is keyed by the conversation's compression ROOT while the fork addresses it by its
own session id, the live TIP; a delete that removed the chain must still let the fork read the
stamped cause, renew and release THAT row (``hermes_state_compression`` matches the globally
unique holder when the lineage walk can no longer reproduce the key).
"""

import os

import pytest

from agent import review_admission
from hermes_state import SessionDB
from hermes_state_compression import BACKGROUND_REVIEW_LEASE_HOLDER_MARK
from hermes_state_errors import SessionTurnLeaseLostError
from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS
import hermes_state_runtime as rt


def _review_holder(tag="abc"):
    return f"pid={os.getpid()}{BACKGROUND_REVIEW_LEASE_HOLDER_MARK}{tag}"


def _foreground_holder():
    return f"pid={os.getpid()}:turn=foreground"


def _mutate(db, epoch, session_id, operation, payload, request_id, **kw):
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
        **kw,
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


def test_delete_asks_a_background_review_holder_to_yield_and_proceeds(tmp_path):
    """A user deleting the conversation right after a turn must not be told the session is
    busy by invisible background work: the fork reviews an immutable snapshot with
    persistence disabled, so the delete loses nothing it would protect, and it removes the
    fork's replay basis outright, so the fork is asked to yield (it stops on its next renewal
    tick and releases the row itself; the lease table carries no foreign key to sessions)."""
    with SessionDB(db_path=tmp_path / "state.db") as db:
        _three_rows(db)
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("s", holder, ttl_seconds=60)
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        receipt = _mutate(db, epoch, "s", "delete", {}, "delete")
        assert "s" in receipt["deleted_ids"]
        assert db.get_session("s") is None
        assert (
            db.session_turn_lease_yield_reason("s", holder)
            == review_admission.REASON_PREEMPTED_BY_TRANSCRIPT_EDIT
        )
        # Asked to yield, never reclaimed: the fork's own exit releases the row.
        assert db.refresh_session_turn_lease("s", holder, ttl_seconds=60) is True


def _compression_chain(db, root="root", tip="tip"):
    """``root`` ended by compression, continued by ``tip``: one lease row, keyed by ``root``."""
    db.create_session(root, source="test")
    db.append_message(root, role="user", content="the start")
    db.end_session(root, "compression")
    db.create_session(tip, source="test", parent_session_id=root)
    db.append_messages_batch(
        tip,
        [
            {"role": "user", "content": "continued"},
            {"role": "assistant", "content": "reply"},
        ],
    )


def _lease_rows(db):
    return [
        (row["conversation_id"], row["holder"], row["yield_reason"])
        for row in db._read_all(
            "SELECT conversation_id, holder, yield_reason FROM session_turn_leases"
        )
    ]


def _assert_fork_still_owns_the_stamped_row(db, holder):
    """What the fork does from its session id (the deleted tip) after the chain is gone: its
    next tick reads the stamped cause and renews; its exit releases the row, so a re-created
    root id (an API client's fixed ``X-Hermes-Session-Id`` IS the root) is not kept waiting."""
    assert (
        db.session_turn_lease_yield_reason("tip", holder)
        == review_admission.REASON_PREEMPTED_BY_TRANSCRIPT_EDIT
    )
    assert db.refresh_session_turn_lease("tip", holder, ttl_seconds=60) is True
    db.release_session_turn_lease("tip", holder)
    assert _lease_rows(db) == []
    assert db.try_acquire_session_turn_lease(
        "root", _foreground_holder(), ttl_seconds=60
    )


def test_canonical_chain_delete_leaves_the_review_its_root_keyed_row(tmp_path):
    """The canonical receipt removes its targets (the tip; the closed root stays): the walk
    from the fork's session id then stops short of the root the row is keyed by."""
    with SessionDB(db_path=tmp_path / "state.db") as db:
        _compression_chain(db)
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("tip", holder, ttl_seconds=60)
        assert _lease_rows(db) == [("root", holder, None)]
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        receipt = _mutate(db, epoch, "tip", "delete", {}, "delete")
        assert "tip" in receipt["deleted_ids"]
        assert db.get_session("tip") is None
        assert _lease_rows(db) == [
            ("root", holder, review_admission.REASON_PREEMPTED_BY_TRANSCRIPT_EDIT)
        ]
        _assert_fork_still_owns_the_stamped_row(db, holder)


def test_facade_chain_delete_leaves_the_review_its_root_keyed_row(tmp_path):
    """The dashboard's delete: the whole chain, guards on."""
    with SessionDB(db_path=tmp_path / "state.db") as db:
        _compression_chain(db)
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("tip", holder, ttl_seconds=60)
        assert (
            db.delete_session(
                "tip", exclude_active_write_guards=True, include_compression_chain=True
            )
            is True
        )
        assert db.get_session("root") is None and db.get_session("tip") is None
        _assert_fork_still_owns_the_stamped_row(db, holder)


def test_delete_still_refuses_a_foreground_holder_without_stamping_it(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        _three_rows(db)
        holder = _foreground_holder()
        assert db.try_acquire_session_turn_lease("s", holder, ttl_seconds=60)
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        with pytest.raises(MUTATION_GUARD_REFUSALS):
            _mutate(db, epoch, "s", "delete", {}, "delete")
        assert db.get_session("s") is not None
        assert (
            db._read_one("SELECT yield_requested_at FROM session_turn_leases")[0]
            is None
        )


def test_facade_delete_asks_a_background_review_holder_to_yield_and_proceeds(tmp_path):
    """``hermes sessions delete``, the dashboard and the TUI delete through the facade with
    ``exclude_active_write_guards``: the same rule, while a prune sweep still spares the row."""
    from hermes_state_errors import SessionActiveWriteGuardError

    with SessionDB(db_path=tmp_path / "state.db") as db:
        _three_rows(db)
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("s", holder, ttl_seconds=60)
        assert db.delete_session("s", exclude_active_write_guards=True) is True
        assert db.get_session("s") is None
        assert (
            db.session_turn_lease_yield_reason("s", holder)
            == review_admission.REASON_PREEMPTED_BY_TRANSCRIPT_EDIT
        )
        db.release_session_turn_lease("s", holder)

        _three_rows(db, "fg")
        assert db.try_acquire_session_turn_lease(
            "fg", _foreground_holder(), ttl_seconds=60
        )
        with pytest.raises(SessionActiveWriteGuardError):
            db.delete_session("fg", exclude_active_write_guards=True)
        assert db.get_session("fg") is not None


def test_facade_delete_refused_by_a_foreground_guard_leaves_the_review_unstamped(
    tmp_path,
):
    """A delegate child's live turn still refuses the parent's delete, and the refusal rolls
    the whole write back, yield stamp included: the review of a row that stays is never asked
    to yield for a delete that did not happen."""
    from hermes_state_errors import SessionActiveWriteGuardError

    with SessionDB(db_path=tmp_path / "state.db") as db:
        _three_rows(db)
        db.create_session(
            "child",
            source="test",
            parent_session_id="s",
            model_config={"_delegate_from": "s"},
        )
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("s", holder, ttl_seconds=60)
        assert db.try_acquire_session_turn_lease(
            "child", _foreground_holder(), ttl_seconds=60
        )
        with pytest.raises(SessionActiveWriteGuardError):
            db.delete_session("s", exclude_active_write_guards=True)
        assert db.get_session("s") is not None
        assert db.session_turn_lease_yield_requested("s", holder) is False


def test_bulk_delete_asks_only_the_reviews_of_the_rows_it_removes_to_yield(tmp_path):
    """The dashboard's bulk delete skips a root a foreground guard keeps (here a delegate
    child's live turn) and removes the rest in the same transaction: the review of a removed
    row is asked to yield, the review of a kept row is left alone (its replay basis stands)."""
    with SessionDB(db_path=tmp_path / "state.db") as db:
        _three_rows(db, "gone")
        _three_rows(db, "kept")
        db.create_session(
            "kept-child",
            source="test",
            parent_session_id="kept",
            model_config={"_delegate_from": "kept"},
        )
        gone_review, kept_review = _review_holder("gone"), _review_holder("kept")
        assert db.try_acquire_session_turn_lease("gone", gone_review, ttl_seconds=60)
        assert db.try_acquire_session_turn_lease("kept", kept_review, ttl_seconds=60)
        assert db.try_acquire_session_turn_lease(
            "kept-child", _foreground_holder(), ttl_seconds=60
        )
        skipped: list[str] = []
        assert (
            db.delete_sessions(
                ["gone", "kept"], exclude_active_write_guards=True, skipped_ids=skipped
            )
            == 1
        )
        assert skipped == ["kept"]
        assert db.get_session("gone") is None
        assert db.get_session("kept") is not None
        assert (
            db.session_turn_lease_yield_reason("gone", gone_review)
            == review_admission.REASON_PREEMPTED_BY_TRANSCRIPT_EDIT
        )
        assert db.session_turn_lease_yield_requested("kept", kept_review) is False


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


def test_branch_prepare_returns_the_snapshot_beside_a_background_review(tmp_path):
    """An ACP editor-MCP session's branch runs the commit's idle rule at its prepare step,
    before the owner starts the child's servers (``gateway/session_mutations.write_mutation``
    -> ``session_local_mcp.prepare_branch_editor``). It answers a background-review lease the
    way the commit does -- the copy lands beside the review -- or a ``/branch`` typed while the
    post-turn review holds the row is refused ``session_busy`` at prepare although the commit
    would land it. A foreground holder still refuses at prepare."""
    from tests.gateway.test_unknown_resolution_physical_worker import _local_reset

    with SessionDB(db_path=tmp_path / "state.db") as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        logical = _local_reset(db, epoch, "logical", "physical")
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("physical", holder, ttl_seconds=60)
        prepared = _mutate(
            db, epoch, logical, "branch", {}, "branch", _prepare_only=True
        )
        assert set(prepared) == {"snapshot"}
        assert prepared["snapshot"]["target"] == "physical"
        assert db.session_turn_lease_yield_requested("physical", holder) is False
        assert db.refresh_session_turn_lease("physical", holder, ttl_seconds=60) is True
        db.release_session_turn_lease("physical", holder)

        assert db.try_acquire_session_turn_lease(
            "physical", _foreground_holder(), ttl_seconds=60
        )
        with pytest.raises(MUTATION_GUARD_REFUSALS):
            _mutate(db, epoch, logical, "branch", {}, "branch", _prepare_only=True)


def test_completion_note_lands_beside_a_background_review_and_still_refuses_a_foreground_holder(
    tmp_path,
):
    """An unviewed completion's note runs at the FIFO head with no viewer attached — a moment the
    session may hold a live review lease. It is an append after the review's snapshot, so it lands
    without asking the holder to yield, exactly once per admission; a foreground holder refuses it."""
    with SessionDB(db_path=tmp_path / "state.db") as db:
        _three_rows(db)
        holder = _review_holder()
        assert db.try_acquire_session_turn_lease("s", holder, ttl_seconds=60)
        row_id = db.append_completion_note(
            "s", "adm-1", "done", "internal_notification", {}
        )
        assert isinstance(row_id, int)
        assert db.session_turn_lease_yield_requested("s", holder) is False
        assert (
            db.append_completion_note(
                "s", "adm-1", "done again", "internal_notification", {}
            )
            == row_id
        )
        assert [m["content"] for m in db.get_messages("s")][-1] == "done"
        db.release_session_turn_lease("s", holder)

        assert db.try_acquire_session_turn_lease(
            "s", _foreground_holder(), ttl_seconds=60
        )
        with pytest.raises(SessionTurnLeaseLostError):
            db.append_completion_note(
                "s", "adm-2", "later", "internal_notification", {}
            )
