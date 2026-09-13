"""M8 T43: Rollback — durable phases, preimage fence, no early reopen."""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_rollback import (
    FENCE_CLOSED_PHASES,
    ROLLBACK_PHASES,
    RollbackError,
    RollbackOperation,
    advance_phase,
    authorize_reopen,
    check_no_early_reopen,
    check_preimage_compatible,
    create_rollback,
    resume_rollback,
)


def make_op():
    """Create a standard rollback operation for testing."""
    return create_rollback(
        operation_id="rb-001",
        owner_token_hash="tok-hash",
        source_live_hash="a" * 64,
        source_code_hash="src" + "a" * 61,
        target_live_hash="b" * 64,
        target_code_hash="tgt" + "b" * 61,
    )


def test_rollback_state_machine():
    """T43: rollback durable phases — no early reopen, no skip.

    All phases advance in order. Fence stays closed.
    Reopen requires additional David-level token.
    """
    op = make_op()

    # All phases in order
    assert op.current_phase == "gate_submits"
    assert op.fence_generation == 1  # fence closed
    assert op.phase_revision == 0

    # Advance through all phases
    phases_to_advance = list(ROLLBACK_PHASES[1:])  # skip gate_submits (starting point)
    for phase in phases_to_advance:
        if phase == "reopened":
            # Reopened requires David-level token first
            authorize_reopen(op, david_token="david-approval")
            advance_phase(op, target_phase=phase, owner_token_hash="tok-hash", expected_revision=op.phase_revision)
        else:
            advance_phase(op, target_phase=phase, owner_token_hash="tok-hash", expected_revision=op.phase_revision)

    assert op.current_phase == "reopened"
    assert op.fence_generation == 0  # fence released
    assert op.is_active is True
    assert len(op.completed_phases) == len(ROLLBACK_PHASES) - 1

    # Phase revision incremented for each advance
    assert op.phase_revision == len(ROLLBACK_PHASES) - 1


def test_rollback_no_skip():
    """T43: cannot skip phases."""
    op = make_op()

    # Try to skip from gate_submits to workers_stopped
    with pytest.raises(RollbackError, match="phase_skip_not_allowed"):
        advance_phase(op, target_phase="workers_stopped", owner_token_hash="tok-hash", expected_revision=0)

    # Try to go backwards
    advance_phase(op, target_phase="fence_draining", owner_token_hash="tok-hash", expected_revision=0)
    with pytest.raises(RollbackError, match="phase_rollback_not_allowed"):
        advance_phase(op, target_phase="gate_submits", owner_token_hash="tok-hash", expected_revision=1)


def test_rollback_owner_cas():
    """T43: owner token mismatch rejected."""
    op = make_op()

    with pytest.raises(RollbackError, match="owner_token_mismatch"):
        advance_phase(op, target_phase="fence_draining", owner_token_hash="wrong", expected_revision=0)

    with pytest.raises(RollbackError, match="phase_revision_mismatch"):
        advance_phase(op, target_phase="fence_draining", owner_token_hash="tok-hash", expected_revision=99)


def test_rollback_fence_closed():
    """T43: fence stays closed from fence_draining through verified.

    No early reopen.
    """
    op = make_op()

    # gate_submits → fence_draining
    advance_phase(op, target_phase="fence_draining", owner_token_hash="tok-hash", expected_revision=0)
    assert check_no_early_reopen(op) is True
    assert op.fence_generation >= 1

    # Through verified
    for phase in ["workers_stopped", "leases_revoked", "snapshot_sealed",
                   "code_switched", "old_writer_receipts_verified", "verified"]:
        advance_phase(op, target_phase=phase, owner_token_hash="tok-hash", expected_revision=op.phase_revision)
        assert check_no_early_reopen(op) is True  # fence stays closed

    # Try to reopen without David-level token
    with pytest.raises(RollbackError, match="reopen_requires_additional_token"):
        advance_phase(op, target_phase="reopened", owner_token_hash="tok-hash", expected_revision=op.phase_revision)

    # Authorize with David-level token
    authorize_reopen(op, david_token="david-approval")
    assert op.is_active is True

    # Now reopen succeeds
    advance_phase(op, target_phase="reopened", owner_token_hash="tok-hash", expected_revision=op.phase_revision)
    assert op.fence_generation == 0
    assert check_no_early_reopen(op) is True  # properly released


def test_rollback_preimage_compatible():
    """T43: compatible preimage means M0 fence-aware source hash.

    Pre-M0 code can never be a rollback target.
    """
    # M0-aware source → compatible
    assert check_preimage_compatible("src_hash", "tgt_hash", is_m0_aware=True) is True

    # Pre-M0 source → not compatible
    assert check_preimage_compatible("src_hash", "tgt_hash", is_m0_aware=False) is False

    # Missing hashes → not compatible
    assert check_preimage_compatible("", "tgt_hash") is False
    assert check_preimage_compatible("src_hash", "") is False

    # Same hash (not a rollback) → not compatible
    assert check_preimage_compatible("same", "same") is False


def test_rollback_resume():
    """T43: crash resumes from durable phase/revision."""
    op = make_op()
    advance_phase(op, target_phase="fence_draining", owner_token_hash="tok-hash", expected_revision=0)
    advance_phase(op, target_phase="workers_stopped", owner_token_hash="tok-hash", expected_revision=1)

    # Resume returns current phase
    phase = resume_rollback(op, owner_token_hash="tok-hash")
    assert phase == "workers_stopped"

    # Wrong owner rejected
    with pytest.raises(RollbackError, match="resume_owner_mismatch"):
        resume_rollback(op, owner_token_hash="wrong")


def test_rollback_invalid_phase():
    """T43: invalid phase name rejected."""
    op = make_op()
    with pytest.raises(RollbackError, match="invalid_phase"):
        advance_phase(op, target_phase="nonexistent", owner_token_hash="tok-hash", expected_revision=0)
