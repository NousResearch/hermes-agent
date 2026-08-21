import pytest

from hermes_cli.ares_candidate_lifecycle import (
    CandidateLifecycleState,
    is_gc_protected,
    transition_allowed,
)


def test_lifecycle_keeps_audit_and_activation_states_distinct():
    assert transition_allowed(
        CandidateLifecycleState.SEALED, CandidateLifecycleState.AWAITING_HOSTILE_AUDIT
    )
    assert transition_allowed(
        CandidateLifecycleState.AWAITING_HOSTILE_AUDIT,
        CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS,
    )
    assert transition_allowed(
        CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS,
        CandidateLifecycleState.AUDIT_PASSED,
    )
    assert is_gc_protected(CandidateLifecycleState.AUDIT_BLOCKED)
    assert not is_gc_protected(CandidateLifecycleState.REJECTED)


@pytest.mark.parametrize(
    "state",
    [
        CandidateLifecycleState.AWAITING_HOSTILE_AUDIT,
        CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS,
        CandidateLifecycleState.AUDIT_BLOCKED,
        CandidateLifecycleState.AUDIT_PASSED,
        CandidateLifecycleState.AWAITING_ACTIVATION,
        CandidateLifecycleState.ACTIVE,
        CandidateLifecycleState.ROLLBACK_REQUIRED,
        CandidateLifecycleState.INCIDENT_HELD,
    ],
)
def test_protected_lifecycle_states_never_become_gc_eligible(state):
    assert is_gc_protected(state)
