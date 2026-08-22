"""Tests for Executive v2 ObjectiveEngine: state machine."""

from __future__ import annotations

import pytest

from agent.executive.objective_engine import (
    ObjectiveEngine,
    PermissionError_,
    StateTransitionError,
)
from agent.executive.types import ObjectiveState


@pytest.fixture
def engine(in_memory_storage):
    return ObjectiveEngine(
        user_id="u-test",
        enabled=True,
        storage=in_memory_storage,
    )


def test_submit_creates_draft_state(engine):
    oid = engine.submit("investiga sobre X")
    state = engine.get_state(oid)
    assert state.state == ObjectiveState.DRAFT
    assert state.objective_text == "investiga sobre X"
    assert state.user_id == "u-test"


def test_normalize_transitions_to_normalized(engine):
    oid = engine.submit("investiga sobre X")
    engine.normalize(oid)
    assert engine.get_state(oid).state == ObjectiveState.NORMALIZED


def test_classify_transitions_to_classified(engine):
    oid = engine.submit("investiga sobre X")
    engine.normalize(oid)
    engine.classify(oid)
    assert engine.get_state(oid).state == ObjectiveState.CLASSIFIED


def test_discover_transitions_to_discovered(engine):
    oid = engine.submit("investiga sobre X")
    engine.normalize(oid)
    engine.classify(oid)
    engine.discover(oid)
    assert engine.get_state(oid).state == ObjectiveState.DISCOVERED


def test_generate_contract_transitions_to_contract_draft(engine):
    oid = engine.submit("implementa una API REST")
    engine.normalize(oid)
    engine.classify(oid)
    engine.discover(oid)
    engine.generate_contract(oid)
    assert engine.get_state(oid).state == ObjectiveState.CONTRACT_DRAFT


def test_persist_transitions_to_persisted(engine):
    oid = engine.submit("implementa una API REST")
    engine.normalize(oid)
    engine.classify(oid)
    engine.discover(oid)
    engine.generate_contract(oid)
    engine.persist(oid)
    assert engine.get_state(oid).state == ObjectiveState.PERSISTED
    # Also confirm via storage.
    assert engine._storage.exists(oid)


def test_idempotent_transitions(engine):
    """Repeated normalize is a no-op."""
    oid = engine.submit("text")
    engine.normalize(oid)
    state1 = engine.get_state(oid)
    engine.normalize(oid)  # no-op
    state2 = engine.get_state(oid)
    assert state1.state == state2.state == ObjectiveState.NORMALIZED
    # transition_id should not have been regenerated.
    assert state1.last_transition_id == state2.last_transition_id


def test_failure_safety(engine):
    """Failure in classify does not break state."""
    oid = engine.submit("text")
    engine.normalize(oid)
    # Corrupt normalized to force failure in classify.
    state = engine.get_state(oid)
    state.normalized = "this is not a dict"  # invalid
    engine.classify(oid)
    # Engine should set FAILED.
    assert engine.get_state(oid).state == ObjectiveState.FAILED
    assert engine.get_state(oid).last_error is not None


def test_audit_trail_records_transitions(engine):
    oid = engine.submit("text")
    engine.normalize(oid)
    engine.classify(oid)
    assert len(engine._transition_log) >= 2


def test_run_pipeline_full(engine):
    oid = engine.run_pipeline("investiga sobre X")
    state = engine.get_state(oid)
    # Pipeline should reach at least CLASSIFIED or beyond.
    assert state.state in (
        ObjectiveState.CLASSIFIED,
        ObjectiveState.DISCOVERED,
        ObjectiveState.CONTRACT_DRAFT,
    )


def test_unknown_objective_id_raises():
    from agent.executive.state_storage import ObjectiveStateStorage
    e = ObjectiveEngine(
        user_id="u", enabled=True, storage=ObjectiveStateStorage()
    )
    with pytest.raises(StateTransitionError):
        e.normalize("nonexistent")


def test_persist_optional_in_run_pipeline(engine):
    """persist_to_state_meta=False (default) means state stays in memory."""
    oid = engine.run_pipeline("investiga sobre X", persist_to_state_meta=False)
    # State is in memory but NOT in storage.
    assert engine.get_state(oid).state != ObjectiveState.PERSISTED
    assert not engine._storage.exists(oid)


def test_persist_optional_in_run_pipeline_with_persist(engine):
    """persist_to_state_meta=True writes to storage."""
    oid = engine.run_pipeline("investiga sobre X", persist_to_state_meta=True)
    assert engine.get_state(oid).state == ObjectiveState.PERSISTED
    assert engine._storage.exists(oid)


def test_archive_removes_from_memory_and_storage(engine):
    oid = engine.run_pipeline("text", persist_to_state_meta=True)
    assert engine._storage.exists(oid)
    engine.archive(oid)
    assert not engine._storage.exists(oid)
    with pytest.raises(StateTransitionError):
        engine.get_state(oid)
