from __future__ import annotations

import sqlite3

import pytest

from hermes_cli.control_plane.nodes import (
    ConcurrencyConflict,
    IdempotencyConflict,
    InvalidTransition,
    NodeRegistry,
)


@pytest.fixture
def registry(tmp_path):
    return NodeRegistry(tmp_path / "control-plane.db", clock=lambda: 1_000)


def enroll(registry, **overrides):
    values = {
        "enrollment_key": "request-1",
        "node_id": "node-1",
        "role": "worker",
        "owner": "ops",
        "actor": "operator:alice",
        "capabilities": {"os": "linux", "gpu": False},
    }
    values.update(overrides)
    return registry.enroll(**values)


def test_enrollment_is_durable_and_idempotent(registry):
    first = enroll(registry)
    retry = enroll(registry)

    assert retry == first
    assert registry.get("node-1") == first
    assert registry.list() == [first]
    assert len(registry.history("node-1")) == 1
    assert first.state == "enrolled"
    assert first.revision == 1


def test_enrollment_key_cannot_alias_different_facts(registry):
    enroll(registry)

    with pytest.raises(IdempotencyConflict):
        enroll(registry, owner="another-team")


def test_retry_without_caller_supplied_node_id_keeps_generated_identity(registry):
    first = enroll(registry, node_id=None)
    retry = enroll(registry, node_id=None)

    assert retry.id == first.id
    assert len(registry.history(first.id)) == 1


def test_lifecycle_is_explicit_and_audited(registry):
    node = enroll(registry)
    node = registry.transition(
        node.id,
        "active",
        actor="service:reconciler",
        expected_revision=node.revision,
        reason="attestation accepted",
    )
    node = registry.transition(
        node.id,
        "quarantined",
        actor="operator:alice",
        expected_revision=node.revision,
        reason="configuration drift",
    )
    node = registry.transition(
        node.id,
        "recovering",
        actor="operator:alice",
        expected_revision=node.revision,
        reason="repair started",
    )
    node = registry.transition(
        node.id,
        "active",
        actor="service:reconciler",
        expected_revision=node.revision,
        reason="policy converged",
    )

    assert node.state == "active"
    assert node.revision == 5
    assert [event.to_state for event in registry.history(node.id)] == [
        "enrolled",
        "active",
        "quarantined",
        "recovering",
        "active",
    ]
    assert registry.verify_audit_chain() is True


def test_illegal_transition_does_not_mutate_state(registry):
    node = enroll(registry)

    with pytest.raises(InvalidTransition):
        registry.transition(
            node.id,
            "recovering",
            actor="operator:alice",
            expected_revision=1,
            reason="skip quarantine",
        )

    assert registry.get(node.id) == node
    assert len(registry.history(node.id)) == 1


def test_stale_revision_is_rejected(registry):
    node = enroll(registry)
    registry.transition(
        node.id,
        "active",
        actor="service:reconciler",
        expected_revision=1,
        reason="ready",
    )

    with pytest.raises(ConcurrencyConflict):
        registry.transition(
            node.id,
            "retired",
            actor="operator:alice",
            expected_revision=1,
            reason="stale request",
        )


def test_retired_node_is_terminal(registry):
    node = enroll(registry)
    node = registry.transition(
        node.id,
        "retired",
        actor="operator:alice",
        expected_revision=1,
        reason="decommissioned",
    )

    with pytest.raises(InvalidTransition):
        registry.transition(
            node.id,
            "active",
            actor="operator:alice",
            expected_revision=node.revision,
            reason="accidental reuse",
        )


def test_audit_chain_detects_tampering(registry):
    node = enroll(registry)
    registry.transition(
        node.id,
        "active",
        actor="service:reconciler",
        expected_revision=1,
        reason="ready",
    )

    with sqlite3.connect(registry.db_path) as conn:
        conn.execute(
            "UPDATE managed_node_events SET actor = ? WHERE sequence = 1",
            ("attacker",),
        )

    assert registry.verify_audit_chain() is False


def test_state_filter_and_unknown_node(registry):
    enroll(registry)

    assert [node.id for node in registry.list(state="enrolled")] == ["node-1"]
    assert registry.list(state="active") == []
    assert registry.get("missing") is None
    with pytest.raises(KeyError):
        registry.transition(
            "missing",
            "active",
            actor="operator:alice",
            expected_revision=1,
            reason="not present",
        )
