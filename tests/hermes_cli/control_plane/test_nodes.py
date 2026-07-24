from __future__ import annotations

import sqlite3

import pytest

from hermes_cli.control_plane.nodes import (
    AuthenticationFailed,
    ConcurrencyConflict,
    IdempotencyConflict,
    InvalidTransition,
    NodeRegistry,
    PolicyConflict,
    ReportConflict,
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
    return registry.enroll(**values).node


def test_enrollment_is_durable_and_idempotent(registry):
    first_issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
        capabilities={"os": "linux", "gpu": False},
    )
    retry_issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
        capabilities={"os": "linux", "gpu": False},
    )
    first = first_issuance.node
    retry = retry_issuance.node

    assert retry == first
    assert first_issuance.credential
    assert retry_issuance.credential is None
    assert registry.authenticate(first.id, first_issuance.credential)
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


def test_authentication_rejection_rotation_revocation_and_persistence(tmp_path):
    db_path = tmp_path / "control-plane.db"
    registry = NodeRegistry(db_path, clock=lambda: 1_000)
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    original = issuance.credential
    assert original is not None
    assert registry.authenticate("node-1", original)
    assert not registry.authenticate("node-1", f"{original}x")
    assert not registry.authenticate("missing", original)

    reopened = NodeRegistry(db_path, clock=lambda: 2_000)
    assert reopened.authenticate("node-1", original)
    rotated = reopened.rotate_credential(
        "node-1",
        actor="operator:bob",
        expected_credential_revision=1,
    )
    replacement = rotated.credential
    assert replacement is not None
    assert replacement != original
    assert not registry.authenticate("node-1", original)
    assert registry.authenticate("node-1", replacement)

    revoked = registry.revoke_credential(
        "node-1",
        actor="operator:bob",
        expected_credential_revision=2,
    )
    assert revoked.credential_status == "revoked"
    assert not reopened.authenticate("node-1", replacement)
    assert registry.verify_audit_chain()


def test_credentials_and_verifiers_are_not_disclosed(registry):
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    raw = issuance.credential
    assert raw is not None
    verifier = registry._credential_verifier(raw)

    public_values = (
        registry.get("node-1"),
        registry.list(),
        registry.history("node-1"),
    )
    serialized = repr(public_values)
    assert raw not in serialized
    assert verifier not in serialized

    with sqlite3.connect(registry.db_path) as conn:
        stored = conn.execute(
            "SELECT verifier FROM managed_node_credentials WHERE node_id = ?",
            ("node-1",),
        ).fetchone()[0]
        assert stored == verifier
        assert raw not in "\n".join(conn.iterdump())


def test_legacy_node_is_migrated_revoked_with_audited_rotation_path(registry):
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    with sqlite3.connect(registry.db_path) as conn:
        conn.execute(
            "DELETE FROM managed_node_credentials WHERE node_id = ?", ("node-1",)
        )

    migrated = registry.get("node-1")
    assert migrated.credential_status == "revoked"
    assert not registry.authenticate("node-1", issuance.credential)
    assert registry.history("node-1")[-1].event_type == (
        "node.credential_migrated_revoked"
    )
    assert registry.verify_audit_chain()

    rotated = registry.rotate_credential(
        "node-1",
        actor="operator:alice",
        expected_credential_revision=1,
    )
    assert rotated.credential is not None
    assert registry.authenticate("node-1", rotated.credential)


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


def test_authenticated_observations_reject_replay_and_persist_latest(tmp_path):
    db_path = tmp_path / "control-plane.db"
    registry = NodeRegistry(db_path, clock=lambda: 2_000)
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    credential = issuance.credential
    assert credential

    first = registry.submit_observation(
        "node-1",
        credential=credential,
        schema_version=1,
        report_sequence=1,
        observed_at=1_900,
        health_state="healthy",
        capabilities={"os": "linux", "runtime": {"docker": True}},
    )
    second = registry.submit_observation(
        "node-1",
        credential=credential,
        schema_version=1,
        report_sequence=3,
        observed_at=1_800,
        health_state="degraded",
        capabilities={"os": "linux"},
    )
    assert first.report_sequence == 1
    assert NodeRegistry(db_path).latest_observation("node-1") == second

    for sequence in (3, 2):
        with pytest.raises(ReportConflict, match="greater than 3"):
            registry.submit_observation(
                "node-1",
                credential=credential,
                schema_version=1,
                report_sequence=sequence,
                observed_at=2_000,
                health_state="healthy",
                capabilities={},
            )
    assert registry.latest_observation("node-1") == second


def test_observation_auth_validation_non_disclosure_and_audit(registry):
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    credential = issuance.credential
    assert credential
    with pytest.raises(AuthenticationFailed):
        registry.submit_observation(
            "node-1",
            credential=f"{credential}x",
            schema_version=1,
            report_sequence=1,
            observed_at=900,
            health_state="healthy",
            capabilities={},
        )
    with pytest.raises(ValueError, match="unsupported observation"):
        registry.submit_observation(
            "node-1",
            credential=credential,
            schema_version=2,
            report_sequence=1,
            observed_at=900,
            health_state="healthy",
            capabilities={},
        )
    with pytest.raises(ValueError, match="must not contain secret field"):
        registry.submit_observation(
            "node-1",
            credential=credential,
            schema_version=1,
            report_sequence=1,
            observed_at=900,
            health_state="healthy",
            capabilities={"runtime": {"token": "must-not-persist"}},
        )

    report = registry.submit_observation(
        "node-1",
        credential=credential,
        schema_version=1,
        report_sequence=1,
        observed_at=900,
        health_state="healthy",
        capabilities={"gpu": False},
    )
    serialized = repr((report, registry.history("node-1")))
    assert credential not in serialized
    with sqlite3.connect(registry.db_path) as conn:
        assert credential not in "\n".join(conn.iterdump())
    assert registry.history("node-1")[-1].actor == "node:node-1"
    assert registry.verify_audit_chain()
    registry.revoke_credential(
        "node-1",
        actor="operator:alice",
        expected_credential_revision=1,
    )
    with pytest.raises(AuthenticationFailed):
        registry.submit_observation(
            "node-1",
            credential=credential,
            schema_version=1,
            report_sequence=2,
            observed_at=950,
            health_state="healthy",
            capabilities={},
        )


def test_policy_revision_and_reconciliation_drift_then_convergence(registry):
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    policy = registry.set_policy(
        "node-1",
        actor="operator:alice",
        schema_version=1,
        desired_health_state="healthy",
        capabilities={"os": "linux", "runtime": {"docker": True}},
        expected_revision=0,
    )
    assert registry.get_policy("node-1") == policy
    with pytest.raises(PolicyConflict, match="revision 1"):
        registry.set_policy(
            "node-1",
            actor="operator:bob",
            schema_version=1,
            desired_health_state="healthy",
            capabilities={},
            expected_revision=0,
        )

    credential = issuance.credential
    assert credential
    registry.submit_observation(
        "node-1",
        credential=credential,
        schema_version=1,
        report_sequence=1,
        observed_at=900,
        health_state="degraded",
        capabilities={"os": "linux"},
    )
    drifted = registry.reconcile("node-1")
    assert not drifted.in_sync
    assert [item["path"] for item in drifted.drift] == [
        "health_state",
        "capabilities.runtime",
    ]

    registry.submit_observation(
        "node-1",
        credential=credential,
        schema_version=1,
        report_sequence=2,
        observed_at=950,
        health_state="healthy",
        capabilities={
            "os": "linux",
            "runtime": {"docker": True},
            "extra": "allowed",
        },
    )
    converged = registry.reconcile("node-1")
    assert converged.in_sync
    assert converged.drift == []
    assert [event.event_type for event in registry.history("node-1")] == [
        "node.enrolled",
        "node.policy_updated",
        "node.observation_accepted",
        "node.observation_accepted",
    ]
    assert registry.verify_audit_chain()
