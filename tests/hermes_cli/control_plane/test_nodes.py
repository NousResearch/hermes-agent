from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from hermes_cli.control_plane.nodes import (
    AuthenticationFailed,
    ConcurrencyConflict,
    CredentialConflict,
    IdempotencyConflict,
    InvalidTransition,
    NodeRegistry,
    PolicyConflict,
    ReportConflict,
    write_txn,
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


def test_requested_node_id_cannot_alias_different_enrollment_key(registry):
    enroll(registry)

    with pytest.raises(IdempotencyConflict, match="different enrollment key"):
        enroll(registry, enrollment_key="request-2")

    assert [node.enrollment_key for node in registry.list()] == ["request-1"]
    assert len(registry.history("node-1")) == 1


def test_concurrent_duplicate_node_id_is_an_idempotency_conflict(tmp_path):
    db_path = tmp_path / "control-plane.db"
    NodeRegistry(db_path).connect().close()

    def enroll_key(key):
        return NodeRegistry(db_path).enroll(
            enrollment_key=key,
            node_id="node-1",
            role="worker",
            owner="ops",
            actor="operator:alice",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(enroll_key, key) for key in ("request-1", "request-2")
        ]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except IdempotencyConflict as exc:
            outcomes.append(exc)

    assert sum(not isinstance(value, Exception) for value in outcomes) == 1
    assert sum(isinstance(value, IdempotencyConflict) for value in outcomes) == 1
    assert len(NodeRegistry(db_path).list()) == 1


def test_enrollment_rejects_nested_secret_capabilities_before_persistence(registry):
    secret = "must-not-persist"

    with pytest.raises(ValueError, match="must not contain secret field"):
        enroll(
            registry,
            capabilities={"runtime": [{"metadata": {"token": secret}}]},
        )

    registry.connect().close()
    with sqlite3.connect(registry.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM managed_nodes").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM managed_node_events").fetchone()[0] == 0
        )
        assert secret not in "\n".join(conn.iterdump())


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_boundaries_reject_nested_non_finite_values_without_persistence(
    registry, non_finite
):
    with pytest.raises(ValueError, match="capabilities must contain JSON values"):
        enroll(
            registry,
            capabilities={"runtime": [{"measurement": non_finite}]},
        )

    issuance = registry.enroll(
        enrollment_key="valid-node",
        node_id="valid-node",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    credential = issuance.credential
    assert credential is not None
    baseline_events = len(registry.history("valid-node"))

    with pytest.raises(ValueError, match="capabilities must contain JSON values"):
        registry.submit_observation(
            "valid-node",
            credential=credential,
            schema_version=1,
            report_sequence=1,
            observed_at=100,
            health_state="healthy",
            capabilities={"runtime": [{"measurement": non_finite}]},
        )
    with pytest.raises(ValueError, match="capabilities must contain JSON values"):
        registry.set_policy(
            "valid-node",
            actor="operator:alice",
            schema_version=1,
            desired_health_state="healthy",
            capabilities={"thresholds": {"maximum": non_finite}},
            expected_revision=0,
        )
    with registry.connect() as conn:
        with pytest.raises(ValueError, match="Out of range float values"):
            with write_txn(conn):
                registry._append_event(
                    conn,
                    node_id="valid-node",
                    event_type="node.test",
                    actor="test",
                    from_state="enrolled",
                    to_state="enrolled",
                    revision=1,
                    occurred_at=1_000,
                    details={"nested": [{"measurement": non_finite}]},
                )

    with registry.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM managed_nodes").fetchone()[0] == 1
        assert (
            conn.execute("SELECT COUNT(*) FROM managed_node_observations").fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM managed_node_policies").fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM managed_node_events").fetchone()[0]
            == baseline_events
        )


def test_finite_numbers_round_trip_across_canonical_json_boundaries(registry):
    finite = {
        "negative": -1.25,
        "nested": [{"zero": 0.0, "positive": 1.5e100}],
    }
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
        capabilities=finite,
    )
    credential = issuance.credential
    assert credential is not None
    assert issuance.node.capabilities == finite

    observation = registry.submit_observation(
        "node-1",
        credential=credential,
        schema_version=1,
        report_sequence=1,
        observed_at=100,
        health_state="healthy",
        capabilities=finite,
    )
    policy = registry.set_policy(
        "node-1",
        actor="operator:alice",
        schema_version=1,
        desired_health_state="healthy",
        capabilities=finite,
        expected_revision=0,
    )

    assert observation.capabilities == finite
    assert policy.capabilities == finite
    assert registry.latest_observation("node-1").capabilities == finite
    assert registry.get_policy("node-1").capabilities == finite
    assert registry.verify_audit_chain()


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
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    node = issuance.node
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
    assert node.credential_status == "revoked"
    assert node.credential_revision == 2
    assert not registry.authenticate(node.id, issuance.credential)
    assert [event.event_type for event in registry.history(node.id)] == [
        "node.enrolled",
        "node.retired",
        "node.credential_revoked",
    ]
    assert registry.history(node.id)[-1].details == {
        "credential_revision": 2,
        "reason": "node retired",
    }


def test_retirement_rejects_observation_and_credential_rotation(registry):
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    registry.transition(
        "node-1",
        "retired",
        actor="operator:alice",
        expected_revision=1,
        reason="decommissioned",
    )

    with pytest.raises(AuthenticationFailed):
        registry.submit_observation(
            "node-1",
            credential=issuance.credential,
            schema_version=1,
            report_sequence=1,
            observed_at=900,
            health_state="healthy",
            capabilities={},
        )
    with pytest.raises(InvalidTransition, match="retired"):
        registry.rotate_credential(
            "node-1",
            actor="operator:alice",
            expected_credential_revision=2,
        )


def test_retirement_rolls_back_lifecycle_and_credential_together(registry, monkeypatch):
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    original_append = registry._append_event

    def fail_credential_event(conn, **kwargs):
        if kwargs["event_type"] == "node.credential_revoked":
            raise RuntimeError("injected audit failure")
        return original_append(conn, **kwargs)

    monkeypatch.setattr(registry, "_append_event", fail_credential_event)
    with pytest.raises(RuntimeError, match="injected"):
        registry.transition(
            "node-1",
            "retired",
            actor="operator:alice",
            expected_revision=1,
            reason="decommissioned",
        )

    node = registry.get("node-1")
    assert node.state == "enrolled"
    assert node.revision == 1
    assert node.credential_status == "active"
    assert registry.authenticate("node-1", issuance.credential)
    assert [event.event_type for event in registry.history("node-1")] == [
        "node.enrolled"
    ]


def test_concurrent_retirement_and_rotation_cannot_leave_retired_credential_active(
    tmp_path,
):
    db_path = tmp_path / "control-plane.db"
    issuance = NodeRegistry(db_path).enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )

    def retire():
        return NodeRegistry(db_path).transition(
            "node-1",
            "retired",
            actor="operator:alice",
            expected_revision=1,
            reason="decommissioned",
        )

    def rotate():
        return NodeRegistry(db_path).rotate_credential(
            "node-1",
            actor="operator:bob",
            expected_credential_revision=1,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(retire), executor.submit(rotate)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except (InvalidTransition, CredentialConflict) as exc:
            outcomes.append(exc)

    node = NodeRegistry(db_path).get("node-1")
    assert node.state == "retired"
    assert node.credential_status == "revoked"
    assert not NodeRegistry(db_path).authenticate("node-1", issuance.credential)
    for outcome in outcomes:
        if hasattr(outcome, "credential") and outcome.credential:
            assert not NodeRegistry(db_path).authenticate("node-1", outcome.credential)


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


def test_audit_chain_treats_corrupt_details_json_as_invalid(registry):
    enroll(registry)

    with sqlite3.connect(registry.db_path) as conn:
        conn.execute(
            "UPDATE managed_node_events SET details_json = ? WHERE sequence = 1",
            ("{not-json",),
        )

    assert registry.verify_audit_chain() is False


def test_audit_chain_detects_tail_and_all_event_deletion(registry):
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
            "DELETE FROM managed_node_events WHERE sequence = "
            "(SELECT MAX(sequence) FROM managed_node_events)"
        )
    assert registry.verify_audit_chain() is False

    with sqlite3.connect(registry.db_path) as conn:
        conn.execute("DELETE FROM managed_node_events")
    assert registry.verify_audit_chain() is False


def test_audit_head_migration_is_idempotent_and_preserves_valid_legacy_chain(
    registry,
):
    enroll(registry)
    with sqlite3.connect(registry.db_path) as conn:
        expected = conn.execute(
            "SELECT sequence, event_hash FROM managed_node_events"
        ).fetchone()
        conn.execute("DROP TABLE managed_node_audit_head")
        conn.execute("PRAGMA user_version = 0")

    registry.connect().close()
    registry.connect().close()
    with sqlite3.connect(registry.db_path) as conn:
        heads = conn.execute(
            "SELECT singleton, event_sequence, event_hash FROM managed_node_audit_head"
        ).fetchall()
    assert heads == [(1, expected[0], expected[1])]
    assert registry.verify_audit_chain()


def test_migrated_database_never_reanchors_a_deleted_audit_head(registry):
    enroll(registry)
    with sqlite3.connect(registry.db_path) as conn:
        conn.execute("DELETE FROM managed_node_audit_head")

    registry.connect().close()
    assert registry.verify_audit_chain() is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("event_sequence", "malformed"),
        ("event_hash", "not-a-hash"),
    ],
)
def test_audit_chain_fails_closed_on_malformed_head(registry, column, value):
    enroll(registry)
    with sqlite3.connect(registry.db_path) as conn:
        conn.execute(
            f"UPDATE managed_node_audit_head SET {column} = ? WHERE singleton = 1",
            (value,),
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


def test_reconcile_reads_policy_and_observation_from_one_snapshot(
    registry, monkeypatch
):
    monkeypatch.setattr("hermes_state.is_sqlite_wal_reset_vulnerable", lambda: False)
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    registry.set_policy(
        "node-1",
        actor="operator:alice",
        schema_version=1,
        desired_health_state="healthy",
        capabilities={},
        expected_revision=0,
    )
    registry.submit_observation(
        "node-1",
        credential=issuance.credential,
        schema_version=1,
        report_sequence=1,
        observed_at=900,
        health_state="healthy",
        capabilities={},
    )

    original_policy = registry._policy
    write_completed = False

    def update_both_after_policy_read(row):
        nonlocal write_completed
        policy = original_policy(row)
        if write_completed:
            return policy
        with sqlite3.connect(registry.db_path) as writer:
            writer.execute("PRAGMA foreign_keys=ON")
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                """
                UPDATE managed_node_policies
                SET desired_health_state = 'degraded', revision = 2, updated_at = 1001
                WHERE node_id = 'node-1'
                """
            )
            writer.execute(
                """
                INSERT INTO managed_node_observations (
                    node_id, report_sequence, schema_version, observed_at,
                    received_at, health_state, capabilities_json
                ) VALUES ('node-1', 2, 1, 901, 1001, 'degraded', '{}')
                """
            )
        write_completed = True
        return policy

    monkeypatch.setattr(registry, "_policy", update_both_after_policy_read)

    result = registry.reconcile("node-1")

    assert write_completed
    assert result.policy.desired_health_state == "healthy"
    assert result.observation.health_state == "healthy"
    assert result.in_sync
    assert result.drift == []
    assert registry.get_policy("node-1").desired_health_state == "degraded"
    assert registry.latest_observation("node-1").health_state == "degraded"
