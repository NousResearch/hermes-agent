"""The guarded import IDs are exactly the rows the writer will insert."""

from tests.fixtures.storage_seed import seed_session


def test_import_normalizes_ids_inside_the_explicit_target_profile(backend, scope_a, scope_b):
    owner, target = backend.open(scope_a), backend.open(scope_b)
    seed_session(owner, "existing-owner", "cli", title="Owned")
    fresh = {"id": "free-id", "source": "cli", "started_at": 100.0, "messages": []}
    collision = {**fresh, "id": "  existing-owner  "}

    result = target.maintenance.import_sessions([fresh, collision])

    assert result.ok
    assert result.imported == 2
    assert set(result.imported_ids) == {"existing-owner", "free-id"}
    assert target.sessions.get("existing-owner").title is None
    assert target.sessions.get("free-id") is not None
    assert owner.sessions.get("existing-owner").title == "Owned"


def test_retired_donor_cannot_accept_more_auxiliary_usage(backend, scope_a, scope_b):
    import pytest

    from lineage_adoption import adopt_session_lineage
    from persistence import PersistenceError
    from persistence.conversation.usage import TokenCounts

    donor, target = backend.open(scope_a), backend.open(scope_b)
    seed_session(donor, "transferred", "cli")
    donor.usage.record_auxiliary_usage("transferred", "compression", model="test-model", tokens=TokenCounts(input_tokens=3))
    assert adopt_session_lineage(target, donor, "transferred")["donor_retired"]

    with pytest.raises(PersistenceError, match="transferred to another profile store"):
        donor.usage.record_auxiliary_usage("transferred", "compression", model="test-model", tokens=TokenCounts(input_tokens=7))
    assert donor.usage.auxiliary_usage_by_task("transferred") == target.usage.auxiliary_usage_by_task("transferred")
    with pytest.raises(PersistenceError, match="transferred to another profile store"):
        donor.sessions.reopen("transferred")
    with pytest.raises(PersistenceError, match="transferred to another profile store"):
        donor.usage.set_billing_route("transferred", provider="test-provider", base_url="https://example.invalid")

    # A legacy write can clear the session stamp, but the donor transfer
    # receipt was committed independently of that mutable row.
    from tests.fixtures.storage_holder import raw_holder

    raw_holder(donor)._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET ended_at=NULL, end_reason=NULL WHERE id=?", ("transferred",),
    ))
    with pytest.raises(PersistenceError, match="transferred to another profile store"):
        donor.sessions.update_model("transferred", "late-model")


def test_adoption_refuses_an_existing_destination_without_a_transfer_record(backend, scope_a, scope_b):
    from lineage_adoption import adopt_session_lineage

    donor, destination = backend.open(scope_a), backend.open(scope_b)
    seed_session(donor, "coincident", "cli", title="Donor")
    seed_session(destination, "coincident", "cli", title="Independent")
    result = adopt_session_lineage(destination, donor, "coincident")
    assert not result["adopted"]
    assert not result["donor_retired"]
    assert donor.sessions.get("coincident").ended_at is None
    assert destination.sessions.get("coincident").title == "Independent"


def test_adoption_ignores_independent_same_id_in_a_third_profile(backend, scope_a, scope_b):
    from hermes_constants import get_default_hermes_root
    from lineage_adoption import adopt_session_lineage
    from persistence_assembly import open_persistence, profile_scope

    donor, destination = backend.open(scope_a), backend.open(scope_b)
    other_home = get_default_hermes_root() / "profiles" / "independent"
    other_home.mkdir()
    with open_persistence(profile_scope(home=other_home)) as other:
        seed_session(donor, "coincident", "cli", title="Donor")
        seed_session(other, "coincident", "cli", title="Independent")
        result = adopt_session_lineage(destination, donor, "coincident")
        assert result["adopted"] and result["donor_retired"]
        assert donor.sessions.get("coincident").end_reason == "adopted_by_profile"
        assert destination.sessions.get("coincident").title == "Donor"
        assert other.sessions.get("coincident").title == "Independent"
        assert other.sessions.get("coincident").ended_at is None
