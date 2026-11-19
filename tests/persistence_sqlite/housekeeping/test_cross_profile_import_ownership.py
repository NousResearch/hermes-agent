"""The guarded import IDs are exactly the rows the writer will insert."""

from tests.fixtures.storage_seed import seed_session


def test_import_refuses_entire_batch_if_normalized_id_belongs_to_another_profile(backend, scope_a, scope_b):
    owner, target = backend.open(scope_a), backend.open(scope_b)
    seed_session(owner, "existing-owner", "cli", title="Owned")
    fresh = {"id": "free-id", "source": "cli", "started_at": 100.0, "messages": []}
    collision = {**fresh, "id": "  existing-owner  "}

    result = target.maintenance.import_sessions([fresh, collision])

    assert not result.ok
    assert result.imported == 0
    assert target.sessions.get("existing-owner") is None
    assert target.sessions.get("free-id") is None
    assert owner.sessions.get("existing-owner").title == "Owned"
    assert target.maintenance.import_sessions([fresh]).imported == 1


def test_retired_donor_cannot_accept_more_auxiliary_usage(backend, scope_a, scope_b):
    import pytest

    from lineage_adoption import adopt_session_lineage
    from persistence import PersistenceError
    from persistence.conversation.usage import TokenCounts

    donor, target = backend.open(scope_a), backend.open(scope_b)
    seed_session(donor, "transferred", "cli")
    donor.usage.record_auxiliary_usage("transferred", "compression", model="test-model", tokens=TokenCounts(input_tokens=3))
    assert adopt_session_lineage(target, donor, "transferred")["donor_retired"]

    with pytest.raises(PersistenceError, match="owned by profile store"):
        donor.usage.record_auxiliary_usage("transferred", "compression", model="test-model", tokens=TokenCounts(input_tokens=7))
    assert donor.usage.auxiliary_usage_by_task("transferred") == target.usage.auxiliary_usage_by_task("transferred")
    with pytest.raises(PersistenceError, match="owned by profile store"):
        donor.usage.set_billing_route("transferred", provider="test-provider", base_url="https://example.invalid")
