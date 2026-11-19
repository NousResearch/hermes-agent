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
