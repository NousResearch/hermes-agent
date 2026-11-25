"""Profile operations keep heartbeat owners and counts consistent."""

import json

import pytest

from persistence import Conflict
from persistence.housekeeping.profile_identity import ProfileIdentityCollision

NOW = 1767225600.0


def test_rekey_and_purge_heartbeat_owners(backend, scope_a):
    store_a = backend.open(scope_a)
    for backend_id, profile in (("old", "alpha"), ("other", "gamma"), ("unowned", "")):
        store_a.maintenance.register_backend_heartbeat(backend_id=backend_id, pid=12345, host="host", profile=profile, started_at=NOW, last_heartbeat=NOW)
    assert store_a.profile_identity.rekey_profile("alpha", "beta").gateway_heartbeats_profile == 1
    assert store_a.profile_identity.rekey_profile("alpha", "beta").gateway_heartbeats_profile == 0
    # The renamed row is the new name's: purging it removes exactly that row.
    assert store_a.profile_identity.purge_profile("beta").gateway_heartbeats == 1
    assert store_a.profile_identity.purge_profile("beta").gateway_heartbeats == 0
    assert [store_a.maintenance.clear_backend_heartbeat(bid) for bid in ("old", "other", "unowned")] == [False, True, True]


def test_rekey_collision_carries_its_text_and_keeps_the_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    for name in ("alpha", "beta"):
        key = f"agent:{name}:telegram:1"
        store_a.routing.upsert_routing_entry(key, json.dumps({"session_key": key, "session_id": name}), routing_scope="telegram")
    before = store_a.routing.load_routing_entries(routing_scope="telegram")
    with pytest.raises(ProfileIdentityCollision) as caught:
        store_a.profile_identity.rekey_profile("alpha", "beta")
    assert isinstance(caught.value, Conflict) and isinstance(caught.value, ValueError)
    assert str(caught.value) == "profile routing collision in scope 'telegram': 'agent:beta:telegram:1'"
    assert (caught.value.entity, caught.value.entity_id) == ("profile", "beta")
    assert store_a.routing.load_routing_entries(routing_scope="telegram") == before
