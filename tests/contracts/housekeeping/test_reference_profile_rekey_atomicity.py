"""Reference atomicity: a profile rename that fails at publish leaves every facet under the old name."""

import pytest

from persistence.housekeeping.profile_identity import profile_namespace
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts.conftest import require_failpoints
from tests.contracts.housekeeping.test_profile_identity_contract import NS, OLD, NEW, bind_receipts, keys_of, seed_profile


def test_a_rename_that_fails_at_publish_moves_nothing(backend, scope_a):
    require_failpoints(backend)
    store_a = backend.open(scope_a)
    seed_profile(backend, scope_a, OLD, "t1")
    api_sid, local_sid = bind_receipts(store_a, OLD, "t1")
    before = keys_of(backend, scope_a, OLD, "t1")
    receipts = (store_a.bindings.api_binding(api_sid), store_a.bindings.receipt(local_sid))
    state = authority_state(backend, scope_a)
    state.failpoints.arm(state.PUBLISH)
    with pytest.raises(InjectedFailure):
        store_a.profile_identity.rekey_profile(OLD, NEW)
    # CONTRACT: a rename is one write, as main's is one transaction; a failure at publish leaves the session, routing, topic and receipt facets all under the old name.
    assert keys_of(backend, scope_a, OLD, "t1") == before
    assert (store_a.bindings.api_binding(api_sid), store_a.bindings.receipt(local_sid)) == receipts
    assert before["session"][0].startswith(profile_namespace(OLD) + f"{NS}:")
    # The retry, with the failpoint spent, moves the whole identity.
    store_a.profile_identity.rekey_profile(OLD, NEW)
    assert keys_of(backend, scope_a, OLD, "t1", owner=NEW)["session"][0] == profile_namespace(NEW) + f"{NS}:t1"
