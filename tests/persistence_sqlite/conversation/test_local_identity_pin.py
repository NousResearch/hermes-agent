"""The local session id the gateway mints (``persistence.authority.bindings.local_session_id``, called by
``gateway/session_local.py``) is the id the SQLite lineage check derives (``validate_local_lineage``) and the
legacy store's ``hermes_state_keys.local_identity``, for the same profile, principal and request id."""

from __future__ import annotations

from persistence.sqlite._state import keys as hermes_state_keys
from persistence import SessionId
from persistence.authority.bindings import local_session_id
from tests.contracts.authority.test_bindings_contract import assert_refusal, make_binding

IDENTITIES = [
    ("default", "user-1", "req-1"),
    ("work", "telegram:42", "launch:2026-10-29T15:00:00"),
    ("default", "usér \"quoted\"", "req,with:separators"),
    ("", "", ""),
]


def test_the_gateway_id_is_the_legacy_local_identity():
    assert local_session_id("default", "user-1", "req-1") == (
        "local-0dc04c9dd69cda75e7756e7308553dde80bdd7e25350d0cd1161fbddd03257d4")
    for identity in IDENTITIES:
        assert local_session_id(*identity) == hermes_state_keys.local_identity(*identity)


def test_the_backend_lineage_check_accepts_the_gateway_id_and_no_other(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="identity-pin")
    sid = local_session_id("default", "user-1", "req-pin")
    store.bindings.reserve(epoch=epoch, binding=make_binding(sid, route="route-pin", request_id="req-pin"), owner=None)
    assert store.bindings.validate_lineage(store.bindings.receipt(sid)) == sid
    other = local_session_id("default", "user-1", "req-other")
    store.bindings.reserve(epoch=epoch, binding=make_binding(other, route="route-other", request_id="req-pin-2"), owner=None)
    assert_refusal(lambda: store.bindings.validate_lineage(store.bindings.receipt(SessionId(other))), "storage_unavailable")
