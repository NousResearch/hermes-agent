"""Binding projections preserve the stored scalar and timestamp semantics."""

from dataclasses import replace
import sqlite3

import pytest

from persistence import FenceStale, PermanentStoreError
from tests.contracts.authority.test_bindings_contract import begin, make_binding
from tests.contracts._scopes import profile_scope


@pytest.mark.parametrize("value, expected", [(7, "7"), (True, "1"), (1.2345678901234567, "1.2345678901234567"), (-(2**63), "-9223372036854775808"), (2**63 - 1, "9223372036854775807")])
def test_binding_chat_id_uses_text_affinity(backend, scope_a, value, expected):
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    binding = make_binding()
    binding = replace(binding, entry={**binding.entry, "origin": {"chat_id": value}})
    store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    assert (store_a.sessions.get(binding.session_id)).chat_id == expected
    assert (store_a.bindings.receipt(binding.session_id)).entry["origin"]["chat_id"] == value


@pytest.mark.parametrize("value, error", [(2**63, OverflowError), (-(2**63) - 1, OverflowError), ([7], PermanentStoreError), ({"id": 7}, PermanentStoreError)])
def test_binding_refuses_unbindable_chat_ids_atomically(backend, scope_a, value, error):
    store_a = backend.open(scope_a)
    epoch = begin(store_a)
    binding = make_binding()
    binding = replace(binding, entry={**binding.entry, "origin": {"chat_id": value}})
    with pytest.raises(FenceStale):
        store_a.bindings.reserve(epoch=0, binding=binding, owner=None)
    with pytest.raises(error) as caught:
        store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    if error is PermanentStoreError:
        assert isinstance(caught.value.__cause__, sqlite3.ProgrammingError)
        assert str(caught.value) == f"Error binding parameter 6: type '{type(value).__name__}' is not supported"
    else:
        assert str(caught.value) == "Python int too large to convert to SQLite INTEGER"
        assert caught.value.__cause__ is None
    assert store_a.sessions.get(binding.session_id) is None
    assert store_a.routing.load_routing_entries() == {}


def test_naive_binding_timestamp_uses_local_timezone(backend, scope_a, monkeypatch):
    store_a = backend.open(scope_a)
    import time

    binding = make_binding()
    binding = replace(binding, entry={**binding.entry, "created_at": "2026-09-16T10:00:00"})
    epoch = begin(store_a)
    try:
        with monkeypatch.context() as clock:
            clock.setenv("TZ", "Asia/Kolkata")
            time.tzset()
            store_a.bindings.reserve(epoch=epoch, binding=binding, owner=None)
            assert (store_a.sessions.get(binding.session_id)).started_at == 1789533000.0
    finally:
        time.tzset()


def test_binding_receipt_without_lineage_defaults_to_its_session():
    from tests.contracts._reference import InMemoryBackend
    from tests.contracts._reference._authority import authority_state

    backend = InMemoryBackend()
    scope = profile_scope("lineage")
    try:
        binding = make_binding()
        epoch = begin(backend.open(scope))
        backend.open(scope).bindings.reserve(epoch=epoch, binding=binding, owner=None)
        assert "lineage" not in authority_state(backend, scope).snapshot["meta"]["binding:" + binding.session_id]
        assert (backend.open(scope).bindings.receipt(binding.session_id)).lineage == (binding.session_id,)
        assert backend.open(scope).bindings.reserve(epoch=epoch, binding=binding, owner=None).binding.lineage == (binding.session_id,)
    finally:
        backend.close()
