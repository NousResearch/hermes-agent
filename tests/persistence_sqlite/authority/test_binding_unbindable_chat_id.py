"""``SessionBindingRepository.reserve`` on SQLite refuses a chat id SQLite cannot bind as main's store does, and the refusal leaves no session row and no routing entry.

An int outside SQLite's 64-bit range raises the driver's own ``OverflowError`` with no cause; a list or dict raises ``PermanentStoreError`` chained to the driver's ``ProgrammingError``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from persistence import FenceStale, PermanentStoreError
from tests.contracts.authority.test_bindings_contract import begin, make_binding


def _reserve_refused(store, value):
    epoch = begin(store)
    binding = make_binding()
    binding = replace(binding, entry={**binding.entry, "origin": {"chat_id": value}})
    with pytest.raises(FenceStale):
        store.bindings.reserve(epoch=0, binding=binding, owner=None)
    with pytest.raises(Exception) as caught:
        store.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    assert store.sessions.get(binding.session_id) is None
    assert store.routing.load_routing_entries() == {}
    return caught.value


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1])
def test_an_out_of_range_chat_id_raises_the_drivers_overflow_error(backend, scope_a, value):
    error = _reserve_refused(backend.open(scope_a), value)
    assert type(error) is OverflowError
    assert str(error) == "Python int too large to convert to SQLite INTEGER"
    assert error.__cause__ is None


@pytest.mark.parametrize("value", [[7], {"id": 7}])
def test_an_unbindable_chat_id_raises_permanent_store_error_from_the_driver(backend, scope_a, value):
    error = _reserve_refused(backend.open(scope_a), value)
    assert type(error) is PermanentStoreError
    assert str(error) == f"Error binding parameter 6: type '{type(value).__name__}' is not supported"
    assert type(error.__cause__) is sqlite3.ProgrammingError
    assert str(error.__cause__) == str(error)
