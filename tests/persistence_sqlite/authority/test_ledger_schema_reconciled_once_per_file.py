"""The delegation ledger reconciles state.db's schema once per database file, not once per ledger call.

The orphan sweep runs every 30 s inside a process that holds its profile's store open (the TUI gateway keeps an eager
``open_persistence`` handle for its lifetime), and each pass makes at least two ledger calls
(``recover_abandoned_delegations`` and ``sweep_orphaned_completions``), each through a handle of its own on the
process-shared store. The schema reconcile (``_init_schema``: the canonical DDL plus the column reconcile) runs when
the profile's ``StoreHolder`` opens; while the process holds the store, every ledger call borrows that one holder, so
repeated sweeps, reads and writes do not reconcile again. Each profile's file is reconciled on its own.
"""

from __future__ import annotations

import time

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from persistence.sqlite import _holder
from persistence_assembly import open_persistence, profile_scope
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def fresh_ledger():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


@pytest.fixture
def reconciles(monkeypatch):
    """The state.db path of each holder open that reconciled the schema, in order."""
    calls: list[str] = []
    real = _holder._init_schema

    def counting(holder):
        calls.append(str(holder.db_path))
        return real(holder)

    monkeypatch.setattr(_holder, "_init_schema", counting)
    return calls


@pytest.fixture
def held_store(reconciles):
    """The profile's store held open for the test, as a long-running process holds it."""
    handle = open_persistence(profile_scope(), eager=True)
    yield handle
    handle.close()


def _state_db() -> str:
    return str(get_hermes_home() / "state.db")


def _dispatch(delegation_id: str) -> None:
    ad._persist_dispatch({"delegation_id": delegation_id, "session_key": "session", "origin_ui_session_id": "ui",
                          "parent_session_id": "parent", "dispatched_at": time.time()})


def _sweep_once() -> None:
    ad.recover_abandoned_delegations()
    ad.sweep_orphaned_completions(None, now=time.time())


def test_repeated_ledger_calls_reconcile_the_schema_once(held_store, reconciles):
    _dispatch("d-1")
    for _ in range(4):
        _sweep_once()
        assert ad.get_durable_delegation("d-1") is not None
    _dispatch("d-2")
    assert reconciles == [_state_db()]


def test_each_database_is_reconciled_on_its_own(held_store, reconciles, tmp_path):
    _dispatch("d-1")
    _sweep_once()
    first = _state_db()
    other = tmp_path / "other-home"
    other.mkdir()
    token = set_hermes_home_override(str(other))
    try:
        with open_persistence(profile_scope(), eager=True):
            _dispatch("d-2")
            _sweep_once()
            _sweep_once()
            assert ad.get_durable_delegation("d-2") is not None
            second = _state_db()
    finally:
        reset_hermes_home_override(token)
    assert reconciles == [first, second]
