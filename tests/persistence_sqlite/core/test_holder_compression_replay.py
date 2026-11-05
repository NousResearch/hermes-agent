"""The holder's write replay on a live foreign compression lease: the legacy error and the contract's
``CompressionInProgress`` are both waited out within the compression budget, then raised.
"""

import os
from pathlib import Path

import pytest

from hermes_state_errors import SessionCompressionInProgressError
from persistence.core.errors import CompressionInProgress
from persistence.sqlite import _registry as registry
from persistence.sqlite._holder import StoreHolder

BUSY = {
    "legacy": lambda: SessionCompressionInProgressError("compression in progress"),
    "contract": lambda: CompressionInProgress("s-1"),
}


@pytest.fixture
def holder():
    registry.close_all()
    store = StoreHolder(Path(os.environ["HERMES_HOME"]) / "state.db")
    yield store
    store.close()
    registry.close_all()


@pytest.mark.parametrize("kind", sorted(BUSY))
def test_a_write_refused_by_a_live_compression_lease_is_replayed(holder, kind):
    attempts = []

    def write(conn):
        attempts.append(1)
        if len(attempts) < 3:
            raise BUSY[kind]()
        conn.execute("INSERT INTO state_meta (key, value) VALUES ('replayed', ?)", (kind,))
        return len(attempts)

    assert holder._execute_write(write) == 3
    assert [tuple(row) for row in holder._read_all("SELECT value FROM state_meta WHERE key = 'replayed'")] == [(kind,)]


@pytest.mark.parametrize("kind", sorted(BUSY))
def test_a_lease_still_held_after_the_compression_budget_raises(holder, kind, monkeypatch):
    monkeypatch.setattr(StoreHolder, "_COMPRESSION_BUSY_WAIT_S", 0.2)
    error = type(BUSY[kind]())
    attempts = []

    def write(conn):
        attempts.append(1)
        raise BUSY[kind]()

    with pytest.raises(error):
        holder._execute_write(write)
    assert len(attempts) > 1
