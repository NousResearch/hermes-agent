"""A borrowed read permit must outlive the handle that borrowed it (#126783).

``_read_budgets`` is a ``WeakValueDictionary``, and a ``_PathReadBudget`` is kept alive
only by the handles that reference it. A handle dropped without ``close()`` -- which
``SessionDB.__enter__``'s docstring concedes happens -- therefore takes the file's permit
ledger with it: the permit it checked out is never returned, the budget is collected, and
the next ``SessionDB`` on that same path mints a fresh ``BoundedSemaphore(_READ_POOL_MAX)``
starting again at the full ceiling. The descriptor the borrowed permit paid for is still
open and now counted nowhere, so peak descriptors per file can exceed ``_READ_POOL_MAX``
-- the leak shape the per-file ceiling exists to bound (#98573), defeated by one GC cycle.
The process-wide ledger does not behave this way: ``_process_read_permits`` is a module
global, so a leaked permit shrinks it for the life of the process. The per-file ledger must
agree with it.

The contract pinned here: permit accounting belongs to the FILE. It is released when the
permit comes back, never merely because the handle that held it became unreachable.
"""

import gc
import weakref

import pytest

from hermes_state import SessionDB, _READ_POOL_MAX


def _spare(permits) -> int:
    """Permits the semaphore can still hand out: borrow every one and hand them
    straight back, because a ``BoundedSemaphore`` offers no non-destructive count."""
    taken = 0
    while permits.acquire(blocking=False):
        taken += 1
    for _ in range(taken):
        permits.release()
    return taken


def _collect() -> None:
    """Drop unreachable handles: a single pass can defer a ``__del__`` with a cycle."""
    for _ in range(3):
        gc.collect()


@pytest.mark.requires_wal
def test_a_handle_collected_holding_a_permit_does_not_reset_the_file_ceiling(tmp_path):
    """A GC'd handle must not hand the next handle a full ceiling it already spent."""
    path = tmp_path / "state.db"
    handle = SessionDB(db_path=path)
    handle_ref = weakref.ref(handle)
    borrowed = handle._checkout_read_conn()
    assert borrowed is not None, "no pooled read path here: this test would prove nothing"
    assert _spare(handle._read_permits) == _READ_POOL_MAX - 1, "the borrow must spend a permit"

    # Abandoned, not closed: the descriptor is still open in `borrowed`, so the permit it
    # is paying for is still live accounting even though its handle is not.
    del handle
    _collect()
    assert handle_ref() is None, "the handle must die for this scenario to be the one reported"

    borrowed.close()  # the descriptor dies with its borrower; the permit stays checked out

    reopened = SessionDB(db_path=path)
    try:
        assert _spare(reopened._read_permits) == _READ_POOL_MAX - 1, (
            "the borrowed permit was forgotten when its handle was collected: the reopened "
            f"handle starts from a full {_READ_POOL_MAX}-permit ceiling while the descriptor "
            "it pays for is counted nowhere, so peak descriptors per file exceed the ceiling"
        )
    finally:
        reopened.close()


@pytest.mark.requires_wal
def test_a_handle_collected_with_an_idle_connection_returns_its_permit(tmp_path):
    """The other half of the same contract: a returned permit must free the budget again.

    The ledger must not become a strong registry that pins a quiet path forever after
    its handles are gone -- that was the reason the budget was weak to begin with.
    """
    path = tmp_path / "state.db"
    handle = SessionDB(db_path=path)
    budget_ref = weakref.ref(handle._read_budget)
    conn = handle._checkout_read_conn()
    assert conn is not None, "no pooled read path here: this test would prove nothing"
    handle._read_pool.put_nowait(conn)  # idle, still holding its permit

    del handle, conn
    _collect()
    assert budget_ref() is None, "a budget whose permits are all back must not be pinned"

    reopened = SessionDB(db_path=path)
    try:
        assert _spare(reopened._read_permits) == _READ_POOL_MAX, (
            "finalisation must return the idle connections' permits"
        )
    finally:
        reopened.close()
