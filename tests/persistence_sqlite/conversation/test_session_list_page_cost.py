"""A recency-ordered session page pays the row projection only for its own rows (#119403).

The sidebar asks ``list_sessions_rich(order_by_last_active=True)`` for a small page of a store
that can hold tens of thousands of sessions (cron runs). Ranking must look at every admitted
row, but the projection (``s.*``, the resolved prompt, the preview subquery over message
content) belongs to the page alone: when it ran for every admitted row before ``LIMIT``, a
35k-session store took 121 s per sidebar refresh.
"""

from contextlib import contextmanager

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, list_sessions_rich


def _seed(db: StoreHolder, count: int) -> None:
    for i in range(count):
        sid = f"s{i:05d}"
        create_session(db, sid, source="cli")
        append_message(db, session_id=sid, role="user", content=f"question {i} " + "x" * 400)
        append_message(db, session_id=sid, role="assistant", content="answer")


def _vm_steps(db: StoreHolder, limit: int) -> int:
    """SQLite VM instructions spent by one sidebar-shaped list call."""
    steps = 0

    def tick() -> int:
        nonlocal steps
        steps += 1
        return 0

    borrow = db._read_ctx

    @contextmanager
    def counted_read(*, reopen: bool = True):
        # Whichever connection the read borrows (pooled WAL reader or the DELETE-mode writer).
        with borrow(reopen=reopen) as conn:
            conn.set_progress_handler(tick, 1)
            try:
                yield conn
            finally:
                conn.set_progress_handler(None, 0)

    db._read_ctx = counted_read  # type: ignore[method-assign]
    try:
        rows = list_sessions_rich(db, order_by_last_active=True, limit=limit, compact_rows=True,
                                  min_message_count=1)
    finally:
        del db._read_ctx
    assert len(rows) == limit
    return steps


@pytest.fixture
def stores(tmp_path):
    small, large = StoreHolder(tmp_path / "small.db"), StoreHolder(tmp_path / "large.db")
    try:
        _seed(small, 30)
        _seed(large, 300)
        yield small, large
    finally:
        small.close()
        large.close()


def test_rows_outside_the_page_are_ranked_but_not_projected(stores):
    small, large = stores
    # Cost each extra admitted row adds to a one-row page ...
    per_row_unprojected = (_vm_steps(large, 1) - _vm_steps(small, 1)) / 270
    # ... versus a row that is ranked AND projected (every row on the page).
    per_row_projected = (_vm_steps(large, 300) - _vm_steps(small, 30)) / 270
    assert per_row_unprojected < 0.7 * per_row_projected, (per_row_unprojected, per_row_projected)
