"""A recency-ordered SQLite session listing ranks every admitted row but projects only its page's rows, and chained
and childless sessions still sort as one list."""

import sqlite3
import time
from contextlib import contextmanager

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionOrder, SessionQuery, SessionWindow
from tests.persistence_sqlite.conftest import store_for

RECENT = SessionQuery(order=SessionOrder.LAST_ACTIVE)


def test_chained_and_childless_sessions_share_one_recency_order(backend, scope_a):
    store = backend.open(scope_a)
    base = time.time() - 1000
    store.sessions.create(NewSession("root", "cli"))
    store.sessions.create(NewSession("tip", "cli", parent_session_id="root"))
    store.sessions.create(NewSession("fresh", "cli"))
    store.sessions.create(NewSession("stale", "cli"))
    store.messages.append("tip", NewMessage("user", "hello from tip", timestamp=base + 90))
    store.messages.append("fresh", NewMessage("user", "hello from fresh", timestamp=base + 60))
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE sessions SET last_activity_at = NULL, message_count = 1")
        for sid, started in (("root", base), ("tip", base + 20), ("fresh", base + 5), ("stale", base + 30)):
            connection.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started, sid))
        connection.execute("UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = 'root'", (base + 10,))
    page = store.sessions.list_sessions(RECENT, SessionWindow(limit=10))
    assert [row.session.session_id for row in page] == ["tip", "fresh", "stale"]


def _seed(store, count: int) -> None:
    for i in range(count):
        sid = f"s{i:05d}"
        store.sessions.create(NewSession(sid, "cli"))
        store.messages.append(sid, NewMessage("user", f"question {i} " + "x" * 400))
        store.messages.append(sid, NewMessage("assistant", "answer"))


def _vm_steps(store, limit: int) -> int:
    """SQLite VM instructions one sidebar-shaped listing spends."""
    steps = 0

    def tick() -> int:
        nonlocal steps
        steps += 1
        return 0

    holder = store.holder()
    borrow = holder._read_ctx

    @contextmanager
    def counted_read(*, reopen: bool = True):
        with borrow(reopen=reopen) as conn:
            conn.set_progress_handler(tick, 1)
            try:
                yield conn
            finally:
                conn.set_progress_handler(None, 0)

    holder._read_ctx = counted_read
    try:
        page = store.sessions.list_sessions(SessionQuery(order=SessionOrder.LAST_ACTIVE, min_message_count=1),
                                            SessionWindow(limit=limit), compact_rows=True)
    finally:
        del holder._read_ctx
    assert len(page) == limit
    return steps


def test_rows_outside_the_page_are_ranked_but_not_projected(backend, scope_a, scope_b):
    small, large = backend.open(scope_a), backend.open(scope_b)
    _seed(small, 30)
    _seed(large, 300)
    per_row_unprojected = (_vm_steps(large, 1) - _vm_steps(small, 1)) / 270
    per_row_projected = (_vm_steps(large, 300) - _vm_steps(small, 30)) / 270
    assert per_row_unprojected < 0.7 * per_row_projected, (per_row_unprojected, per_row_projected)


def test_cwdless_excluded_sources_filter_before_listing_limit(backend, scope_a):
    """Messaging rows without a cwd do not consume the project-tree page window."""
    store = backend.open(scope_a)
    for sid, source, cwd in (
        ("tele-no-cwd", "telegram", None),
        ("tele-empty-cwd", "telegram", ""),
        ("tele-space-cwd", "telegram", "   "),
        ("tele-with-cwd", "telegram", "/workspace/app"),
        ("tele-root-only", "telegram", None),
        ("local-row", "cli", "/workspace/other"),
    ):
        store.sessions.create(NewSession(sid, source, cwd=cwd))
        store.messages.append(sid, NewMessage("user", sid))

    # Put cwd-less messaging rows first. A post-LIMIT Python filter would return
    # an incomplete/empty page; the SQL predicate must remove them before LIMIT.
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE sessions SET git_repo_root = ?, last_activity_at = ? WHERE id = ?", ("/workspace/app", 450, "tele-root-only"))
        for last_active, sid in ((500, "tele-no-cwd"), (400, "tele-empty-cwd"), (300, "tele-space-cwd"), (200, "tele-with-cwd"), (100, "local-row")):
            connection.execute("UPDATE sessions SET last_activity_at = ? WHERE id = ?", (last_active, sid))

    query = SessionQuery(
        order=SessionOrder.LAST_ACTIVE,
        cwdless_excluded_sources=("telegram",),
    )
    rows = store.sessions.list_sessions(query, SessionWindow(limit=3))
    ids = {row.session.session_id for row in rows}

    assert ids == {"tele-root-only", "tele-with-cwd", "local-row"}
    assert "tele-no-cwd" not in ids
    assert "tele-empty-cwd" not in ids
    assert "tele-space-cwd" not in ids
