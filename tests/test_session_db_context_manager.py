"""``SessionDB`` must support ``with``, so an owning scope releases its fds.

A SessionDB handle cannot be released by dropping the last reference. Once its
background token writer starts, the instance pins ITSELF two ways: the writer
thread's target is a bound method, and ``queue_token_counts`` registers
``atexit.register(self._drain_token_queue_at_exit)``, which only ``close()``
unregisters. A dropped-but-pinned handle keeps its ``state.db``/``-wal``/
``-shm`` descriptors for the life of the process, and ``__del__`` never runs
for it -- so the safety net is dead code for exactly the instances that leak.

That is why owning call sites are expected to close explicitly; the ownership
comments in ``run_agent.py`` and ``tui_gateway/methods_session.py`` say so in
those words. This module pins the ergonomic half of that contract: an owner can
scope a handle with ``with`` and be exception-safe by construction, instead of
hand-writing a ``try/finally`` at each of the ~59 instantiation sites (#88033).

Following ``test_session_db_read_conn_pool.py``, these assert on the
``sqlite_safe_read`` tracking registry rather than on raw descriptor counts:
SQLite's unix VFS parks a closed descriptor on a per-inode reuse list while any
connection still holds POSIX locks on that inode, so descriptor counts lag the
real connection count and make such assertions flaky.
"""

import pytest

from hermes_state import SessionDB


def _live_count(path) -> int:
    """Live-connection count the tracking registry holds for *path*."""
    import hermes_cli.sqlite_safe_read as mod

    with mod._live_lock:
        return mod._live_connections.get(mod._key(path), 0)


def test_with_block_closes_the_handle(tmp_path):
    """The whole point: leaving the scope releases the connection."""
    path = tmp_path / "state.db"

    with SessionDB(db_path=path) as db:
        db.create_session(session_id="s1", source="cli", model="m")
        db.append_message("s1", role="user", content="hello")
        assert db._conn is not None
        assert _live_count(path) > 0

    assert db._conn is None
    assert _live_count(path) == 0


def test_enter_returns_the_same_handle(tmp_path):
    """``with SessionDB(...) as db`` must bind the instance, not a wrapper.

    Returning anything else would silently break every attribute access in the
    body, so this is cheap insurance on the one line that is easy to get wrong.
    """
    db = SessionDB(db_path=tmp_path / "state.db")

    with db as entered:
        assert entered is db

    assert db._conn is None


def test_exception_inside_the_block_still_closes_and_propagates(tmp_path):
    """Exception safety is the reason to prefer ``with`` over a bare close().

    The handle must be released on the failure path (the path that leaked
    hardest, since it is the one callers forget), and ``__exit__`` must NOT
    suppress: swallowing a caller's error to release a descriptor would trade
    one bug for a worse one.
    """
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)

    with pytest.raises(ValueError, match="boom"):
        with db:
            db.create_session(session_id="s1", source="cli", model="m")
            raise ValueError("boom")

    assert db._conn is None
    assert _live_count(path) == 0


def test_closing_inside_the_block_leaves_the_exit_clean(tmp_path):
    """``close()`` is documented idempotent; the scope must not fight a caller.

    A call site converted to ``with`` may still hold an explicit ``close()``
    (or hit one on an internal error path). The second close from ``__exit__``
    must be a no-op rather than an error.
    """
    path = tmp_path / "state.db"

    with SessionDB(db_path=path) as db:
        db.create_session(session_id="s1", source="cli", model="m")
        db.close()
        assert db._conn is None

    assert _live_count(path) == 0
