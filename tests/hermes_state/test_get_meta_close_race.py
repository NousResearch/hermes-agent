"""A ``get_meta`` read that races ``close()`` is answered, not an ``AttributeError``.

Every other caller on the writer handle reopens when ``close()`` cleared ``_conn``
mid-flight (a teardown owner racing a worker that still has work to land,
#94736). ``get_meta`` was the exception: it dereferenced ``self._conn``
unconditionally, so the same race surfaced as
``AttributeError: 'NoneType' object has no attribute 'execute'`` — a crash in
unrelated code (``fts_rebuild_step`` reads progress this way) rather than the
project's loud, bounded reopen.

Contract: a meta read whose connection was closed underneath it still returns the
stored value (or None for an absent key) through the standard reopen path.
"""

from hermes_state import SessionDB


def test_get_meta_reopens_after_close_raced_the_reader(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.set_meta("probe", "value")
        # close() cleared the handle while a reader was still in flight.
        db._conn = None

        assert db.get_meta("probe") == "value"
        assert db.get_meta("no-such-key") is None
    finally:
        db.close()
