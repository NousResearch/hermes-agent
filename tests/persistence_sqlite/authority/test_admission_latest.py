"""``latest_admission`` on SQLite: one newest-first cursor per status, merged by ``seq``, read only as far as the first
row carrying the marker, every cursor closed inside its one read context."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from tests.persistence_sqlite.authority._admission_support import admit

_MARKER = "native_text_v1"
_OLDER = 30


class _CountedCursor:
    def __init__(self, cursor: Any, probe: dict[str, Any]) -> None:
        self._cursor = cursor
        self._probe = probe
        probe["cursors"].append(self)
        self.closed = False

    def __iter__(self) -> _CountedCursor:
        return self

    def __next__(self) -> Any:
        row = next(self._cursor)
        self._probe["rows"] += 1
        return row

    def close(self) -> None:
        self.closed = True
        self._cursor.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _CountedConnection:
    def __init__(self, conn: Any, probe: dict[str, Any]) -> None:
        self._conn = conn
        self._probe = probe

    def execute(self, *args: Any) -> _CountedCursor:
        return _CountedCursor(self._conn.execute(*args), self._probe)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


@contextmanager
def _counted_reads(holder: Any) -> Iterator[dict[str, Any]]:
    """Count the rows read from every cursor the read contexts opened on ``holder`` hand out, and whether each cursor
    was closed before its context exited."""
    probe: dict[str, Any] = {"rows": 0, "cursors": [], "open_at_exit": 0, "reads": 0}
    read = holder._read_ctx

    @contextmanager
    def counted_read(*args: Any, **kwargs: Any) -> Iterator[_CountedConnection]:
        probe["reads"] += 1
        with read(*args, **kwargs) as conn:
            yield _CountedConnection(conn, probe)
            probe["open_at_exit"] += sum(not cursor.closed for cursor in probe["cursors"])

    holder._read_ctx = counted_read
    try:
        yield probe
    finally:
        del holder._read_ctx


def _history(db, epoch):
    """s1: ``_OLDER`` cancelled rows without the marker, then a cancelled row carrying it, then a queued row without it;
    s2: a newer queued row carrying the marker."""
    db.create_session("s1", source="cli")
    db.create_session("s2", source="cli")
    for index in range(_OLDER):
        old = admit(db, "s1", f"old-{index}")
        db.store.admissions.cancel(epoch=epoch, admission_id=old["admission_id"])
    marked = admit(db, "s1", "marked", payload={_MARKER: {"text": "marked"}})
    db.store.admissions.cancel(epoch=epoch, admission_id=marked["admission_id"])
    admit(db, "s1", "newest")
    admit(db, "s2", "elsewhere", payload={_MARKER: {"text": "elsewhere"}})
    return marked


def test_latest_admission_stops_at_the_first_marked_row(seeded):
    _path, store, marked = seeded(_history)
    with _counted_reads(store.holder()) as probe:
        latest = store.admissions.latest_admission(session_id="s1", payload_marker=_MARKER)
    assert latest.admission_id == marked["admission_id"]
    # One cursor per status; the queued row and the marked terminal row are the only rows read, the older history
    # never leaves SQLite.
    assert (probe["reads"], len(probe["cursors"]), probe["rows"]) == (1, 4, 2)
    assert probe["open_at_exit"] == 0


def test_latest_admission_without_a_match_reads_the_whole_session_and_closes_every_cursor(seeded):
    _path, store, _marked = seeded(_history)
    with _counted_reads(store.holder()) as probe:
        assert store.admissions.latest_admission(session_id="s1", payload_marker="absent") is None
    assert (probe["reads"], len(probe["cursors"]), probe["rows"]) == (1, 4, _OLDER + 2)
    assert probe["open_at_exit"] == 0
