"""``history`` on SQLite reads its per-status streams from one snapshot: a row whose status moves backward between two
of them (here a returned claim, started back to queued) is still on the page, exactly once."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from tests.persistence_sqlite.authority._admission_support import EPOCH, admit


class _ConnectionThatWritesAfterTheFirstStream:
    def __init__(self, conn: Any, write: Callable[[], None]) -> None:
        self._conn = conn
        self._write = write
        self._streams = 0

    def execute(self, *args: Any) -> Any:
        cursor = self._conn.execute(*args)
        if "status=?" in args[0]:
            self._streams += 1
            if self._streams == 1:
                self._write()
        return cursor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


@contextmanager
def _write_after_the_first_stream(holder: Any, write: Callable[[], None]) -> Iterator[None]:
    read = holder._read_ctx

    @contextmanager
    def interleaved_read(*args: Any, **kwargs: Any) -> Iterator[_ConnectionThatWritesAfterTheFirstStream]:
        with read(*args, **kwargs) as conn:
            yield _ConnectionThatWritesAfterTheFirstStream(conn, write)

    holder._read_ctx = interleaved_read
    try:
        yield
    finally:
        del holder._read_ctx


def _terminal_started_terminal(db, epoch):
    db.create_session("s1", source="cli")
    first = admit(db, "s1", "first")
    db.store.admissions.cancel(epoch=epoch, admission_id=first["admission_id"])
    admit(db, "s1", "middle")
    middle = db.store.admissions.claim(epoch=epoch, session_id="s1")
    last = admit(db, "s1", "last")
    db.store.admissions.cancel(epoch=epoch, admission_id=last["admission_id"])
    return middle


def test_a_claim_returned_between_two_status_streams_stays_on_the_history_page(seeded):
    _path, store, middle = seeded(_terminal_started_terminal)

    def return_the_claim() -> None:
        store.admissions.return_claim(epoch=EPOCH, admission_id=middle.admission_id, generation=middle.generation,
                                      queued_owner_epoch=middle.queued_owner_epoch)

    with _write_after_the_first_stream(store.holder(), return_the_claim):
        page = store.admissions.history(session_id="s1", after_seq=0, limit=10)
    assert [row.request_id for row in page] == ["first", "middle", "last"]
    assert page[1].status == "started"
    assert store.admissions.get(admission_id=middle.admission_id).status == "queued"
