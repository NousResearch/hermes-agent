"""Seeds and probes the admission backend tests share: the seed calls, run through the backend (:class:`BackendSeed`), record projections, and a failure injected inside a verb's write after its first mutating statement."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import pytest

from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.persistence_sqlite.authority._support import dump

#: The ``(epoch, instance)`` the seeds begin.
EPOCH = 1


class InjectedFailure(Exception):
    """Raised inside a verb's write by :func:`failing_write`."""


def record_dict(record: Any) -> Any:
    """A contract record as a dict of its fields without ``queued_owner_epoch``; ``None`` and ``False`` as they are."""
    if record is None or record is False:
        return record
    values = dataclasses.asdict(record)
    values.pop("queued_owner_epoch", None)
    return values


class BackendSeed:
    """The ``SessionDB``-shaped calls the shared seeds make, run through the backend's verbs on ``store`` (raw SQL on its holder)."""

    def __init__(self, store) -> None:
        self.store = store
        self.db_path = store.store_path

    def create_session(self, session_id, source, **fields):
        seed_session(self.store, session_id, source, **fields)

    def append_message(self, session_id, **message):
        return seed_messages(self.store, session_id, message)[0]

    def delete_session(self, session_id):
        self.store.sessions.delete(session_id, media_root=str(self.db_path.parent / "media"))

    def _execute_write(self, fn):
        return self.store.holder()._execute_write(fn)


def begin_epoch(db, instance_id):
    return db.store.runtime.begin_epoch(instance_id=instance_id)


def recover(db, epoch):
    return db.store.admissions.recover(epoch=epoch)


def admit(db, session_id, request_id, payload=None, principal_id="p", epoch=EPOCH, **kwargs):
    payload = payload or {"text": request_id}
    record, _created = db.store.admissions.admit(epoch=epoch, principal_id=principal_id, session_id=session_id,
                                                 request_id=request_id, payload=payload, **kwargs)
    return record_dict(record)


def claim(db, session_id, epoch=EPOCH):
    return record_dict(db.store.admissions.claim(epoch=epoch, session_id=session_id))


def get(db, admission_id):
    return record_dict(db.store.admissions.get(admission_id=admission_id))


def settle(db, row, outcome="completed", result=None, epoch=EPOCH):
    """Settle the claimed ``row`` under ``epoch``."""
    call = dict(epoch=epoch, admission_id=row["admission_id"], generation=row["generation"], outcome=outcome,
                result=result)
    return record_dict(db.store.admissions.settle(**call))


def steer(db, row, claimed, epoch=EPOCH):
    """Hand the queued ``row`` to the running ``claimed`` row."""
    call = dict(epoch=epoch, admission_id=row["admission_id"], generation=claimed["generation"])
    record, transferred = db.store.admissions.steer(**call)
    return record_dict(record), transferred


def register(db, session_id, generation, execution_id="exec-1", epoch=EPOCH):
    """Register a compute worker on ``session_id`` at ``generation``."""
    call = dict(epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, kind="compute",
                adoption_secret="secret-1")
    return db.store.worker_executions.register(**call)


class _FailingConnection:
    """A connection whose first statement after the first mutating one raises :class:`InjectedFailure`."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self.mutated = False

    def _statement(self, sql: str) -> None:
        if self.mutated:
            raise InjectedFailure(sql)

    def _after(self, sql: str) -> None:
        if sql.lstrip().split(None, 1)[0].upper() in ("INSERT", "UPDATE", "DELETE", "REPLACE"):
            self.mutated = True

    def execute(self, sql, *args):
        self._statement(sql)
        cursor = self._conn.execute(sql, *args)
        self._after(sql)
        return cursor

    def executemany(self, sql, *args):
        self._statement(sql)
        cursor = self._conn.executemany(sql, *args)
        self._after(sql)
        return cursor

    def __getattr__(self, name):
        return getattr(self._conn, name)


@contextmanager
def failing_write(holder) -> Iterator[None]:
    """Every write on ``holder`` fails right after its first mutating statement (or at the end of its body when
    nothing follows it), so the transaction rolls back with that statement applied."""
    write = holder._execute_write

    def wrapped(fn, *args, **kwargs):
        def body(conn):
            proxy = _FailingConnection(conn)
            fn(proxy)
            assert proxy.mutated, "the write ran no mutating statement"
            raise InjectedFailure("end of the write")
        return write(body, *args, **kwargs)

    holder._execute_write = wrapped
    try:
        yield
    finally:
        del holder._execute_write


def assert_rolls_back(path, holder, call: Callable[[], Any]) -> None:
    """``call`` fails inside its write after a mutating statement and leaves every authority table unchanged."""
    before = dump(path)
    with failing_write(holder):
        with pytest.raises(InjectedFailure):
            call()
    assert dump(path) == before
