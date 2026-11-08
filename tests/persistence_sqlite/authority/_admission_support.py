"""Seeds and probes the admission backend tests share: the seed calls, run on the legacy store for a test that compares
with it and through the backend for any other (:class:`BackendSeed`), deterministic admission ids, record projections,
and a failure injected inside a verb's write after its first mutating statement."""

from __future__ import annotations

import dataclasses
import itertools
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest

from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.persistence_sqlite.authority._support import dump

#: The ``(epoch, instance)`` the seeds begin.
EPOCH = 1


class InjectedFailure(Exception):
    """Raised inside a verb's write by :func:`failing_write`."""


def record_dict(record: Any) -> Any:
    """A contract record as the legacy ``_row`` dict (``ClaimedAdmission`` without the field legacy lacks)."""
    if record is None or record is False:
        return record
    values = dataclasses.asdict(record)
    values.pop("queued_owner_epoch", None)
    return values


class BackendSeed:
    """The ``SessionDB`` calls the shared seeds make, run through the backend's verbs on ``store`` (raw SQL on its
    holder), for a test that does not compare with the legacy store."""

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
    if isinstance(db, BackendSeed):
        return db.store.runtime.begin_epoch(instance_id=instance_id)
    from hermes_state_runtime import begin_runtime_epoch
    return begin_runtime_epoch(db, instance_id=instance_id)


def recover(db, epoch):
    if isinstance(db, BackendSeed):
        return db.store.admissions.recover(epoch=epoch)
    from hermes_state_runtime import recover_session_inputs
    return recover_session_inputs(db, epoch=epoch)


def admit(db, session_id, request_id, payload=None, principal_id="p", epoch=EPOCH, **kwargs):
    payload = payload or {"text": request_id}
    if isinstance(db, BackendSeed):
        record, _created = db.store.admissions.admit(epoch=epoch, principal_id=principal_id, session_id=session_id,
                                                     request_id=request_id, payload=payload, **kwargs)
        return record_dict(record)
    from hermes_state_runtime import admit_session_input
    return admit_session_input(db, epoch=epoch, principal_id=principal_id, session_id=session_id,
                               request_id=request_id, payload=payload, **kwargs)


def claim(db, session_id, epoch=EPOCH):
    if isinstance(db, BackendSeed):
        return record_dict(db.store.admissions.claim(epoch=epoch, session_id=session_id))
    from hermes_state_runtime import claim_session_input
    return claim_session_input(db, epoch=epoch, session_id=session_id)


def get(db, admission_id):
    if isinstance(db, BackendSeed):
        return record_dict(db.store.admissions.get(admission_id=admission_id))
    from hermes_state_runtime import get_session_admission
    return get_session_admission(db, admission_id=admission_id)


def settle(db, row, outcome="completed", result=None, epoch=EPOCH):
    """Settle the claimed ``row`` under ``epoch``."""
    call = dict(epoch=epoch, admission_id=row["admission_id"], generation=row["generation"], outcome=outcome,
                result=result)
    if isinstance(db, BackendSeed):
        return record_dict(db.store.admissions.settle(**call))
    from hermes_state_runtime import settle_session_input
    return settle_session_input(db, **call)


def steer(db, row, claimed, epoch=EPOCH):
    """Hand the queued ``row`` to the running ``claimed`` row."""
    call = dict(epoch=epoch, admission_id=row["admission_id"], generation=claimed["generation"])
    if isinstance(db, BackendSeed):
        record, transferred = db.store.admissions.steer(**call)
        return record_dict(record), transferred
    from hermes_state_runtime import steer_session_input
    return steer_session_input(db, **call)


def register(db, session_id, generation, execution_id="exec-1", epoch=EPOCH):
    """Register a compute worker on ``session_id`` at ``generation``."""
    call = dict(epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, kind="compute",
                adoption_secret="secret-1")
    if isinstance(db, BackendSeed):
        return db.store.worker_executions.register(**call)
    from hermes_state_runtime import register_worker_execution
    return register_worker_execution(db, **call)


def authority(db, epoch=EPOCH):
    """The parts of a legacy gateway authority the copied gateway functions read."""
    return SimpleNamespace(db=db, epoch=epoch, sessions={})


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
