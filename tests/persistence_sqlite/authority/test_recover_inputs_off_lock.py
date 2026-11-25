"""The boot recovery scans of ``recover_session_inputs`` run without the state.db write lock.

At every gateway start ``recover_session_inputs`` ran four full-table predicates as ``UPDATE ...
WHERE`` inside one write transaction. No index covers a status plus an epoch, and a
``session_admissions`` scan reads every payload's overflow pages, so the boot held the write lock
for the length of four scans even when nothing was lost, and every other writer waited behind it.

The scans are reads now; a boot that finds nothing opens no write, and the write updates only what
the scans found, by key, repeating each predicate. The clean-boot case fails with the source change
reverted (it always opened a write); the rest pin that the recovery itself is unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from persistence.sqlite import _holder
from persistence.sqlite.authority._admission_store import (admit_session_input, claim_session_input,
                                                          get_session_admission, recover_session_inputs)
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.workers import _runtime_workers
from persistence.sqlite.workers._executions import register_worker_execution


@pytest.fixture
def writes(monkeypatch) -> list[Any]:
    """Every ``_execute_write`` the store performs from now on."""
    seen: list[Any] = []
    real = _holder.StoreHolder._execute_write

    def execute(holder_self, fn, *a, **k):
        seen.append(fn)
        return real(holder_self, fn, *a, **k)

    monkeypatch.setattr(_holder.StoreHolder, "_execute_write", execute)
    return seen


def _boot(holder, *, child: bool = True, root: bool = True, started: bool = True):
    """A first gateway's epoch with the kinds of work a restart can strand; returns its admission id."""
    create_session(holder, "s", "cli")
    create_session(holder, "child", "cli", parent_session_id="s", model_config={"_delegate_from": "s"})
    create_session(holder, "work", "cli")
    epoch = begin_runtime_epoch(holder, instance_id="first")
    if child:
        register_worker_execution(holder, epoch=epoch, execution_id="c1", session_id="child", generation=0,
                                  kind="child", adoption_secret="child-secret")
    if root:
        register_worker_execution(holder, epoch=epoch, execution_id="w1", session_id="s", generation=0,
                                  kind="compute", adoption_secret="root-secret")
    admission_id = ""
    if started:
        # A session with no worker of its own: a claim is refused while one is registered.
        row, _created = admit_session_input(holder, epoch=epoch, principal_id="human", session_id="work",
                                            request_id="input", payload={"text": "hi"})
        admission_id = row["admission_id"]
        assert claim_session_input(holder, epoch=epoch, session_id="work")
    return admission_id


def _worker_status(holder, execution_id: str) -> str:
    with holder._read_ctx() as conn:
        return conn.execute("SELECT status FROM worker_executions WHERE execution_id=?", (execution_id,)).fetchone()[0]


def test_a_clean_boot_opens_no_write_transaction(store, writes):
    holder = store.holder()
    create_session(holder, "s", "cli")
    begin_runtime_epoch(holder, instance_id="first")
    restart = begin_runtime_epoch(holder, instance_id="restart")
    writes.clear()

    assert recover_session_inputs(holder, epoch=restart) == 0
    assert writes == [], "a boot with nothing to recover must not take the write lock"


def test_every_lost_kind_is_recovered_exactly_as_before(store):
    holder = store.holder()
    admission_id = _boot(holder)
    restart = begin_runtime_epoch(holder, instance_id="restart")

    assert recover_session_inputs(holder, epoch=restart) == 1
    assert _worker_status(holder, "c1") == "terminal"  # a child dies with its owner
    assert _worker_status(holder, "w1") == "unknown"  # only its root can be recovered
    assert get_session_admission(holder, admission_id=admission_id)["status"] == "unknown"


def test_a_child_that_stopped_being_stale_between_the_scan_and_the_write_is_left_alone(store, monkeypatch):
    holder = store.holder()
    _boot(holder, root=False, started=False)
    restart = begin_runtime_epoch(holder, instance_id="restart")
    real = _runtime_workers.stale_child_ids

    def raced(conn, epoch):
        ids = real(conn, epoch)
        # It settled by itself between the scan and the write.
        holder._execute_write(lambda c: c.execute("UPDATE worker_executions SET status='terminal' WHERE execution_id='c1'"))
        return ids

    monkeypatch.setattr(_runtime_workers, "stale_child_ids", raced)
    recover_session_inputs(holder, epoch=restart)
    assert _worker_status(holder, "c1") == "terminal"


def test_an_old_epoch_is_still_refused(store):
    holder = store.holder()
    _boot(holder, child=False, root=False, started=False)
    first = 1
    begin_runtime_epoch(holder, instance_id="restart")
    with pytest.raises(Exception) as refused:
        recover_session_inputs(holder, epoch=first)
    assert "stale_epoch" in str(refused.value)
