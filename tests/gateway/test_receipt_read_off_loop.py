"""Receipt storage waits do not stall siblings or cross live-session generations."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import threading

import pytest

from gateway.session_authority import LiveSession
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.test_prompt_queued_controls import _call, _connection
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit


@contextmanager
def _exhaust_readers(db):
    """Borrow actual SQLite readers until the bounded pool degrades to its writer."""
    readers = []
    try:
        while (reader := db._checkout_read_conn()) is not None:
            readers.append(reader)
        assert readers, "the fixture must exercise the real WAL reader budget"
        yield
    finally:
        for reader in readers:
            db._close_read_conn(reader)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["prompt.receipt", "prompt.cancel"])
async def test_receipt_reader_exhaustion_keeps_dispatch_responsive(tmp_path, monkeypatch, method):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    scope = ContextVar("receipt_test_scope", default=None)
    token = scope.set("request-profile")
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    holding, release, released = threading.Event(), threading.Event(), threading.Event()
    observed, reads, frames = [], [], []
    authority.sessions["s"].event_stream.observers.add(frames.append)

    def hold_writer():
        def transaction(conn):
            conn.execute("UPDATE sessions SET title=title WHERE id='s'")
            holding.set()
            release.wait(5)  # Safety ceiling: a blocked event loop cannot release the writer.
            released.set()
        db._execute_write(transaction)

    try:
        with db:
            queued = await _submit(authority, "queued")
            connection = _connection(authority)
            with _exhaust_readers(db):
                writer = asyncio.create_task(asyncio.to_thread(hold_writer))
                assert await asyncio.to_thread(holding.wait, 5)
                read_ctx = db._read_ctx

                def probe():
                    observed.append(released.is_set())
                    release.set()

                @contextmanager
                def announced_read(*args, **kwargs):
                    reads.append((threading.get_ident(), scope.get()))
                    loop.call_soon_threadsafe(probe)
                    with read_ctx(*args, **kwargs) as conn:
                        yield conn

                monkeypatch.setattr(db, "_read_ctx", announced_read)
                try:
                    reply = await _call(connection, method, admission_id=queued.admission_id)
                finally:
                    release.set()
                    await asyncio.wait_for(writer, 10)
            assert observed and observed[0] is False, "receipt storage wait blocked the owner loop"
            assert reads and reads[0][1] == "request-profile"
            assert reads[0][0] != loop_thread
            assert "error" not in reply, reply
            if method == "prompt.receipt":
                assert reply["result"]["status"] == "queued"
            else:
                assert (reply["result"]["status"], reply["result"]["outcome"]) == ("terminal", "cancelled")
                assert [f["params"]["type"] for f in frames].count("message.complete") == 1
            assert scope.get() == "request-profile"
    finally:
        release.set()
        scope.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["remove", "remount", "retire", "settling"])
async def test_receipt_rechecks_live_state_after_storage_read(tmp_path, monkeypatch, change):
    """Read the real row, then let the owner change its mount/publication before returning it."""
    from gateway import session_authority

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    loop = asyncio.get_running_loop()
    advanced = threading.Event()
    live = authority.sessions["s"]
    read = session_authority.get_session_admission

    with db:
        queued = await _submit(authority, "queued")
        durable = get_session_admission(db, admission_id=queued.admission_id)
        if change == "settling":
            from hermes_state_runtime import cancel_session_input
            cancelled = cancel_session_input(db, epoch=authority.epoch, admission_id=queued.admission_id)
            assert cancelled["status"] == "terminal"

        def advance():
            if change == "remove":
                authority.sessions.pop("s")
            elif change == "remount":
                authority.sessions["s"] = LiveSession(live.source, "new-route")
            elif change == "retire":
                authority.retired = True
            else:
                # The completion has committed but its frame is not yet published.
                with live.event_stream.lock:
                    live.settling[queued.admission_id] = durable
            advanced.set()

        def read_before_owner_change(*args, **kwargs):
            row = read(*args, **kwargs)
            loop.call_soon_threadsafe(advance)
            assert advanced.wait(5), "receipt read held the loop while the owner needed to advance"
            return row

        monkeypatch.setattr(session_authority, "get_session_admission", read_before_owner_change)
        if change == "settling":
            receipt = await authority.receipt(ACTOR, REF, queued.admission_id)
            assert receipt.status == "queued", "receipt exposed storage state before its completion frame"
        else:
            with pytest.raises(RuntimeStoreError) as error:
                await authority.receipt(ACTOR, REF, queued.admission_id)
            assert error.value.reason == ("runtime_retired" if change == "retire" else "not_found")
