"""Neither attach shape nor the normal Desktop RPC may park the owner loop behind a pooled-read fallback.

The metadata case exhausts the real read pool before attach; the history case takes the last
reader at local_history's entry. A real writer callback owns the store's transaction and mutex;
no history data or reader is stubbed.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from gateway.session_contract import SessionActor
from hermes_state import _READ_POOL_MAX
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority

pytestmark = pytest.mark.usefixtures("authority_cwd_in_tmp")


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["omitted-metadata", "message-history", "normal-rpc"])
async def test_attach_waiting_on_the_writer_keeps_loop_and_publication_free(tmp_path, monkeypatch, site):
    from gateway import session_local_recovery

    history = site == "message-history"
    if site == "normal-rpc":
        from gateway import session_local
        from gateway.session_controls import AuthorityConnection
        from tests.gateway.test_resume_omit_messages import _local_session
        authority, viewer, ref, physical = await _local_session(tmp_path, monkeypatch)
        db = authority.db
        monkeypatch.setattr(session_local, "local_session_info", lambda *a: {"desktop_protocol": "hermes-gateway-v1"})
        connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
        connection.actor = viewer
        authority.events[viewer.transport_id] = object()
    else:
        db, authority = _authority(tmp_path, monkeypatch)
        ref, physical = REF, REF.session_id
        viewer = SessionActor("human", "owned", frozenset({"session:read"}), "viewer")
    db.append_message(physical, "user", "stored history")
    stream = authority.sessions[ref.session_id].event_stream
    loop = asyncio.get_running_loop()
    writer_entered, history_entered = threading.Event(), threading.Event()
    release, released = threading.Event(), threading.Event()
    probes_done, witnesses = asyncio.Event(), {}
    held_readers, attach = [], None
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="attach-history-writer")
    # Only a safety ceiling: success requires both loop probes BEFORE the writer is released.
    hold_ceiling = 30.0

    def hold_transaction(_connection):
        writer_entered.set()
        try:
            release.wait(hold_ceiling)
        finally:
            released.set()

    def publish():
        stream.publish(ref.session_id, {"running": False}, event_type="session.info")
        witnesses["publication"] = released.is_set()

    def heartbeat():
        witnesses["heartbeat"] = released.is_set()
        probes_done.set()

    def schedule_probes():
        loop.call_soon(publish)
        loop.call_later(0.05, heartbeat)

    original_history = session_local_recovery.local_history

    def contended_history(*args, **kwargs):
        if history_entered.is_set():
            return original_history(*args, **kwargs)
        # Preflight used the remaining reader. Occupy it at the final native history entry, unless
        # the snapshot already pinned it: all eight permits are real in either implementation.
        # A coherent worker snapshot may already own the eighth reader at this entry.
        last_reader = None if db._pinned_read_conn() is not None else db._checkout_read_conn()
        assert last_reader is not None or db._pinned_read_conn() is not None
        try:
            assert db._checkout_read_conn() is None, "the real read pool was not exhausted"
            history_entered.set()
            loop.call_soon_threadsafe(schedule_probes)
            return original_history(*args, **kwargs)
        finally:
            if last_reader is not None:
                db._close_read_conn(last_reader)

    if history:
        monkeypatch.setattr(session_local_recovery, "local_history", contended_history)
    transaction = None
    try:
        held_readers = [db._checkout_read_conn() for _ in range(_READ_POOL_MAX - int(history))]
        assert all(reader is not None for reader in held_readers)
        transaction = writer.submit(db._execute_write, hold_transaction)
        assert await asyncio.to_thread(writer_entered.wait, 10), "the writer callback was not entered"
        operation = (connection.dispatch({"jsonrpc": "2.0", "id": 1, "method": "session.resume", "params": {
            "session_id": ref.session_id, "omit_messages": True}}) if site == "normal-rpc"
            else authority.attach(viewer, ref, history=history))
        attach = asyncio.create_task(operation)
        if not history:
            # Queue behind attach's first step: a synchronous metadata read blocks these probes,
            # while a worker wait yields to them. All eight real readers stay checked out.
            loop.call_soon(schedule_probes)
        await asyncio.wait_for(probes_done.wait(), hold_ceiling + 10)
        if history:
            assert history_entered.is_set(), "attach never reached the native history reader"
    finally:
        release.set()
        await asyncio.to_thread(writer.shutdown, wait=True)
        try:
            if transaction is not None:
                transaction.result()
            if attach is not None:
                snapshot = await asyncio.wait_for(attach, 10)
        finally:
            for reader in held_readers:
                if reader is not None:
                    db._close_read_conn(reader)
            db.close()

    if site == "normal-rpc":
        assert "error" not in snapshot, snapshot
        assert snapshot["result"]["messages"] == [] and snapshot["result"]["messages_omitted"] is True
    else:
        assert [row["content"] for row in snapshot.history] == (["stored history"] if history else [])
        assert snapshot.handle.ref == ref
    assert witnesses == {"publication": False, "heartbeat": False}, (
        "attach blocked loop publication/heartbeat until the held writer released: "
        f"{witnesses}"
    )
