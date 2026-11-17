"""Externally spoken turns survive retries and reconnect without starting work (#121045)."""
import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tests.waiting import async_wait_until


def connection(db, home, *, capabilities=None):
    epoch = begin_runtime_epoch(db, instance_id='test-owner')
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(home),
                                 instance_id='test-owner', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(None, 'route')
    auth = {'user_id': 'owner'}
    if capabilities is not None:
        auth['capabilities'] = capabilities
    return AuthorityConnection(authority, object(), auth)


def request(client_id='utterance-1', content='Spoken words', role='user'):
    return {'id': 1, 'method': 'session.append_message', 'params': {
        'session_id': 's', 'client_id': client_id, 'role': role, 'content': content}}


@pytest.mark.asyncio
async def test_append_reconnect_retry_and_conflict_use_real_store(tmp_path):
    path = tmp_path / 'state.db'
    with SessionDB(db_path=path) as db:
        db.create_session('s', source='test')
        owner = connection(db, tmp_path)
        try:
            first = await owner.dispatch(request())
            assert 'result' in first, first
            second = await owner.dispatch(request('utterance-2', 'Actually, make that tomorrow'))
            assert 'result' in second, second
            assert await owner.dispatch(request()) == first
            conflict = await owner.dispatch(request(content='Changed words'))
            assert conflict['error']['message'] == 'client_id_conflict'
            assert [row['content'] for row in db.get_messages('s')] == ['Spoken words', 'Actually, make that tomorrow']
            assert db.get_session('s')['message_count'] == 2
            assert db._conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0
        finally:
            await owner.close()

    with SessionDB(db_path=path) as db:
        reconnected = connection(db, tmp_path)
        try:
            assert await reconnected.dispatch(request()) == first
            assert len(db.get_messages('s')) == 2
            from gateway.run import _build_gateway_agent_history
            history, _ = _build_gateway_agent_history(db.get_messages_as_conversation('s'))
            assert all(text in str(history) for text in ['Spoken words', 'Actually, make that tomorrow'])
        finally:
            await reconnected.close()


@pytest.mark.asyncio
async def test_disconnect_does_not_abandon_committed_write(tmp_path):
    path = tmp_path / 'state.db'
    with SessionDB(db_path=path) as db, sqlite3.connect(path) as blocker:
        db.create_session('s', source='test')
        owner = connection(db, tmp_path)
        blocker.execute('BEGIN IMMEDIATE')
        call = asyncio.create_task(owner.dispatch(request()))
        try:
            await async_wait_until(lambda: bool(getattr(owner.authority, '_mutation_tasks', ())),
                                   timeout=5, what='append registered as retained owner work')
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            blocker.rollback()
            await asyncio.gather(*owner.authority._mutation_tasks)
            reply = await owner.dispatch(request())
            assert 'result' in reply, reply
            assert len(db.get_messages('s')) == 1
        finally:
            blocker.rollback()
            await owner.close()


@pytest.mark.asyncio
async def test_compaction_and_compression_keep_receipts_and_spoken_tail(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        owner = connection(db, tmp_path)
        try:
            db.append_message('s', 'user', 'Earlier context')
            watermark = db.get_active_message_watermark('s')
            first = await owner.dispatch(request())
            assert 'result' in first, first
            db.archive_and_compact('s', [{'role': 'user', 'content': 'Summary'}], watermark=watermark)
            assert [m['content'] for m in db.get_messages('s')] == ['Summary', 'Spoken words']
            assert await owner.dispatch(request()) == first
            db.end_session('s', 'compression')
            db.create_session('tip', source='test', parent_session_id='s')
            second = await owner.dispatch(request('second', 'After compression'))
            assert second['result']['session_id'] == 'tip', second
            assert await owner.dispatch(request()) == first
            assert [m['content'] for m in db.get_messages('tip')] == ['After compression']
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_parallel_retries_and_storage_failure_are_atomic(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        owner = connection(db, tmp_path)
        try:
            replies = await asyncio.gather(*(owner.dispatch(request()) for _ in range(8)))
            assert all(reply == replies[0] for reply in replies)
            assert len(db.get_messages('s')) == 1
            db._conn.execute("CREATE TRIGGER fail_append BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
            failed = await owner.dispatch(request('failed', 'Never committed'))
            assert failed['error']['message'] == 'storage_unavailable'
            assert db.get_session('s')['message_count'] == 1
            db._conn.execute('DROP TRIGGER fail_append')
            assert 'result' in await owner.dispatch(request('failed', 'Never committed'))
            assert len(db.get_messages('s')) == 2
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_append_keeps_active_worker_and_refuses_reader(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        owner = connection(db, tmp_path)
        reader = AuthorityConnection(owner.authority, object(), {'user_id': 'reader', 'capabilities': ['session:read']})
        release = asyncio.Event()
        live = owner.authority.sessions['s']
        live.task = asyncio.create_task(release.wait())
        assert db.try_acquire_session_turn_lease('s', 'active-worker')
        try:
            denied = await reader.dispatch(request())
            assert denied['error']['message'] == 'permission_denied'
            invalid = await owner.dispatch(request(role='tool'))
            assert invalid['error']['code'] == 4000
            appended = await owner.dispatch(request())
            assert 'result' in appended, appended
            assert not live.task.done()
            assert not db.try_acquire_session_turn_lease('s', 'another-worker')
            db.append_message('s', 'assistant', 'Task completed', turn_lease_holder='active-worker')
            assert [row['content'] for row in db.get_messages('s')] == ['Spoken words', 'Task completed']
        finally:
            release.set()
            await live.task
            db.release_session_turn_lease('s', 'active-worker')
            await reader.close()
            await owner.close()
