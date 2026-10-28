"""N50: session.create's creation commit and an API turn's settings write run off the owner loop.

Same class as test_owner_loop_writes_class.py: each row drives the real verb end to end and
records the thread every named SQLite write ran on; on the owner-loop thread a contended writer
lock (up to _WRITE_PATIENCE_S) freezes every session, timer and socket the owner serves.
"""
import asyncio
import threading

import pytest

from tests.gateway.test_owner_loop_writes_class import _owner


async def _create(authority, owner, tmp_path, api):
    reply = await owner.dispatch({'id': 1, 'method': 'session.create', 'params': {
        'request_id': 'plain', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}})
    assert 'result' in reply, reply


async def _api_turn(authority, owner, tmp_path, api):
    from gateway.session_api_turn import admit_api_turn_async
    _, ref, row = await admit_api_turn_async(api, session_id='api', user_message='hi', conversation_history=[])
    await authority._drain(ref)
    from hermes_state_runtime import get_session_admission
    assert get_session_admission(authority.db, admission_id=row['admission_id'])['status'] == 'terminal'


def _writes_state_meta(prefix, threads):
    """Wrap a ``_execute_write`` callback so a statement naming *prefix* records its thread."""
    class Conn:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=()):
            if 'state_meta' in sql and any(isinstance(p, str) and p.startswith(prefix) for p in params):
                threads.append(threading.get_ident())
            return self._conn.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._conn, name)
    return Conn


def _record(monkeypatch, prefix):
    from hermes_state import SessionDB
    threads = []
    wrap = _writes_state_meta(prefix, threads)
    real = SessionDB._execute_write
    monkeypatch.setattr(SessionDB, '_execute_write',
                        lambda self, fn, *a, **k: real(self, lambda conn: fn(wrap(conn)), *a, **k))
    return threads


async def _handle(event):
    from gateway.session_results import execution_result
    execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
    return 'ok'


@pytest.mark.asyncio
async def test_session_create_commit_never_runs_on_the_owner_loop_thread(tmp_path, monkeypatch):
    from hermes_state_local import POLICY_PREFIX
    store, authority, owner = _owner(tmp_path, monkeypatch, _handle)
    threads = _record(monkeypatch, POLICY_PREFIX)
    try:
        async with asyncio.timeout(20):
            await _create(authority, owner, tmp_path, None)
        assert threads, 'the creation commit never ran'
        assert threading.get_ident() not in threads, 'the creation commit ran on the owner loop thread'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_api_settings_write_never_runs_on_the_owner_loop_thread(api, owner, monkeypatch, tmp_path):
    owner.runner._handle_message = _handle
    threads = _record(monkeypatch, 'gateway.api.settings.v1.')
    async with asyncio.timeout(20):
        await _api_turn(owner, None, tmp_path, api)
    assert threads, 'the API settings write never ran'
    assert threading.get_ident() not in threads, 'the API settings write ran on the owner loop thread'
