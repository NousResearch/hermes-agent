"""Native automation bookkeeping must not hold the owner loop behind a SQLite writer."""
import asyncio
import json
import threading

import pytest

from tests.gateway.test_resume_latest_local_cli import _call, _owner


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['before', 'after', 'publish'])
async def test_automation_yields_while_a_writer_holds_metadata(tmp_path, monkeypatch, phase):
    from gateway import session_viewer_automation as automation
    from hermes_cli import goals
    from hermes_cli.goals import GoalState

    store, authority, viewer = _owner(tmp_path / 'home', monkeypatch)
    db = authority.db
    loop = asyncio.get_running_loop()
    writing, release, expired = threading.Event(), threading.Event(), threading.Event()
    observed = asyncio.Event()
    observations = []
    writer = request = None
    try:
        created = await _call(viewer, 'session.create', request_id='automation', source='gui',
                              cwd=str(tmp_path), toolsets=[], model='m')
        sid = created['session_id']
        old = GoalState('committed goal', status='active', created_at=10).to_json()
        new = GoalState('new goal', status='paused', created_at=10).to_json()
        db.set_meta('goal:' + sid, old)
        monkeypatch.setattr(goals, '_get_session_db', lambda: db)
        updates = []
        authority.sessions[sid].event_stream.observers.add(
            lambda frame: frame['params']['type'] == 'session.control.update'
            and updates.append(frame['params']['payload']))

        def update(conn):
            conn.execute('UPDATE state_meta SET value=? WHERE key=?', (new, 'goal:' + sid))
            writing.set()
            if not release.wait(10):
                expired.set()

        def start_writer():
            nonlocal writer
            writer = threading.Thread(target=db._execute_write, args=(update,))
            writer.start()
            assert writing.wait(10), 'fixture writer did not enter its transaction'

        invoke = automation.invoke

        def run_then_write(*args, **kwargs):
            result = invoke(*args, **kwargs)
            start_writer()
            return result

        if phase == 'before':
            await asyncio.to_thread(start_writer)
        else:
            # The manager stays real; hold a fresh writer only after it returns,
            # exercising the second comparison read or the mutation's publication.
            monkeypatch.setattr(automation, 'invoke', run_then_write)

        get_meta = db.get_meta
        scheduled = threading.Event()

        def probe():
            observations.append(expired.is_set())
            observed.set()

        def read(key):
            if key == 'goal:' + sid and writing.is_set() and not scheduled.is_set():
                scheduled.set()
                loop.call_soon_threadsafe(probe)
            return get_meta(key)

        monkeypatch.setattr(db, 'get_meta', read)
        call = 'pause' if phase == 'publish' else 'state'
        request = asyncio.create_task(_call(viewer, 'session.automation', session_id=sid,
                                            kind='goal', call=call))
        await asyncio.wait_for(observed.wait(), 15)
        assert observations == [False], f'{phase} metadata read blocked the owner loop on the writer mutex'
        assert isinstance(await _call(viewer, 'ping'), dict)
        assert not expired.is_set(), 'sibling native RPC waited for the unrelated writer'
        release.set()
        await asyncio.to_thread(writer.join, 15)
        assert not writer.is_alive()
        reply = await asyncio.wait_for(request, 15)
        expected = 'new goal' if phase == 'before' else 'committed goal'
        assert json.loads(reply['state'])['goal'] == expected
        assert json.loads(db.get_meta('goal:' + sid))['goal'] == 'new goal'
        if phase != 'before':
            assert updates[-1]['control']['goal']['title'] == 'new goal'
    finally:
        release.set()
        if writer is not None:
            await asyncio.to_thread(writer.join, 15)
            assert not writer.is_alive()
        if request is not None:
            await asyncio.wait_for(request, 15)
        await viewer.close()
        store.close_all_db_handles()
