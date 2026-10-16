"""Temporary chats on the canonical gateway: ``session.create {temporary}`` births a hidden, unlisted,
temporary row; the last viewer closing it deletes it; a dropped viewer leaves it for the unviewed
sweep; it can never be branched into a saved chat. Real authority + SQLite."""
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection
from tests.gateway.test_local_source_labels import _authority


async def _create(conn, tmp_path, request_id, **extra):
    return await conn.dispatch({'id': request_id, 'method': 'session.create', 'params': {
        'request_id': request_id, 'source': 'tui', 'cwd': str(tmp_path), **extra}})


async def _open(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    return authority, store, conn


async def _close(authority, conn):
    task = getattr(authority, '_temporary_sweeper', None)
    if task is not None:
        task.cancel()
    await conn.close()
    authority.db.close()


@pytest.mark.asyncio
async def test_temporary_create_is_hidden_unlisted_and_advertised(tmp_path, monkeypatch):
    authority, _store, conn = await _open(tmp_path, monkeypatch)
    try:
        described = await conn.dispatch({'id': 0, 'method': 'runtime.describe', 'params': {}})
        assert 'temporary' in described['result']['session_create']['parameters']
        saved = (await _create(conn, tmp_path, 'saved'))['result']['stored_session_id']
        created = await _create(conn, tmp_path, 'temp', temporary=True)
        temp = created['result']['stored_session_id']
        db = authority.db
        assert db.is_session_temporary(temp) and db.get_session(temp)['hidden'] == 1
        assert not db.is_session_temporary(saved)
        listed = await conn.dispatch({'id': 9, 'method': 'session.list', 'params': {'include_hidden': True}})
        assert [row['id'] for row in listed['result']['sessions']] == [saved]
        from gateway.session_local_title import resolve_latest_session
        live = await conn.dispatch({'id': 10, 'method': 'session.list', 'params': {'include_hidden': True}})
        assert temp not in [row['id'] for row in live['result']['sessions']]
        # `hermes --tui -c` resumes the latest tui chat: the saved one, never the newer temporary one.
        assert resolve_latest_session(authority, conn.actor, 'tui').session_id == saved
        from gateway.session_execution import _session_is_temporary
        runner = SimpleNamespace(_session_db=db)
        assert _session_is_temporary(runner, temp) and not _session_is_temporary(runner, saved)
    finally:
        await _close(authority, conn)


@pytest.mark.asyncio
async def test_temporary_create_refuses_bad_flags(tmp_path, monkeypatch):
    authority, _store, conn = await _open(tmp_path, monkeypatch)
    try:
        titled = await _create(conn, tmp_path, 'titled', temporary=True, title='My notes')
        assert titled['error']['message'] == 'invalid_params', 'a title may resolve to a saved chat'
        bad = await _create(conn, tmp_path, 'bad', temporary='yes')
        assert bad['error']['message'] == 'invalid_params'
    finally:
        await _close(authority, conn)


@pytest.mark.asyncio
async def test_closing_viewer_deletes_dropped_viewer_keeps_for_the_sweep(tmp_path, monkeypatch):
    from gateway.session_temporary import TEMPORARY_UNVIEWED_TTL_SECONDS, sweep_unviewed
    authority, _store, conn = await _open(tmp_path, monkeypatch)
    db = authority.db
    try:
        ids = {}
        for name in ('closed', 'dropped'):
            created = await _create(conn, tmp_path, name, temporary=True)
            ids[name] = (created['result']['stored_session_id'], created['result']['subscription_id'])

        async def detach(name, **extra):
            sid, sub = ids[name]
            return await conn.dispatch({'id': 2, 'method': 'session.detach', 'params': {
                'session_id': sid, 'subscription_id': sub, **extra}})

        assert 'result' in await detach('closed', end_reason='tui_close')
        closed = ids['closed'][0]
        assert db.get_session(closed) is None and closed not in authority.sessions

        assert 'result' in await detach('dropped')  # a socket drop: the client may come straight back
        dropped = ids['dropped'][0]
        assert db.get_session(dropped) is not None and db.get_session(dropped)['ended_at'] is None
        clocks = {}
        await sweep_unviewed(authority, clocks, now=1000.0)
        assert db.get_session(dropped) is not None, 'the TTL starts when the sweep first sees it unviewed'
        await sweep_unviewed(authority, clocks, now=1000.0 + TEMPORARY_UNVIEWED_TTL_SECONDS)
        assert db.get_session(dropped) is None and dropped not in authority.sessions
    finally:
        await _close(authority, conn)


@pytest.mark.asyncio
async def test_viewed_temporary_chat_survives_the_sweep(tmp_path, monkeypatch):
    from gateway.session_temporary import TEMPORARY_UNVIEWED_TTL_SECONDS, sweep_unviewed
    authority, _store, conn = await _open(tmp_path, monkeypatch)
    try:
        temp = (await _create(conn, tmp_path, 'temp', temporary=True))['result']['stored_session_id']
        clocks = {}
        for now in (0.0, TEMPORARY_UNVIEWED_TTL_SECONDS * 2):
            await sweep_unviewed(authority, clocks, now=now)
        assert authority.db.get_session(temp) is not None and clocks == {}
    finally:
        await _close(authority, conn)


@pytest.mark.asyncio
async def test_saved_chat_detach_still_ends_and_is_never_deleted(tmp_path, monkeypatch):
    authority, _store, conn = await _open(tmp_path, monkeypatch)
    try:
        created = await _create(conn, tmp_path, 'saved')
        sid, sub = created['result']['stored_session_id'], created['result']['subscription_id']
        await conn.dispatch({'id': 2, 'method': 'session.detach', 'params': {
            'session_id': sid, 'subscription_id': sub, 'end_reason': 'tui_close'}})
        from hermes_state_local import local_receipt
        row = authority.db.get_session(local_receipt(authority.db, sid)['entry']['session_id'])
        assert row is not None and row['end_reason'] == 'tui_close'
        from gateway.session_temporary import delete_all_unviewed
        assert delete_all_unviewed(authority) == [] and authority.db.get_session(sid) is not None
    finally:
        await _close(authority, conn)


@pytest.mark.asyncio
async def test_temporary_chat_cannot_be_branched(tmp_path, monkeypatch):
    authority, _store, conn = await _open(tmp_path, monkeypatch)
    try:
        created = (await _create(conn, tmp_path, 'temp', temporary=True))['result']
        reply = await conn.dispatch({'id': 3, 'method': 'session.mutate', 'params': {
            'session_id': created['stored_session_id'], 'request_id': 'b1',
            'expected_revision': created['revision'], 'expected_generation': created['execution_generation'],
            'operation': 'branch', 'payload': {}}})
        assert reply['error']['message'] == 'temporary_chat_cannot_branch', reply
    finally:
        await _close(authority, conn)


@pytest.mark.asyncio
async def test_shutdown_deletes_every_temporary_chat_it_owns(tmp_path, monkeypatch):
    from gateway.session_temporary import delete_all_unviewed
    authority, _store, conn = await _open(tmp_path, monkeypatch)
    try:
        temp = (await _create(conn, tmp_path, 'temp', temporary=True))['result']['stored_session_id']
        assert delete_all_unviewed(authority) == [temp]
        assert authority.db.get_session(temp) is None
    finally:
        await _close(authority, conn)


@pytest.mark.asyncio
async def test_closed_temporary_chat_leaves_no_bytes_in_the_database_files(tmp_path, monkeypatch):
    # The gateway's delete must take the scrubbed path (FTS5 secure-delete + WAL reset), not a plain
    # delete_session that leaves the chat in the WAL and the FTS segments.
    authority, _store, conn = await _open(tmp_path, monkeypatch)
    db = authority.db
    canary = "zqxcanary" + "c0ffee5eed42"
    try:
        created = await _create(conn, tmp_path, 'scrub', temporary=True)
        sid, sub = created['result']['stored_session_id'], created['result']['subscription_id']
        db.append_messages_batch(sid, [{'role': 'user', 'content': f'secret {canary} ' + 'padding ' * 900}])
        path = db.db_path

        def on_disk():
            return b''.join(p.read_bytes() for p in (path, path.with_name(path.name + '-wal')) if p.exists()).lower()

        assert canary.encode() in on_disk()
        assert 'result' in await conn.dispatch({'id': 2, 'method': 'session.detach', 'params': {
            'session_id': sid, 'subscription_id': sub, 'end_reason': 'tui_close'}})
        assert db.get_session(sid) is None
        assert canary.encode() not in on_disk()
    finally:
        await _close(authority, conn)
