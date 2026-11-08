"""Canonical closeout preserves actual authenticated authority and normal primary admission."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_cli.conversation_closeout import CloseoutStore
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.mark.asyncio
async def test_canonical_rpc_uses_normal_submit_and_archives_only_old_root(tmp_path, monkeypatch):
    import gateway.session_closeout as closeout
    monkeypatch.setattr(closeout, 'PRESERVATION_WAIT_SECONDS', 0)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('old', source='test')
        db.create_session('fresh', source='test')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path),
                                     instance_id='current', db=db, epoch=epoch)
        authority.sessions['old'] = LiveSession(None, 'route')
        connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        connection.subscriptions['old'] = 'sub'
        calls = []
        async def primary(ref, params):
            operation = params['submission_id'].removeprefix('closeout-review-')
            calls.append(operation)
            store = CloseoutStore(db, epoch=epoch)
            assert store.claim_capture(operation, 'old')
            store.record_capture(operation, 'old', memory='accepted', source_id='source', tasks='reconciled')
            return {'status': 'queued'}
        monkeypatch.setattr(connection, 'submit', primary)
        monkeypatch.setattr(closeout, 'canonical_settled', lambda *args: True)
        request = {'id': 1, 'method': 'session.closeout', 'params': {
            'session_id': 'old', 'request_id': 'request', 'intent': 'new'}}
        response = await connection.dispatch(request)
        assert 'result' in response, response
        operation = response['result']['closeout_id']
        await asyncio.gather(*authority._closeout_jobs)
        replay = await connection.dispatch(request)
        assert replay['result']['archived'] and replay['result']['destinations']['memory'] == 'accepted'
        assert calls == [operation] and not db.get_session('fresh')['archived']
        await connection.close()


@pytest.mark.asyncio
async def test_canonical_closeout_requires_control_authority(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('old', source='test')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path),
                                     instance_id='current', db=db, epoch=epoch)
        authority.sessions['old'] = LiveSession(None, 'route')
        connection = AuthorityConnection(authority, object(), {'user_id': 'owner', 'capabilities': ['session:read']})
        response = await connection.dispatch({'id': 1, 'method': 'session.closeout', 'params': {
            'session_id': 'old', 'request_id': 'request', 'intent': 'archive'}})
        assert response['error']['data']['reason'] == 'permission_denied'
        assert not db.get_session('old')['archived']
        await connection.close()
