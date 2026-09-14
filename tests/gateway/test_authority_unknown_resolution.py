"""An owner restart leaves a started admission `unknown`; a client must be able to acknowledge it."""
import asyncio
import queue
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import SessionRef, Submission
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import claim_session_input, list_session_admissions


class Peer:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


async def _restarted_owner_with_unknown_head(tmp_path, monkeypatch, executed):
    """Real authority + SQLite: a started turn interrupted by an owner epoch change plus a queued follower."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db

    async def answer(event):
        executed.append(event.text)
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='unknown-head')
    db.create_session('s', source='telegram')
    first = await initialize_session_authority(runner, profile_id='p', instance_id='owner-1')
    first.sessions['s'] = LiveSession(source, 's')
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(first, Peer(), {'user_id': 'human'}).actor
    # Admit the head and claim it without draining: the owner dies mid-turn.
    first._schedule = lambda ref: None
    head = await first.submit(actor, Submission('head', ref, {'text': 'HEAD'}, 'queue'))
    follower = await first.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))
    claimed = claim_session_input(db, epoch=first.epoch, session_id='s')
    assert claimed['admission_id'] == head.admission_id
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')
    authority.sessions['s'] = LiveSession(source, 's')
    rows = {r['admission_id']: r for r in list_session_admissions(db, session_id='s')}
    assert rows[head.admission_id]['status'] == 'unknown' and rows[follower.admission_id]['status'] == 'queued'
    return store, authority, ref, rows[head.admission_id], follower


@pytest.mark.asyncio
async def test_resolve_unknown_releases_the_paused_follower_exactly_once(tmp_path, monkeypatch):
    executed = []
    store, authority, ref, unknown, follower = await _restarted_owner_with_unknown_head(tmp_path, monkeypatch, executed)
    peer = Peer()
    viewer = AuthorityConnection(authority, peer, {'user_id': 'human'})
    try:
        resumed = (await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}}))['result']
        assert [r['status'] for r in resumed['pending']] == ['unknown', 'queued']
        # The follower stays paused behind the unknown head; nothing runs on its own.
        await asyncio.sleep(0.05)
        assert executed == []
        reply = await viewer.dispatch({'id': 2, 'method': 'prompt.resolve_unknown', 'params': {
            'session_id': 's', 'admission_id': unknown['admission_id'],
            'execution_generation': unknown['generation']}})
        receipt = reply['result']
        assert receipt['status'] == 'terminal' and receipt['outcome'] == 'interrupted', reply
        await authority.sessions['s'].task
        assert executed == ['FOLLOWER']
        statuses = {r['admission_id']: (r['status'], r['outcome']) for r in
                    list_session_admissions(authority.db, session_id='s', pending_only=False)}
        assert statuses[follower.admission_id] == ('terminal', 'completed')
        # The viewer learns about the release through the ordinary pending fanout.
        infos = []
        while not infos or infos[-1]['params']['type'] != 'message.complete':
            infos.append(await asyncio.to_thread(peer.frames.get, True, 5))
        first_info = next(f for f in infos if f['params']['type'] == 'session.info')
        assert all(r['status'] != 'unknown' for r in first_info['params']['payload']['pending'])
        # Acknowledging twice is not a second resolution.
        again = await viewer.dispatch({'id': 3, 'method': 'prompt.resolve_unknown', 'params': {
            'session_id': 's', 'admission_id': unknown['admission_id'],
            'execution_generation': unknown['generation']}})
        assert again['error']['data']['reason'] == 'stale_generation'
        assert executed == ['FOLLOWER']
    finally:
        await viewer.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_resolve_unknown_is_fenced_by_generation_and_capability(tmp_path, monkeypatch):
    executed = []
    store, authority, ref, unknown, follower = await _restarted_owner_with_unknown_head(tmp_path, monkeypatch, executed)
    owner = AuthorityConnection(authority, Peer(), {'user_id': 'human'})
    reader = AuthorityConnection(authority, Peer(), {'user_id': 'human', 'capabilities': ['session:read', 'session:submit']})
    try:
        for connection in (owner, reader):
            await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        params = {'session_id': 's', 'admission_id': unknown['admission_id'], 'execution_generation': unknown['generation']}
        denied = await reader.dispatch({'id': 2, 'method': 'prompt.resolve_unknown', 'params': params})
        assert denied['error']['data']['reason'] == 'permission_denied'
        stale = await owner.dispatch({'id': 3, 'method': 'prompt.resolve_unknown',
                                     'params': {**params, 'execution_generation': unknown['generation'] + 1}})
        assert stale['error']['data']['reason'] == 'stale_generation'
        foreign = await owner.dispatch({'id': 4, 'method': 'prompt.resolve_unknown',
                                       'params': {**params, 'session_id': 'other'}})
        assert 'error' in foreign
        await asyncio.sleep(0.05)
        rows = {r['admission_id']: r['status'] for r in list_session_admissions(authority.db, session_id='s')}
        assert rows[unknown['admission_id']] == 'unknown' and rows[follower.admission_id] == 'queued'
        assert executed == []
    finally:
        await owner.close()
        await reader.close()
        store.close_all_db_handles()
