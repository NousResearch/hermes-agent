"""A served login can adopt its own legacy TUI Bot Chat before automation submits to it."""
import json
from pathlib import Path

import pytest

from tests.gateway import test_peer_dm_bot_chat_headless as _headless_gateway
from tests.gateway.test_legacy_stamped_bot_chat import _Transport

headless = _headless_gateway.headless
AUTHOR = _headless_gateway.AUTHOR


def _native_login(authority, login):
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
    from gateway.session_controls import AuthorityConnection
    subject = 'auth:v1:' + json.dumps(['basic', '', login], separators=(',', ':'))
    return AuthorityConnection(authority, _Transport(), {
        'provider': 'local', 'user_id': subject, 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'capabilities': sorted(_PURPOSE_CAPABILITIES['interactive']),
        'native_bootstrap': True, 'profile_scope': 'profile'}, operator=True)


@pytest.mark.asyncio
async def test_same_login_adopts_tui_history_then_peer_and_local_deliveries_keep_its_id(headless):
    from gateway.session_authorities import served_profile_name
    db = headless.authority.db
    sid = 'legacy-tui-bot-chat'
    db.create_session(sid, source='tui', model='fixture', user_id='basic:alice')
    db.set_session_title(sid, 'Bot Chat')
    db.append_message(sid, 'user', 'earlier history')
    native = _native_login(headless.authority, 'alice')
    try:
        resumed = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert 'error' not in resumed, resumed
        assert db.get_session_by_title('Bot Chat')['id'] == sid
        assert db.get_messages_as_conversation(sid)[0]['content'] == 'earlier history'
        peer = await headless.cli.post(f'/api/sessions/{sid}/chat', json={'message': 'peer input', 'author': AUTHOR})
        assert peer.status == 200, await peer.text()
        from gateway.session_bot import deliver
        local = await deliver(native, {
            'id': 'a' * 32, 'profile': served_profile_name(Path(db.db_path).parent),
            'session_id': sid, 'message': 'local or cron input'})
        assert local['session_id'] == sid
        from tests.waiting import async_wait_until
        await async_wait_until(lambda: len(headless.executed) == 2, timeout=10, interval=.05)
        assert [(entry[0], entry[2]) for entry in headless.executed] == [(sid, 'peer input'), (sid, 'local or cron input')]
        assert len(db._read_all('SELECT id FROM sessions')) == 1
    finally:
        await native.close()


@pytest.mark.asyncio
async def test_foreign_login_cannot_adopt_tui_history(headless):
    db = headless.authority.db
    sid = 'legacy-tui-bot-chat'
    db.create_session(sid, source='tui', model='fixture', user_id='basic:alice')
    db.set_session_title(sid, 'Bot Chat')
    native = _native_login(headless.authority, 'bob')
    try:
        response = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert response.get('error', {}).get('message') == 'not_found', response
        assert db.get_session_by_title('Bot Chat')['id'] == sid
        assert not db.get_session(sid)['chat_id']
    finally:
        await native.close()


@pytest.mark.asyncio
async def test_remote_login_cannot_perform_native_adoption(headless):
    from gateway.session_controls import AuthorityConnection
    db = headless.authority.db
    sid = 'remote-tui-bot-chat'
    db.create_session(sid, source='tui', model='fixture', user_id='basic:alice')
    db.set_session_title(sid, 'Bot Chat')
    remote = AuthorityConnection(headless.authority, _Transport(), {
        'provider': 'basic', 'user_id': 'alice', 'profile_id': headless.authority.profile_id,
        'instance_id': headless.authority.instance_id}, operator=True)
    try:
        response = await remote.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert response.get('error', {}).get('message') == 'not_found', response
        assert not db.get_session(sid)['chat_id']
        assert db.get_session_by_title('Bot Chat')['id'] == sid
    finally:
        await remote.close()


@pytest.mark.parametrize('source', ['cli', 'tui', 'gui', 'desktop', 'oneshot', 'tool', 'acp'])
def test_same_login_validation_for_every_legacy_local_source(tmp_path, source):
    from hermes_state import SessionDB
    from hermes_state_local_migration import validate_legacy_row
    from hermes_state_runtime import RuntimeStoreError
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('old', source=source, user_id='basic:alice')
        row = db.get_session('old')
        validate_legacy_row(db, row, subject='auth:v1:["basic","","alice"]')
        for subject in ('uid:1000', 'auth:v1:["basic","","bob"]', 'auth:v1:["basic","issuer","alice"]'):
            with pytest.raises(RuntimeStoreError, match='not_found'):
                validate_legacy_row(db, row, subject=subject)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_same_login_cannot_take_history_while_legacy_desktop_owner_is_live(headless):
    from hermes_cli.active_sessions import try_acquire_active_session, release_active_session
    db = headless.authority.db
    sid = 'busy-tui-bot-chat'
    db.create_session(sid, source='tui', model='fixture', user_id='basic:alice')
    db.set_session_title(sid, 'Bot Chat')
    lease, refusal = try_acquire_active_session(session_id=sid, surface='desktop',
        registry_home=Path(db.db_path).parent, config={})
    assert lease is not None, refusal
    native = _native_login(headless.authority, 'alice')
    try:
        response = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert response.get('error', {}).get('message') == 'runtime_coordination_required', response
        assert not db.get_session(sid)['chat_id']
        assert db.get_session_by_title('Bot Chat')['id'] == sid
    finally:
        release_active_session(lease)
        await native.close()
