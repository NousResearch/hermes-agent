"""Room participant controls reach the canonical owner without starting a turn."""
import asyncio
import json
from pathlib import Path

import pytest

from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from tui_gateway.hosted_room_driver import HostedRoomBinding


@pytest.fixture
def room_owner(hosted_owner, monkeypatch):
    authority, loop, _, _ = hosted_owner
    authority.profile_id = str(Path(authority.db.db_path).parent)
    monkeypatch.setattr(authority.runner, '_adapter_for_source',
                        lambda source: authority.runner.adapters.get(source.platform), raising=False)
    service = CanonicalHostedRoomService(authority, loop)
    authority.hosted_room_service = service
    home = Path(authority.profile_id)
    monkeypatch.setattr(service, 'profile_homes', lambda: {'default': home, 'other': home / 'profiles' / 'other'})
    monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True})
    service.authorize_room('alice', 'room', create=True)
    room = service.create_room(room_id='room', name='Room', members=[
        {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
        {'member_id': 'two', 'profile': 'other', 'handle': 'two'},
    ])
    producer = service._resolve_member_transport(
        HostedRoomBinding('room', room['authority_gateway_id'], room['authority_epoch']),
        {'payload': {'target_profile': 'default', 'target_member_id': 'one'}})

    def call(*, actor='alice', capabilities=None, **params):
        identity = {'provider': 'local', 'user_id': actor, 'profile_id': authority.profile_id,
                    'instance_id': authority.instance_id,
                    'capabilities': capabilities if capabilities is not None else ['session:read', 'session:control']}
        connection = AuthorityConnection(authority, object(), identity)
        request = {'id': 1, 'method': 'groups.member.yolo',
                   'params': {'room_id': 'room', 'member_id': 'one', **params}}
        return asyncio.run_coroutine_threadsafe(connection.dispatch(request), loop).result(10)

    yield authority, producer, call
    from tools.approval import clear_session
    for live in authority.sessions.values():
        clear_session(live.route)


def test_member_yolo_persists_explicit_retries_without_admission(room_owner, monkeypatch):
    from tools import approval
    from hermes_state_runtime import list_session_admissions

    authority, producer, call = room_owner
    missing = call(enabled=True)
    assert missing['error']['message'] == 'not_found', missing
    assert authority.sessions == {}
    sid = producer.create(profile='default', source='bot_room', title='Group: room')['session_id']
    live = authority.sessions[sid]
    entry = authority.runner.session_store.lookup_by_session_key(live.route)
    assert call()['result']['enabled'] is False
    for enabled in (True, True, False, False):
        result = call(enabled=enabled)['result']
        assert result == {'room_id': 'room', 'member_id': 'one', 'session_key': entry.session_id,
                          'enabled': enabled, 'effective_enabled': enabled}
        assert approval.is_session_yolo_enabled(live.route) is enabled
        store = authority.runner.session_store
        saved = json.loads(store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())[live.route])
        assert saved['yolo_toggle'] is enabled
    # Profile/process bypass is distinct from this participant's explicit override.
    monkeypatch.setattr(approval, '_YOLO_MODE_FROZEN', True)
    result = call()['result']
    assert result['enabled'] is False and result['effective_enabled'] is True
    assert list_session_admissions(authority.db, session_id=sid, pending_only=False) == []


def test_member_yolo_authorization_and_failed_write_leave_bypass_off(room_owner, monkeypatch):
    from tools.approval import is_session_yolo_enabled

    authority, producer, call = room_owner
    sid = producer.create(profile='default', source='bot_room', title='Group: room')['session_id']
    route = authority.sessions[sid].route
    for params in ({'actor': 'bob'}, {'capabilities': ['session:read']},
                   {'member_id': 'missing'}, {'enabled': 'true'}, {'profile': 'foreign'}):
        result = call(**{'enabled': True, **params})
        assert 'error' in result, result
        assert not is_session_yolo_enabled(route)
    assert call(capabilities=['session:read'])['result']['enabled'] is False

    def fail(*args, **kwargs):
        raise OSError('read-only test store')

    with monkeypatch.context() as scoped:
        scoped.setattr(authority.runner.session_store, '_save_entry', fail)
        result = call(enabled=True)
        assert result['error']['message'] == 'storage_unavailable', result
    assert not is_session_yolo_enabled(route)
    assert call()['result']['enabled'] is False
    assert call(enabled=True)['result']['enabled'] is True
