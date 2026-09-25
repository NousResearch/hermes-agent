"""Exact approval rights and original-runtime reads over the real state RPC.

Only transport/coordinator boundaries are inert; no runtime is started.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as tasks
from gateway.session_hosted_service import CanonicalHostedRoomService
from tests.gateway.test_canonical_peer_backoff_retry import LocalRPC, Target, current, rpc, tick


@pytest.fixture
def projection_case(tmp_path, monkeypatch):
    # Text-only equivalent of the inert peer Retry setup: the lower Route owner
    # intentionally has no Output attachment prerequisite.
    import time
    from gateway import hosted_rooms as rooms
    from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping, issue_room_grant
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    from tui_gateway.hosted_room_driver import HostedRoomRuntime
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError('runtime/listener/network startup forbidden')
    monkeypatch.setattr(HostedRoomRuntime, 'start', forbidden)
    monkeypatch.setattr('tui_gateway.hosted_room_peer_http._open_roomlink_url', forbidden)
    with SessionDB(tmp_path / 'state.db') as db:
        runner = SimpleNamespace(_draining=False)
        authority = SessionAuthority(runner, db=db, profile_id=str(tmp_path), instance_id='fixture',
            epoch=begin_runtime_epoch(db, instance_id='fixture'))
        runner.session_authority = authority
        service = CanonicalHostedRoomService(authority, None)
        authority.hosted_room_service = service
        connection = AuthorityConnection(authority, SimpleNamespace(), {'user_id': 'alice'})
        service.authorize_room(connection.actor.subject, 'room', create=True)
        service.local_profiles = lambda: ('default',)
        catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(installation_id='peer-install',
            persistent_process=True, target_profile='reviewer', attachments=True))
        gateway = rooms.local_authority_gateway_id()
        scope = dict(room_id='room', home_install_id=gateway, authority_gateway_id=gateway,
            authority_epoch=1, member_id='peer', target_profile='reviewer')
        peer = Target(catalog, scope)
        peer.mode = 'refused-dispatch'
        grant = issue_room_grant(peer.secret, grant_id='grant', **scope,
            target_install_id=catalog.installation_id,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            permissions=('dispatch', 'status', 'stop', 'attachment.stage'), ttl_seconds=3600)
        route = PeerMemberRoute(home_install_id=gateway, member_id='peer',
            target_install_id=catalog.installation_id, target_profile='reviewer',
            capability_digest=catalog.catalog_digest,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            cancellation_scope_id='cancel', trace_id='trace', grant=grant, attachments=True)
        service.create_room(room_id='room', name='Projection', members=[
            dict(member_id='peer', profile='reviewer', handle='peer', target=dict(kind='peer',
                peer_id='peer-install', installation_id='peer-install', profile='reviewer',
                capability_digest=catalog.catalog_digest)),
            dict(member_id='healthy', profile='default', handle='healthy')])
        service.register_peer_route(room_id='room', member_id='peer', route=route, client=peer,
            target_url=peer.base_url, catalog=catalog)
        service.member_rpcs[('room', 'healthy', 'default', connection.actor.subject, str(tmp_path))] = LocalRPC()
        service.runtime.clock = time.time
        service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        service.send(room_id='room', event_id='request', payload=dict(thread_id='thread', text='Review together'))
        original = tasks.list_tasks(db.db_path, room_id='room')[0]
        try:
            yield SimpleNamespace(service=service, connection=connection, authority=authority,
                                  original=original, peer=peer)
        finally:
            service.runtime._thread = None


@pytest.fixture
def populated(projection_case):
    c = projection_case
    tick(c, 4)
    assert tasks.is_proven_nonadmission(current(c))
    c.service.send(room_id='room', event_id='approval-turn',
        payload=dict(thread_id='approval-thread', text='@healthy review this'))
    task, = tasks.list_tasks(c.service.db_path, room_id='room', status='queued')
    runtime = c.service.runtime
    tasks.start_task(c.service.db_path, task['identity'], runtime._leases['room'],
        expected_cancel_generation=0, clock=runtime.clock)
    task = tasks.get_task(c.service.db_path, task['identity'])
    assert task['status'] == 'running' and task['execution_generation'] == 1
    runtime._report_pending_action(task, session_id='approval-session', info=dict(
        run_id='approval-run', pending_approval=dict(request_id='approval-request',
            choices=['once', 'deny'], command='review fixture')))
    c.approval = dict(c.service._pending_actions[('room', 'healthy')])
    assert c.approval['kind'] == 'approval'
    assert c.approval['task_id'] == task['identity'].task_id
    assert c.approval['execution_generation'] == task['execution_generation']
    return c


@pytest.mark.parametrize('rights', [(), ('session:approve',), ('session:control',),
                                    ('session:approve', 'session:control')],
                         ids=['read', 'approve', 'control', 'both'])
@pytest.mark.parametrize('phase', ['projection', 'final_filter'])
def test_state_actions_match_independent_rpc_capabilities(populated, monkeypatch, rights, phase):
    c = populated
    c.connection.actor = replace(c.connection.actor,
        capabilities=frozenset(('session:read', *rights)))
    allowed = ({'approval'} if 'session:approve' in rights else set()) | (
        {'retry', 'discard'} if 'session:control' in rights else set())
    real_status = c.service.status
    def status(*args, **kwargs):
        result = real_status(*args, **kwargs)
        if phase == 'projection':
            assert {a['kind'] for a in result['pending_actions']
                    if a['kind'] not in {'output_retry', 'output_cleanup'}} <= allowed
        else:
            # A stale inherited projection must also be filtered on read exit.
            result['pending_actions'] = [c.approval,
                dict(kind='retry', task_id='stale-retry'),
                dict(kind='discard', task_id='stale-discard')]
        return result
    monkeypatch.setattr(c.service, 'status', status)
    reply = rpc(c, 'groups.state')
    actions = reply['result']['driver_status']['pending_actions']
    expected = allowed if phase == 'final_filter' else allowed - {'discard'}
    assert {a['kind'] for a in actions
            if a['kind'] not in {'output_retry', 'output_cleanup'}} == expected
    assert [a for a in actions if a['kind'] == 'approval'] == (
        [c.approval] if 'session:approve' in rights else [])
    approved = []
    c.service.rpc = SimpleNamespace(approve=lambda **kw: approved.append(kw) or {'approved': True})
    request = {key: c.approval[key] for key in (
        'member_id', 'task_id', 'execution_generation', 'request_id')}
    response = rpc(c, 'groups.approve', dict(room_id='room', choice='once', **request))
    if 'session:approve' in rights:
        assert 'result' in response, response
        assert approved == [dict(session_id='approval-session', request_id='approval-request', choice='once')]
    else:
        assert response['error']['data']['reason'] == 'permission_denied'
        assert approved == []
    if 'session:control' not in rights:
        for method in ('groups.retry', 'groups.discard'):
            response = rpc(c, method, dict(room_id='room', **{k: request[k] for k in (
                'member_id', 'task_id', 'execution_generation')}))
            assert response['error']['data']['reason'] == 'permission_denied'


@pytest.mark.parametrize('withdraw_at', ['before', 'after_projection'])
def test_withdrawal_removes_populated_approval_and_retry(populated, monkeypatch, withdraw_at):
    c = populated
    real_status = c.service.status
    def status(*args, **kwargs):
        result = real_status(*args, **kwargs)
        assert {'approval', 'retry'} <= {a['kind'] for a in result['pending_actions']}
        c.authority.runner._draining = True
        return result
    if withdraw_at == 'before':
        c.authority.runner._draining = True
    else:
        monkeypatch.setattr(c.service, 'status', status)
    reply = rpc(c, 'groups.state')
    assert reply['result']['room']['room_id'] == 'room'
    assert not any(a['kind'] in {'approval', 'retry', 'discard'}
        for a in reply['result']['driver_status']['pending_actions'])


@pytest.mark.parametrize('replace_at', ['before', 'between_status', 'lease_validation', 'after_projection'])
def test_state_never_observes_replacement_runtime(populated, monkeypatch, replace_at):
    c = populated
    original = c.service.runtime
    replacement = CanonicalHostedRoomService(c.authority, None).runtime
    assert replacement is not original
    original._blocked_rooms.add('room')
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('replacement runtime must never supply status or leases')
    class ForeignLeases(dict):
        get = forbidden
    monkeypatch.setattr(replacement, 'status', forbidden)
    monkeypatch.setattr(replacement, '_leases', ForeignLeases())
    monkeypatch.setattr(replacement, 'clock', forbidden)
    real_status = c.service.status
    def status(*args, **kwargs):
        if replace_at == 'between_status':
            c.service.runtime = replacement
        result = real_status(*args, **kwargs)
        if replace_at == 'after_projection':
            c.service.runtime = replacement
        return result
    monkeypatch.setattr(c.service, 'status', status)
    if replace_at == 'before':
        c.service.runtime = replacement
    if replace_at == 'lease_validation':
        from gateway import session_hosted_peer_retry
        validate = session_hosted_peer_retry._validate
        def replace_after_validation(*args, **kwargs):
            result = validate(*args, **kwargs)
            c.service.runtime = replacement
            return result
        monkeypatch.setattr(session_hosted_peer_retry, '_validate', replace_after_validation)
    original._wake.clear()
    try:
        response = rpc(c, 'groups.state')
        result = response['result']
        assert result['room']['room_id'] == 'room'
        assert result['driver_status']['running'] is True
        assert result['driver_status']['blocked'] is True
        assert result['driver_status']['counts']['deferred'] == 1
        assert result['driver_status']['counts']['running'] == 1
        assert not any(a['kind'] in {'approval', 'retry', 'discard'}
            for a in result['driver_status']['pending_actions'])
        assert not calls and not original._wake.is_set() and not replacement._wake.is_set()
    finally:
        c.service.runtime = original
