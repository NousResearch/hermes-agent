"""Background results traverse real profile sockets, admissions and room routing without a Desktop."""
import pytest

from tests.gateway import test_hosted_mux_runtime as fixtures
from tests.waiting import wait_until

mux = fixtures.mux


@pytest.mark.parametrize('producer', ['async_delegation', 'completion'])
@pytest.mark.parametrize('boundary', ['active', 'stop', 'unserved', 'cursor-recovery', 'superseded', 'disband'])
def test_background_report_returns_to_room_and_wakes_mentioned_peer(mux, monkeypatch, producer, boundary):
    from gateway import hosted_rooms, session_finite

    from gateway.session_authorities import owner_scope
    from gateway.session_authority import SessionAuthority
    from gateway.session_hosted_service import ensure_hosted_service
    from hermes_state_runtime import list_session_admissions

    runner, homes, _, call = mux
    actual_execute = session_finite.execute_finite_admission
    executions = []
    handled = []

    async def execute(authority, ref, row):
        executions.append((authority.profile_id, row['request_id']))
        if 'local_automation_v1' in row['payload']:
            return await actual_execute(authority, ref, row)
        return 'Work dispatched' if authority.profile_id == str(homes['beta']) else 'Review received'

    async def handle(event):
        assert event.internal is True
        handled.append(event.text)
        return '@reviewer implementation ready'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(runner, '_handle_message', handle)
    for authority in runner.session_authorities:
        monkeypatch.setattr(authority, '_schedule', SessionAuthority._schedule.__get__(authority))
    call(ensure_hosted_service(runner))
    source = runner.session_authorities.require(homes['alpha'])
    target = runner.session_authorities.require(homes['beta'])
    service = source.hosted_room_service
    with owner_scope(source):
        service.authorize_room('alice', 'room', create=True)
        service.create_room(room_id='room', name='Room', members=[
            {'member_id': 'reviewer', 'profile': 'alpha', 'handle': 'reviewer'},
            {'member_id': 'builder', 'profile': 'beta', 'handle': 'builder'}])
        if boundary == 'unserved':
            service.send(room_id='room', event_id='prior', payload={'text': '@reviewer context', 'thread_id': 'prior'})
    if boundary == 'unserved':
        assert wait_until(lambda: any(e['kind'] == 'room.activity' for e in
            hosted_rooms.read_events(source.db.db_path, room_id='room')['events']), timeout=10)
    with owner_scope(source):
        service.send(room_id='room', event_id='start', payload={'text': '@builder work', 'thread_id': 'original'})

    def events():
        return hosted_rooms.read_events(source.db.db_path, room_id='room', include_disbanded=True)['events']

    assert wait_until(lambda: any(e['kind'] == 'room.activity' and e['payload']['thread_id'] == 'original'
                                for e in events()), timeout=10)
    sid, = target.sessions
    original, = list_session_admissions(target.db, session_id=sid, pending_only=False)
    live = target.sessions[sid]
    assert all(actor.transport_id == 'hosted-owner-transport' for actor in live.subscribers.values())
    if boundary == 'unserved':
        profile_homes = service.profile_homes
        monkeypatch.setattr(service, 'profile_homes', lambda: {
            name: home for name, home in profile_homes().items() if name != 'alpha'})
    if boundary == 'stop':
        with owner_scope(source):
            service.stop_room('room', cancel_id='user-stop')
    if boundary == 'superseded':
        with owner_scope(source):
            service.send(room_id='room', event_id='new', payload={'text': '@reviewer new request', 'thread_id': 'original'})
    if boundary == 'disband':
        state = service._room('room')
        hosted_rooms.disband_room(service.db_path, room_id='room',
            expected_gateway_id=state['authority_gateway_id'], expected_epoch=state['authority_epoch'])
    if boundary == 'cursor-recovery':
        source.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER fail_completion_cursor BEFORE INSERT "
            "ON state_meta WHEN NEW.key LIKE 'gateway.hosted.completion_cursor.v1:%' "
            "BEGIN SELECT RAISE(ABORT,'fixture crash after publication'); END"))

    async def offer():
        with owner_scope(target):
            event = {'type': producer, 'session_key': live.route, 'parent_session_id': sid,
                     'origin_admission_id': original['admission_id']}
            if producer == 'async_delegation':
                from tests.gateway.test_completion_admission import pending
                event.update(pending(live.route, 'deleg-room'))
                return await runner._deliver_async_delegation_group([event])
            event.update(session_id='proc-room', started_at=1.0, command='make', exit_code=0,
                         output='PRIVATE PROCESS OUTPUT')
            return await runner._deliver_completion_notification('PRIVATE PROCESS OUTPUT', event)

    assert call(offer()) is True
    if boundary in {'stop', 'superseded', 'disband'}:
        assert wait_until(lambda: len(list_session_admissions(target.db, session_id=sid, pending_only=False)) == 2
                          and list_session_admissions(target.db, session_id=sid, pending_only=False)[-1]['status'] == 'terminal',
                          timeout=10)
        assert handled == []
        assert not any(e['payload'].get('text') == '@reviewer implementation ready' for e in events())
        return
    if boundary == 'unserved':
        assert wait_until(lambda: any(e['payload'].get('text') == '@reviewer implementation ready' for e in events()), timeout=10)
        return
    if boundary == 'cursor-recovery':
        assert wait_until(lambda: any(e['payload'].get('text') == '@reviewer implementation ready' for e in events()), timeout=10)
        source.db._execute_write(lambda conn: conn.execute('DROP TRIGGER fail_completion_cursor'))
    assert wait_until(lambda: any(e['payload'].get('text') == 'Review received' for e in events()), timeout=15)
    reports = [e for e in events() if e['payload'].get('text') == '@reviewer implementation ready']
    assert len(reports) == 1
    assert reports[0]['actor']['id'] == 'builder'
    assert reports[0]['actor']['kind'] == 'member'
    assert reports[0]['payload']['thread_id'] == 'original'
    assert 'PRIVATE PROCESS OUTPUT' not in str(events())
    assert sum(home == str(homes['alpha']) for home, _ in executions) == 1
    assert sum(home == str(homes['beta']) for home, _ in executions) == 2
    # A lost producer acknowledgement cannot rerun the parent or post another report.
    call(offer())
    service._prepare_with_reports(next(b for b in service.bindings() if b.room_id == 'room'))
    assert len([e for e in events() if e['payload'].get('text') == '@reviewer implementation ready']) == 1
