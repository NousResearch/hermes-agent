"""A finished room continuation is a member report, never an impersonated human."""
import pytest

from gateway import hosted_room_discussion as discussion, hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


@pytest.fixture
def room(tmp_path):
    path = tmp_path / 'room.db'
    state = hosted_rooms.create_room(path, room_id='room', name='Room', members=[
        {'member_id': 'builder', 'profile': 'build', 'handle': 'builder'},
        {'member_id': 'reviewer', 'profile': 'review', 'handle': 'reviewer'}],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    return path, state


def test_continuation_report_routes_only_mentioned_peer_in_original_thread(room):
    path, state = room
    report = hosted_rooms.append_event(path, room_id='room', event_id='completion:one',
        kind='message.user', actor={'kind': 'member', 'id': 'builder', 'profile': 'build'},
        payload={'text': '@reviewer ready for review', 'thread_id': 'original'},
        authority_gateway_id=state['authority_gateway_id'], authority_epoch=state['authority_epoch'])
    checkpoint = HostedRoomPolicyCheckpoint(path)
    snapshot = checkpoint.snapshot(room_id='room', latest_seq=report['seq'])
    decision = discussion.plan_next_task(state, snapshot.events, local_profiles=('build', 'review'))
    assert decision.status == 'task'
    assert decision.task is not None
    assert decision.task.member.member_id == 'reviewer'
    assert decision.task.identity.thread_id == 'original'
    assert '@builder: @reviewer ready for review' in decision.task.payload['prompt']
    assert 'User (user):' not in decision.task.payload['prompt']


def test_continuation_without_peer_mention_does_not_broadcast(room):
    path, state = room
    hosted_rooms.append_event(path, room_id='room', event_id='completion:quiet', kind='message.user',
        actor={'kind': 'member', 'id': 'builder', 'profile': 'build'},
        payload={'text': '@builder finished', 'thread_id': 'original'},
        authority_gateway_id=state['authority_gateway_id'], authority_epoch=state['authority_epoch'])
    events = hosted_rooms.read_events(path, room_id='room')['events']
    decision = discussion.plan_next_task(state, events, local_profiles=('build', 'review'))
    assert decision.status == 'settled'


def test_forged_completion_member_identity_is_refused(room):
    path, state = room
    with pytest.raises((discussion.DiscussionValidationError, hosted_rooms.HostedRoomError)):
        hosted_rooms.append_event(path, room_id='room', event_id='completion:forged', kind='message.user',
            actor={'kind': 'member', 'id': 'builder', 'profile': 'review'},
            payload={'text': '@reviewer ready', 'thread_id': 'original'},
            authority_gateway_id=state['authority_gateway_id'], authority_epoch=state['authority_epoch'])
        discussion.plan_next_task(state, hosted_rooms.read_events(path, room_id='room')['events'],
                                  local_profiles=('build', 'review'))


def test_background_work_captures_origin_admission():
    from gateway.session_display import submission_turn_scope
    from tools.async_delegation import _capture_routing_origin
    from tools.process_registry import ProcessRegistry
    with submission_turn_scope('admission-original', 'hosted:original'):
        assert _capture_routing_origin()['origin_admission_id'] == 'admission-original'
        process = ProcessRegistry._new_session('make test', 'task', 'task', 'route', None)
        assert ProcessRegistry._watch_event_base(process)['origin_admission_id'] == 'admission-original'


@pytest.mark.parametrize('final_text', ['@reviewer ready', '界' * 24000], ids=['short', 'long-utf8'])
def test_room_completion_is_read_back_as_final_parent_report(hosted_owner, monkeypatch, final_text):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input, claim_session_input, settle_session_input
    authority, loop, principal, _ = hosted_owner
    authority.runner._adapter_for_source = lambda source: authority.runner.adapters[source.platform]
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='builder', profile='default',
                                principal=principal, authorize=lambda *args: True)
    coords = {'profile': 'default', 'source': 'bot_room'}
    sid = rpc.create(**coords, title='Group: room')['session_id']
    original = rpc.submit(**coords, session_id=sid, prompt='start work',
        task=TaskIdentity('room', 'task', 'original', 'turn'), execution_generation=1,
        on_terminal=lambda row: None)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    assert isinstance(row, dict)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
        generation=row['generation'], outcome='completed', result={'result': {'final_response': 'working'}})
    live = authority.sessions[sid]
    descriptor = {'identity': '["async_delegation","child",""]', 'owner': sid, 'route': live.route,
                  'target': sid, 'origin_admission_id': original['admission_id'], 'resumes_parent': True}
    admitted = admit_session_input(authority.db, epoch=authority.epoch, session_id=sid,
        principal_id='automation:' + live.route, request_id=descriptor['identity'],
        payload={'text': 'PRIVATE CHILD OUTPUT', 'local_automation_v1': descriptor})
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    assert isinstance(row, dict)
    from gateway.session_completion_notes import note_if_unviewed
    monkeypatch.setattr('gateway.session_hosted_completions.completion_allowed', lambda *args: True)
    assert asyncio_run(loop, note_if_unviewed(authority, SessionRef(authority.profile_id, sid), row)) is None
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=admitted['admission_id'],
        generation=row['generation'], outcome='completed', result={'result': {'final_response': final_text}})
    reports = rpc.completions(**coords, session_id=sid, after_seq=0)
    assert reports['cursor'] == admitted['seq']
    assert len(reports['reports']) == 1
    text = reports['reports'][0]['text']
    assert len(text.encode('utf-8')) <= discussion.MAX_USER_TEXT_BYTES
    assert discussion.validate_user_payload({'text': text, 'thread_id': 'original'})['text'] == text
    if final_text == '@reviewer ready':
        assert text == final_text
    assert reports['reports'][0]['task']['thread_id'] == 'original'
    assert 'PRIVATE CHILD OUTPUT' not in str(reports)
    assert rpc.completions(**coords, session_id=sid, after_seq=reports['cursor'])['reports'] == []


def asyncio_run(loop, coroutine):
    import asyncio
    return asyncio.run_coroutine_threadsafe(coroutine, loop).result(10)


def test_pruned_origin_task_refuses_local_completion(room, monkeypatch):
    from types import SimpleNamespace
    from gateway.hosted_room_driver import TaskConflictError, TaskIdentity
    from gateway import session_hosted_completions as completions
    from hermes_state import SessionDB
    path, _ = room
    task = TaskIdentity('room', 'pruned', 'original', 'turn')
    monkeypatch.setattr(completions, 'room_origin', lambda *args: ({'principal_id': 'alice'}, task, 1))
    def missing(*args):
        raise TaskConflictError('task not found')
    monkeypatch.setattr('gateway.hosted_room_driver.get_task', missing)
    with SessionDB(path) as db:
        authority = SimpleNamespace(db=db, hosted_room_service=SimpleNamespace(
            db_path=path, _owner=lambda room_id: 'alice'))
        assert completions.completion_allowed(authority, SimpleNamespace(session_id='session'), {}) is False


@pytest.mark.parametrize('revoked', [None, 'stop', 'member', 'disband', 'generation', 'superseded', 'authority', 'stop-race'])
def test_final_report_publishes_once_and_respects_room_fences(room, revoked, monkeypatch):
    import json
    import time
    from types import SimpleNamespace
    from gateway import hosted_room_driver as tasks
    from gateway.session_hosted_completions import publish_report
    path, state = room
    source = hosted_rooms.append_event(path, room_id='room', event_id='start', kind='message.user',
        actor={'kind': 'user', 'id': 'desktop'}, payload={'text': '@builder work', 'thread_id': 'original'},
        authority_gateway_id=state['authority_gateway_id'], authority_epoch=1)
    decision = discussion.plan_next_task(state, [source], local_profiles=('build', 'review'))
    assert decision.task is not None
    plan = decision.task
    tasks.admit_task(path, plan.identity, payload=dict(plan.payload), clock=time.time)
    lease = tasks.acquire_lease(path, room_id='room', gateway_id=state['authority_gateway_id'],
        authority_epoch=1, process_generation='test', ttl_seconds=60, clock=time.time)
    tasks.start_task(path, plan.identity, lease, expected_cancel_generation=0, clock=time.time)
    selector = {'room_id': 'room', 'member_id': 'builder', 'profile': 'build'}
    with __import__('sqlite3').connect(path) as conn:
        conn.execute("UPDATE hosted_room_driver_tasks SET status='settled'")
    hosted_rooms.append_event(path, room_id='room', event_id='source-done', kind='room.activity',
        actor={'kind': 'gateway', 'id': state['authority_gateway_id']},
        payload={'status': 'settled', 'reason_code': 'all_members_passed',
                 'thread_id': 'original', 'discussion_event_id': source['event_id']},
        authority_gateway_id=state['authority_gateway_id'], authority_epoch=1)
    service = SimpleNamespace(db_path=path, _owner=lambda room_id: 'alice')
    report = {'admission_id': 'completion-1', 'task': __import__('dataclasses').asdict(plan.identity),
              'execution_generation': 1, 'text': '@reviewer ready'}
    if revoked in ('stop', 'superseded'):
        hosted_rooms.append_event(path, room_id='room', event_id='fence',
            kind='room.stop_requested' if revoked == 'stop' else 'message.user',
            actor={'kind': 'gateway', 'id': state['authority_gateway_id']} if revoked == 'stop'
                  else {'kind': 'user', 'id': 'desktop'},
            payload={} if revoked == 'stop' else {'text': 'new request', 'thread_id': 'original'},
            authority_gateway_id=state['authority_gateway_id'], authority_epoch=1)
    elif revoked == 'stop-race':
        append = hosted_rooms.append_event
        def stopped_before_commit(*args, **kwargs):
            append(path, room_id='room', event_id='racing-stop', kind='room.stop_requested',
                actor={'kind': 'gateway', 'id': state['authority_gateway_id']}, payload={'cancel_id': 'race'},
                authority_gateway_id=state['authority_gateway_id'], authority_epoch=1)
            return append(*args, **kwargs)
        monkeypatch.setattr(hosted_rooms, 'append_event', stopped_before_commit)
    elif revoked:
        import sqlite3
        with sqlite3.connect(path) as conn:
            if revoked == 'member':
                conn.execute('UPDATE hosted_rooms SET members_json=?', (json.dumps(state['members'][1:]),))
            elif revoked == 'disband':
                conn.execute('UPDATE hosted_rooms SET disbanded_at=1')
            elif revoked == 'authority':
                conn.execute('UPDATE hosted_rooms SET authority_epoch=2')
            else:
                conn.execute('UPDATE hosted_room_driver_tasks SET execution_generation=2')
    first = publish_report(service, selector, report)
    again = publish_report(service, selector, report)
    if revoked:
        assert first is None and again is None
    else:
        assert first is not None and again is not None
        assert first['payload']['thread_id'] == 'original'
        assert first['actor']['kind'] == 'member'
        assert again['seq'] == first['seq']
        page = hosted_rooms.read_events(path, room_id='room')['events']
        assert sum(e['event_id'] == first['event_id'] for e in page) == 1
        routed = discussion.plan_next_task(state, page, local_profiles=('build', 'review'))
        assert routed.task is not None and routed.task.member.member_id == 'reviewer'
