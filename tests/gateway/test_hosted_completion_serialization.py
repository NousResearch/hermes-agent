"""Completion reports queue behind their source discussion and each prior peer handoff."""
import time
import pytest
from dataclasses import asdict
from types import SimpleNamespace

from gateway import hosted_rooms, hosted_room_discussion as discussion, hosted_room_driver as tasks
from hermes_state import SessionDB
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from tests.gateway import test_hosted_room_completions as fixtures

room = fixtures.room


def test_bounded_discussion_keeps_report_pending_until_settled(room):
    from contextlib import closing
    from gateway.session_hosted_completions import _require_idle_thread, _ReportPending
    path, state = room
    gateway = state['authority_gateway_id']
    hosted_rooms.append_event(path, room_id='room', event_id='start', kind='message.user',
        actor={'kind': 'user', 'id': 'desktop'}, payload={'text': '@builder work', 'thread_id': 'original'},
        authority_gateway_id=gateway, authority_epoch=1)
    def activity(event_id, status, reason):
        hosted_rooms.append_event(path, room_id='room', event_id=event_id, kind='room.activity',
            actor={'kind': 'gateway', 'id': gateway}, payload={'status': status, 'reason_code': reason,
                'thread_id': 'original', 'discussion_event_id': 'start'},
            authority_gateway_id=gateway, authority_epoch=1)
    activity('capped', 'bounded', 'max_rounds')
    with closing(hosted_rooms._read_connection(path)) as conn:
        with pytest.raises(_ReportPending):
            _require_idle_thread(conn, 'room', 'original')
    activity('done', 'settled', 'all_members_passed')
    assert tasks.list_tasks(path, room_id='room') == []
    with closing(hosted_rooms._read_connection(path)) as conn:
        _require_idle_thread(conn, 'room', 'original')


def test_reports_on_one_thread_preserve_original_and_each_peer_handoff(room):
    from gateway.session_hosted_completions import _sweep_member
    path, state = room
    gateway = state['authority_gateway_id']
    source = hosted_rooms.append_event(path, room_id='room', event_id='start', kind='message.user',
        actor={'kind': 'user', 'id': 'desktop'}, payload={'text': '@builder work', 'thread_id': 'original'},
        authority_gateway_id=gateway, authority_epoch=1)
    plan = discussion.plan_next_task(state, [source], local_profiles=('build', 'review')).task
    assert plan is not None
    tasks.admit_task(path, plan.identity, payload=dict(plan.payload), clock=time.time)
    lease = tasks.acquire_lease(path, room_id='room', gateway_id=gateway, authority_epoch=1,
        process_generation='test', ttl_seconds=60, clock=time.time)
    attempt = tasks.start_task(path, plan.identity, lease, expected_cancel_generation=0, clock=time.time)
    reports = [{'admission_id': f'completion-{i}', 'task': asdict(plan.identity),
                'execution_generation': 1, 'text': f'@reviewer report {i}'} for i in (1, 2)]
    rpc = SimpleNamespace(ref=SimpleNamespace(profile_id='build'),
        resolve_exact=lambda **kw: {'session_id': 'member-session'},
        completions=lambda **kw: {'cursor': 2, 'reports': reports})
    binding = SimpleNamespace(room_id='room')
    member = {'member_id': 'builder', 'profile': 'build'}
    with SessionDB(path) as db:
        service = SimpleNamespace(db_path=path, authority=SimpleNamespace(db=db),
            _resolve_member_transport=lambda *args: rpc)
        def sweep():
            _sweep_member(service, binding, member, {})
        def events():
            return hosted_rooms.read_events(path, room_id='room')['events']
        checkpoint = HostedRoomPolicyCheckpoint(path)
        def next_peer():
            snapshot = checkpoint.snapshot(room_id='room', latest_seq=events()[-1]['seq'])
            return discussion.plan_next_task(state, snapshot.events, local_profiles=('build', 'review')).task
        def finish(plan, text):
            publication = discussion.plan_publication(state, events(), plan, status='settled',
                result={'text': text}, local_profiles=('build', 'review'))
            for event in publication.events:
                hosted_rooms.append_event(path, **event.append_kwargs('room'))
            hosted_rooms.append_event(path, room_id='room', event_id='done:' + plan.discussion_event_id,
                kind='room.activity', actor={'kind': 'gateway', 'id': gateway},
                payload={'status': 'settled', 'reason_code': 'all_members_passed',
                         'thread_id': 'original', 'discussion_event_id': plan.discussion_event_id},
                authority_gateway_id=gateway, authority_epoch=1)
        sweep()
        assert not [e for e in events() if e['actor']['kind'] == 'member']
        tasks.settle_task(path, attempt, settlement_id='settle', status='settled',
                          result={'text': 'Original reply'}, clock=time.time)
        sweep()
        assert not [e for e in events() if e['actor']['kind'] == 'member']
        finish(plan, 'Original reply')
        sweep()
        assert [e['payload']['text'] for e in events() if e['kind'] == 'message.user'] == [
            '@builder work', '@reviewer report 1']
        peer = next_peer()
        assert peer is not None and peer.member.member_id == 'reviewer'
        finish(peer, 'Review one')
        sweep()
        peer = next_peer()
        assert peer is not None and peer.member.member_id == 'reviewer'
        assert peer.discussion_event_id != plan.discussion_event_id
        assert '@reviewer report 2' in peer.payload['prompt']
        assert any(e['payload'].get('text') == 'Original reply' for e in events())
        assert len([e for e in events() if e['event_id'].startswith('completion:')]) == 2
